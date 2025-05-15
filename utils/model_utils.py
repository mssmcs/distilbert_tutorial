# utils/model_utils.py
import os
import time
import torch
import numpy as np
from datetime import datetime
from transformers import (
    DistilBertForSequenceClassification,
    Trainer, TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

def create_model(model_name: str = "distilbert-base-uncased", num_labels: int = 2) -> DistilBertForSequenceClassification:
    """
    Create and initialize a DistilBERT model.
    
    Args:
        model_name: Name of the pre-trained model to use
        num_labels: Number of output labels
    
    Returns:
        Initialized model
    """
    print(f"Initializing DistilBERT model for sequence classification (num_labels={num_labels})...")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Print model architecture details
    print(f"Model created: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print model structure summary
    print("\nModel structure:")
    print("  DistilBERT base model")
    print("  → Transformer with 6 layers")
    print("  → Each layer has self-attention and feed-forward components")
    print("  → Pre-trained on masked language modeling task")
    print("  → Sequence classification head on top")
    
    return model

def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for prediction.
    
    Args:
        pred: Prediction outputs from the model (predictions can be logits or labels)
    
    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    predictions = pred.predictions
    
    # Debug: Print input shapes and samples
    print(f"eval_pred.label_ids shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"eval_pred.predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
    print(f"eval_pred.predictions sample: {predictions[:5]}")
    
    # If predictions are logits (2D array), convert to labels
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=-1)
    elif predictions.ndim != 1:
        raise ValueError(f"Expected 1D labels or 2D logits array, got shape {predictions.shape}")
    
    # Ensure predictions are integers
    predictions = predictions.astype(np.int64)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    # Add more detailed metrics
    conf_mat = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = conf_mat.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def create_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    logging_steps: int = 10,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    seed: int = 42
):
    """
    Create training arguments for the Trainer.
    
    Args:
        Numerous parameters to configure training
    
    Returns:
        TrainingArguments object
    """
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create full output directories
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    logging_dir = os.path.join(output_dir, f"logs_{timestamp}")
    
    # Create output directories
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    # Based on Transformers 4.51.3 compatibility test
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        # Using eval_strategy instead of evaluation_strategy
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        # Make sure these both have the same value
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="f1",
        seed=seed,
    )
    
    print("\n=== Training Configuration ===")
    print(f"Output directory: {run_output_dir}")
    print(f"Training epochs: {num_train_epochs}")
    print(f"Batch size (train): {per_device_train_batch_size}")
    print(f"Batch size (eval): {per_device_eval_batch_size}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Weight decay: {weight_decay}")
    print(f"Logging steps: {logging_steps}")
    print(f"Evaluation strategy: {evaluation_strategy}")
    print(f"Save strategy: {save_strategy}")
    print(f"Load best model at end: {load_best_model_at_end}")
    print(f"Random seed: {seed}")
    
    return training_args, run_output_dir, logging_dir

def train_model(
    model,
    train_dataset,
    eval_dataset,
    training_args,
    compute_metrics_fn: Callable = compute_metrics
) -> Tuple[Trainer, Dict]:
    """
    Train a model with detailed progress tracking.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        compute_metrics_fn: Function to compute metrics
    
    Returns:
        Trained Trainer object and dictionary of training metrics
    """
    print("\n=== Starting Model Training ===")
    print(f"Training set size: {len(train_dataset)} examples")
    print(f"Evaluation set size: {len(eval_dataset)} examples")
    
    # Print class distribution in training set
    train_labels = [example["label"].item() for example in train_dataset]
    train_pos_count = sum(1 for label in train_labels if label == 1)
    train_neg_count = sum(1 for label in train_labels if label == 0)
    print(f"Training set class distribution: {train_pos_count} positive, {train_neg_count} negative")
    
    # Create and configure trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
    )
    
    print("Trainer initialized. Beginning training...")
    start_time = time.time()
    
    # Train model
    train_result = trainer.train()
    
    # Calculate training time
    train_time = time.time() - start_time
    hours, remainder = divmod(train_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Evaluate model on the evaluation dataset
    print("\n=== Final Evaluation ===")
    eval_metrics = trainer.evaluate()
    
    print("\nTraining metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nEvaluation metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Return trainer and metrics
    combined_metrics = {
        "train": train_result.metrics,
        "eval": eval_metrics,
        "training_time_seconds": train_time
    }
    
    return trainer, combined_metrics

def save_model_and_tokenizer(
    model,
    tokenizer,
    output_dir: str,
    model_name: str = "finetuned_distilbert"
) -> str:
    """
    Save the model and tokenizer.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Base output directory
        model_name: Name for the saved model
    
    Returns:
        Path to saved model
    """
    # Create timestamp for model name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    
    # Create directory
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    print(f"Saving model to {model_path}...")
    model.save_pretrained(model_path)
    
    # Save tokenizer
    print(f"Saving tokenizer to {model_path}...")
    tokenizer.save_pretrained(model_path)
    
    print(f"Model and tokenizer saved to {model_path}")
    
    return model_path

def load_trained_model_and_tokenizer(model_path: str) -> Tuple:
    """
    Load a trained model and its tokenizer.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    print("Model loaded.")
    
    # Load tokenizer
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded.")
    
    return model, tokenizer