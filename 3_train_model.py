# 3_train_model.py
import os
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from utils.tokenization_utils import get_tokenizer
from utils.model_utils import (
    create_model,
    create_training_args,
    compute_metrics,
    train_model,
    save_model_and_tokenizer
)
from utils.visualization_utils import (
    plot_training_history,
    save_figure
)

def main():
    """
    Train a DistilBERT model on the IMDb dataset.
    """
    print("=" * 80)
    print("STEP 3: MODEL TRAINING".center(80))
    print("=" * 80)
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Check if tokenized datasets exist
    if not os.path.exists("data/train_tokenized.pt"):
        print("Tokenized datasets not found. Please run 2_prepare_data.py first.")
        return
    
    # Try to register the Dataset class as safe for PyTorch 2.6+
    try:
        from datasets.arrow_dataset import Dataset
        torch.serialization.add_safe_globals([Dataset])
        print("Registered Dataset class as safe for serialization")
    except (ImportError, AttributeError) as e:
        print(f"Note: Could not register Dataset class - {e}")
    
    # Load tokenized datasets with robust error handling
    print("Loading tokenized datasets...")
    
    # Try loading with weights_only parameter for PyTorch 2.6+
    try:
        train_dataset = torch.load("data/train_tokenized.pt", weights_only=False)
        valid_dataset = torch.load("data/valid_tokenized.pt", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        print("Falling back to standard load method without weights_only parameter")
        train_dataset = torch.load("data/train_tokenized.pt")
        valid_dataset = torch.load("data/valid_tokenized.pt")
    
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Validation set: {len(valid_dataset)} examples")
    
    # Load or create tokenizer
    if os.path.exists("data/tokenizer"):
        print("Loading tokenizer from data/tokenizer...")
        tokenizer = get_tokenizer(model_name="data/tokenizer")
    else:
        print("Creating new tokenizer...")
        tokenizer = get_tokenizer(model_name="distilbert-base-uncased")
    
    # Create model
    model = create_model(model_name="distilbert-base-uncased", num_labels=2)
    
    # Create training arguments
    training_args, run_output_dir, _ = create_training_args(
        output_dir="models",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        seed=42
    )
    
    # Train model
    trainer, metrics = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        training_args=training_args,
        compute_metrics_fn=compute_metrics
    )
    
    # Save model and tokenizer
    model_path = save_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        output_dir="models",
        model_name="finetuned_distilbert"
    )
    
    # Save training metrics
    metrics_path = os.path.join(run_output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Training metrics saved to {metrics_path}")
    
    # Create training history visualization
    print("\nCreating training history visualization...")
    
    # Extract metric histories from logs
    try:
        log_history = trainer.state.log_history
        
        # Process logs to extract metrics by epoch
        train_loss = []
        eval_loss = []
        eval_accuracy = []
        eval_f1 = []
        eval_precision = []
        eval_recall = []
        
        for entry in log_history:
            if 'loss' in entry and 'eval_loss' not in entry:
                # This is a training log entry
                train_loss.append(entry['loss'])
            elif 'eval_loss' in entry:
                # This is an evaluation log entry
                eval_loss.append(entry['eval_loss'])
                if 'eval_accuracy' in entry:
                    eval_accuracy.append(entry['eval_accuracy'])
                if 'eval_f1' in entry:
                    eval_f1.append(entry['eval_f1'])
                if 'eval_precision' in entry:
                    eval_precision.append(entry['eval_precision'])
                if 'eval_recall' in entry:
                    eval_recall.append(entry['eval_recall'])
        
        # Create history dictionary
        history = {
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'eval_f1': eval_f1,
            'eval_precision': eval_precision,
            'eval_recall': eval_recall
        }
        
        # Save history
        history_path = os.path.join(run_output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        # Plot history
        history_fig = plot_training_history(history)
        history_fig_path = "visualizations/training_history.png"
        save_figure(history_fig, history_fig_path)
        print(f"Training history visualization saved to {history_fig_path}")
        
    except Exception as e:
        print(f"Error creating training history visualization: {e}")
    
    print("\nModel training complete.")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()