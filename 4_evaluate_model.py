# 4_evaluate_model.py
import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.model_utils import load_trained_model_and_tokenizer, compute_metrics
from utils.visualization_utils import plot_confusion_matrix, save_figure

def main():
    """
    Evaluate the trained model on the test set.
    """
    print("=" * 80)
    print("STEP 4: MODEL EVALUATION".center(80))
    print("=" * 80)
    
    # Create output directories
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Check if test dataset exists
    if not os.path.exists("data/test_tokenized.pt"):
        print("Test dataset not found. Please run 2_prepare_data.py first.")
        return
    
    # Get the most recent model directory
    model_dirs = [d for d in os.listdir("models") if d.startswith("finetuned_distilbert_")]
    if not model_dirs:
        print("No trained model found. Please run 3_train_model.py first.")
        return
    
    model_dir = sorted(model_dirs)[-1]  # Most recent by name (which includes timestamp)
    model_path = os.path.join("models", model_dir)
    
    # Load model and tokenizer
    model, tokenizer = load_trained_model_and_tokenizer(model_path)
    
    # Try to register the Dataset class as safe for PyTorch 2.6+
    try:
        from datasets.arrow_dataset import Dataset
        torch.serialization.add_safe_globals([Dataset])
        print("Registered Dataset class as safe for serialization")
    except (ImportError, AttributeError) as e:
        print(f"Note: Could not register Dataset class - {e}")
    
    # Load test dataset with robust error handling
    print("Loading test dataset...")
    try:
        test_dataset = torch.load("data/test_tokenized.pt", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        print("Falling back to standard load method without weights_only parameter")
        test_dataset = torch.load("data/test_tokenized.pt")
    
    # Debug: Verify test dataset
    print(f"Test set: {len(test_dataset)} examples")
    print(f"Sample test_dataset labels: {[test_dataset[i]['label'] for i in range(5)]}")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    model.eval()
    
    # Get predictions
    y_true = []
    y_pred = []
    y_scores = []
    
    batch_size = 32
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    print(f"Processing {num_batches} batches with batch size {batch_size}...")
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            
            # Prepare batch
            batch = {
                key: torch.stack([test_dataset[j][key] for j in range(start_idx, end_idx)]).to(device)
                for key in ['input_ids', 'attention_mask']
            }
            
            # Get labels
            labels = torch.tensor([test_dataset[j]['label'] for j in range(start_idx, end_idx)]).to(device)
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Debug: Print logits shape
            print(f"Batch {i+1}: logits shape: {logits.shape}")
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()  # Convert to list
            scores = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            
            # Debug: Print predictions
            print(f"Batch {i+1}: predictions shape: {torch.argmax(logits, dim=1).shape}, values: {predictions}")
            
            # Add to lists
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions)
            y_scores.extend(scores[:, 1].tolist())  # Positive class scores
            
            # Print progress
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"  Processed {i+1}/{num_batches} batches ({(i+1)/num_batches*100:.1f}%)")
    
    # Debug: Print final lengths and samples
    print(f"Final y_true length: {len(y_true)}, y_pred length: {len(y_pred)}, y_scores length: {len(y_scores)}")
    print(f"y_pred type: {type(y_pred)}, sample: {y_pred[:5]}")
    print(f"np.array(y_pred) shape: {np.array(y_pred).shape}, dtype: {np.array(y_pred).dtype}")
    print(f"Unique y_pred values: {set(y_pred)}")
    
    # Calculate metrics
    metrics = compute_metrics(type('obj', (), {
        'label_ids': np.array(y_true, dtype=np.int64),
        'predictions': np.array(y_pred, dtype=np.int64)
    }))
    
    # Create classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative", "Positive"],
        digits=4
    )
    
    # Print results
    print("\n=== Test Set Evaluation Results ===\n")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save results
    results = {
        "metrics": metrics,
        "classification_report": report,
        "model_path": model_path
    }
    
    results_path = "results/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            **{"metrics": metrics, "model_path": model_path},
            **{"classification_report": report}
        }, f, indent=4)
    print(f"Evaluation results saved to {results_path}")
    
    # Create confusion matrix visualization
    confusion_fig = plot_confusion_matrix(y_true, y_pred, "Test Set Confusion Matrix")
    confusion_fig_path = "visualizations/confusion_matrix.png"
    save_figure(confusion_fig, confusion_fig_path)
    print(f"Confusion matrix visualization saved to {confusion_fig_path}")
    
    # Save predictions
    predictions_path = "results/test_predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump({
            "true_labels": y_true,
            "predicted_labels": y_pred,
            "positive_scores": [float(score) for score in y_scores]
        }, f)
    print(f"Predictions saved to {predictions_path}")
    
    print("\nModel evaluation complete.")

if __name__ == "__main__":
    main()