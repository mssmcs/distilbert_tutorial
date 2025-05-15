# 5_visualize_results.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from utils.visualization_utils import save_figure

def main():
    """
    Create advanced visualizations from model results.
    """
    print("=" * 80)
    print("STEP 5: RESULT VISUALIZATION".center(80))
    print("=" * 80)
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Check if predictions exist
    if not os.path.exists("results/test_predictions.json"):
        print("Test predictions not found. Please run 4_evaluate_model.py first.")
        return
    
    # Load predictions
    with open("results/test_predictions.json", 'r') as f:
        predictions = json.load(f)
    
    y_true = predictions["true_labels"]
    y_pred = predictions["predicted_labels"]
    y_scores = predictions["positive_scores"]
    
    print(f"Loaded predictions for {len(y_true)} examples.")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. ROC Curve
    roc_fig = create_roc_curve(y_true, y_scores)
    roc_fig_path = "visualizations/roc_curve.png"
    save_figure(roc_fig, roc_fig_path)
    print(f"ROC curve saved to {roc_fig_path}")
    
    # 2. Precision-Recall Curve
    pr_fig = create_precision_recall_curve(y_true, y_scores)
    pr_fig_path = "visualizations/precision_recall_curve.png"
    save_figure(pr_fig, pr_fig_path)
    print(f"Precision-Recall curve saved to {pr_fig_path}")
    
    # 3. Error Analysis (Prediction Scores Distribution)
    error_fig = create_error_analysis(y_true, y_pred, y_scores)
    error_fig_path = "visualizations/error_analysis.png"
    save_figure(error_fig, error_fig_path)
    print(f"Error analysis visualization saved to {error_fig_path}")
    
    # 4. Threshold Analysis
    threshold_fig = create_threshold_analysis(y_true, y_scores)
    threshold_fig_path = "visualizations/threshold_analysis.png"
    save_figure(threshold_fig, threshold_fig_path)
    print(f"Threshold analysis visualization saved to {threshold_fig_path}")
    
    print("\nVisualization complete.")

def create_roc_curve(y_true, y_scores):
    """
    Create a ROC curve visualization.
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=12)
    
    # Annotate some key points
    # Find point closest to 0.1 FPR
    idx_01 = np.argmin(np.abs(fpr - 0.1))
    ax.plot(fpr[idx_01], tpr[idx_01], 'ro')
    ax.annotate(
        f'FPR=0.1, TPR={tpr[idx_01]:.4f}',
        xy=(fpr[idx_01], tpr[idx_01]),
        xytext=(fpr[idx_01] + 0.1, tpr[idx_01] - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10
    )
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_precision_recall_curve(y_true, y_scores):
    """
    Create a Precision-Recall curve visualization.
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.4f})')
    
    # Calculate F1 score at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    # Plot best F1 point
    ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'ro', markersize=8)
    ax.annotate(
        f'Best F1: {f1_scores[best_f1_idx]:.4f}\nThreshold: {best_threshold:.4f}',
        xy=(recall[best_f1_idx], precision[best_f1_idx]),
        xytext=(recall[best_f1_idx] - 0.2, precision[best_f1_idx] - 0.1),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10
    )
    
    # Set axis limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower left', fontsize=12)
    
    # Add a horizontal line at precision = n_pos / total
    no_skill = sum(y_true) / len(y_true)
    ax.plot([0, 1], [no_skill, no_skill], 'k--', lw=2, label=f'No skill: {no_skill:.4f}')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_error_analysis(y_true, y_pred, y_scores):
    """
    Create a visualization for error analysis.
    """
    # Separate scores by prediction correctness
    correct_indices = np.array(y_true) == np.array(y_pred)
    incorrect_indices = ~correct_indices
    
    correct_scores = np.array(y_scores)[correct_indices]
    incorrect_scores = np.array(y_scores)[incorrect_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms
    bins = 20
    ax.hist(
        correct_scores,
        bins=bins,
        alpha=0.5,
        label=f'Correct predictions ({len(correct_scores)})',
        color='green'
    )
    ax.hist(
        incorrect_scores,
        bins=bins,
        alpha=0.5,
        label=f'Incorrect predictions ({len(incorrect_scores)})',
        color='red'
    )
    
    # Set axis labels and title
    ax.set_xlabel('Positive Class Probability Score', fontsize=14)
    ax.set_ylabel('Number of Predictions', fontsize=14)
    ax.set_title('Distribution of Prediction Scores by Correctness', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add annotation with error rate
    error_rate = len(incorrect_scores) / len(y_true) * 100
    ax.annotate(
        f'Error rate: {error_rate:.2f}%',
        xy=(0.98, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_threshold_analysis(y_true, y_scores):
    """
    Create a visualization for threshold analysis.
    """
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0, 1, 100)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for threshold in thresholds:
        y_pred_at_threshold = (np.array(y_scores) >= threshold).astype(int)
        
        # True positives, false positives, etc.
        tp = np.sum((y_pred_at_threshold == 1) & (np.array(y_true) == 1))
        fp = np.sum((y_pred_at_threshold == 1) & (np.array(y_true) == 0))
        tn = np.sum((y_pred_at_threshold == 0) & (np.array(y_true) == 0))
        fn = np.sum((y_pred_at_threshold == 0) & (np.array(y_true) == 1))
        
        # Calculate metrics
        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)
    
    # Find best F1 threshold
    best_f1_idx = np.argmax(f1)
    best_f1_threshold = thresholds[best_f1_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot metrics
    ax.plot(thresholds, accuracy, 'b-', lw=2, label='Accuracy')
    ax.plot(thresholds, precision, 'g-', lw=2, label='Precision')
    ax.plot(thresholds, recall, 'r-', lw=2, label='Recall')
    ax.plot(thresholds, f1, 'c-', lw=2, label='F1 Score')
    
    # Highlight best F1 threshold
    ax.axvline(x=best_f1_threshold, color='black', linestyle='--', lw=2, label=f'Best F1 threshold: {best_f1_threshold:.4f}')
    
    # Default threshold (0.5)
    ax.axvline(x=0.5, color='purple', linestyle=':', lw=2, label='Default threshold: 0.5')
    
    # Set axis labels and title
    ax.set_xlabel('Threshold', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Metrics at Different Probability Thresholds', fontsize=16, pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Add annotation with best values
    best_values_text = (
        f'Best F1: {f1[best_f1_idx]:.4f}\n'
        f'At threshold: {best_f1_threshold:.4f}\n'
        f'Accuracy: {accuracy[best_f1_idx]:.4f}\n'
        f'Precision: {precision[best_f1_idx]:.4f}\n'
        f'Recall: {recall[best_f1_idx]:.4f}'
    )
    
    ax.annotate(
        best_values_text,
        xy=(best_f1_threshold, f1[best_f1_idx]),
        xytext=(best_f1_threshold + 0.2, f1[best_f1_idx] - 0.2),
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Adjust layout
    fig.tight_layout()
    
    return fig