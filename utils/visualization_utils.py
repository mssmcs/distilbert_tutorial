# utils/visualization_utils.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List, Tuple, Optional, Union, Any

# Set style
plt.style.use('ggplot')

def plot_label_distribution(labels: List[int], title: str = "Label Distribution") -> plt.Figure:
    """
    Plot the distribution of labels in a dataset.
    
    Args:
        labels: List of labels (0 or 1)
        title: Title for the plot
    
    Returns:
        Figure object
    """
    # Count labels
    label_counts = {
        "Negative (0)": labels.count(0),
        "Positive (1)": labels.count(1)
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bar chart
    bars = ax.bar(
        label_counts.keys(),
        label_counts.values(),
        color=['#ff9999', '#66b3ff']
    )
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f'{height:,}',
            ha='center',
            va='bottom',
            fontsize=12
        )
    
    # Add percentage labels inside bars
    total = sum(label_counts.values())
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = height / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height / 2,
            f'{percentage:.1f}%',
            ha='center',
            va='center',
            fontsize=12,
            color='black',
            fontweight='bold'
        )
    
    # Add title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_ylim(0, max(label_counts.values()) * 1.15)  # Add some space for the labels
    
    # Add grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def plot_text_length_distribution(
    text_lengths: List[int],
    title: str = "Text Length Distribution",
    bins: int = 30
) -> plt.Figure:
    """
    Plot the distribution of text lengths.
    
    Args:
        text_lengths: List of text lengths
        title: Title for the plot
        bins: Number of bins for histogram
    
    Returns:
        Figure object
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    sns.histplot(text_lengths, bins=bins, kde=True, ax=ax, color='#5975a4')
    
    # Add statistics lines
    mean_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    
    # Plot vertical lines for mean and median
    ax.axvline(x=mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax.axvline(x=median_length, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_length:.1f}')
    
    # Add statistics as text
    stats_text = (
        f"Min: {min(text_lengths):,}\n"
        f"Max: {max(text_lengths):,}\n"
        f"Mean: {mean_length:.1f}\n"
        f"Median: {median_length:.1f}"
    )
    
    # Place text box in upper right corner
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Text Length (characters)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Format x-axis with commas for thousands
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def plot_tokenization_length_distribution(
    token_lengths: List[int],
    max_length: int,
    title: str = "Token Length Distribution"
) -> plt.Figure:
    """
    Plot the distribution of tokenized text lengths.
    
    Args:
        token_lengths: List of token counts per example
        max_length: Maximum token length used during tokenization
        title: Title for the plot
    
    Returns:
        Figure object
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histogram
    sns.histplot(token_lengths, bins=30, kde=True, ax=ax, color='#5975a4')
    
    # Add statistics lines
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    
    # Plot vertical lines
    ax.axvline(x=mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax.axvline(x=median_length, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_length:.1f}')
    ax.axvline(x=max_length, color='orange', linestyle='-', linewidth=2, label=f'Max Length: {max_length}')
    
    # Add statistics as text
    truncated_count = sum(1 for length in token_lengths if length >= max_length)
    truncated_percent = truncated_count / len(token_lengths) * 100
    
    stats_text = (
        f"Min: {min(token_lengths)}\n"
        f"Max: {max(token_lengths)}\n"
        f"Mean: {mean_length:.1f}\n"
        f"Median: {median_length:.1f}\n"
        f"Truncated: {truncated_count} ({truncated_percent:.1f}%)"
    )
    
    # Place text box in upper right corner
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Number of Tokens', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def plot_training_history(history: Dict[str, List[float]]) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with keys 'train_loss', 'eval_loss', 'eval_accuracy', etc.
                Each key maps to a list of values per epoch
    
    Returns:
        Figure object
    """
    # Create plot with two subplots (loss and metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot losses
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss subplot
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['eval_loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Metrics subplot
    if 'eval_accuracy' in history:
        ax2.plot(epochs, history['eval_accuracy'], 'go-', label='Accuracy')
    if 'eval_f1' in history:
        ax2.plot(epochs, history['eval_f1'], 'mo-', label='F1 Score')
    if 'eval_precision' in history:
        ax2.plot(epochs, history['eval_precision'], 'co-', label='Precision')
    if 'eval_recall' in history:
        ax2.plot(epochs, history['eval_recall'], 'yo-', label='Recall')
    
    ax2.set_title('Validation Metrics', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # Add min/max annotations for validation metrics
    if 'eval_accuracy' in history:
        max_acc = max(history['eval_accuracy'])
        max_acc_epoch = history['eval_accuracy'].index(max_acc) + 1
        ax2.annotate(f'Max: {max_acc:.4f}',
                    xy=(max_acc_epoch, max_acc),
                    xytext=(max_acc_epoch, max_acc + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
    
    Returns:
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Negative (0)", "Positive (1)"]
    )
    disp.plot(cmap='Blues', ax=ax)
    
    # Customize
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    
    # Add text annotations with percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            ax.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                   ha='center', va='center', fontsize=12)
    
    # Calculate and display overall metrics
    true_positive = cm[1, 1]
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 Score: {f1:.4f}"
    )
    
    # Add metrics textbox
    fig.text(
        0.85, 0.5, metrics_text,
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def visualize_token_attention(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
    layer_index: int = 5,  # Last layer by default
    head_index: int = 0
) -> plt.Figure:
    """
    Visualize token attention for a specific text.
    
    Args:
        tokenizer: DistilBERT tokenizer
        model: DistilBERT model
        text: Input text to visualize attention for
        max_length: Maximum token length
        layer_index: Transformer layer to visualize (0-5 for DistilBERT)
        head_index: Attention head to visualize
    
    Returns:
        Figure object
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    
    # Get token IDs and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Convert IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Get visible tokens (non-padding tokens)
    visible_tokens = []
    visible_indices = []
    for i, (token, mask) in enumerate(zip(tokens, attention_mask[0])):
        if mask == 1:  # Token is not padding
            visible_tokens.append(token)
            visible_indices.append(i)
    
    # Run model and extract attention
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    
    # Get attention from specified layer and head
    # Shape: (batch_size, num_heads, seq_length, seq_length)
    attention = outputs.attentions[layer_index][0, head_index].cpu().numpy()
    
    # Filter attention to only visible tokens
    visible_attention = attention[visible_indices, :][:, visible_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot attention heatmap
    im = ax.imshow(visible_attention, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(visible_tokens)))
    ax.set_yticks(np.arange(len(visible_tokens)))
    ax.set_xticklabels(visible_tokens, rotation=90, fontsize=10)
    ax.set_yticklabels(visible_tokens, fontsize=10)
    
    # Turn off tick marks
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    
    # Add title
    ax.set_title(f"Attention Matrix (Layer {layer_index+1}, Head {head_index+1})", fontsize=16, pad=20)
    
    # Add labels
    ax.set_xlabel("Token (Target)", fontsize=14)
    ax.set_ylabel("Token (Source)", fontsize=14)
    
    # Add grid
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    
    # Draw grid lines
    ax.set_xticks(np.arange(visible_attention.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(visible_attention.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a figure to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save figure
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")