# 1_explore_dataset.py
import os
import matplotlib.pyplot as plt
from utils.data_utils import (
    load_imdb_dataset, 
    display_dataset_examples,
    analyze_dataset_statistics,
    save_dataset_statistics
)
from utils.visualization_utils import (
    plot_label_distribution,
    plot_text_length_distribution,
    save_figure
)

def main():
    """
    Explore and analyze the IMDb dataset.
    """
    print("=" * 80)
    print("STEP 1: DATASET EXPLORATION".center(80))
    print("=" * 80)
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Load dataset
    dataset = load_imdb_dataset(subset_size=None)  # Use full dataset for exploration
    
    # Display examples
    display_dataset_examples(dataset['train'], num_examples=3)
    
    # Analyze statistics
    train_stats = analyze_dataset_statistics(dataset['train'])
    test_stats = analyze_dataset_statistics(dataset['test'])
    
    # Save statistics
    save_dataset_statistics(train_stats, "data/train_stats.json")
    save_dataset_statistics(test_stats, "data/test_stats.json")
    
    # Create visualizations
    
    # 1. Label distribution
    train_labels = [example["label"] for example in dataset['train']]
    train_label_fig = plot_label_distribution(train_labels, "Train Set Label Distribution")
    save_figure(train_label_fig, "visualizations/train_label_distribution.png")
    
    test_labels = [example["label"] for example in dataset['test']]
    test_label_fig = plot_label_distribution(test_labels, "Test Set Label Distribution")
    save_figure(test_label_fig, "visualizations/test_label_distribution.png")
    
    # 2. Text length distribution
    train_text_lengths = [len(example["text"]) for example in dataset['train']]
    train_length_fig = plot_text_length_distribution(
        train_text_lengths, 
        "Train Set Text Length Distribution"
    )
    save_figure(train_length_fig, "visualizations/train_text_length_distribution.png")
    
    test_text_lengths = [len(example["text"]) for example in dataset['test']]
    test_length_fig = plot_text_length_distribution(
        test_text_lengths, 
        "Test Set Text Length Distribution"
    )
    save_figure(test_length_fig, "visualizations/test_text_length_distribution.png")
    
    print("\nDataset exploration complete. Visualizations saved to 'visualizations/' directory.")

if __name__ == "__main__":
    main()