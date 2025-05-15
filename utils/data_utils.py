# utils/data_utils.py
import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Union

def load_imdb_dataset(subset_size: Optional[int] = None, seed: int = 42) -> Dict:
    """
    Load and preprocess the IMDb dataset.
    
    Args:
        subset_size: Number of samples to use for train and test (if None, use all data)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing 'train' and 'test' datasets
    """
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    print(f"Original dataset loaded - Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Create subsets if requested
    if subset_size is not None:
        train_subset = dataset["train"].shuffle(seed=seed).select(range(subset_size))
        test_subset = dataset["test"].shuffle(seed=seed).select(range(subset_size // 5))
        print(f"Created subsets - Train: {len(train_subset)} samples, Test: {len(test_subset)} samples")
        return {'train': train_subset, 'test': test_subset}
    
    return {'train': dataset['train'], 'test': dataset['test']}

def display_dataset_examples(dataset, num_examples: int = 5) -> None:
    """
    Display a sample of examples from the dataset.
    
    Args:
        dataset: Dataset to display examples from
        num_examples: Number of examples to display
    """
    print(f"\n=== Sample data ({num_examples} examples) ===")
    for i in range(min(num_examples, len(dataset))):
        # Get a sample
        example = dataset[i]
        text = example["text"]
        label = "Positive" if example["label"] == 1 else "Negative"
        
        # Truncate if too long
        if len(text) > 300:
            text = text[:300] + "..."
        
        print(f"\nExample {i+1} ({label}):")
        print(f"{text}")
        print("-" * 40)

def analyze_dataset_statistics(dataset) -> Dict:
    """
    Analyze and return statistics about the dataset.
    
    Args:
        dataset: Dataset to analyze
    
    Returns:
        Dictionary containing various statistics
    """
    print("\n=== Dataset Statistics ===")
    
    # Count labels
    labels = [example["label"] for example in dataset]
    label_counts = {
        "Positive": labels.count(1),
        "Negative": labels.count(0)
    }
    
    # Calculate text lengths
    text_lengths = [len(example["text"]) for example in dataset]
    
    # Calculate word counts
    word_counts = [len(example["text"].split()) for example in dataset]
    
    stats = {
        "total_examples": len(dataset),
        "label_distribution": label_counts,
        "text_length": {
            "min": min(text_lengths),
            "max": max(text_lengths),
            "mean": sum(text_lengths) / len(text_lengths),
            "median": sorted(text_lengths)[len(text_lengths) // 2]
        },
        "word_count": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": sum(word_counts) / len(word_counts),
            "median": sorted(word_counts)[len(word_counts) // 2]
        }
    }
    
    # Print statistics
    print(f"Total examples: {stats['total_examples']}")
    print(f"Label distribution: {label_counts['Positive']} positive, {label_counts['Negative']} negative")
    print(f"Text length (chars): min={stats['text_length']['min']}, max={stats['text_length']['max']}, avg={stats['text_length']['mean']:.1f}")
    print(f"Word count: min={stats['word_count']['min']}, max={stats['word_count']['max']}, avg={stats['word_count']['mean']:.1f}")
    
    return stats

def save_dataset_statistics(stats: Dict, output_path: str) -> None:
    """
    Save dataset statistics to a JSON file.
    
    Args:
        stats: Dictionary of statistics
        output_path: Path to save the statistics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to serializable format
    serializable_stats = {
        key: value if not isinstance(value, dict) else 
             {k: round(v, 2) if isinstance(v, float) else v 
              for k, v in value.items()}
        for key, value in stats.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_stats, f, indent=4)
    
    print(f"Dataset statistics saved to {output_path}")

def split_dataset(dataset, valid_ratio: float = 0.1, seed: int = 42) -> Tuple:
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        valid_ratio: Ratio of validation set size to original dataset size
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, valid_dataset)
    """
    if valid_ratio <= 0 or valid_ratio >= 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    
    # Shuffle the dataset
    dataset_shuffled = dataset.shuffle(seed=seed)
    
    # Calculate split point
    valid_size = int(len(dataset) * valid_ratio)
    train_size = len(dataset) - valid_size
    
    # Split dataset
    train_dataset = dataset_shuffled.select(range(train_size))
    valid_dataset = dataset_shuffled.select(range(train_size, len(dataset_shuffled)))
    
    print(f"Split dataset: {len(train_dataset)} training examples, {len(valid_dataset)} validation examples")
    
    return train_dataset, valid_dataset