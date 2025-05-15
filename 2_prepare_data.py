# 2_prepare_data.py
import os
import pickle
import torch
import numpy as np
from utils.data_utils import load_imdb_dataset, split_dataset
from utils.tokenization_utils import (
    get_tokenizer,
    tokenize_example,
    display_tokenization,
    tokenize_dataset
)
from utils.visualization_utils import (
    plot_tokenization_length_distribution,
    save_figure
)

def main():
    """
    Prepare the IMDb dataset for training.
    """
    print("=" * 80)
    print("STEP 2: DATA PREPARATION".center(80))
    print("=" * 80)
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Load dataset
    subset_size = 5000  # Use subset for faster processing
    dataset = load_imdb_dataset(subset_size=subset_size)
    
    # Split training set into train and validation
    train_dataset, valid_dataset = split_dataset(dataset['train'], valid_ratio=0.1)
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(model_name="distilbert-base-uncased")
    
    # Demonstrate tokenization with a sample
    print("\n--- Tokenization Example ---")
    sample_text = train_dataset[0]["text"]
    tokenized_sample = tokenize_example(tokenizer, sample_text)
    display_tokenization(tokenized_sample)
    
    # Tokenize datasets
    max_length = 512  # Maximum sequence length for DistilBERT
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length=max_length)
    valid_tokenized = tokenize_dataset(valid_dataset, tokenizer, max_length=max_length)
    test_tokenized = tokenize_dataset(dataset['test'], tokenizer, max_length=max_length)
    
    # Get token length statistics for visualization
    print("\nCollecting token length statistics...")
    sample_size = min(1000, len(train_dataset))
    train_token_lengths = []
    
    for i in range(sample_size):
        sample_ids = train_tokenized[i]['input_ids'].tolist()
        # Count non-padding tokens
        length = sum(1 for token_id in sample_ids if token_id != tokenizer.pad_token_id)
        train_token_lengths.append(length)
    
    # Create visualization
    token_length_fig = plot_tokenization_length_distribution(
        train_token_lengths,
        max_length,
        "Token Length Distribution (Sample of Training Set)"
    )
    save_figure(token_length_fig, "visualizations/token_length_distribution.png")
    
    # Save tokenized datasets
    print("\nSaving tokenized datasets...")
    
    # Try to register the Dataset class as safe for PyTorch 2.6+
    try:
        from datasets.arrow_dataset import Dataset
        torch.serialization.add_safe_globals([Dataset])
        print("Registered Dataset class as safe for serialization")
    except (ImportError, AttributeError) as e:
        print(f"Note: Could not register Dataset class - {e}")
    
    # Save datasets - try with weights_only parameter for PyTorch 2.6+
    try:
        torch.save(train_tokenized, "data/train_tokenized.pt", weights_only=False)
        torch.save(valid_tokenized, "data/valid_tokenized.pt", weights_only=False)
        torch.save(test_tokenized, "data/test_tokenized.pt", weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        print("Falling back to standard save method without weights_only parameter")
        torch.save(train_tokenized, "data/train_tokenized.pt")
        torch.save(valid_tokenized, "data/valid_tokenized.pt")
        torch.save(test_tokenized, "data/test_tokenized.pt")
    
    # Save tokenizer for later use
    tokenizer.save_pretrained("data/tokenizer")
    
    print("\nData preparation complete.")
    print(f"Train set: {len(train_tokenized)} examples")
    print(f"Validation set: {len(valid_tokenized)} examples")
    print(f"Test set: {len(test_tokenized)} examples")
    print("Tokenized datasets saved to 'data/' directory.")

if __name__ == "__main__":
    main()