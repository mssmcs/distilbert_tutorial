# utils/tokenization_utils.py
import torch
from transformers import DistilBertTokenizer
from typing import List, Dict, Tuple, Optional, Union, Any

def get_tokenizer(model_name: str = "distilbert-base-uncased") -> DistilBertTokenizer:
    """
    Initialize and return a tokenizer.
    
    Args:
        model_name: Name of the pre-trained model to use
    
    Returns:
        Initialized tokenizer
    """
    print(f"Initializing tokenizer from {model_name}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    print("Tokenizer initialized.")
    return tokenizer

def tokenize_example(tokenizer, text: str, max_length: int = 512) -> Dict:
    """
    Tokenize a single text example and display detailed information.
    
    Args:
        tokenizer: Tokenizer to use
        text: Text to tokenize
        max_length: Maximum length for padding/truncation
    
    Returns:
        Tokenized output
    """
    # Tokenize
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Get token IDs
    input_ids = tokenized['input_ids'][0].tolist()
    
    # Convert token IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Add detailed information
    result = {
        'input_ids': tokenized['input_ids'][0],
        'attention_mask': tokenized['attention_mask'][0],
        'tokens': tokens,
        'original_text': text,
        'num_tokens': len([t for t in tokens if t != '[PAD]']),
        'was_truncated': len(tokens) == max_length and tokens[-1] != '[PAD]'
    }
    
    return result

def display_tokenization(tokenization_result: Dict) -> None:
    """
    Display detailed information about a tokenized example.
    
    Args:
        tokenization_result: Output from tokenize_example
    """
    print("\n=== Tokenization Example ===")
    print(f"Original text: {tokenization_result['original_text'][:100]}...")
    print(f"Number of tokens: {tokenization_result['num_tokens']}")
    print(f"Was truncated: {tokenization_result['was_truncated']}")
    
    print("\nFirst 10 tokens:")
    for i in range(min(10, len(tokenization_result['tokens']))):
        token = tokenization_result['tokens'][i]
        token_id = tokenization_result['input_ids'][i].item()
        attn = tokenization_result['attention_mask'][i].item()
        print(f"  {i}: Token='{token}', ID={token_id}, Attention={attn}")
    
    if tokenization_result['was_truncated']:
        print("\n[Note: Text was truncated to fit the maximum length]")

def tokenize_dataset(dataset, tokenizer, max_length: int = 512, batch_size: int = 1000) -> Dict:
    """
    Tokenize an entire dataset.
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum length for padding/truncation
        batch_size: Batch size for processing
    
    Returns:
        Tokenized dataset
    """
    print(f"Tokenizing dataset with {len(dataset)} examples...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        desc="Tokenizing"
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    
    print("Tokenization complete.")
    
    # Collect statistics
    truncation_count = 0
    token_lengths = []
    
    # Sample for counting truncation (to avoid processing the entire dataset)
    sample_size = min(1000, len(dataset))
    sampled_indices = torch.randperm(len(dataset))[:sample_size].tolist()
    
    for idx in sampled_indices:
        tokens = tokenizer.convert_ids_to_tokens(tokenized_dataset[idx]['input_ids'].tolist())
        token_length = len([t for t in tokens if t != '[PAD]'])
        token_lengths.append(token_length)
        if token_length == max_length and tokens[-2] != '[PAD]':  # Check if truncated
            truncation_count += 1
    
    print(f"Tokenization stats (based on {sample_size} samples):")
    print(f"  Average token length: {sum(token_lengths)/len(token_lengths):.1f}")
    print(f"  Max token length: {max(token_lengths)}")
    print(f"  Examples truncated: {truncation_count} ({truncation_count/sample_size*100:.1f}%)")
    
    return tokenized_dataset