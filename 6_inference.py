# 6_inference.py
import os
import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from utils.tokenization_utils import tokenize_example, display_tokenization
from utils.visualization_utils import visualize_token_attention, save_figure

def predict(text, model, tokenizer, detailed=False):
    """
    Make a prediction for a single text input.
    
    Args:
        text: Input text
        model: DistilBERT model
        tokenizer: DistilBERT tokenizer
        detailed: Whether to return detailed information
    
    Returns:
        Dictionary with prediction results
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    # Move tensors to the correct device
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True if detailed else None
        )
    
    # Get prediction and confidence
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
    
    # Get label
    label = "Positive" if predicted_class == 1 else "Negative"
    
    # Create result
    result = {
        "text": text,
        "prediction": label,
        "confidence": confidence,
        "class_probabilities": {
            "Negative": probabilities[0].item(),
            "Positive": probabilities[1].item()
        }
    }
    
    # Add detailed information if requested
    if detailed:
        # Convert token IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get attention weights
        attentions = [layer[0].cpu().numpy() for layer in outputs.attentions]
        
        # Add to result
        result["tokens"] = tokens
        result["attention_weights"] = attentions
    
    return result

def main():
    """
    Run inference on new text inputs.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference with the trained DistilBERT model.")
    parser.add_argument(
        "--text",
        type=str,
        help="Text to analyze (if not provided, will use example texts)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the trained model (if not provided, will use most recent model)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Whether to show detailed analysis"
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Whether to save visualizations"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("STEP 6: MODEL INFERENCE".center(80))
    print("=" * 80)
    
    # Create output directory for visualizations
    if args.save_viz:
        os.makedirs("visualizations/inference", exist_ok=True)
    
    # Find model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Get the most recent model directory
        model_dirs = [d for d in os.listdir("models") if d.startswith("finetuned_distilbert_")]
        if not model_dirs:
            print("No trained model found. Please run 3_train_model.py first.")
            return
        
        model_dir = sorted(model_dirs)[-1]  # Most recent by name (includes timestamp)
        model_path = os.path.join("models", model_dir)
    
    print(f"Using model from: {model_path}")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")
    
    # Get text to analyze
    if args.text:
        texts = [args.text]
    else:
        # Use example texts
        texts = [
            "This movie was fantastic! The acting was great and the plot kept me engaged throughout.",
            "I hated this film. The story was boring and the characters were one-dimensional.",
            "The movie was okay. Some parts were good, but others were too slow."
        ]
    
    # Run inference
    print("\n=== Inference Results ===\n")
    
    for i, text in enumerate(texts):
        print(f"Example {i+1}:")
        print(f"Text: {text[:100]}...")
        
        # Get prediction
        result = predict(text, model, tokenizer, detailed=args.detailed)
        
        # Print prediction
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Class probabilities: Positive={result['class_probabilities']['Positive']:.4f}, "
              f"Negative={result['class_probabilities']['Negative']:.4f}")
        
        # Detailed analysis
        if args.detailed:
            # Display tokenization
            tokenized = tokenize_example(tokenizer, text)
            display_tokenization(tokenized)
            
            # Visualize attention
            print("\nGenerating attention visualization...")
            
            for layer_idx in [0, 2, 5]:  # First, middle, and last layer
                for head_idx in [0]:     # First attention head
                    # Create visualization
                    attention_fig = visualize_token_attention(
                        tokenizer=tokenizer,
                        model=model,
                        text=text,
                        layer_index=layer_idx,
                        head_index=head_idx
                    )
                    
                    # Save visualization if requested
                    if args.save_viz:
                        fig_path = f"visualizations/inference/attention_example{i+1}_layer{layer_idx+1}_head{head_idx+1}.png"
                        save_figure(attention_fig, fig_path)
                        print(f"  Attention visualization saved to {fig_path}")
                    else:
                        plt.show()
                        plt.close()
            
            # Print sentiment-important tokens
            # Get last layer attention weights for first head
            attn_weights = result["attention_weights"][-1][0]  # Last layer, first head
            
            # Get most attended tokens (excluding [CLS], [SEP], and [PAD])
            tokens = result["tokens"]
            special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
            attn_scores = []
            
            for i, token in enumerate(tokens):
                if token not in special_tokens and i < len(tokens) - 1:  # Skip padding at the end
                    # Get average attention to this token from all other non-special tokens
                    attn_score = np.mean([
                        attn_weights[j, i]
                        for j in range(len(tokens))
                        if tokens[j] not in special_tokens
                    ])
                    attn_scores.append((token, attn_score))
            
            # Sort by attention score
            attn_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Print top attended tokens
            print("\nTop attended tokens:")
            for token, score in attn_scores[:10]:
                print(f"  {token}: {score:.4f}")
        
        print("-" * 40)
    
    print("\nInference complete.")

if __name__ == "__main__":
    main()