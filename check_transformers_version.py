# check_transformers_version.py
"""
This script checks details about your Transformers library installation to help troubleshoot issues.
"""
import sys
import importlib

def main():
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check Transformers version
    try:
        import transformers
        print(f"\nTransformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        
        # Check TrainingArguments
        from transformers import TrainingArguments
        print("\nChecking TrainingArguments parameters...")
        
        # Create a directory to test TrainingArguments
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different parameters
            params_to_test = [
                "evaluation_strategy",
                "eval_strategy",
                "save_strategy",
                "metric_for_best_model",
                "load_best_model_at_end"
            ]
            
            for param in params_to_test:
                try:
                    # Try to create TrainingArguments with this parameter
                    args = {"output_dir": temp_dir}
                    args[param] = "epoch" if param != "metric_for_best_model" else "accuracy"
                    _ = TrainingArguments(**args)
                    print(f"  ✓ Parameter '{param}' is supported")
                except TypeError as e:
                    print(f"  ✗ Parameter '{param}' is NOT supported: {e}")
        
    except ImportError as e:
        print(f"Error importing transformers: {e}")
    
    # Check for other relevant libraries
    libraries_to_check = [
        "torch", 
        "datasets", 
        "numpy", 
        "sklearn"
    ]
    
    print("\nChecking other relevant libraries:")
    for lib in libraries_to_check:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, "__version__", "unknown version")
            print(f"  {lib}: {version}")
        except ImportError:
            print(f"  {lib}: Not installed")
    
    print("\nTransformers version check complete.")

if __name__ == "__main__":
    main()