# check_pytorch_details.py
"""
This script checks details about your PyTorch installation to help troubleshoot issues.
"""
import sys
import torch
import importlib.util

def check_module_exists(module_name):
    """Check if a module can be imported"""
    return importlib.util.find_spec(module_name) is not None

def main():
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"PyTorch path: {torch.__file__}")
    
    # Check if serialization module has add_safe_globals
    has_add_safe_globals = hasattr(torch.serialization, 'add_safe_globals')
    print(f"\nPyTorch has add_safe_globals: {has_add_safe_globals}")
    
    # Try to import the Dataset class and check if it's available
    print("\nChecking for datasets package and Dataset class:")
    datasets_available = check_module_exists('datasets')
    print(f"datasets package available: {datasets_available}")
    
    dataset_class_available = False
    arrow_dataset_available = False
    
    if datasets_available:
        try:
            import datasets
            print(f"datasets version: {datasets.__version__}")
            
            arrow_dataset_available = check_module_exists('datasets.arrow_dataset')
            print(f"datasets.arrow_dataset module available: {arrow_dataset_available}")
            
            if arrow_dataset_available:
                from datasets.arrow_dataset import Dataset
                dataset_class_available = True
                print("Successfully imported Dataset class from datasets.arrow_dataset")
        except ImportError as e:
            print(f"Error importing datasets components: {e}")
    
    # Check for safe_globals in torch.serialization
    has_safe_globals = hasattr(torch.serialization, 'safe_globals')
    print(f"\nPyTorch has safe_globals context manager: {has_safe_globals}")
    
    # Try to test torch.save with weights_only
    print("\nTesting torch.save with weights_only parameter:")
    try:
        tensor = torch.tensor([1, 2, 3])
        # Try to save with weights_only parameter in a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile() as temp:
            try:
                torch.save(tensor, temp.name, weights_only=True)
                print("torch.save with weights_only parameter works!")
            except TypeError as e:
                print(f"torch.save with weights_only parameter failed: {e}")
    except Exception as e:
        print(f"Error during torch.save test: {e}")
    
    print("\nDetailed PyTorch info check complete.")

if __name__ == "__main__":
    main()