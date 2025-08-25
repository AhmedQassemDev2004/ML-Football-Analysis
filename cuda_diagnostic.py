import torch
import sys
import subprocess
import platform

def check_cuda_availability():
    print("=== CUDA Diagnostic Report ===\n")
    
    # System Info
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}\n")
    
    # PyTorch Info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA is NOT available")
        print(f"PyTorch CUDA compiled version: {torch.version.cuda}")
    
    print("\n=== NVIDIA System Info ===")
    try:
        # Try to get nvidia-smi output
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("NVIDIA GPU detected:")
            print(result.stdout)
        else:
            print("nvidia-smi command failed or not found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available or timed out")
    
    print("\n=== Recommendations ===")
    if not torch.cuda.is_available():
        print("To fix CUDA issues:")
        print("1. Check if you have NVIDIA GPU drivers installed")
        print("2. Install CUDA-enabled PyTorch:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Or visit: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    check_cuda_availability()
