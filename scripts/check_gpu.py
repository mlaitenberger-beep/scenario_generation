"""GPU and Dependency Verification Script

Run this script on your server to verify:
1. All required dependencies are installed
2. PyTorch can detect CUDA/GPUs
3. GPU memory and specifications
4. Basic tensor operations work on GPU

Usage:
    python scripts/check_gpu.py
"""

import sys
import os

print("=" * 60)
print("DEPENDENCY & GPU VERIFICATION")
print("=" * 60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check critical imports
print("\n[1/6] Checking core dependencies...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"  ✗ NumPy: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except ImportError as e:
    print(f"  ✗ Pandas: {e}")
    sys.exit(1)

try:
    import sklearn
    print(f"  ✓ scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"  ✗ scikit-learn: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"  ✗ Matplotlib: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"  ✓ PyYAML")
except ImportError as e:
    print(f"  ✗ PyYAML: {e}")
    sys.exit(1)

try:
    import tqdm
    print(f"  ✓ tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"  ✗ tqdm: {e}")
    sys.exit(1)

# Check PyTorch (critical for GPU)
print("\n[2/6] Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")
    print("\nInstall PyTorch: https://pytorch.org/get-started/locally/")
    sys.exit(1)

# Check einops (required by Diffusion-TS)
print("\n[3/6] Checking Diffusion-TS dependencies...")
try:
    import einops
    print(f"  ✓ einops {einops.__version__}")
except ImportError as e:
    print(f"  ✗ einops: {e}")
    print("  Install: pip install einops")
    sys.exit(1)

try:
    import ema_pytorch
    print(f"  ✓ ema-pytorch")
except ImportError as e:
    print(f"  ✗ ema-pytorch: {e}")
    print("  Install: pip install ema-pytorch")
    sys.exit(1)

# Check CUDA availability
print("\n[4/6] Checking CUDA/GPU availability...")
print(f"  CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Name: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"    Compute capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
else:
    print("  ⚠ No CUDA GPUs detected - will use CPU (much slower)")
    print("  If you expect GPU access, check:")
    print("    1. NVIDIA drivers installed: nvidia-smi")
    print("    2. CUDA toolkit installed")
    print("    3. PyTorch CUDA version matches your CUDA version")

# Test GPU tensor operations
print("\n[5/6] Testing GPU tensor operations...")
if torch.cuda.is_available():
    try:
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print(f"  ✓ Matrix multiplication on GPU successful")
        print(f"  ✓ Result shape: {z.shape}, device: {z.device}")
        
        # Memory check
        allocated = torch.cuda.memory_allocated(0) / 1e6
        cached = torch.cuda.memory_reserved(0) / 1e6
        print(f"  ✓ GPU memory allocated: {allocated:.2f} MB")
        print(f"  ✓ GPU memory cached: {cached:.2f} MB")
        
    except Exception as e:
        print(f"  ✗ GPU tensor test failed: {e}")
else:
    print("  ⊘ Skipping (no GPU available)")

# Check MPS (Apple Silicon) support
print("\n[6/6] Checking MPS (Apple Silicon) support...")
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"  ✓ MPS available (Apple Silicon GPU)")
    try:
        device = torch.device('mps')
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.matmul(x, y)
        print(f"  ✓ MPS tensor operations working")
    except Exception as e:
        print(f"  ⚠ MPS available but test failed: {e}")
else:
    print("  ⊘ MPS not available (not on Apple Silicon)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if torch.cuda.is_available():
    print("✓ All dependencies installed")
    print(f"✓ GPU acceleration available: {torch.cuda.device_count()} GPU(s)")
    print(f"✓ Primary device: cuda:0 ({torch.cuda.get_device_name(0)})")
    print("\nYou can run training with GPU acceleration!")
    print("The model will automatically use GPU when available.")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("✓ All dependencies installed")
    print("✓ MPS (Apple Silicon) acceleration available")
    print("\nNote: Some operations may fall back to CPU on MPS.")
else:
    print("✓ All dependencies installed")
    print("⚠ No GPU acceleration available - will use CPU")
    print("\nTraining will be significantly slower on CPU.")
    print("Consider using a GPU-enabled server for faster training.")

print("\nTo verify GPU usage during training, monitor:")
if torch.cuda.is_available():
    print("  - Run 'nvidia-smi' in another terminal")
    print("  - GPU utilization should increase during training")
    print("  - Watch GPU memory usage grow")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("  - Open Activity Monitor > Window > GPU History")
    print("  - GPU usage should increase during training")
else:
    print("  - CPU usage via 'top' or 'htop'")

print("=" * 60)
