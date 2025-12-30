#!/usr/bin/env python3
"""
Colab Environment Fix

Execute this FIRST before running any training scripts

Usage:
    import requests
    url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_setup_fix.py'
    script = requests.get(url).text
    exec(script)
"""

print("\n" + "="*80)
print("COLAB ENVIRONMENT FIX")
print("="*80 + "\n")

import subprocess
import sys

# Step 1: Fix numpy
print("Step 1: Fixing NumPy...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy==1.24.3", "-q"])
    print("  ✓ NumPy 1.24.3 installed")
except Exception as e:
    print(f"  Note: {e}")
    pass

# Step 2: Fix Pandas
print("\nStep 2: Fixing Pandas...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pandas==2.0.3", "-q"])
    print("  ✓ Pandas 2.0.3 installed")
except Exception as e:
    print(f"  Note: {e}")
    pass

# Step 3: Install scikit-learn
print("\nStep 3: Installing scikit-learn...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "-q"])
    print("  ✓ scikit-learn installed")
except Exception as e:
    print(f"  Note: {e}")
    pass

# Step 4: Verify installation
print("\nStep 4: Verifying installation...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"  ✗ NumPy error: {e}")

try:
    import pandas as pd
    print(f"  ✓ Pandas {pd.__version__}")
except Exception as e:
    print(f"  ✗ Pandas error: {e}")

try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"  ✗ TensorFlow error: {e}")

try:
    from sklearn.utils.class_weight import compute_class_weight
    print(f"  ✓ scikit-learn")
except Exception as e:
    print(f"  ✗ scikit-learn error: {e}")

try:
    from huggingface_hub import hf_hub_download
    print(f"  ✓ huggingface_hub")
except Exception as e:
    print(f"  Note: huggingface_hub not found, will be installed automatically")

print("\n" + "="*80)
print("ENVIRONMENT FIX COMPLETE!")
print("="*80)

print("\nNow you can run the training script:")
print("""
SYMBOL = 'BTC'
INTERVAL = '1h'
EPOCHS = 50
LOOKBACK = 40

import requests
url = 'https://raw.githubusercontent.com/caizongxun/v2bot/main/optimization/colab_training_final.py'
script = requests.get(url, timeout=120).text
exec(script, {'SYMBOL': SYMBOL, 'INTERVAL': INTERVAL, 'EPOCHS': EPOCHS, 'LOOKBACK': LOOKBACK})
""")

print("\n" + "="*80 + "\n")
