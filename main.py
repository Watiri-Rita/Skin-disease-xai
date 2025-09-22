# main.py

import tensorflow as tf
import torch
import numpy as np

def check_environment():
    print("✅ Environment Check:")
    print("TensorFlow version:", tf.__version__)
    print("PyTorch version:", torch.__version__)
    print("NumPy version:", np.__version__)

if __name__ == "__main__":
    check_environment()
