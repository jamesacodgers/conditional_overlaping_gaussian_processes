#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data utilities for Original GP LVM experiments
"""

import os
import urllib.request
import tarfile
import shutil
import numpy as np
import torch
from sklearn.datasets import make_swiss_roll
from torchvision import datasets

def download_oilflow_data_if_needed():
    """Download oilflow data if not present"""
    data_url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    
    # Check if data files already exist
    if os.path.exists('data/oil_data.txt') and os.path.exists('data/oil_labels.txt'):
        print("Oilflow data files already exist.")
        return
    
    print("Downloading oilflow data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download and extract data
    urllib.request.urlretrieve(data_url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')
        f.extract('DataTrnFrctns.txt')
    
    # Copy to expected location
    shutil.copy2('DataTrn.txt', 'data/oil_data.txt')
    shutil.copy2('DataTrnLbls.txt', 'data/oil_labels.txt')
    shutil.copy2('DataTrnFrctns.txt', 'data/oil_fractions.txt')
    
    # Clean up temporary files (keep DataTrnFrctns.txt for plotting functions)
    if os.path.exists('3PhData.tar.gz'):
        os.remove('3PhData.tar.gz')
    if os.path.exists('DataTrn.txt'):
        os.remove('DataTrn.txt')
    if os.path.exists('DataTrnLbls.txt'):
        os.remove('DataTrnLbls.txt')
    # Note: Keep DataTrnFrctns.txt in current directory for plotting functions
    
    print("Oilflow data downloaded and set up successfully!")

def generate_swiss_roll_data(n_samples=1000, noise=0.1, random_state=42):
    """Generate Swiss roll dataset
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    noise : float
        Standard deviation of Gaussian noise
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    Y : torch.Tensor
        High-dimensional observations (n_samples x 3)
    X_true : torch.Tensor
        True 2D latent coordinates
    labels : np.array
        Color labels based on position
    """
    # Generate swiss roll data
    Y_np, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Convert to torch tensors
    Y = torch.tensor(Y_np, dtype=torch.float64)
    
    # Use the parameter t as labels (for visualization)
    labels = t
    
    # Create a simple 2D latent representation from the roll parameter
    # This is just for reference - the model will learn its own latent space
    X_true = torch.tensor(np.column_stack([t, np.zeros_like(t)]), dtype=torch.float64)
    
    return Y, X_true, labels

def load_mnist_data():
    """
    Load MNIST data using PyTorch's torchvision (no TensorFlow needed).
    Returns data in the same format as load_real_data('mnist') from generalised_gplvm_code.
    
    Consistent with original load_real_data('mnist'):
    - Y as torch.float64 tensor (instead of float32 for better precision)
    - labels as numpy array (not tensor)
    - q = None, X = None
    
    Returns
    -------
    n : int
        Total number of samples (70000)
    d : int
        Data dimensionality (784 = 28*28)
    q : None
        Not used (for compatibility with load_real_data interface)
    X : None
        Not used (for compatibility with load_real_data interface)
    Y : torch.Tensor
        Image data flattened to vectors (70000 x 784), dtype=float64
    labels : np.ndarray
        Digit labels 0-9 (70000,), dtype=int64
    """
    # Download MNIST dataset using torchvision (will download on first run)
    train_dataset = datasets.MNIST(root='./data/MNIST', train=True, download=True)
    test_dataset = datasets.MNIST(root='./data/MNIST', train=False, download=True)
    
    # Extract data and labels
    train_data = train_dataset.data.numpy()  # (60000, 28, 28)
    train_labels = train_dataset.targets.numpy()  # (60000,)
    test_data = test_dataset.data.numpy()  # (10000, 28, 28)
    test_labels = test_dataset.targets.numpy()  # (10000,)
    
    # Combine train and test datasets
    Y = np.vstack([train_data.reshape(len(train_data), -1), 
                   test_data.reshape(len(test_data), -1)])  # (70000, 784)
    labels = np.hstack([train_labels, test_labels])  # (70000,)
    
    # Convert Y to torch tensor (float64 for precision), keep labels as numpy
    Y = torch.tensor(Y, dtype=torch.float64)
    
    # Match original load_real_data interface
    n = len(Y)  # 70000
    d = len(Y.T)  # 784 (28*28) - use len(Y.T) to match original
    q = X = None  # Match original: q = X = None
    
    print(f"MNIST data loaded: {n} samples, {d} dimensions (28x28 images)")
    
    return n, d, q, X, Y, labels
