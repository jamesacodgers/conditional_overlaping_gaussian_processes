#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data utilities for Original GP LVM experiments
"""

import os
import urllib.request
import tarfile
import shutil

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
    
    # Copy to expected location
    shutil.copy2('DataTrn.txt', 'data/oil_data.txt')
    shutil.copy2('DataTrnLbls.txt', 'data/oil_labels.txt')
    
    # Clean up temporary files
    if os.path.exists('3PhData.tar.gz'):
        os.remove('3PhData.tar.gz')
    if os.path.exists('DataTrn.txt'):
        os.remove('DataTrn.txt')
    if os.path.exists('DataTrnLbls.txt'):
        os.remove('DataTrnLbls.txt')
    
    print("Oilflow data downloaded and set up successfully!")
