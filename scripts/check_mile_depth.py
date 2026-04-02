#!/usr/bin/env python3
"""
Quick script to check the depth values in a mile deployment to see if they're negative or positive.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyologger.utils.folder_manager import load_configuration

# Configuration
DATASET_ID = "mile-adult-sese_vdr_argentina_RD-KM"
DEPLOYMENT_ID = "2013-11-09_mile-008"
DATA_ROOT = "/Volumes/WORK-SSD/Datasets/Unpublished"
DEPLOYMENT_PATH = os.path.join(DATA_ROOT, DATASET_ID, DEPLOYMENT_ID)
PKL_PATH = os.path.join(DEPLOYMENT_PATH, "outputs", "data.pkl")

print("="*80)
print("CHECKING DEPTH VALUES IN MILE DEPLOYMENT")
print("="*80)
print(f"Dataset: {DATASET_ID}")
print(f"Deployment: {DEPLOYMENT_ID}")
print(f"Path: {PKL_PATH}")

if not os.path.exists(PKL_PATH):
    print(f"\n❌ data.pkl not found at: {PKL_PATH}")
    sys.exit(1)

print("\n📂 Loading data.pkl...")
with open(PKL_PATH, 'rb') as f:
    data_pkl = pickle.load(f)

print(f"✅ Loaded successfully")

# Check what signals are available
print("\n" + "="*80)
print("AVAILABLE SIGNALS")
print("="*80)
for signal_name in data_pkl.signal_data.keys():
    print(f"  - {signal_name}")

# Check depth signal
if 'depth' in data_pkl.signal_data:
    depth_df = data_pkl.signal_data['depth']
    # Find the actual depth column (exclude datetime columns)
    numeric_cols = depth_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("\n❌ No numeric columns found in depth signal")
        sys.exit(1)
    depth_col = numeric_cols[0]  # Usually 'depth' or similar
    depth_values = depth_df[depth_col].dropna()
    
    print("\n" + "="*80)
    print("DEPTH SIGNAL ANALYSIS")
    print("="*80)
    print(f"Column name: {depth_col}")
    print(f"Total values: {len(depth_values)}")
    print(f"Non-NaN values: {len(depth_values)}")
    
    # Statistics
    print(f"\nDepth statistics:")
    print(f"  Min: {depth_values.min():.2f} m")
    print(f"  Max: {depth_values.max():.2f} m")
    print(f"  Mean: {depth_values.mean():.2f} m")
    print(f"  Median: {depth_values.median():.2f} m")
    
    # Count positive vs negative
    negative_count = (depth_values < 0).sum()
    positive_count = (depth_values > 0).sum()
    zero_count = (depth_values == 0).sum()
    
    print(f"\nSign distribution:")
    print(f"  Negative values: {negative_count} ({negative_count/len(depth_values)*100:.1f}%)")
    print(f"  Zero values: {zero_count} ({zero_count/len(depth_values)*100:.1f}%)")
    print(f"  Positive values: {positive_count} ({positive_count/len(depth_values)*100:.1f}%)")
    
    # Determine if fix is needed
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    if negative_count > positive_count:
        print("❌ PROBLEM: Depth values are mostly NEGATIVE")
        print("   This deployment needs to be reprocessed with conversion_factor: -1")
    else:
        print("✅ GOOD: Depth values are mostly POSITIVE")
        print("   This deployment appears to have correct depth signs")
    
    # Check transformation log
    if 'depth' in data_pkl.signal_info:
        trans_log = data_pkl.signal_info['depth'].get('transformation_log', [])
        print(f"\nTransformation log: {trans_log}")
    
else:
    print("\n❌ No 'depth' signal found in data.pkl")
    print("   Available signals:", list(data_pkl.signal_data.keys()))

# Check if there's a pressure signal that hasn't been converted to depth
if 'pressure' in data_pkl.signal_data:
    pressure_df = data_pkl.signal_data['pressure']
    pressure_col = 'pressure' if 'pressure' in pressure_df.columns else pressure_df.columns[0]
    pressure_values = pressure_df[pressure_col].dropna()
    
    print("\n" + "="*80)
    print("PRESSURE SIGNAL FOUND")
    print("="*80)
    print("There's also a pressure signal that might need to be converted to depth")
    print(f"  Min: {pressure_values.min():.2f}")
    print(f"  Max: {pressure_values.max():.2f}")
    print(f"  Mean: {pressure_values.mean():.2f}")
