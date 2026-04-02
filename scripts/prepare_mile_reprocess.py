#!/usr/bin/env python3
"""
Backup and remove NetCDF files for mile deployments to force reprocessing with the depth sign fix.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
DATASET_ID = "mile-adult-sese_vdr_argentina_RD-KM"
DEPLOYMENT_ID = "2013-11-09_mile-008"
DATA_ROOT = "/Volumes/WORK-SSD/Datasets/Unpublished"
DEPLOYMENT_PATH = os.path.join(DATA_ROOT, DATASET_ID, DEPLOYMENT_ID)
OUTPUTS_PATH = os.path.join(DEPLOYMENT_PATH, "outputs")

print("="*80)
print("BACKUP AND REMOVE NETCDF FILES FOR REPROCESSING")
print("="*80)
print(f"Dataset: {DATASET_ID}")
print(f"Deployment: {DEPLOYMENT_ID}")
print(f"Outputs path: {OUTPUTS_PATH}")

if not os.path.exists(OUTPUTS_PATH):
    print(f"\n❌ Outputs directory not found: {OUTPUTS_PATH}")
    exit(1)

# Find all .nc files
nc_files = list(Path(OUTPUTS_PATH).glob("*.nc"))

if not nc_files:
    print("\n✅ No .nc files found - deployment ready for fresh processing")
    exit(0)

print(f"\n📁 Found {len(nc_files)} NetCDF file(s):")
for nc_file in nc_files:
    print(f"   {nc_file.name}")

# Ask for confirmation
response = input("\n📝 Backup and remove these files to force reprocessing? (yes/no): ").strip().lower()
if response not in ["yes", "y"]:
    print("❌ Cancelled by user.")
    exit(0)

# Create backup directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = os.path.join(OUTPUTS_PATH, f"backup_nc_{timestamp}")
os.makedirs(backup_dir, exist_ok=True)

print(f"\n📦 Creating backup directory: {backup_dir}")

# Backup and remove each file
for nc_file in nc_files:
    backup_path = os.path.join(backup_dir, nc_file.name)
    print(f"  Moving: {nc_file.name} → backup_nc_{timestamp}/{nc_file.name}")
    shutil.move(str(nc_file), backup_path)

print(f"\n✅ Backed up {len(nc_files)} file(s) to: backup_nc_{timestamp}/")
print("✅ NetCDF files removed from outputs/ - ready for reprocessing")

print("\n" + "="*80)
print("NEXT STEP")
print("="*80)
print("Run the calibration workflow:")
print(f"  python workflows/01_calibrate_pressure.py --dataset {DATASET_ID} --deployment {DEPLOYMENT_ID}")
