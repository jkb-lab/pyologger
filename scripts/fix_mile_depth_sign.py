#!/usr/bin/env python3
"""
Fix negative depth values for mile-adult-sese deployments by setting conversion_factor: -1
in the dataset-level defaults of parameter_log.json.

This script will:
1. Check if parameter_log.json exists in the mile dataset directory
2. Read and display current dive_detection_settings
3. Apply the fix by setting conversion_factor: -1 in dataset defaults
4. Create a timestamped backup before making changes
5. Save the updated configuration

Usage:
    python scripts/fix_mile_depth_sign.py
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path

# Configuration
DATASET_ID = "mile-adult-sese_vdr_argentina_RD-KM"
DATA_ROOT = "/Volumes/WORK-SSD/Datasets/Unpublished"
DATASET_PATH = os.path.join(DATA_ROOT, DATASET_ID)
PARAM_LOG_PATH = os.path.join(DATASET_PATH, "parameter_log.json")

# Mile deployment IDs from config.yaml
MILE_DEPLOYMENTS = [
    "2012-11-02_mile-001",
    "2012-11-02_mile-002",
    "2012-11-01_mile-003",
    "2013-11-10_mile-006",
    "2013-11-09_mile-008",
    "2015-11-05_mile-011",
    "2015-11-05_mile-013",
    "2015-11-06_mile-014"
]


def check_drive_mounted():
    """Check if the WORK-SSD drive is mounted."""
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Error: Data directory not found: {DATA_ROOT}")
        print("   Please ensure the WORK-SSD external drive is mounted.")
        return False
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset directory not found: {DATASET_PATH}")
        return False
    return True


def read_parameter_log():
    """Read and parse parameter_log.json."""
    if not os.path.exists(PARAM_LOG_PATH):
        print(f"ℹ️  parameter_log.json does not exist yet: {PARAM_LOG_PATH}")
        return None
    
    with open(PARAM_LOG_PATH, 'r') as f:
        return json.load(f)


def display_current_settings(config):
    """Display current dive_detection_settings for all deployments."""
    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    
    for entry in config:
        deployment_id = entry.get("deployment_id")
        dive_settings = entry.get("dive_detection_settings", {})
        
        conversion_factor = dive_settings.get("conversion_factor", "not set")
        disable_sign_flip = dive_settings.get("disable_automatic_sign_flipping", "not set")
        
        if deployment_id == "__dataset_defaults__":
            print(f"\n📁 DATASET DEFAULTS:")
        else:
            print(f"\n📄 {deployment_id}:")
        
        print(f"   conversion_factor: {conversion_factor}")
        print(f"   disable_automatic_sign_flipping: {disable_sign_flip}")


def create_backup(filepath):
    """Create a timestamped backup of the parameter_log.json file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace(".json", f"_backup_{timestamp}.json")
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backup created: {backup_path}")
    return backup_path


def apply_fix(config):
    """Apply the fix: set conversion_factor: -1 in dataset defaults."""
    # Find or create dataset defaults entry
    defaults_entry = None
    for entry in config:
        if entry.get("deployment_id") == "__dataset_defaults__":
            defaults_entry = entry
            break
    
    if defaults_entry is None:
        # Create new defaults entry
        defaults_entry = {
            "deployment_id": "__dataset_defaults__",
            "deployment_folder_path": DATASET_PATH,
            "logger_ids": [],
            "settings": {}
        }
        config.insert(0, defaults_entry)
        print("\n✨ Created new __dataset_defaults__ entry")
    
    # Ensure dive_detection_settings section exists
    if "dive_detection_settings" not in defaults_entry:
        defaults_entry["dive_detection_settings"] = {}
    
    # Apply the fix
    defaults_entry["dive_detection_settings"]["conversion_factor"] = -1.0
    
    # Also ensure automatic sign flipping is not disabled
    if "disable_automatic_sign_flipping" in defaults_entry["dive_detection_settings"]:
        if defaults_entry["dive_detection_settings"]["disable_automatic_sign_flipping"]:
            print("   ℹ️  Note: disable_automatic_sign_flipping was true, leaving it (conversion_factor takes precedence)")
    
    print("\n🔧 APPLIED FIX:")
    print("   Set conversion_factor: -1.0 in dataset defaults")
    print("   This will multiply depth values by -1, converting negative to positive")
    
    return config


def check_deployment_overrides(config):
    """Check if any deployment-specific entries override the conversion_factor."""
    print("\n" + "="*80)
    print("CHECKING DEPLOYMENT-SPECIFIC OVERRIDES")
    print("="*80)
    
    overrides_found = False
    for entry in config:
        deployment_id = entry.get("deployment_id")
        if deployment_id == "__dataset_defaults__":
            continue
        
        dive_settings = entry.get("dive_detection_settings", {})
        if "conversion_factor" in dive_settings or "disable_automatic_sign_flipping" in dive_settings:
            overrides_found = True
            print(f"\n⚠️  {deployment_id} has deployment-specific overrides:")
            if "conversion_factor" in dive_settings:
                print(f"   conversion_factor: {dive_settings['conversion_factor']}")
            if "disable_automatic_sign_flipping" in dive_settings:
                print(f"   disable_automatic_sign_flipping: {dive_settings['disable_automatic_sign_flipping']}")
    
    if not overrides_found:
        print("\n✅ No deployment-specific overrides found.")
        print("   All deployments will inherit conversion_factor: -1.0 from dataset defaults.")


def save_parameter_log(config, filepath):
    """Save the updated parameter_log.json."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n✅ Saved updated configuration to: {filepath}")


def main():
    print("="*80)
    print("FIX NEGATIVE DEPTH VALUES - MILE DEPLOYMENTS")
    print("="*80)
    print(f"\nDataset: {DATASET_ID}")
    print(f"Deployments affected: {len(MILE_DEPLOYMENTS)}")
    print(f"   {', '.join(MILE_DEPLOYMENTS[:3])} ...")
    
    # Step 1: Check drive is mounted
    if not check_drive_mounted():
        return 1
    
    # Step 2: Read current configuration
    config = read_parameter_log()
    
    if config is None:
        print("\n❌ parameter_log.json does not exist.")
        print("   The file should be created automatically when you first run 01_calibrate_pressure.py")
        print("   for any mile deployment. Run that workflow first, then run this script.")
        return 1
    
    # Step 3: Display current settings
    display_current_settings(config)
    
    # Step 4: Ask for confirmation
    print("\n" + "="*80)
    print("PROPOSED FIX")
    print("="*80)
    print("Set conversion_factor: -1.0 in dataset defaults section")
    print("This will flip all depth values from negative to positive for all mile deployments.")
    print("\nA timestamped backup will be created before making changes.")
    
    response = input("\n📝 Proceed with fix? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("❌ Cancelled by user.")
        return 0
    
    # Step 5: Create backup
    print("\n" + "="*80)
    print("APPLYING FIX")
    print("="*80)
    create_backup(PARAM_LOG_PATH)
    
    # Step 6: Apply fix
    updated_config = apply_fix(config)
    
    # Step 7: Check for deployment-specific overrides
    check_deployment_overrides(updated_config)
    
    # Step 8: Save updated configuration
    save_parameter_log(updated_config, PARAM_LOG_PATH)
    
    # Step 9: Display summary
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Reprocess all mile deployments to apply the fix:")
    print("   cd /Users/jessiekb/Documents/GitHub/EcoViz_DiveDB/pyologger")
    print("   source ../venv/bin/activate")
    print("")
    for deployment_id in MILE_DEPLOYMENTS:
        print(f"   python workflows/01_calibrate_pressure.py --dataset {DATASET_ID} --deployment {deployment_id}")
    print("\n2. Verify the fix by loading a deployment in the dashboard and checking depth values are positive.")
    print("\n3. Check the terminal output during processing - it should show:")
    print('   "✅ Baseline-adjusted depth is not mostly negative; no sign change applied."')
    print("   (because conversion_factor already flipped the sign before the automatic check)")
    
    print("\n✅ Configuration update complete!")
    return 0


if __name__ == "__main__":
    exit(main())
