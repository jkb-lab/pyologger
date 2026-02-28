#!/usr/bin/env python3
"""
Delete outputs folders for a specific dataset ID.

Usage:
    python scripts/delete_outputs.py --dataset <dataset_id> [--dry-run]

Example:
    python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --dry-run
    python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB
"""

import os
import shutil
import yaml
import argparse
from pathlib import Path


def load_config():
    """Load config.yaml from pyologger root directory."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def delete_outputs_for_dataset(dataset_id, dry_run=True):
    """
    Delete all outputs folders for deployments in the specified dataset.
    
    Parameters
    ----------
    dataset_id : str
        Dataset ID from config.yaml
    dry_run : bool
        If True, only show what would be deleted without actually deleting
    """
    config = load_config()
    data_dir = config["paths"]["local_private_data"]
    
    # Check if dataset exists in config
    if dataset_id not in config["datasets"]:
        print(f"❌ Dataset '{dataset_id}' not found in config.yaml")
        print(f"\nAvailable datasets:")
        for ds in config["datasets"].keys():
            print(f"  - {ds}")
        return
    
    deployments = config["datasets"][dataset_id]["deployments"]
    dataset_path = Path(data_dir) / dataset_id
    
    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_path}")
        return
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN MODE - No files will be deleted")
    else:
        print(f"DELETE MODE - Files will be permanently removed")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_id}")
    print(f"Path: {dataset_path}")
    print(f"Deployments: {len(deployments)}")
    print(f"{'='*80}\n")
    
    deleted_count = 0
    skipped_count = 0
    total_size = 0
    
    for deployment_id in deployments:
        # Handle deployment folders with suffixes (e.g., _media)
        deployment_folder = None
        exact_match = dataset_path / deployment_id
        
        if exact_match.exists():
            deployment_folder = exact_match
        else:
            # Look for folders starting with deployment_id
            for entry in dataset_path.iterdir():
                if entry.is_dir() and entry.name.startswith(deployment_id):
                    # Require _ or end of string after ID to avoid false matches
                    suffix = entry.name[len(deployment_id):]
                    if not suffix or suffix[0] == '_':
                        deployment_folder = entry
                        break
        
        if not deployment_folder:
            print(f"⚠️  {deployment_id}: Deployment folder not found")
            skipped_count += 1
            continue
        
        outputs_folder = deployment_folder / "outputs"
        
        if not outputs_folder.exists():
            print(f"⚠️  {deployment_id}: No outputs folder")
            skipped_count += 1
            continue
        
        # Calculate size
        folder_size = sum(f.stat().st_size for f in outputs_folder.rglob('*') if f.is_file())
        total_size += folder_size
        size_mb = folder_size / (1024 * 1024)
        
        # Count files
        file_count = sum(1 for _ in outputs_folder.rglob('*') if _.is_file())
        
        if dry_run:
            print(f"📁 {deployment_id}: Would delete {file_count} files ({size_mb:.2f} MB)")
        else:
            try:
                shutil.rmtree(outputs_folder)
                print(f"✅ {deployment_id}: Deleted {file_count} files ({size_mb:.2f} MB)")
                deleted_count += 1
            except Exception as e:
                print(f"❌ {deployment_id}: Error deleting - {e}")
                skipped_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if dry_run:
        print(f"Would delete: {len(deployments) - skipped_count} deployment outputs")
    else:
        print(f"Deleted: {deleted_count} deployment outputs")
    
    print(f"Skipped: {skipped_count} deployments")
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB ({total_size / (1024 * 1024 * 1024):.2f} GB)")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("💡 Run without --dry-run to actually delete the files")


def main():
    parser = argparse.ArgumentParser(
        description="Delete outputs folders for a specific dataset ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --dry-run
  
  # Actually delete the files
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB
  
  # Delete for multiple datasets
  python scripts/delete_outputs.py --dataset mile-adult-sese_vdr_argentina_RD-KM
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset ID from config.yaml"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    delete_outputs_for_dataset(args.dataset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
