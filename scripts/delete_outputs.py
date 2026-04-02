#!/usr/bin/env python3
"""
Delete files inside deployment outputs folders for a specific dataset.

Usage:
    python scripts/delete_outputs.py --dataset <dataset_id> [--dry-run] [--trash | --delete] [--remove-folder]

Example:
    python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --dry-run --delete
    python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --delete
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


def iter_deployment_folders(dataset_path: Path):
    for entry in sorted(dataset_path.iterdir()):
        if entry.is_dir() and not entry.name.startswith("00_"):
            yield entry


def is_hidden_path(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part.startswith(".") for part in rel_parts)


def trash_dir() -> Path:
    return Path.home() / ".Trash"


def unique_trash_path(filename: str) -> Path:
    target = trash_dir() / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 1
    while True:
        candidate = trash_dir() / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_file_to_trash(file_path: Path, outputs_folder: Path, deployment_name: str):
    rel_path = file_path.relative_to(outputs_folder)
    safe_name = "__".join((deployment_name, *rel_path.parts))
    target = unique_trash_path(safe_name)
    shutil.move(str(file_path), str(target))


def empty_trash(*, dry_run: bool):
    trash = trash_dir()
    items = list(trash.iterdir()) if trash.exists() else []
    if dry_run:
        print(f"Would empty Trash: {len(items)} items")
        return
    for item in items:
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    print(f"Emptied Trash: {len(items)} items")


def delete_outputs_for_dataset(dataset_id, deployment_id=None, dry_run=True, use_trash=False, remove_folder=False):
    """
    Delete all files inside outputs folders for deployments in the specified dataset.
    
    Parameters
    ----------
    dataset_id : str
        Dataset ID from config.yaml
    dry_run : bool
        If True, only show what would be deleted without actually deleting
    """
    config = load_config()
    data_dir = config["paths"]["local_private_data"]
    
    dataset_path = Path(data_dir) / dataset_id

    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_path}")
        available_datasets = sorted(
            entry.name for entry in Path(data_dir).iterdir()
            if entry.is_dir() and not entry.name.startswith("00_")
        )
        if available_datasets:
            print("\nAvailable datasets:")
            for ds in available_datasets:
                print(f"  - {ds}")
        return

    deployment_folders = list(iter_deployment_folders(dataset_path))
    if deployment_id:
        deployment_folders = [path for path in deployment_folders if path.name == deployment_id]
        if not deployment_folders:
            print(f"❌ Deployment '{deployment_id}' not found in dataset: {dataset_id}")
            return
    if not deployment_folders:
        print(f"❌ No deployment folders found in dataset: {dataset_path}")
        return
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN MODE - No files will be removed")
    elif use_trash:
        print(f"TRASH MODE - Files will be moved to {trash_dir()}")
    else:
        print(f"DELETE MODE - Files will be permanently removed")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_id}")
    print(f"Path: {dataset_path}")
    if deployment_id:
        print(f"Deployment: {deployment_id}")
    print(f"Deployments: {len(deployment_folders)}")
    print(f"{'='*80}\n")
    
    deleted_count = 0
    skipped_count = 0
    total_size = 0
    hidden_file_total = 0
    
    for deployment_folder in deployment_folders:
        deployment_id = deployment_folder.name
        outputs_folder = deployment_folder / "outputs"

        if not outputs_folder.exists():
            print(f"⚠️  {deployment_id}: No outputs folder")
            skipped_count += 1
            continue

        files_to_delete = sorted(path for path in outputs_folder.rglob("*") if path.is_file())
        counted_files = sorted(path for path in files_to_delete if not is_hidden_path(path, outputs_folder))
        hidden_files = sorted(path for path in files_to_delete if is_hidden_path(path, outputs_folder))
        hidden_file_total += len(hidden_files)
        dirs_to_prune = sorted(
            [path for path in outputs_folder.rglob("*") if path.is_dir()],
            key=lambda p: len(p.parts),
            reverse=True,
        )

        if not files_to_delete and not dirs_to_prune:
            if dry_run and remove_folder:
                print(f"📁 {deployment_id}: Would remove empty outputs folder")
            elif remove_folder:
                try:
                    outputs_folder.rmdir()
                    print(f"✅ {deployment_id}: Removed empty outputs folder")
                    deleted_count += 1
                except OSError as e:
                    print(f"❌ {deployment_id}: Error removing empty outputs folder - {e}")
                    skipped_count += 1
                continue
            else:
                print(f"⚠️  {deployment_id}: outputs folder is already empty")
                skipped_count += 1
                continue

        folder_size = sum(path.stat().st_size for path in counted_files)
        total_size += folder_size
        size_mb = folder_size / (1024 * 1024)

        file_count = len(counted_files)

        if dry_run:
            action = "move to Trash" if use_trash else "delete"
            print(f"📁 {deployment_id}: Would {action} {file_count} files ({size_mb:.2f} MB)")
            for file_path in counted_files:
                rel_path = file_path.relative_to(outputs_folder)
                rel_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   - {rel_path} ({rel_size_mb:.2f} MB)")
            if remove_folder:
                print("   - [outputs folder itself]")
        else:
            try:
                for file_path in files_to_delete:
                    if use_trash:
                        move_file_to_trash(file_path, outputs_folder, deployment_id)
                    else:
                        file_path.unlink()
                for dir_path in dirs_to_prune:
                    try:
                        dir_path.rmdir()
                    except OSError:
                        pass
                if remove_folder:
                    try:
                        outputs_folder.rmdir()
                    except OSError as e:
                        raise OSError(f"could not remove outputs folder '{outputs_folder}': {e}") from e
                verb = "Moved to Trash" if use_trash else "Deleted"
                print(f"✅ {deployment_id}: {verb} {file_count} files ({size_mb:.2f} MB)")
                deleted_count += 1
            except Exception as e:
                print(f"❌ {deployment_id}: Error cleaning outputs - {e}")
                skipped_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if dry_run:
        print(f"Would clean: {len(deployment_folders) - skipped_count} deployment outputs")
    else:
        print(f"Cleaned: {deleted_count} deployment outputs")
    
    print(f"Skipped: {skipped_count} deployments")
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB ({total_size / (1024 * 1024 * 1024):.2f} GB)")
    print(f"Hidden files excluded from counts but still targeted: {hidden_file_total}")
    print(f"{'='*80}\n")
    
    if dry_run:
        if use_trash:
            print("💡 Run without --dry-run and with --trash to move files to Trash")
        else:
            print("💡 Run without --dry-run to actually delete the files")


def main():
    parser = argparse.ArgumentParser(
        description="Delete files inside outputs folders for a specific dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be deleted
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --dry-run
  
  # Actually delete the files permanently
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --delete

  # Delete files and remove each outputs folder itself
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --delete --remove-folder

  # Move files to Trash instead of deleting permanently
  python scripts/delete_outputs.py --dataset mian-juv-nese_sleep_lml-ano_JKB --trash

  # Empty Trash as a separate explicit step
  python scripts/delete_outputs.py --empty-trash

  # Clean only one deployment's outputs
  python scripts/delete_outputs.py --dataset pale-adult-lion_vid-accel_africa_TW --deployment 2014-09-24_pale-015 --dry-run --delete
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset folder name under local_private_data"
    )

    parser.add_argument(
        "--deployment",
        type=str,
        help="Optional deployment folder name to limit cleanup to one deployment"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be removed without actually doing it"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--trash",
        action="store_true",
        help="Move files to macOS Trash instead of permanently deleting them"
    )
    mode_group.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete files immediately, skipping Trash"
    )

    parser.add_argument(
        "--empty-trash",
        action="store_true",
        help="Empty macOS Trash. Can be run by itself as a separate second step."
    )

    parser.add_argument(
        "--remove-folder",
        action="store_true",
        help="Also remove each outputs folder itself after cleaning its contents"
    )
    
    args = parser.parse_args()

    if args.empty_trash and not args.dataset:
        empty_trash(dry_run=args.dry_run)
        return

    if not args.dataset:
        parser.error("--dataset is required unless using --empty-trash by itself")

    use_trash = bool(args.trash)

    delete_outputs_for_dataset(
        args.dataset,
        deployment_id=args.deployment,
        dry_run=args.dry_run,
        use_trash=use_trash,
        remove_folder=args.remove_folder,
    )

    if args.empty_trash:
        empty_trash(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
