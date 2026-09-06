"""Set up an NDP demo download into a pyologger dataset/deployment hierarchy.

NDP Launcher downloads land in a dataset-named sibling folder next to this
repo, e.g.:
    _User-Persistent-Storage_CephBlock_/
    +-- pyologger/                                  <- this repo
    +-- subset-of-data-for-brain-activity-.../       <- NDP download, name varies

containing files named like:
    2020-04-10_mian-002_data.pkl
    2020-04-10_mian-002.nc
    2020-04-10_mian-002_trimmed.EDF

This script finds those files under --source and moves them into a new
`pyologger_demo_data/` folder that sits *next to* this repo (not inside it,
so the repo stays clean):
    _User-Persistent-Storage_CephBlock_/
    +-- pyologger/
    +-- pyologger_demo_data/
    |   +-- 00_Metadata/metadata_snapshot.pkl
    |   +-- <dataset_id>/<deployment_id>/outputs/data.pkl
    |   +-- <dataset_id>/<deployment_id>/outputs/<deployment_id>_output.nc
    |   +-- <dataset_id>/<deployment_id>/01_raw-data/<deployment_id>_NL-02_001.edf
    +-- subset-of-data-for-brain-activity-.../       <- emptied out, now sourced

using the dataset/deployment mapping in demo_config.yaml, then installs
demo_config.yaml as config.yaml inside pyologger/ (only if config.yaml doesn't
already exist, so it never clobbers a real config). demo_config.yaml points
`paths.local_private_data` at `../pyologger_demo_data` to match.

It also installs .env from .env.example (only if .env doesn't already exist),
commenting out variables the demo doesn't need (Notion tokens/database IDs --
metadata comes from the pre-trimmed snapshot fetched above, not live Notion --
and a few pyologger-specific paths unused by the demo) so participants are
only asked to fill in IMMICH_API_KEY/IMMICH_BASE_URL if they want mini-dash's
video playback, and can otherwise run the demo untouched.

If the raw EDF wasn't part of the NDP download, it's fetched from the public
Pelican/OSDF namespace (osdf:///jkb-lab-public/demo/) instead. A trimmed,
demo-scoped metadata_snapshot.pkl (built with scripts/trim_metadata_snapshot.py)
is also fetched into 00_Metadata/, so `workflows/00_load_data.py` can run
fully offline (no Notion token needed) if students want to try the Snakemake
pipeline from raw data.

Usage:
    # From inside this repo -- searches every sibling folder next to it for
    # the demo files, whatever NDP happened to name the download folder, and
    # builds the hierarchy in a sibling pyologger_demo_data/ folder:
    cd pyologger && python setup_demo.py

    # Or point at specific folders directly:
    python setup_demo.py --source /path/to/downloads --dest /path/to/demo_data
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "demo_config.yaml"
DEFAULT_DEST_DIRNAME = "pyologger_demo_data"
PELICAN_DEMO_NAMESPACE = "osdf:///jkb-lab-public/demo"

# Variable name prefixes/exact names not needed for the demo -- these get
# commented out in the installed .env rather than removed, so the file still
# documents what they're for if a participant wants to go further (e.g. wire
# up their own Notion workspace). Everything else in .env.example (notably
# IMMICH_API_KEY, IMMICH_BASE_URL, CONFIG_PATH) is left active.
ENV_DEMO_UNUSED_PREFIXES = ("NOTION_",)
ENV_DEMO_UNUSED_EXACT = {
    "PROCESSED_DATA_DIR",
    "PREPROCESSED_DATA_FILENAME",
    "SEGMENTATION_RUNS_PATH",
    "HOST_DELTA_LAKE_PATH",
    "PYOLOGGER_ORIENTATION_MAX_RAW_ROWS",
    "PYOLOGGER_ORIENTATION_MAX_OUTPUT_ROWS",
    "PYOLOGGER_VIDEO_DIR",
    "PYOLOGGER_IMMICH_ENV",
}


def build_deployment_to_dataset(config: dict) -> dict[str, str]:
    mapping = {}
    for dataset_id, dataset_cfg in config.get("datasets", {}).items():
        for deployment_id in dataset_cfg.get("deployments", []):
            mapping[deployment_id] = dataset_id
    return mapping


def find_deployment_files(source: Path, deployment_id: str) -> dict[str, Path]:
    found = {}
    pkl_matches = list(source.rglob(f"{deployment_id}_data.pkl"))
    nc_matches = list(source.rglob(f"{deployment_id}.nc"))
    edf_matches = list(source.rglob(f"{deployment_id}_trimmed.EDF")) or list(
        source.rglob(f"{deployment_id}_trimmed.edf")
    )
    if pkl_matches:
        found["pkl"] = pkl_matches[0]
    if nc_matches:
        found["nc"] = nc_matches[0]
    if edf_matches:
        found["edf"] = edf_matches[0]
    return found


def resolve_default_source(deployment_ids: list[str], dest: Path) -> Path:
    """Find the NDP download folder without depending on its (variable) name.

    NDP Launcher drops downloads into a dataset-named folder that sits next
    to this repo, not inside it and not with a predictable name. So instead
    of assuming a fixed path, search every sibling of this repo for one that
    actually contains the expected demo files, and use whichever one matches.
    Falls back to the current directory if no sibling matches (e.g. the
    files were downloaded directly into cwd).
    """
    repo_parent = SCRIPT_DIR.parent
    candidates = [
        p for p in repo_parent.iterdir()
        if p.is_dir() and p != SCRIPT_DIR and p.resolve() != dest.resolve()
    ]

    for candidate in candidates:
        for deployment_id in deployment_ids:
            if find_deployment_files(candidate, deployment_id):
                return candidate

    return Path(".")


def fetch_from_pelican(remote_name: str, target: Path) -> bool:
    """Fetch a file from the public Pelican demo namespace."""
    remote_path = f"{PELICAN_DEMO_NAMESPACE}/{remote_name}"
    print(f"[fetch] {remote_path} -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["pelican", "object", "get", remote_path, str(target)],
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[warn] could not fetch {remote_name} from Pelican: {e}")
        return False


def organize(source: Path, dest: Path, config: dict, dry_run: bool = False) -> None:
    deployment_to_dataset = build_deployment_to_dataset(config)

    for deployment_id, dataset_id in deployment_to_dataset.items():
        files = find_deployment_files(source, deployment_id)
        if not files:
            print(f"[skip] no downloaded files found for {deployment_id}")
            continue

        deployment_dir = dest / dataset_id / deployment_id
        outputs_dir = deployment_dir / "outputs"
        rawdata_dir = deployment_dir / "01_raw-data"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        if "pkl" in files:
            target = outputs_dir / "data.pkl"
            print(f"[move] {files['pkl']} -> {target}")
            if not dry_run:
                shutil.move(str(files["pkl"]), target)
        else:
            print(f"[warn] missing *_data.pkl for {deployment_id}")

        if "nc" in files:
            target = outputs_dir / f"{deployment_id}_output.nc"
            print(f"[move] {files['nc']} -> {target}")
            if not dry_run:
                shutil.move(str(files["nc"]), target)
        else:
            print(f"[warn] missing *.nc for {deployment_id}")

        edf_target = rawdata_dir / f"{deployment_id}_NL-02_001.edf"
        if "edf" in files:
            rawdata_dir.mkdir(parents=True, exist_ok=True)
            print(f"[move] {files['edf']} -> {edf_target}")
            if not dry_run:
                shutil.move(str(files["edf"]), edf_target)
        elif not dry_run:
            fetch_from_pelican(f"{deployment_id}_trimmed.EDF", edf_target)
        else:
            print(f"[dry-run] would fetch missing EDF for {deployment_id} from Pelican")

    metadata_target = dest / "00_Metadata" / "metadata_snapshot.pkl"
    if not dry_run:
        fetch_from_pelican("metadata_snapshot.pkl", metadata_target)
    else:
        print(f"[dry-run] would fetch demo metadata_snapshot.pkl -> {metadata_target}")


def install_config(config_path: Path, dry_run: bool = False) -> None:
    dest_config = SCRIPT_DIR / "config.yaml"
    if dest_config.exists():
        print(f"[skip] {dest_config} already exists, not overwriting")
        return
    print(f"[copy] {config_path} -> {dest_config}")
    if not dry_run:
        shutil.copy(config_path, dest_config)


def _is_demo_unused(var_name: str) -> bool:
    if var_name in ENV_DEMO_UNUSED_EXACT:
        return True
    return any(var_name.startswith(prefix) for prefix in ENV_DEMO_UNUSED_PREFIXES)


def _comment_out_unused_env_lines(text: str, config_path: Path) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            var_name = stripped.split("=", 1)[0].strip()
            if var_name == "CONFIG_PATH":
                lines.append(f"CONFIG_PATH={config_path}")
                continue
            if _is_demo_unused(var_name):
                lines.append(f"# {line}")
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def install_env(config_path: Path, dry_run: bool = False) -> None:
    env_example = SCRIPT_DIR / ".env.example"
    dest_env = SCRIPT_DIR / ".env"
    if dest_env.exists():
        print(f"[skip] {dest_env} already exists, not overwriting")
        return
    if not env_example.exists():
        print(f"[warn] {env_example} not found, skipping .env setup")
        return

    print(f"[copy] {env_example} -> {dest_env} (Notion/unused vars commented out, CONFIG_PATH filled in)")
    if not dry_run:
        commented = _comment_out_unused_env_lines(env_example.read_text(), config_path)
        dest_env.write_text(commented)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="Folder to search for downloaded files (default: auto-detect a sibling of this repo containing the demo files, falling back to the current directory)")
    parser.add_argument("--dest", default=None, help=f"Folder to build the dataset/deployment hierarchy in (default: {DEFAULT_DEST_DIRNAME}/, a sibling of this repo)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to demo_config.yaml (default: alongside this script)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without touching files")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    deployment_ids = list(build_deployment_to_dataset(config).keys())

    dest = Path(args.dest).resolve() if args.dest else (SCRIPT_DIR.parent / DEFAULT_DEST_DIRNAME).resolve()

    if args.source:
        source = Path(args.source).resolve()
    else:
        source = resolve_default_source(deployment_ids, dest).resolve()
        print(f"[auto] using source folder: {source}")

    organize(source, dest, config, dry_run=args.dry_run)
    install_config(config_path, dry_run=args.dry_run)
    install_env(SCRIPT_DIR / "config.yaml", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
