# Run with shell command: python3 pyologger/workflows/00_load_data.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import json
import re
import pickle
import tempfile
import argparse
import sys
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Ensure direct workflow execution resolves the repo-local pyologger package.
WORKFLOW_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKFLOW_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import pyologger utilities
from pyologger.utils.folder_manager import *
from pyologger.utils.param_manager import ParamManager
from pyologger.load_data.datareader import DataReader
from pyologger.load_data.metadata import Metadata
from pyologger.io_operations.base_exporter import *

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load data")
parser.add_argument("--dataset", type=str, help="Dataset folder name")
parser.add_argument("--deployment", type=str, help="Deployment ID")
args = parser.parse_args()

# Load important file paths and configurations
config, data_dir, color_mapping_path, montage_path = load_configuration()

# Step 1: Select dataset folder
if args.dataset:
    dataset_folder = os.path.join(data_dir, args.dataset)
else:
    print("No dataset provided. Selecting from available datasets.")
    dataset_folder = select_folder(data_dir, "Select a dataset folder:")

if not os.path.exists(dataset_folder):
    raise ValueError(f"❌ Dataset folder {dataset_folder} not found!")

overwrite = False  # Force refresh from Notion if True

# Paths
metadata_dir = os.path.join(data_dir, "00_Metadata")
os.makedirs(metadata_dir, exist_ok=True)
metadata_pickle_path = os.path.join(metadata_dir, "metadata_snapshot.pkl")

relations_map_dir = config["paths"].get("local_repo_path", metadata_dir)
os.makedirs(relations_map_dir, exist_ok=True)
relations_map_path = os.path.join(relations_map_dir, "relations_map.json")


def atomic_write_bytes(path: str, data: bytes):
    """Write bytes atomically to avoid partial/corrupt files."""
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dirpath, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def atomic_write_text(path: str, text: str):
    atomic_write_bytes(path, text.encode("utf-8"))


def load_all_tables(md_obj):
    """
    Convenience: pull all the DB/data source tables from a Metadata instance.
    Returns a dict so you can unpack if you want.
    """
    return {
        "deployment_db": md_obj.get_metadata("deployment_DB"),
        "logger_db": md_obj.get_metadata("logger_DB"),
        "recording_db": md_obj.get_metadata("recording_DB"),
        "animal_db": md_obj.get_metadata("animal_DB"),
        "dataset_db": md_obj.get_metadata("dataset_DB"),
        "procedure_db": md_obj.get_metadata("procedure_DB"),
        "observation_db": md_obj.get_metadata("observation_DB"),
        "collaborator_db": md_obj.get_metadata("collaborator_DB"),
        "location_db": md_obj.get_metadata("location_DB"),
        "montage_db": md_obj.get_metadata("montage_DB"),
        "signal_db": md_obj.get_metadata("signal_DB"),
        "attachment_db": md_obj.get_metadata("attachment_DB"),
        "originalchannel_db": md_obj.get_metadata("originalchannel_DB"),
        "standardizedchannel_db": md_obj.get_metadata("standardizedchannel_DB")
    }


def pickle_needs_refresh(path: str, max_age_days: int = 14) -> bool:
    if not os.path.exists(path):
        return True
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) > timedelta(days=max_age_days)
    except Exception:
        # If anything is weird with the file, refresh.
        return True


# Decide whether to pull fresh data from Notion
needs_refresh = overwrite or pickle_needs_refresh(metadata_pickle_path, max_age_days=14)

if needs_refresh:
    # 1) Build a fresh Metadata instance (hits Notion and populates self.metadata etc.)
    metadata = Metadata()

    # 2) Recompute relations map (uses databases.retrieve schemas and relation fields)
    metadata.map_database_relations()
    relations_map = metadata.relations_map

    # 3) Save relations map atomically (so it’s never half-written)
    try:
        atomic_write_text(relations_map_path, json.dumps(relations_map, indent=4))
        print(f"Relations map saved at: {relations_map_path}")
    except Exception as e:
        print(f"[WARN] Failed to write relations_map.json: {e}")

    # 4) Strip live client / transient runtime caches before pickling
    metadata.notion = None
    if hasattr(metadata, "data_source_cache"):
        metadata.data_source_cache = {}

    # Optional: embed a tiny snapshot header for sanity
    snapshot_meta = {
        "notion_version": getattr(metadata, "notion_version", None),
        "created_at": datetime.now().isoformat(),
        "class": "Metadata",
    }
    payload = {"snapshot_meta": snapshot_meta, "metadata_obj": metadata}

    # 5) Snapshot full metadata object to disk atomically
    try:
        atomic_write_bytes(metadata_pickle_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        print(f"[REFRESH] Metadata snapshot saved at: {metadata_pickle_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write metadata snapshot: {e}")
        raise
else:
    # Load cached snapshot instead of hitting Notion
    print(f"[CACHE] Using existing metadata snapshot at: {metadata_pickle_path}")
    try:
        with open(metadata_pickle_path, "rb") as file:
            payload = pickle.load(file)
        # Backward-compat: support old format (raw Metadata pickled directly)
        if isinstance(payload, dict) and "metadata_obj" in payload:
            metadata = payload["metadata_obj"]
            snapshot_meta = payload.get("snapshot_meta", {})
        else:
            metadata = payload
            snapshot_meta = {}
        # Note: metadata.notion is None here (by design). We're only reading dfs, so it's fine.
    except Exception as e:
        print(f"[WARN] Cache unreadable ({e}). Falling back to fresh pull.")
        metadata = Metadata()
        metadata.map_database_relations()
        relations_map = metadata.relations_map
        try:
            atomic_write_text(relations_map_path, json.dumps(relations_map, indent=4))
        except Exception as ee:
            print(f"[WARN] Failed to write relations_map.json on fallback: {ee}")
        metadata.notion = None
        if hasattr(metadata, "data_source_cache"):
            metadata.data_source_cache = {}
        payload = {
            "snapshot_meta": {
                "notion_version": getattr(metadata, "notion_version", None),
                "created_at": datetime.now().isoformat(),
                "class": "Metadata",
            },
            "metadata_obj": metadata,
        }
        atomic_write_bytes(metadata_pickle_path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        print(f"[REFRESH] Metadata snapshot saved at: {metadata_pickle_path}")


# Expose each table for downstream code in this session
tables = load_all_tables(metadata)

deployment_db = tables["deployment_db"]
logger_db = tables["logger_db"]
recording_db = tables["recording_db"]
animal_db = tables["animal_db"]
dataset_db = tables["dataset_db"]
procedure_db = tables["procedure_db"]
observation_db = tables["observation_db"]
collaborator_db = tables["collaborator_db"]
location_db = tables["location_db"]
montage_db = tables["montage_db"]
signal_db = tables["signal_db"]
attachment_db = tables["attachment_db"]
originalchannel_db = tables["originalchannel_db"]
standardizedchannel_db = tables["standardizedchannel_db"]

# Optional: quick sanity print of row counts
try:
    counts = {k: (v.shape[0] if v is not None else 0) for k, v in tables.items()}
    print("[Metadata tables] row counts:", json.dumps(counts, indent=2))
except Exception:
    pass

# Step 3: Select deployment folder
if args.deployment:
    deployment_id = args.deployment
    print(f"✅ Using provided deployment ID: {deployment_id}")
    
    # Find deployment folder with suffix support (similar to DataReader._find_deployment_folder)
    # First try exact match
    deployment_folder = os.path.join(dataset_folder, deployment_id)
    if not os.path.exists(deployment_folder):
        # Look for folders starting with deployment_id (handles suffixes)
        found = False
        try:
            for entry in os.listdir(dataset_folder):
                full_path = os.path.join(dataset_folder, entry)
                if os.path.isdir(full_path) and entry.startswith(deployment_id):
                    # Make sure it's not just a prefix match (require _ or end of string after ID)
                    if entry == deployment_id or (len(entry) > len(deployment_id) and entry[len(deployment_id)] == '_'):
                        print(f"   📁 Found deployment folder with suffix: {entry}")
                        deployment_folder = full_path
                        found = True
                        break
        except (FileNotFoundError, PermissionError):
            pass
        
        if not found:
            raise ValueError(f"❌ Deployment folder not found for ID: {deployment_id}")
else:
    print("No deployment provided. Selecting from available deployments.")
    deployment_folder = select_folder(dataset_folder, "Select a deployment folder:")

if not os.path.exists(deployment_folder):
    raise ValueError(f"❌ Deployment folder {deployment_folder} not found!")

# Extract deployment_id and animal_id from the folder name
# Pattern: YYYY-MM-DD_animalid-NNNN or YYYY-MM-DD_animalid-NNNN_suffix
match = re.match(r"(\d{4}-\d{2}-\d{2}_[a-z]+-\d+)", os.path.basename(deployment_folder), re.IGNORECASE)
if match:
    deployment_id = match.group(1)  # Extract YYYY-MM-DD_animalID
    animal_id = deployment_id.split("_")[1]  # Extract animal ID (everything after first _)
    print(f"✅ Extracted deployment ID: {deployment_id}, Animal ID: {animal_id}")
else:
    raise ValueError(f"❌ Unable to extract deployment ID from folder: {deployment_folder}")

deployment_info, loggers_used = metadata.extract_essential_metadata(deployment_id)

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        hit = cols.get(str(c).strip().lower())
        if hit is not None:
            return hit
    return None


def _enrich_signal_metadata_from_standardized_db(data_obj, std_db: pd.DataFrame) -> int:
    """
    Attach standardized channel label/description metadata onto data_pkl.signal_info[*].metadata[*].
    Returns number of channel metadata entries updated.
    """
    if not isinstance(std_db, pd.DataFrame) or std_db.empty:
        return 0

    parent_col = _pick_col(std_db, ["Parent signal", "parent_signal"])
    channel_col = _pick_col(std_db, ["Channel ID", "Standardized Channel ID", "standardized_channel_id", "channel_id"])
    label_col = _pick_col(std_db, ["Signal Label", "Label", "signal_label", "label"])
    desc_col = _pick_col(std_db, ["Signal Description", "Description", "signal_description", "description"])
    if parent_col is None or channel_col is None:
        return 0

    # Use explicit composite key signal.channel to avoid collisions across duplicated
    # standardized channel IDs that belong to different parent signals.
    lookup: dict[str, dict[str, str]] = {}
    for _, row in std_db.iterrows():
        parent = str(row.get(parent_col, "") or "").strip().lower()
        channel = str(row.get(channel_col, "") or "").strip().lower()
        if not parent or not channel:
            continue
        label_text = str(row.get(label_col, "") or "").strip() if label_col else ""
        desc_text = str(row.get(desc_col, "") or "").strip() if desc_col else ""
        key = f"{parent}.{channel}"
        existing = lookup.get(key, {"label": "", "description": ""})
        # Keep first non-empty value so blank duplicate rows do not erase metadata.
        if label_text and not existing.get("label"):
            existing["label"] = label_text
        if desc_text and not existing.get("description"):
            existing["description"] = desc_text
        lookup[key] = existing

    updated = 0
    signal_info = getattr(data_obj, "signal_info", {}) or {}
    for signal_id, sinfo in signal_info.items():
        if not isinstance(sinfo, dict):
            continue
        metadata = sinfo.get("metadata")
        if not isinstance(metadata, dict):
            continue
        sig_key = str(signal_id).strip().lower()
        for channel_id, cmeta in metadata.items():
            if not isinstance(cmeta, dict):
                continue
            ch_key = str(channel_id).strip().lower()
            enrich = lookup.get(f"{sig_key}.{ch_key}")
            if not enrich:
                # Fallback: some channels carry parent_signal that may differ from the
                # signal_info dict key naming.
                parent_key = str(cmeta.get("parent_signal") or "").strip().lower()
                if parent_key:
                    enrich = lookup.get(f"{parent_key}.{ch_key}")
            if not enrich:
                continue
            changed = False
            if enrich["label"] and str(cmeta.get("label") or "").strip() != enrich["label"]:
                cmeta["label"] = enrich["label"]
                changed = True
            if enrich["description"] and str(cmeta.get("description") or "").strip() != enrich["description"]:
                cmeta["description"] = enrich["description"]
                changed = True
            if changed:
                metadata[channel_id] = cmeta
                updated += 1
        sinfo["metadata"] = metadata
        signal_info[signal_id] = sinfo
    data_obj.signal_info = signal_info
    return updated


# Step 5: Initialize DataReader - will find actual folder with suffix if it exists
data_pkl = DataReader(
    dataset_folder=dataset_folder,
    deployment_id=deployment_id,
    data_subfolder="01_raw-data",
    montage_path=montage_path
)

# Use the actual deployment folder that DataReader found (may have suffix)
actual_deployment_folder = data_pkl.deployment_folder

# If the actual folder has a suffix, create a symlink for Snakemake compatibility FIRST
if os.path.basename(actual_deployment_folder) != deployment_id:
    symlink_folder = os.path.join(dataset_folder, deployment_id)
    if not os.path.exists(symlink_folder):
        os.symlink(os.path.basename(actual_deployment_folder), symlink_folder)
        print(f"🔗 Created symlink: {deployment_id} -> {os.path.basename(actual_deployment_folder)}")

pkl_path = os.path.join(actual_deployment_folder, "outputs", "data.pkl")

# Check if processed data already exists
overwrite_data = False
existing_pickle_ok = False
if os.path.exists(pkl_path) and not overwrite_data:
    try:
        with open(pkl_path, "rb") as f:
            data_pkl = pickle.load(f)
        existing_pickle_ok = True
        print(f"📦 Loaded processed DataReader object from: {pkl_path}")
        # Initialize param_manager for loaded data
        param_manager = ParamManager(deployment_folder=actual_deployment_folder, deployment_id=deployment_id)
    except (EOFError, pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError) as e:
        backup_path = f"{pkl_path}.corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.replace(pkl_path, backup_path)
            print(f"⚠️ Existing data.pkl is unreadable ({type(e).__name__}: {e}).")
            print(f"🛟 Backed up corrupt pickle to: {backup_path}")
        except Exception as backup_error:
            print(f"⚠️ Existing data.pkl is unreadable ({type(e).__name__}: {e}).")
            print(f"⚠️ Could not move corrupt pickle to backup ({backup_error}). Will overwrite in place.")

if not existing_pickle_ok:
    # Read data files
    data_pkl.read_files(
        deployment_info=deployment_info,
        loggers_used=loggers_used,
        save_parq=False,
        save_netcdf=True
    )
    
    # Create output folder in the actual deployment folder (with suffix)
    os.makedirs(os.path.join(actual_deployment_folder, "outputs"), exist_ok=True)
    param_manager = ParamManager(deployment_folder=actual_deployment_folder, deployment_id=deployment_id)
    
    # Check if we have data
    if not data_pkl.signal_data:
        print(f"⚠️ No signal data found for deployment {deployment_id}.")
        param_manager.add_to_config("current_processing_step", "Processing Step 00: No data found.")
        param_manager.export_config()
        
        # Save empty DataReader object for downstream compatibility
        with open(pkl_path, "wb") as file:
            pickle.dump(data_pkl, file)
        print(f"DataReader object (empty) saved to {pkl_path}.")
        
        # Create placeholder NetCDF for Snakemake
        exporter = BaseExporter(data_pkl)
        netcdf_file_path = os.path.join(actual_deployment_folder, 'outputs', f'{deployment_id}_step00.nc')
        exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path)
        print(f"📊 Saved empty deployment data to NetCDF: {netcdf_file_path}")
        
        # Exit - no data to process further
        import sys
        sys.exit(0)
    
    param_manager.add_to_config("current_processing_step", "Processing Step 00: Data imported.")
    param_manager.export_config()

# Get timezone
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

# Ensure standardized channel label/description metadata is carried in signal_info
# so NetCDF exports include it as global attrs.
updated_channel_metadata = _enrich_signal_metadata_from_standardized_db(data_pkl, standardizedchannel_db)
if updated_channel_metadata:
    print(f"🧾 Enriched signal metadata from standardizedchannel_db for {updated_channel_metadata} channel(s).")


def _normalize_timestamp(ts_value, timezone_str):
    """Return a timezone-aware pandas Timestamp normalized to deployment timezone."""
    ts = pd.Timestamp(ts_value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(timezone_str)
    return ts.tz_convert(timezone_str)


# Load time settings
time_settings = param_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

if time_settings:
    print("Time settings present.")

should_recompute_time_settings = False
if not time_settings or any(v is None for v in time_settings.values()):
    should_recompute_time_settings = True
else:
    try:
        existing_overlap_start = _normalize_timestamp(time_settings["overlap_start_time"], timezone)
        existing_overlap_end = _normalize_timestamp(time_settings["overlap_end_time"], timezone)
        existing_zoom_start = _normalize_timestamp(time_settings["zoom_window_start_time"], timezone)
        existing_zoom_end = _normalize_timestamp(time_settings["zoom_window_end_time"], timezone)

        if (
            existing_overlap_start is None
            or existing_overlap_end is None
            or existing_zoom_start is None
            or existing_zoom_end is None
            or existing_overlap_start > existing_overlap_end
            or existing_zoom_start > existing_zoom_end
        ):
            print("⚠️ Existing time settings are invalid. Recomputing from signal timestamps.")
            should_recompute_time_settings = True
        else:
            print("Time settings are valid.")
    except Exception as e:
        print(f"⚠️ Failed to parse existing time settings ({e}). Recomputing.")
        should_recompute_time_settings = True

if should_recompute_time_settings:
    print("Adding timestamps to config.")
    zoom_time_window = 5  # minutes

    # Extract start and end times for all signals
    start_times = []
    end_times = []
    for df in data_pkl.signal_data.values():
        if 'datetime' not in df.columns or df.empty:
            continue
        start_ts = _normalize_timestamp(df['datetime'].min(), timezone)
        end_ts = _normalize_timestamp(df['datetime'].max(), timezone)
        if start_ts is not None:
            start_times.append(start_ts)
        if end_ts is not None:
            end_times.append(end_ts)

    # Check if we have any valid timestamps
    if not start_times or not end_times:
        print("⚠️ No valid timestamps found in signal data. Skipping time settings.")
        time_settings = {
            "overlap_start_time": None,
            "overlap_end_time": None,
            "zoom_window_start_time": None,
            "zoom_window_end_time": None,
        }
    else:
        # Compute common start, end, and zoom window
        overlap_start_time = max(start_times)
        overlap_end_time = min(end_times)
        if overlap_start_time > overlap_end_time:
            # No strict overlap across all signals: fall back to a valid deployment-wide range.
            union_start_time = min(start_times)
            union_end_time = max(end_times)
            print(
                "⚠️ No common overlap across signals. "
                f"Computed overlap was inverted ({overlap_start_time} > {overlap_end_time}). "
                f"Using union range instead: {union_start_time} to {union_end_time}."
            )
            overlap_start_time = union_start_time
            overlap_end_time = union_end_time

        midpoint = overlap_start_time + (overlap_end_time - overlap_start_time) / 2
        zoom_window_start, zoom_window_end = midpoint - timedelta(minutes=zoom_time_window / 2), midpoint + timedelta(minutes=zoom_time_window / 2)

        # Update settings
        time_settings = {
            "overlap_start_time": str(overlap_start_time),
            "overlap_end_time": str(overlap_end_time),
            "zoom_window_start_time": str(zoom_window_start),
            "zoom_window_end_time": str(zoom_window_end),
        }
    
    param_manager.add_to_config(entries=time_settings, section="settings")

if any(v is None for v in time_settings.values()):
    print("YES")
time_settings

# Check if selected start and end times exist in the config file
truncate_times = param_manager.get_from_config(
    ["selected_start_time", "selected_end_time"],
    section="settings"
)

truncation_wrote_pickle = False
if not any(v is None for v in truncate_times.values()):
    print("Truncating with provided cropping times.")
    # Update overlap window with selected range
    OVERLAP_START_TIME = _normalize_timestamp(truncate_times['selected_start_time'], timezone)
    OVERLAP_END_TIME = _normalize_timestamp(truncate_times['selected_end_time'], timezone)
    if OVERLAP_START_TIME > OVERLAP_END_TIME:
        print(
            f"⚠️ selected_start_time ({OVERLAP_START_TIME}) is after "
            f"selected_end_time ({OVERLAP_END_TIME}). Swapping values."
        )
        OVERLAP_START_TIME, OVERLAP_END_TIME = OVERLAP_END_TIME, OVERLAP_START_TIME

    # Truncate signal data
    for signal, df in data_pkl.signal_data.items():
        # Truncate based on selected time range
        truncated_df = df[(df.iloc[:, 0] >= OVERLAP_START_TIME) & (df.iloc[:, 0] <= OVERLAP_END_TIME)].copy()
        data_pkl.signal_data[signal] = truncated_df  # Save truncated version to new variable

    # Recalculate Zoom Window (5-minute window in the middle)
    midpoint = OVERLAP_START_TIME + (OVERLAP_END_TIME - OVERLAP_START_TIME) / 2
    ZOOM_WINDOW_START_TIME = midpoint - timedelta(minutes=2.5)
    ZOOM_WINDOW_END_TIME = midpoint + timedelta(minutes=2.5)

    # Save new time settings
    time_settings_update = {
        "overlap_start_time": str(OVERLAP_START_TIME),
        "overlap_end_time": str(OVERLAP_END_TIME),
        "zoom_window_start_time": str(ZOOM_WINDOW_START_TIME),
        "zoom_window_end_time": str(ZOOM_WINDOW_END_TIME)
    }
    param_manager.add_to_config(entries=time_settings_update, section="settings")

    pkl_path = os.path.join(actual_deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)
    truncation_wrote_pickle = True

# Persist metadata enrichment even when no truncation branch writes data.pkl.
if updated_channel_metadata and not truncation_wrote_pickle:
    pkl_path = os.path.join(actual_deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

# Step 8: Update processing step
param_manager.add_to_config("current_processing_step", "Processing Step 00: Data imported.")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(actual_deployment_folder, 'outputs', f'{deployment_id}_step00.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format
