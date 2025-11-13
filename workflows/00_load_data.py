# Run with shell command: python pyologger/workflows/00_load_data.py --dataset oror-adult-orca_hr-sr-vid_sw_JKB-PP --deployment 2024-01-16_oror-002
import os
import json
import re
import pickle
import tempfile
import argparse
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

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
    deployment_folder = os.path.join(dataset_folder, deployment_id)
    print(f"✅ Using provided deployment ID: {deployment_id}")
else:
    print("No deployment provided. Selecting from available deployments.")
    deployment_folder = select_folder(dataset_folder, "Select a deployment folder:")

if not os.path.exists(deployment_folder):
    raise ValueError(f"❌ Deployment folder {deployment_folder} not found!")

# Extract deployment_id and animal_id from the folder name
match = re.match(r"(\d{4}-\d{2}-\d{2}_[a-z]{4}-\d{3})", os.path.basename(deployment_folder), re.IGNORECASE)
if match:
    deployment_id = match.group(1)  # Extract YYYY-MM-DD_animalID
    animal_id = deployment_id.split("_")[1]  # Extract animal ID
    print(f"✅ Extracted deployment ID: {deployment_id}, Animal ID: {animal_id}")
else:
    raise ValueError(f"❌ Unable to extract deployment ID from folder: {deployment_folder}")

deployment_info, loggers_used = metadata.extract_essential_metadata(deployment_id)

# Step 5: Initialize DataReader with dataset folder, deployment ID, and optional data subfolder
data_pkl = DataReader(dataset_folder=dataset_folder, deployment_id=deployment_id, data_subfolder="01_raw-data", montage_path=montage_path)

# Step 6: Initialize config manager
param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
param_manager.add_to_config("current_processing_step", "Processing Step 00: Data import pending.")
param_manager.export_config()

overwrite_data = False
pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
if os.path.exists(pkl_path) and not overwrite_data:
    with open(pkl_path, "rb") as f:
        data_pkl = pickle.load(f)
    print(f"📦 Loaded processed DataReader object from: {pkl_path}")
else:
    data_pkl = DataReader(
        dataset_folder=dataset_folder,
        deployment_id=deployment_id,
        data_subfolder="01_raw-data",
        montage_path=montage_path
    )
    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
    param_manager.add_to_config("current_processing_step", "Processing Step 00: Data import pending.")

    data_pkl.read_files(
        deployment_info=deployment_info,
        loggers_used=loggers_used,
        save_parq=False,
        save_netcdf=True
    )

# Get timezone
timezone = data_pkl.deployment_info.get("Time Zone", "UTC")

# Load time settings
time_settings = param_manager.get_from_config(
    ["overlap_start_time", "overlap_end_time", "zoom_window_start_time", "zoom_window_end_time"],
    section="settings"
)

if time_settings:
    print("Time settings present.")
# If any required time settings are missing, compute and update them
if not any(v is None for v in time_settings.values()):
    print("Time settings not empty.")
else:
    print("Adding timestamps to config.")
    zoom_time_window = 5  # minutes

    # Extract start and end times for all signals
    start_times = [df['datetime'].min() for df in data_pkl.signal_data.values()]
    end_times = [df['datetime'].max() for df in data_pkl.signal_data.values()]

    # Compute common start, end, and zoom window
    overlap_start_time = max(start_times)
    overlap_end_time = min(end_times)
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

if not any(v is None for v in truncate_times.values()):
    print("Truncating with provided cropping times.")
    # Update overlap window with selected range
    OVERLAP_START_TIME = pd.Timestamp(truncate_times['selected_start_time']).tz_convert(timezone)
    OVERLAP_END_TIME = pd.Timestamp(truncate_times['selected_end_time']).tz_convert(timezone)

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

    pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
    with open(pkl_path, "wb") as file:
        pickle.dump(data_pkl, file)

# Step 8: Update processing step
param_manager.add_to_config("current_processing_step", "Processing Step 00: Data imported.")

exporter = BaseExporter(data_pkl) # Create a BaseExporter instance using data pickle object
netcdf_file_path = os.path.join(deployment_folder, 'outputs', f'{deployment_id}_step00.nc') # Define the export path
exporter.save_to_netcdf(data_pkl, filepath=netcdf_file_path) # Save to NetCDF format
