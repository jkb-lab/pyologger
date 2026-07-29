from __future__ import annotations

import os
import pickle
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pyologger.utils.param_manager import ParamManager

def resolve_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    if config_path:
        return Path(config_path).expanduser().resolve()
    env_path = os.getenv("CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "config.yaml"


def resolve_segmentation_runs_path(
    config_path: str | os.PathLike[str] | None = None,
    segmentation_runs_path: str | os.PathLike[str] | None = None,
) -> Path:
    if segmentation_runs_path:
        return Path(segmentation_runs_path).expanduser().resolve()
    env_path = os.getenv("SEGMENTATION_RUNS_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return resolve_config_path(config_path).with_name("segmentation_runs.yaml")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_combined_config(
    config_path: str | os.PathLike[str] | None = None,
    segmentation_runs_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    main_config_path = resolve_config_path(config_path)
    config = _load_yaml_file(main_config_path)
    runs_path = resolve_segmentation_runs_path(main_config_path, segmentation_runs_path)
    if runs_path.exists():
        runs_payload = _load_yaml_file(runs_path)
        merged = copy.deepcopy(config)
        if isinstance(runs_payload.get("segmentation_runs"), dict):
            merged["segmentation_runs"] = copy.deepcopy(runs_payload["segmentation_runs"])
        for shared_key in (
            "context_filter_definitions",
            "supervised_context_filter_definitions",
            "supervised_label_group_definitions",
            "segmentation_event_definitions",
        ):
            if isinstance(runs_payload.get(shared_key), dict):
                merged[shared_key] = copy.deepcopy(runs_payload[shared_key])
        config = merged
    return config, main_config_path, runs_path


def load_configuration():
    from dotenv import load_dotenv
    load_dotenv()
    config, _, _ = load_combined_config()
    data_dir = config["paths"]["local_private_data"]
    color_mapping_path = os.path.join(config["paths"]["local_repo_path"], "color_mappings.json")
    montage_path = os.path.join(config["paths"]["local_repo_path"], "montage_log.json")
    return config, data_dir, color_mapping_path, montage_path

def select_folder(base_dir, prompt="Select a folder:"):
    """Prompts the user to select a dataset or deployment folder."""
    folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith("00_")])
    
    if not folders:
        raise ValueError(f"No valid folders found in {base_dir}.")
    
    print(prompt)
    for i, folder in enumerate(folders):
        print(f"{i}: {folder}")
    
    selected_index = int(input("Enter the number of the folder you want to select: "))
    
    if 0 <= selected_index < len(folders):
        return os.path.join(base_dir, folders[selected_index])
    else:
        raise ValueError("Invalid selection. Please restart and choose a valid folder.")
    
def select_and_load_deployment_streamlit(data_dir):
    """Streamlit-based deployment selection with dataset and deployment filtering."""
    import streamlit as st

    # Get available datasets (excluding those starting with "00_")
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_")])
    if not datasets:
        st.error("❌ No valid datasets found.")
        st.stop()

    # Determine which dataset to show as the current selection.
    # Priority: dataset_selection (user changed it on a subpage) >
    #           preferred_dataset_selection (home page button) > index 0.
    current = st.session_state.get("dataset_selection")
    preferred = st.session_state.get("preferred_dataset_selection")
    if current not in datasets:
        current = preferred if preferred in datasets else datasets[0]

    current_index = datasets.index(current)
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset", datasets, index=current_index
    )
    # Write back so other parts of the app can read the current choice.
    st.session_state["dataset_selection"] = selected_dataset
    st.session_state["preferred_dataset_selection"] = selected_dataset

    # Reset deployment choice when dataset changes to prevent stale selections.
    last_dataset = st.session_state.get("_last_dataset_selection")
    if last_dataset != selected_dataset:
        st.session_state["_last_dataset_selection"] = selected_dataset
        if "deployment_selection" in st.session_state:
            del st.session_state["deployment_selection"]

    dataset_folder = os.path.join(data_dir, selected_dataset)

    # Get available deployments (excluding those starting with "00_")
    deployments = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d)) and not d.startswith("00_")])
    if not deployments:
        st.error("❌ No valid deployments found in the selected dataset.")
        st.stop()

    # Deployment selection (guard against stale session value from a previous dataset)
    current_deployment = st.session_state.get("deployment_selection")
    if current_deployment not in deployments:
        st.session_state["deployment_selection"] = deployments[0]

    selected_deployment = st.sidebar.selectbox(
        "Select Deployment",
        deployments,
        key="deployment_selection"
    )
    deployment_folder = os.path.join(dataset_folder, selected_deployment)

    # Extract metadata
    deployment_id = selected_deployment
    dataset_id = selected_dataset
    try:
        animal_id = deployment_id.split("_")[1]
    except IndexError:
        st.error(f"❌ Unable to extract animal ID from deployment ID: {deployment_id}")
        st.stop()

    # Load data.pkl
    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        st.error(f"❌ Data pickle file not found: {pkl_path}")
        st.stop()

    try:
        with open(pkl_path, "rb") as file:
            data_pkl = pickle.load(file)
    except (EOFError, pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError) as e:
        st.error(
            "❌ Failed to load data.pkl (likely incomplete or corrupted).\n"
            f"Path: {pkl_path}\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            "Run recovery:\n"
            "python workflows/00_load_data.py --dataset <dataset> --deployment <deployment>"
        )
        st.stop()

    # Initialize ParamManager
    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)

    return animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager

def resolve_deployment_context(data_dir, dataset_id=None, deployment_id=None):
    """Resolve dataset/deployment paths without loading data.pkl."""
    # Get available datasets (excluding those starting with "00_")
    datasets = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("00_")])
    if not datasets:
        raise ValueError("❌ No valid datasets found.")

    # Allow selection by either index or folder name
    if dataset_id is None:
        print("\nAvailable Datasets:")
        for i, dataset in enumerate(datasets):
            print(f"{i}: {dataset}")
        dataset_input = input("Enter dataset index or name: ")

        if dataset_input.isdigit() and 0 <= int(dataset_input) < len(datasets):
            dataset_id = datasets[int(dataset_input)]
        elif dataset_input in datasets:
            dataset_id = dataset_input
        else:
            raise ValueError("❌ Invalid dataset selection.")

    dataset_folder = os.path.join(data_dir, dataset_id)

    # Get available deployments (excluding those starting with "00_")
    deployments = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d)) and not d.startswith("00_")])
    if not deployments:
        raise ValueError(f"❌ No valid deployments found in {dataset_id}.")

    # Allow selection by either index or folder name
    if deployment_id is None:
        print("\nAvailable Deployments:")
        for i, deployment in enumerate(deployments):
            print(f"{i}: {deployment}")
        deployment_input = input("Enter deployment index or name: ")

        if deployment_input.isdigit() and 0 <= int(deployment_input) < len(deployments):
            deployment_id = deployments[int(deployment_input)]
        elif deployment_input in deployments:
            deployment_id = deployment_input
        else:
            raise ValueError("❌ Invalid deployment selection.")

    deployment_folder = os.path.join(dataset_folder, deployment_id)

    # Extract metadata
    try:
        animal_id = deployment_id.split("_")[1]
    except IndexError:
        raise ValueError(f"❌ Unable to extract animal ID from deployment ID: {deployment_id}")

    param_manager = ParamManager(deployment_folder=deployment_folder, deployment_id=deployment_id)
    return animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager

def select_and_load_deployment(data_dir, dataset_id=None, deployment_id=None):
    """Command-line or function-based deployment selection. Allows selection via index or folder name."""

    animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager = resolve_deployment_context(
        data_dir,
        dataset_id=dataset_id,
        deployment_id=deployment_id,
    )

    # Load data.pkl
    pkl_path = os.path.join(deployment_folder, "outputs", "data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"❌ Data pickle file not found: {pkl_path}")

    try:
        with open(pkl_path, "rb") as file:
            data_pkl = pickle.load(file)
    except (EOFError, pickle.UnpicklingError, ModuleNotFoundError, AttributeError, ImportError) as e:
        raise RuntimeError(
            "❌ Failed to load data.pkl (likely incomplete or corrupted).\n"
            f"Path: {pkl_path}\n"
            f"Error: {type(e).__name__}: {e}\n"
            "Recovery: rerun Step 00 to rebuild the pickle:\n"
            f"python workflows/00_load_data.py --dataset {dataset_id} --deployment {deployment_id}"
        ) from e

    return animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager


def match_to_metadata(
    deployments: dict,
    animal_db,
    deployment_db,
    recording_db,
    animal_db_mapping_column: str = "Domain IDs"
):
    """
    Map TOPPIDs to standardized metadata identifiers using database lookups.
    
    This function performs a multi-stage lookup to connect project-level TOPPIDs
    to database-standard identifiers (Animal ID, Deployment ID, Recording ID).
    Essential for organizing data according to the pyologger metadata schema.
    
    Mapping flow:
        TOPPID (project identifier) 
            → Animal ID (via animal_db[animal_db_mapping_column])
            → Deployment ID (via deployment_db['Animal ID'])
            → Recording ID (via recording_db['Deployment ID'])
    
    Parameters
    ----------
    deployments : dict
        Dictionary with TOPPID keys containing deployment data. Typically contains
        'merged_df', 'stroke_df', 'tlld_1hz', and other analysis data.
    animal_db : pd.DataFrame
        Animal database with 'Project ID' or 'Domain IDs' and 'Animal ID' columns. 
        Links project identifiers (TOPPIDs) to standardized animal IDs.
    deployment_db : pd.DataFrame
        Deployment database with 'Animal ID' and 'Deployment ID' columns. Links
        animals to specific field deployments.
    recording_db : pd.DataFrame
        Recording database with 'Deployment ID' and 'Recording ID' columns. Links
        deployments to specific data recording sessions.
    animal_db_mapping_column : str, optional
        Column name in animal_db to match TOPPIDs against. Common values are
        'Domain IDs' or 'Project ID'. Default is 'Domain IDs'.
    
    Returns
    -------
    dict
        Mapping results for each TOPPID with the following structure:
        
        For successful mappings:
            {
                'toppid': {
                    'status': 'success',
                    'animal_id': str,
                    'deployment_id': str,
                    'recording_id': str
                }
            }
        
        For failed mappings:
            {
                'toppid': {
                    'status': 'failed',
                    'reason': str,  # e.g., 'No animal match', 'No deployment match'
                    'animal_id': str (optional, if lookup failed after animal stage),
                    'deployment_id': str (optional, if lookup failed after deployment stage)
                }
            }
    
    Notes
    -----
    - The function performs lookups sequentially, stopping at the first failure
    - If multiple recordings exist for a deployment, the first one is used (with a warning)
    - Failed mappings are tracked with detailed reasons for debugging
    - Progress is printed to console with status indicators (✓ for success, ⚠️ for warnings/failures)
    
    Examples
    --------
    >>> mapping_results = match_to_metadata(
    ...     deployments={'2011016': {...}, '2011017': {...}},
    ...     animal_db=animal_db,
    ...     deployment_db=deployment_db,
    ...     recording_db=recording_db,
    ...     animal_db_mapping_column="Domain IDs"
    ... )
    >>> print(mapping_results['2011016']['deployment_id'])
    'MIAN-2011_PB01_2011-02-12'
    
    See Also
    --------
    create_dataset_structure : Uses mapping results to create standardized folder structure
    """
    import pandas as pd
    
    print(f"\n{'='*70}")
    print(f"MATCHING TOPPIDs TO METADATA")
    print(f"{'='*70}\n")
    
    mapping_results = {}
    
    for toppid in deployments.keys():
        print(f"\nProcessing TOPPID: {toppid}")
        print("-" * 50)
        
        # Step 1: TOPPID -> Animal ID via animal_db
        animal_match = animal_db[animal_db[animal_db_mapping_column].astype(str) == str(toppid)]
        
        if animal_match.empty:
            print(f"⚠️  No Animal ID found for TOPPID {toppid} in animal_db['{animal_db_mapping_column}']")
            mapping_results[toppid] = {"status": "failed", "reason": "No animal match"}
            continue
        
        animal_id = animal_match.iloc[0]["Animal ID"]
        print(f"  ✓ Found Animal ID: {animal_id}")
        
        # Step 2: Animal ID -> Deployment ID via deployment_db
        deployment_match = deployment_db[deployment_db["Animal ID"] == animal_id]
        
        if deployment_match.empty:
            print(f"⚠️  No Deployment ID found for Animal ID {animal_id} in deployment_db")
            mapping_results[toppid] = {
                "status": "failed",
                "animal_id": animal_id,
                "reason": "No deployment match"
            }
            continue
        
        deployment_id = deployment_match.iloc[0]["Deployment ID"]
        print(f"  ✓ Found Deployment ID: {deployment_id}")
        
        # Step 3: Deployment ID -> Recording ID via recording_db
        recording_match = recording_db[recording_db["Deployment ID"] == deployment_id]
        
        if recording_match.empty:
            print(f"⚠️  No Recording ID found for Deployment ID {deployment_id} in recording_db")
            mapping_results[toppid] = {
                "status": "failed",
                "animal_id": animal_id,
                "deployment_id": deployment_id,
                "reason": "No recording match"
            }
            continue
        
        # Use the first recording ID if multiple exist
        recording_id = recording_match.iloc[0]["Recording ID"]
        if len(recording_match) > 1:
            print(f"  ⚠️  Multiple recordings found, using first: {recording_id}")
        else:
            print(f"  ✓ Found Recording ID: {recording_id}")
        
        # Store successful mapping
        mapping_results[toppid] = {
            "status": "success",
            "animal_id": animal_id,
            "deployment_id": deployment_id,
            "recording_id": recording_id
        }
    
    # Summary
    successful = sum(1 for r in mapping_results.values() if r['status'] == 'success')
    failed = sum(1 for r in mapping_results.values() if r['status'] == 'failed')
    
    print(f"\n{'='*70}")
    print(f"MAPPING COMPLETE")
    print(f"{'='*70}")
    print(f"Total TOPPIDs: {len(deployments)}")
    print(f"Successful mappings: {successful}")
    print(f"Failed mappings: {failed}")
    print(f"{'='*70}\n")
    
    return mapping_results


def create_dataset_structure(
    dataset_id: str,
    deployments: dict,
    mapping_results: dict,
    data_dir: str,
    save_parquet: bool = True,
    dry_run: bool = False,
    max_interpolation_gap: str | pd.Timedelta = "2min",
):
    """
    Create standardized dataset folder structure organized by deployment.
    
    Generates a hierarchical folder structure following pyologger conventions:
    - Top level: dataset folder
    - Second level: deployment-specific folders
    - Third level: metadata and raw data subfolders
    
    This structure enables consistent organization across projects and facilitates
    data discovery, version control, and collaborative analysis.
    
    Folder structure created:
        {dataset_id}_new/
            ├── {deployment_id_1}/
            │   ├── 00_metadata/          # Deployment-specific metadata files
            │   └── 01_raw-data/          # Raw data files
            │       └── {recording_id}_merged.parquet
            ├── {deployment_id_2}/
            │   ├── 00_metadata/
            │   └── 01_raw-data/
            │       └── {recording_id}_merged.parquet
            └── ...
    
    Parameters
    ----------
    dataset_id : str
        Base name for the dataset folder. Typically follows format:
        'species-stage-location_sensor-type_initials' 
        (e.g., 'mian-adult-nese_tdr-sr_TA-AT-YN-DC')
    deployments : dict
        Dictionary with TOPPID keys containing deployment data. Each entry should
        contain a 'merged_df' key with the dataframe to save (if save_parquet=True).
    mapping_results : dict
        Pre-computed TOPPID-to-metadata mappings from match_to_metadata(). Contains
        'deployment_id' and 'recording_id' for each successfully mapped TOPPID.
    data_dir : str
        Root data directory path where the dataset folder will be created.
    save_parquet : bool, default=True
        If True, saves each deployment's merged dataframe as a parquet file named
        by its recording ID. If False, only creates the folder structure.
    dry_run : bool, default=False
        If True, prints what would be created without actually creating folders
        or saving files. Useful for previewing structure before execution.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'dataset_folder': Path to the created dataset folder
        - 'deployment_folders': Dict with deployment IDs as keys, each containing:
            - 'folder': Path to deployment folder
            - 'toppids': List of TOPPIDs associated with this deployment
            - 'recording_ids': List of recording IDs for this deployment
            - 'parquet_files': List of saved parquet file paths (if save_parquet=True)
    
    Notes
    -----
    - Skips TOPPIDs with failed mappings (status != 'success' in mapping_results)
    - Handles multiple TOPPIDs per deployment (e.g., multiple tags on same animal)
    - Files are named by recording ID for database traceability
    - Creates parent directories automatically if they don't exist
    - In dry_run mode, no filesystem changes are made
    
    Warnings
    --------
    - Overwrites existing parquet files with the same recording ID
    - If mapping_results is incomplete, some deployments will be skipped
    - Large dataframes may take significant time to write as parquet
    
    Examples
    --------
    Basic usage:
    
    >>> # First, match TOPPIDs to metadata
    >>> mapping_results = match_to_metadata(deployments, animal_db, 
    ...                                      deployment_db, recording_db)
    >>> 
    >>> # Then create folder structure
    >>> result = create_dataset_structure(
    ...     dataset_id='mian-adult-nese_tdr-sr_TA-AT-YN-DC',
    ...     deployments=deployments,
    ...     mapping_results=mapping_results,
    ...     data_dir='/path/to/data',
    ...     save_parquet=True,
    ...     dry_run=False
    ... )
    >>> 
    >>> print(f"Created {len(result['deployment_folders'])} deployment folders")
    Created 12 deployment folders
    
    Dry run preview:
    
    >>> result = create_dataset_structure(
    ...     dataset_id='test_dataset',
    ...     deployments=deployments,
    ...     mapping_results=mapping_results,
    ...     data_dir='/path/to/data',
    ...     dry_run=True  # Preview without creating files
    ... )
    DRY RUN MODE - No folders or files will be created
    ...
    
    See Also
    --------
    match_to_metadata : Generates mapping_results required by this function
    """
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    def _resolve_time_column(df: pd.DataFrame) -> str | None:
        for candidate in ("stroke_time_utc", "tlld_time_utc", "datetime_utc", "datetime", "time"):
            if candidate in df.columns:
                return candidate
        return None

    def _infer_sampling_interval(dt_utc: pd.Series) -> pd.Timedelta:
        diffs = dt_utc.sort_values().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            raise ValueError("Cannot infer sampling interval from fewer than 2 unique timestamps.")
        return diffs.min()

    def _preserve_time_dtype(index_utc: pd.DatetimeIndex, original_time: pd.Series) -> pd.Series:
        parsed_original = pd.to_datetime(original_time, errors="coerce")
        original_tz = getattr(parsed_original.dt, "tz", None)
        if original_tz is None:
            return pd.Series(index_utc.tz_convert(None), index=index_utc)
        return pd.Series(index_utc.tz_convert(original_tz), index=index_utc)

    def _fill_short_time_gaps(merged_df: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame, int]:
        if merged_df is None or merged_df.empty:
            return merged_df, 0

        time_col = _resolve_time_column(merged_df)
        if time_col is None:
            raise ValueError(f"{label}: no datetime column found for gap interpolation.")

        prepared = merged_df.copy()
        prepared[time_col] = pd.to_datetime(prepared[time_col], utc=True, errors="coerce")
        prepared = prepared.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        if prepared.empty:
            return prepared, 0

        duplicated = prepared[time_col].duplicated(keep=False)
        if duplicated.any():
            dup_count = int(duplicated.sum())
            raise ValueError(f"{label}: found {dup_count} duplicate timestamps in {time_col}; cannot interpolate gaps.")

        if len(prepared) < 2:
            return prepared, 0

        sampling_interval = _infer_sampling_interval(prepared[time_col])
        gap_limit = pd.to_timedelta(max_interpolation_gap)
        diffs = prepared[time_col].diff().dropna()
        gap_sizes = diffs - sampling_interval
        violating = gap_sizes[gap_sizes >= gap_limit]
        if not violating.empty:
            first_bad_idx = violating.index[0]
            prev_time = prepared.loc[first_bad_idx - 1, time_col]
            next_time = prepared.loc[first_bad_idx, time_col]
            raise ValueError(
                f"{label}: gap from {prev_time} to {next_time} is {violating.iloc[0]}, "
                f"which meets or exceeds the {gap_limit} interpolation limit."
            )

        original_index = pd.DatetimeIndex(prepared[time_col])
        full_index = pd.date_range(
            start=prepared[time_col].iloc[0],
            end=prepared[time_col].iloc[-1],
            freq=sampling_interval,
            tz="UTC",
        )
        reindexed = prepared.set_index(time_col).reindex(full_index)
        inserted_mask = ~reindexed.index.isin(original_index)
        inserted_rows = int(inserted_mask.sum())

        if inserted_rows == 0:
            out = prepared.copy()
            out[time_col] = _preserve_time_dtype(
                pd.DatetimeIndex(out[time_col]),
                merged_df[time_col],
            ).to_numpy()
            return out.reset_index(drop=True), 0

        reindexed[time_col] = _preserve_time_dtype(reindexed.index, merged_df[time_col]).to_numpy()

        numeric_cols = [
            col for col in prepared.columns
            if col != time_col and is_numeric_dtype(prepared[col]) and not is_bool_dtype(prepared[col])
        ]
        bool_cols = [col for col in prepared.columns if col != time_col and is_bool_dtype(prepared[col])]
        other_cols = [col for col in prepared.columns if col not in set(numeric_cols + bool_cols + [time_col])]

        if numeric_cols:
            interpolated = reindexed[numeric_cols].interpolate(method="time", limit_area="inside")
            reindexed.loc[inserted_mask, numeric_cols] = interpolated.loc[inserted_mask, numeric_cols]

        if other_cols:
            filled = reindexed[other_cols].ffill().bfill()
            reindexed.loc[inserted_mask, other_cols] = filled.loc[inserted_mask, other_cols]

        if bool_cols:
            bool_filled = reindexed[bool_cols].ffill().bfill()
            reindexed.loc[inserted_mask, bool_cols] = bool_filled.loc[inserted_mask, bool_cols]

        out = reindexed.reset_index(drop=True)
        return out, inserted_rows
    
    # Create main dataset folder
    dataset_folder = Path(data_dir) / f"{dataset_id}_new"
    
    if not dry_run:
        dataset_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    if dry_run:
        print(f"DRY RUN MODE - No folders or files will be created")
        print(f"{'='*70}")
    print(f"Creating dataset structure: {dataset_folder}")
    print(f"{'='*70}\n")
    
    deployment_folders = {}
    
    for toppid, deployment_data in deployments.items():
        # Skip failed mappings
        if toppid not in mapping_results or mapping_results[toppid]['status'] != 'success':
            print(f"\n⚠️  Skipping TOPPID {toppid} (no valid mapping)")
            continue
        
        mapping = mapping_results[toppid]
        deployment_id = mapping['deployment_id']
        recording_id = mapping['recording_id']
        
        print(f"\nProcessing TOPPID: {toppid}")
        print("-" * 50)
        print(f"  Deployment ID: {deployment_id}")
        print(f"  Recording ID: {recording_id}")
        
        # Create deployment folder structure
        deployment_folder = dataset_folder / str(deployment_id)
        metadata_folder = deployment_folder / "00_metadata"
        rawdata_folder = deployment_folder / "01_raw-data"
        
        # Create directories
        if not dry_run:
            metadata_folder.mkdir(parents=True, exist_ok=True)
            rawdata_folder.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {deployment_folder.relative_to(data_dir)}/")
        else:
            print(f"  [DRY RUN] Would create: {deployment_folder.relative_to(data_dir)}/")
        
        print(f"    - 00_metadata/")
        print(f"    - 01_raw-data/")
        
        # Save merged data as parquet if requested (named by Recording ID)
        parquet_path = None
        if save_parquet and "merged_df" in deployment_data:
            merged_df, inserted_rows = _fill_short_time_gaps(
                deployment_data["merged_df"],
                label=f"TOPPID {toppid} / deployment {deployment_id}",
            )
            parquet_filename = f"{recording_id}_merged.parquet"
            parquet_path = rawdata_folder / parquet_filename
            
            if not dry_run:
                merged_df.to_parquet(parquet_path, index=False)
                print(
                    f"  ✓ Saved: 01_raw-data/{parquet_filename} "
                    f"({len(merged_df):,} rows, interpolated {inserted_rows:,} gap rows)"
                )
            else:
                print(
                    f"  [DRY RUN] Would save: 01_raw-data/{parquet_filename} "
                    f"({len(merged_df):,} rows, interpolated {inserted_rows:,} gap rows)"
                )
        
        # Track deployment folders (handle multiple TOPPIDs per deployment)
        if deployment_id not in deployment_folders:
            deployment_folders[deployment_id] = {
                "folder": deployment_folder,
                "toppids": [],
                "recording_ids": [],
                "parquet_files": []
            }
        deployment_folders[deployment_id]["toppids"].append(toppid)
        deployment_folders[deployment_id]["recording_ids"].append(recording_id)
        if parquet_path:
            deployment_folders[deployment_id]["parquet_files"].append(parquet_path)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"DATASET STRUCTURE CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Unique deployments created: {len(deployment_folders)}")
    
    if deployment_folders:
        print(f"\nDeployment folders created:")
        for dep_id, info in deployment_folders.items():
            rec_ids = ', '.join(info['recording_ids'])
            print(f"  - {dep_id}: {len(info['toppids'])} TOPPID(s), Recording IDs: [{rec_ids}]")
    
    print(f"{'='*70}\n")
    
    return {
        "dataset_folder": dataset_folder,
        "deployment_folders": deployment_folders
    }


def map_deployment_to_dataset(root_dir: str | Path) -> dict[str, str]:
    """Map deployment IDs to dataset IDs by scanning the dataset root directory.

    Walks the first two levels of *root_dir* (dataset / deployment) and returns
    a dictionary ``{deployment_id: dataset_id}``.  Folders whose names start
    with ``00_`` are skipped (metadata directories by convention).
    """
    root_dir = Path(root_dir)
    dep_to_ds: dict[str, str] = {}
    if not root_dir.is_dir():
        return dep_to_ds

    for dataset_dir in sorted(root_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("00_"):
            continue
        for deployment_dir in dataset_dir.iterdir():
            if deployment_dir.is_dir() and not deployment_dir.name.startswith("00_"):
                dep_to_ds.setdefault(deployment_dir.name, dataset_dir.name)
    return dep_to_ds


def _trim_event_df_to_window(
    event_df: pd.DataFrame | None,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame | None:
    """Trim an event table to an analysis window, preserving overlapping events."""
    if event_df is None or len(event_df) == 0 or "datetime" not in event_df.columns:
        return event_df

    out = event_df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce", utc=True)

    end_col = None
    for candidate in ("end_datetime", "end_time", "datetime_end", "end"):
        if candidate in out.columns:
            out[candidate] = pd.to_datetime(out[candidate], errors="coerce", utc=True)
            end_col = candidate
            break

    if end_col is not None:
        event_end = out[end_col]
    else:
        event_end = out["datetime"].copy()
        duration_col = next(
            (c for c in ("duration_s", "duration_sec", "duration_pos", "duration") if c in out.columns),
            None,
        )
        if duration_col is not None:
            duration_s = pd.to_numeric(out[duration_col], errors="coerce").fillna(0.0)
            event_end = out["datetime"] + pd.to_timedelta(duration_s, unit="s")

    mask = (
        out["datetime"].notna()
        & (out["datetime"] <= end_utc)
        & event_end.notna()
        & (event_end >= start_utc)
    )
    return out.loc[mask].sort_values("datetime").reset_index(drop=True)


def trim_data_pkl_to_accelerometer_window(
    data_pkl: Any,
    *,
    anchor_signal: str = "accelerometer",
    verbose: bool = True,
) -> tuple[Any, pd.Timestamp | None, pd.Timestamp | None]:
    """Trim all signal/event tables in *data_pkl* to the anchor signal's time window.

    Returns ``(data_pkl, start_utc, end_utc)`` where the start/end timestamps
    reflect the anchor signal's extent.  If the anchor signal is missing the
    object is returned unmodified with ``(None, None)``.
    """
    if not hasattr(data_pkl, "signal_data") or data_pkl.signal_data is None:
        return data_pkl, None, None

    anchor_df = data_pkl.signal_data.get(anchor_signal)
    if anchor_df is None or len(anchor_df) == 0 or "datetime" not in anchor_df.columns:
        return data_pkl, None, None

    anchor_dt = pd.to_datetime(anchor_df["datetime"], utc=True, errors="coerce").dropna()
    if anchor_dt.empty:
        return data_pkl, None, None

    start_utc = anchor_dt.min()
    end_utc = anchor_dt.max()

    for signal_name, signal_df in list(data_pkl.signal_data.items()):
        if signal_df is None or len(signal_df) == 0 or "datetime" not in signal_df.columns:
            continue
        dt = pd.to_datetime(signal_df["datetime"], utc=True, errors="coerce")
        keep = dt.notna() & (dt >= start_utc) & (dt <= end_utc)
        trimmed = signal_df.loc[keep].copy()
        if "datetime" in trimmed.columns:
            ref_dt = pd.to_datetime(signal_df["datetime"], errors="coerce")
            if getattr(ref_dt.dt, "tz", None) is None:
                trimmed["datetime"] = dt.loc[keep].dt.tz_convert(None)
            else:
                trimmed["datetime"] = dt.loc[keep].dt.tz_convert(ref_dt.dt.tz)
        data_pkl.signal_data[signal_name] = trimmed.reset_index(drop=True)

    if hasattr(data_pkl, "event_data") and isinstance(data_pkl.event_data, pd.DataFrame):
        trimmed_events = _trim_event_df_to_window(data_pkl.event_data, start_utc, end_utc)
        if trimmed_events is not None:
            ref_signal = next(
                (
                    sig_df
                    for sig_df in data_pkl.signal_data.values()
                    if isinstance(sig_df, pd.DataFrame)
                    and len(sig_df) > 0
                    and "datetime" in sig_df.columns
                ),
                None,
            )
            if ref_signal is not None and len(trimmed_events) > 0:
                ref_dt = pd.to_datetime(ref_signal["datetime"], errors="coerce")
                ref_tz = ref_dt.dt.tz
                for col in ("datetime", "end_datetime"):
                    if col in trimmed_events.columns:
                        dt_col = pd.to_datetime(trimmed_events[col], errors="coerce", utc=True)
                        trimmed_events[col] = (
                            dt_col.dt.tz_convert(None) if ref_tz is None else dt_col.dt.tz_convert(ref_tz)
                        )
            data_pkl.event_data = trimmed_events

    if verbose:
        print(f"Trimmed data.pkl to accelerometer window: {start_utc} to {end_utc}")
    return data_pkl, start_utc, end_utc
