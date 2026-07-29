# AI Agent Documentation — pyologger

> **Purpose**: Token-efficient reference for AI agents working on the pyologger biologging processing and segmentation pipeline.

## Quick Reference

- **Project**: Multi-logger biologging data processing, segmentation, and DiveDB upload pipeline
- **Entry Point**: `select_and_load_deployment(data_dir, dataset_id, deployment_id)` (`utils/folder_manager.py`)
- **Tech Stack**: Python, Snakemake, pandas, numpy, scikit-learn, LightGBM, pycatch22, UMAP, xarray, PyArrow, Streamlit, Plotly
- **Key Dependencies**: `ParamManager`, `DataReader`, `SegmentationPipeline`, `find_segments`
- **Config**: `config.yaml` (paths, datasets) + `segmentation_runs.yaml` (named segmentation runs); loaded via `load_combined_config()`

## Architecture Overview

### Data Flow

```
raw tag files (CATS/WC/StarOddi/CSV/…)
    → DataReader (load_data/datareader.py)
    → data.pkl (DataPkl object) + _00_processed.nc
    → workflows/01–06_*.py  (calibrate, stroke, HR, export)
    → _step01.nc … output.nc
    → DiveDB DataUploader (notebooks/00_data_to_diveDB.ipynb)
    → Apache Iceberg lake
```

### Segmentation Flow

```
config.yaml + segmentation_runs.yaml
    → RunContext (run_name, scope, features, unsupervised, supervised, summary cfg)
    → Stage 09: cross-dataset QC  → qc/qc_channels.csv
    → Stage 10: algorithmic segs  → segments/algorithmic_segments.parquet
    → Stage 11: feature gen       → features/features_filtered.parquet
    → Stage 12: unsupervised      → clustering/clustered_windows.parquet
    → Stage 13: supervised        → supervised/supervised_predictions.parquet
    → Stage 14: summary           → summary/method_budget_daily.parquet
                                     summary/method_budget_hourly.parquet
```

## File Map

| File | Purpose | Key Exports | Approx Lines |
|------|---------|-------------|-------------|
| `utils/folder_manager.py` | Deployment selection and config loading | `select_and_load_deployment`, `resolve_deployment_context`, `load_configuration`, `load_combined_config`, `match_to_metadata`, `create_dataset_structure`, `map_deployment_to_dataset`, `trim_data_pkl_to_accelerometer_window` | 870 |
| `load_data/datareader.py` | Raw tag file reader; produces data.pkl | `DataReader` | ~600 |
| `load_data/metadata.py` | Notion metadata fetcher | `Metadata` | ~300 |
| `analyze_data/segmentation_pipeline.py` | Full segmentation pipeline orchestrator | `RunContext`, stage runner functions | ~2000 |
| `analyze_data/find_segments.py` | Segment detection from labeled columns | `find_segments`, `find_adjusted_segments` | ~200 |
| `analyze_data/segmentation_run_summaries.py` | Parquet summary writers | `write_segmentation_run_summary_parquets` | ~300 |
| `analyze_data/analyze_segments.py` | Post-hoc segment analysis | — | ~200 |
| `utils/param_manager.py` | Per-deployment parameter persistence | `ParamManager` | ~300 |
| `utils/time_manager.py` | Datetime and sampling utilities | `calculate_sampling_frequency` | ~200 |
| `utils/event_manager.py` | Event dataframe helpers | `create_state_event` | ~200 |
| `utils/deployment_source.py` | Load data.pkl or NetCDF, resolve signals | `DeploymentSource` | ~400 |
| `utils/segmentation_run_config.py` | Normalize run config dicts | `normalize_segmentation_run_cfg` | ~150 |
| `utils/segmentation_qc.py` | Channel unit validation | `load_standardized_channel_unit_map`, `normalize_unit_token` | ~200 |
| `utils/cluster_colors.py` | Cluster color mapping | `build_ordered_cluster_color_map` | ~80 |
| `utils/workflow_netcdf.py` | NetCDF path resolution for workflows | `latest_processing_netcdf_path` | ~80 |
| `utils/montage_manager.py` | Montage log read/write | `MontageManager` | ~200 |
| `utils/data_manager.py` | data.pkl manipulation helpers | `clear_intermediate_signals` | ~200 |
| `utils/align_signals.py` | Signal alignment utilities | — | ~150 |
| `utils/solar_utils.py` | Solar position utilities | — | ~100 |
| `process_data/peak_detect.py` | Stroke and heartbeat peak detection | `detect_strokes`, `detect_heartbeats` | ~600 |
| `process_data/feature_generation_utils.py` | Catch22 + stats feature generation | — | ~400 |
| `process_data/sampling.py` | Resampling utilities | — | ~200 |
| `process_data/odba.py` | ODBA calculation | — | ~100 |
| `calibrate_data/calibrate_acc_mag.py` | Acc/mag calibration | — | ~300 |
| `calibrate_data/tag2animal.py` | Tag-to-animal frame rotation | — | ~200 |
| `calibrate_data/zoc.py` | Zero-offset correction | — | ~200 |
| `plot_data/plotter.py` | Interactive Plotly signal viewer | `plot_tag_data_interactive5` | ~800 |
| `plot_data/sleep_budget_plot.py` | Sleep/behavior budget plots | — | ~200 |
| `io_operations/base_importer.py` | Shared importer logic: channel standardization, signal grouping, EDF read/decimation | `BaseImporter`, `group_edf_signals_by_frequency`, `_prefetch_edf_signals`, `resolve_signal_name`, `DEFAULT_EDF_TARGET_FREQUENCIES`, `RECOMPUTED_SIGNALS` | ~950 |
| `io_operations/{evolocus,manitty}_importer.py` | EDF importers; also ingest a CSV/Parquet export from the same logger when present | `EvolocusImporter`, `ManittyImporter` | ~150 each |
| `io_operations/csv_importer.py` | Generic CSV/Parquet importer (delegated to by the EDF importers) | `CSVImporter` | ~150 |
| `io_operations/*_importer.py` | Other format-specific importers (CATS, WC, StarOddi, Vectronics, …) | format subclasses | ~1200 total |
| `dash/integrated/integrated_dash.py` | Integrated Dash app: interactive plot, time selector, 3D model, synchronized video, segmentation UI | Dash `app`, Flask routes `/local-video`, `/immich-video` | ~12k |
| `dash/integrated/segmentation_helpers.py` | Segmentation workflow helpers for the Dash app | `DEFAULT_SEGMENTATION_DATASET/DEPLOYMENT`, workflow preset fns | ~1600 |
| `dash/integrated/model_3d.py` | 3D orientation model data for the Dash app | `build_orientation_data_json`, `fetch_3d_model_info` | ~600 |

## Module Reference

### `utils/folder_manager.py`

**Purpose**: Deployment resolution, config loading, and data.pkl access

**Key Functions**:
- `select_and_load_deployment(data_dir, dataset_id=None, deployment_id=None)` → `(animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager)` — CLI/function-based loader; prompts interactively if IDs not given
- `select_and_load_deployment_streamlit(data_dir)` → same 7-tuple — Streamlit sidebar version
- `resolve_deployment_context(data_dir, dataset_id=None, deployment_id=None)` → `(animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, param_manager)` — paths without loading data.pkl
- `load_configuration()` → `(config, data_dir, color_mapping_path, montage_path)` — loads config.yaml via dotenv
- `load_combined_config(config_path=None, segmentation_runs_path=None)` → `(config_dict, main_config_path, runs_path)` — merges config.yaml + segmentation_runs.yaml
- `resolve_config_path(config_path=None)` → `Path` — resolves via arg → `CONFIG_PATH` env var → repo default
- `match_to_metadata(deployments, animal_db, deployment_db, recording_db, animal_db_mapping_column="Domain IDs")` → `dict` — maps TOPPID → animal/deployment/recording IDs
- `create_dataset_structure(dataset_id, deployments, mapping_results, data_dir, save_parquet=True, dry_run=False, max_interpolation_gap="2min")` → `dict` — creates `{dataset_id}_new/` folder tree
- `map_deployment_to_dataset(root_dir)` → `dict[deployment_id, dataset_id]` — scans two-level directory tree
- `trim_data_pkl_to_accelerometer_window(data_pkl, anchor_signal="accelerometer", verbose=True)` → `(data_pkl, start_utc, end_utc)` — clips all signals to anchor signal's time window

### `load_data/datareader.py`

**Purpose**: Read raw tag files (multi-format) and build the initial data.pkl

**Key Classes**:
- `DataReader`: Primary data loading class

**Key Methods**:
- `__init__(deployment_folder_path)` — initialize with deployment directory
- `check_deployment_folder(deployment_db, data_dir)` → `(deployment_folder, deployment_id)` — validate folder against metadata
- `read_files(metadata, save_csv, save_parq, save_edf, montage_path, save_netcdf)` → `None` — reads all files, builds data.pkl, optionally exports

### `io_operations/base_importer.py`

**Purpose**: Shared importer logic — channel standardization, signal grouping, EDF reading

**Key Methods** (EDF path):

- `group_edf_signals_by_frequency(signals, startdate, starttime, time_zone, skip_full=False, channel_metadata=None)` → `dict[int, pd.DataFrame]` — groups signals by **resulting** rate (not source rate), so one source group may split into several frames. Builds each group as a preallocated `float32` matrix, reading `EdfSignal.data` exactly once per signal.
- `_prefetch_edf_signals(signals)` — fills `_digital` for all signals in a contiguous column run with one chunked pass over edfio's data-record buffer, replacing one strided pass per signal (~1.6x faster). Silently falls back to edfio's per-signal path if the layout is unexpected.
- `get_edf_target_frequencies()` → `dict[str, float]` — resolves decimation targets: class defaults → `config.yaml` `edf_import.target_frequencies` → `parameter_log.json` `settings.edf_target_frequencies`. `null` disables a signal type.
- `_edf_decimation_step(label, source_hz, targets, channel_metadata)` → `int` — integer factor only; returns 1 if the target would give a non-integer rate (downstream stores `int(logger_info['fs'])`).
- `_edf_signal_column(signal, step, n_out)` → `np.ndarray` — `scipy.signal.decimate(ftype="fir", zero_phase=True)`; plain slicing only as a logged fallback.
- `resolve_signal_name(signal_name)` → `str` — appends `_2` for signals in `RECOMPUTED_SIGNALS` when `is_derived_logger()` (montage id contains `derived`), reserving canonical names for pyologger's own output.

**Constants**:

- `DEFAULT_EDF_TARGET_FREQUENCIES = {"eeg": 100, "eog": 100, "emg": 100, "ecg": 250}`
- `RECOMPUTED_SIGNALS = {stroke_rate, heart_rate, depth, prh, velocity, position, location}`

**Gotcha**: `edfio.EdfSignal.data` is a property that re-materializes a full `float64` array on
every access and is never cached. Read it once per signal. Do **not** set `signal._digital = None`
to free memory — edfio consumes `_lazy_loader` on first read, so this makes the signal
permanently unreadable (`ValueError: Signal data not set`).

See `docs/source/edf_import.rst` for the full design.

### `analyze_data/segmentation_pipeline.py`

**Purpose**: Full segmentation pipeline orchestrator for all stages 09–14

**Key Classes**:
- `RunContext`: Dataclass — `run_name`, `run_cfg`, `global_cfg`, `data_root`, `output_root`, `analysis_id`, `scope`

**Key Stage Functions** (called by Snakemake via CLI):
- Stage entry points invoked as `python workflows/10_algorithmic_segmentation.py --config config.yaml --run <name>`
- Internal helpers: `_load_yaml(path)`, `_ensure_dir(path)`, `_deployment_timezone_name(data_pkl)`, `_normalize_window_columns(df, tz_name)`, `_resolve_primary_budget_method(method_meta_df, metrics_df, run_cfg)`

### `analyze_data/find_segments.py`

**Purpose**: Detect contiguous segments in a signal column

**Key Functions**:
- `find_segments(data, column, criteria, min_duration=None, animal_id_col='animal_id')` → `pd.DataFrame` — loop-based segment finder; handles multi-animal DataFrames; respects animal ID boundaries
- `find_adjusted_segments(...)` → `pd.DataFrame` — variant with boundary adjustment logic

### `utils/time_manager.py`

**Key Functions**:
- `calculate_sampling_frequency(datetime_series)` → `float` — median-interval Hz estimator; returns `np.nan` if undetermined; robust to duplicates and disorder

### `utils/deployment_source.py`

**Purpose**: Unified loader for data.pkl or NetCDF with signal resolution

**Key Classes**:
- `DeploymentSource`: Thread-safe loader with lazy NetCDF fallback

**Key Methods**:
- `__init__(deployment_folder, deployment_id)` — initialize; does not load until accessed
- `get_signal(signal_name)` → `pd.DataFrame | None` — returns signal from data.pkl or latest NetCDF
- `get_event_data()` → `pd.DataFrame | None` — returns event table

### `process_data/peak_detect.py`

**Purpose**: Stroke rate and heartbeat detection from accelerometer/ECG signals

**Module-level constants**: `BROAD_LOW_CUTOFF=1`, `BROAD_HIGH_CUTOFF=35`, `NARROW_LOW_CUTOFF=5`, `NARROW_HIGH_CUTOFF=20`, `FILTER_ORDER=2`, `SPIKE_THRESHOLD=400`, `QUADRUPED_SPECIES_CODES` (set — `"pale"` etc.)

**Key Functions**:
- `detect_strokes(data_pkl, mode="stroke_rate", ...)` → `pd.DataFrame` — bandpass filter + peak detection on acc; auto-converts to `stride_rate` for quadrupeds
- `detect_heartbeats(data_pkl, ...)` → `pd.DataFrame` — ECG peak detection via wfdb
- `_should_convert_stroke_to_stride_rate(data_pkl, mode)` → `bool` — checks species code against `QUADRUPED_SPECIES_CODES`

### `dash/integrated/integrated_dash.py`

**Purpose**: The Integrated Dash app — interactive multi-signal plot, time selector,
3D orientation model, synchronized video, and the segmentation UI (`/segmentation`).
Formerly `dash/minimal_interactive/app.py` (renamed 2026-07).

**Launch** (from `pyologger/`):

```bash
# open the configured default deployment (no args required)
python dash/integrated/integrated_dash.py

# or target a specific deployment / port
python dash/integrated/integrated_dash.py --dataset <id> --deployment <id> --port 8061
```

- `--dataset` / `--deployment` are **optional**. `_resolve_launch_target()` picks:
  explicit args → `DEFAULT_SEGMENTATION_DATASET`/`DEFAULT_SEGMENTATION_DEPLOYMENT`
  (from `segmentation_helpers.py`) → first dataset on disk with a deployment that
  has `outputs/data.pkl`. Prints `[launch] Opening dataset=… deployment=…`.
- Routes: `/` (main viewer), `/segmentation`, `/water` (experimental WebGL tank).

**Synchronized video** (matches a playhead time to a video clip and plays it):

- **Source precedence** — `_build_clip_index()`: Immich album `DepID_<deployment_id>`
  (via `DiveDB.services.immich_service.ImmichService`) → local files under
  `VIDEO_DIR` (override with `PYOLOGGER_VIDEO_DIR`). Prints `[video] Using N … clip(s)`.
- **Clip index** — each clip: `{name, start_epoch, end_epoch, source, url}`. Immich
  clips use the raw `find_media` `fileCreatedAt` (true UTC) — NOT
  `prepare_video_options_for_react` (which mislabels local time as UTC). Local clips
  parse `…_HH-MM-SS_HH-MM-SS.mp4` filenames in the deployment timezone.
- **Serving** — Flask routes stream with HTTP Range support (seeking):
  - `/immich-video/<asset_id>` proxies Immich `/assets/{id}/video/playback` with the
    `x-api-key` header server-side (key never reaches the browser); forwards `Range`.
  - `/local-video/<name>` serves local files; lazily runs `qt-faststart`/`ffmpeg
    -movflags +faststart` into a `.faststart/` cache (fixes black frames from a
    trailing `moov` atom).
- **Immich creds** — `IMMICH_API_KEY` / `IMMICH_BASE_URL`; `_ensure_immich_env()`
  loads them from `PYOLOGGER_IMMICH_ENV` or `../EcoPhysVideoViz/.env` if unset.

**Playback clock** (single play button drives plot playhead + video):

- `assets/playback-manager.js` (`window.IntegratedPlayback`) is a rAF clock used only
  when the playhead is in a gap (no video). When a clip is loaded the `<video>` is the
  **master clock** — a `timeupdate` listener drives `playhead-time` via
  `dash_clientside.set_props`, and the sync callback never seeks the element while it
  plays (seeking mid-play was the cause of blank-frame-with-audio).
- Stores/ids: `playhead-time`, `is-playing`, `playback-rate`, `play-pause-btn`,
  `playback-interval`, `sync-video`, `sync-video-current-clip`. Spacebar toggles play.
- The main-plot yellow playhead line moves clientside (`Plotly.relayout`) during
  playback; the server redraw is skipped while `is-playing`.

**Coverage strips** (EcoPhysVideoViz-style CSS bars): `_video_coverage_bars()` builds
one `.coverage-bar` per clip positioned by `--seg-start/--seg-end` against the strip's
`--view-min/--view-max`. Full-range strip under the window slider; window-aligned strip
under the main plot (padded to Plotly's `_fullLayout._size` margins).

## Segmentation Pipeline

### Stage Table

| Stage | Script | Inputs | Key Outputs |
|-------|--------|--------|-------------|
| 09 QC | `09_cross_dataset_qc.py` | data.pkl files in scope | `qc/qc_channels.csv` |
| 10 Algorithmic | `10_algorithmic_segmentation.py` | data.pkl, config | `segments/algorithmic_segments.parquet` |
| 11 Features | `11_feature_generation.py` | data.pkl, algorithmic segments | `features/features_raw.parquet`, `features_filtered.parquet`, `feature_index.parquet`, `feature_correlation_matrix.parquet`, `dropped_correlated_features.csv` |
| 12 Unsupervised | `12_unsupervised_segmentation.py` | features_filtered.parquet | `clustering/clustered_windows.parquet` |
| 13 Supervised | `13_supervised_segmentation.py` | features, event labels | `supervised/supervised_predictions.parquet`, holdout diagnostics, feature importances |
| 14 Summary | `14_summary.py` | all prior outputs | `summary/method_budget_daily.parquet`, `method_budget_hourly.parquet` |

### `segmentation_runs.yaml` Structure

```yaml
segmentation_runs:
  <run_name>:
    scope:
      analysis_id: <str>          # output subdirectory name
      dataset_ids: [<str>, ...]   # datasets to include
      deployment_ids: [<str>, ...] # specific deployments (subset of datasets)
      expected_signals: [<str>, ...] # signals required to be present
      groups:                     # optional grouping for cross-group analysis
        <group_label>: [<deployment_id>, ...]

    features:
      normalization_level: deployment | global
      cluster_length_mode: fixed | dive | event
      duration_s: <int>           # window size for fixed mode
      min_samples_per_chunk: <int>
      max_chunks_per_deployment: <int>
      max_chunks_total: <int>
      random_state: <int>
      strict_qc: <bool>
      sampling_compatibility_tolerance: <float | null>
      channel_feature_spec:
        <signal>.<channel>:
          transforms: [raw | smoothed | derivative]
          feature_set: full | medium | minimal
          required: <bool>        # default true

    unsupervised:
      n_clusters: <int>           # KMeans k
      umap_n_neighbors: <int>
      umap_min_dist: <float>
      umap_sample_frac: <float>

    supervised:
      label_source: <event_key>   # event key to use as ground truth
      classifier: rf | gbm | svm | knn
      export_rf_events: <bool>    # write rf_* state events back to data.pkl

    summary:
      primary_method_kind: unsupervised | supervised
      primary_method_id: <str>
      budget_label_mode: canonical | raw
      context_filter_pass_mode: none | measured_only | full
      actograms:
        daily: <bool>
        continuous: <bool>
```

### DUR_BINS and Cluster Config Patterns

```yaml
# Fixed-window features (e.g. 10 s windows for HR analysis)
features:
  cluster_length_mode: fixed
  duration_s: 10
  channel_feature_spec:
    heart_rate.heart_rate:
      transforms: [raw]
      feature_set: full

# Dive-based windows
features:
  cluster_length_mode: dive
  channel_feature_spec:
    depth.depth:
      transforms: [raw]
      feature_set: medium
      required: false

# Unsupervised KMeans + UMAP
unsupervised:
  n_clusters: 5
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  umap_sample_frac: 0.2
```

## Data Structures

### DataPkl Object Fields

| Field | Type | Contents |
|-------|------|----------|
| `signal_data` | `dict[str, pd.DataFrame]` | Signal name → DataFrame with `datetime` + channel columns |
| `signal_info` | `dict[str, dict]` | Signal name → metadata (channels, units, freq, logger_id, stats, processing_log) |
| `event_data` | `pd.DataFrame` | Event table: `datetime`, `type`, `key`, `value`, `duration`, `short_description`, `long_description` |
| `derived_data` | `dict[str, pd.DataFrame]` | Derived signal name → DataFrame (same structure as signal_data) |
| `derived_info` | `dict[str, dict]` | Derived signal metadata (mirrors signal_info; includes `derived_from_signals`) |
| `param_manager` | `ParamManager` | Per-deployment parameter store (reads/writes JSON sidecar) |
| `deployment_info` | `dict` | Deployment metadata from Notion (deployment ID, dates, time zone, etc.) |
| `animal_info` | `dict` | Animal metadata from Notion (Animal_ID, species, etc.) |

**`signal_data` DataFrame columns**: first column always `datetime` (tz-aware pandas Timestamp); remaining columns are standardized channel names (e.g. `ax`, `ay`, `az` for accelerometer; `ecg` for ECG; `depth` for pressure).

**`event_data` schema**:

| Column | Type | Notes |
|--------|------|-------|
| `datetime` | tz-aware Timestamp | Event start |
| `type` | str | `"point"` or `"state"` |
| `key` | str | Standardized event key (e.g. `heartbeat_manual_ok`, `sleep_state_auto`) |
| `value` | float | Optional numeric value |
| `duration` | float | Seconds; 0 for point events |
| `short_description` | str | Brief label |
| `long_description` | str | Optional extended annotation |

### Segmentation Output Parquet Schemas

**`clustering/clustered_windows.parquet`**:

| Column | Notes |
|--------|-------|
| `window_start` | tz-aware Timestamp |
| `window_end` | tz-aware Timestamp |
| `deployment_id` | str |
| `dataset_id` | str |
| `organism_id` | str (canonical; replaces `animal_id` in new outputs) |
| `cluster` | int — KMeans cluster label |
| `umap_x`, `umap_y` | float — UMAP embedding coordinates |
| feature columns | float — normalized feature values |

**`supervised/supervised_predictions.parquet`**:

| Column | Notes |
|--------|-------|
| `window_start`, `window_end` | tz-aware Timestamps |
| `deployment_id`, `dataset_id`, `organism_id` | str |
| `predicted_label` | str — classifier output |
| `confidence` | float — calibrated probability of predicted class |
| `true_label` | str or NaN — ground truth (holdout rows only) |

## Common Patterns

### Loading a Deployment

```python
from pyologger.utils.folder_manager import load_configuration, select_and_load_deployment

config, data_dir, color_mapping_path, montage_path = load_configuration()

animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = \
    select_and_load_deployment(
        data_dir,
        dataset_id="oror-adult-orca_hr-sr-vid_sw_JKB-PP",
        deployment_id="2023-06-13_oror-002",
    )
```

### Accessing Signals from data_pkl

```python
# Get a signal DataFrame
depth_df = data_pkl.signal_data["depth"]
# depth_df.columns: ["datetime", "depth"]

acc_df = data_pkl.signal_data["accelerometer"]
# acc_df.columns: ["datetime", "ax", "ay", "az"]

# Signal metadata
freq = data_pkl.signal_info["depth"]["sampling_frequency"]  # Hz

# Event data
sleep_events = data_pkl.event_data[data_pkl.event_data["key"] == "sleep_state_auto"]
```

### Running Segmentation

```bash
# From pyologger/
MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); \
  print(c['paths']['local_private_meta_analysis_data'])")/segmentation

RUN_NAME=finescale_hr
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/14_summary.done
```

```python
# Reset and rerun from features stage
# From shell:
# scripts/reset_segmentation_run.sh finescale_hr --scope features
```

### Finding Segments Programmatically

```python
from pyologger.analyze_data.find_segments import find_segments

dive_segments = find_segments(
    data=depth_df,
    column="depth",
    criteria=lambda v: v > 10.0,  # depth > 10 m
    min_duration=30.0,             # at least 30 seconds
)
# Returns DataFrame: window_start, window_end, duration, ...
```

### Loading Config Directly

```python
from pyologger.utils.folder_manager import load_combined_config

config, config_path, runs_path = load_combined_config()
data_root = config["paths"]["local_private_data"]
seg_runs = config.get("segmentation_runs", {})
```

## Maintenance Guidelines

### When to Update This Documentation

- **New files**: Add entry to File Map table
- **New functions**: Add signature to Module Reference (Key Functions/Methods only)
- **New segmentation stage**: Add row to Stage Table and update segmentation flow diagram
- **Config key changes**: Update `segmentation_runs.yaml` Structure section
- **New data.pkl fields**: Update DataPkl Object Fields table
- **New output Parquets**: Add schema to Segmentation Output Parquet Schemas
- **New patterns**: Add to Common Patterns (high-level only)

### Documentation Standards

1. **Token-efficient**: tables over prose; function signatures, not implementations
2. **Include**: file purposes, key exports, function signatures, data flow, schema fields
3. **Exclude**: internal implementation details, step-by-step how-it-works prose
4. **Front-load**: Quick Reference first; detailed reference follows

### Update Checklist

- [ ] New method → Module Reference (signature only)
- [ ] New file → File Map table
- [ ] Signature changed → Module Reference
- [ ] Schema changed → Data Structures section
- [ ] New stage → Stage Table
- [ ] New run config key → `segmentation_runs.yaml` Structure section

**Remember**: `organism_id` is canonical in all new outputs; `animal_id` is legacy for NetCDF/DiveDB upload only.
