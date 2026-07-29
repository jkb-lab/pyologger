# pyologger

Tag-processing and segmentation pipeline for multi-logger biologging data: loads raw tag files into `data.pkl`, runs calibration and feature workflows via Snakemake, segments behavior using unsupervised/supervised ML, and exports to DiveDB (Apache Iceberg lake).

**Technical reference:** [AI_DOCS.md](AI_DOCS.md)

## Layout

| Area | Path | Purpose |
|------|------|---------|
| Python package | `pyologger/` | All library code (see subdir layout below) |
| Snakemake processing | `workflows/00–06_*.py`, `Snakefile` | Per-deployment processing steps |
| Snakemake segmentation | `workflows/09–14_*.py` | Cross-dataset segmentation pipeline |
| Streamlit UI | `streamlit/`, `streamlit/pages/` | Interactive data exploration pages |
| Dash app | `dash/integrated/integrated_dash.py` | Integrated Dash app: interactive plot, time selector, 3D orientation, synchronized video (Immich/local), segmentation UI. Launch: `python dash/integrated/integrated_dash.py` (deployment optional). See `AI_DOCS.md`. |
| Notebooks | `notebooks/` | Analysis, review, and upload workflows |
| Config | `config.yaml` | Paths, datasets, and run_selection flags |
| Segmentation runs | `segmentation_runs.yaml` | Named segmentation run configs |
| Snakefile | `Snakefile` | Main DAG for all workflows |

### `pyologger/` package submodules

| Submodule | Purpose |
|-----------|---------|
| `load_data/` | `DataReader` — reads raw tag files into `data.pkl` |
| `io_operations/` | Format-specific importers (CATS, WC, StarOddi, CSV, KML, etc.) |
| `calibrate_data/` | Accelerometer/magnetometer calibration, ZOC, tag2animal rotation |
| `process_data/` | Stroke detection, HR peak detection, feature generation, sampling |
| `analyze_data/` | Segmentation pipeline, find_segments, run summaries |
| `plot_data/` | Interactive plotter, sleep budget plots, environmental covariates |
| `utils/` | `folder_manager`, `param_manager`, `time_manager`, `event_manager`, `deployment_source`, etc. |

## Core abstractions

| Abstraction | What it is |
|-------------|-----------|
| `data_pkl` | Central per-deployment object (`DataPkl`): holds `signal_data`, `signal_info`, `event_data`, `param_manager`, `deployment_info`, `animal_info` |
| `select_and_load_deployment` | Entry point in `folder_manager.py` — resolves dataset/deployment, loads `data.pkl`, returns `(animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager)` |
| `segmentation_runs.yaml` | Named run configs (`scope`, `features`, `unsupervised`, `supervised`, `summary`) consumed by stages 09–14 |
| `config.yaml` | Repo-level config: `paths`, `datasets`, `run_selection`, `overwrite_step01_from_nc` |

## Processing pipeline stages (00–06)

All stages run from `pyologger/` using `Snakefile`:

| Stage | Script | Output |
|-------|--------|--------|
| 00 | `workflows/00_load_data.py` | `outputs/data.pkl`, `outputs/{dep}_00_processed.nc` |
| 00b | `workflows/00_backup_data.py` | Backup of raw data |
| 01 | `workflows/01_calibrate_pressure.py` | `outputs/{dep}_step01.nc` (pressure/ZOC) |
| 02 | `workflows/02_calibrate_accmag.py` | Calibrated acc/mag |
| 03 | `workflows/03_tag2animal.py` | Animal-frame rotation |
| 04 | `workflows/04_stroke_detect.py` | Stroke rate signal in `data.pkl` |
| 05 | `workflows/05_heartbeat_detect.py` | Heart rate signal in `data.pkl` |
| 06 | `workflows/06_export_data.py` | Export NetCDF for DiveDB upload |

## Segmentation pipeline stages (09–14)

Separate Snakemake DAG driven by `segmentation_runs.yaml`. Marker files live outside the repo under `paths.local_private_meta_analysis_data`.

| Stage | Script | Purpose |
|-------|--------|---------|
| 09 | `workflows/09_cross_dataset_qc.py` | Channel congruency QC across deployments |
| 10 | `workflows/10_algorithmic_segmentation.py` | Rule-based dive/state segmentation |
| 11 | `workflows/11_feature_generation.py` | Per-window feature extraction (catch22 + stats) |
| 12 | `workflows/12_unsupervised_segmentation.py` | KMeans/UMAP clustering |
| 13 | `workflows/13_supervised_segmentation.py` | RF/GBM/SVM classification + holdout diagnostics |
| 14 | `workflows/14_summary.py` | Budget summaries, actograms, Parquet outputs |

## Running Snakemake

### Processing (steps 00–06)

```bash
# Run a single step for one deployment
snakemake -s Snakefile \
  /Volumes/WORK-SSD/Datasets/Unpublished/<dataset>/<deployment>/outputs/<deployment>_step01.nc \
  --cores 1
```

### Segmentation (stages 09–14)

Full guide: [Segmentation_Workflow_Guide.md](Segmentation_Workflow_Guide.md)

```bash
# Set convenience vars
MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); \
  print(c['paths']['local_private_meta_analysis_data'])")/segmentation
RUN_NAME=<run_name>

# Full run
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/14_summary.done

# Reset and rerun from a stage
scripts/reset_segmentation_run.sh <run_name> --scope supervised
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/14_summary.done
```

## Key data paths (config.yaml `paths`)

| Key | Default value |
|-----|--------------|
| `local_private_data` | `/Volumes/WORK-SSD/Datasets/Unpublished` |
| `local_private_meta_analysis_data` | `/Volumes/WORK-SSD/Datasets/Unpublished/00_Meta-Analysis` |
| `local_private_media` | `/Volumes/WORK-SSD/Media-Datasets/Unpublished` |
| `local_public_data` | `/Volumes/WORK-SSD/Datasets/Published` |
| `local_repo_path` | `/Users/<user>/…/EcoViz_DiveDB/pyologger` |
| `delta_lake.local` | `/Volumes/WORK-SSD/DeltaLake` |

Segmentation stage markers go under:
`<local_private_meta_analysis_data>/segmentation/<run_name>/`

## Streamlit pages

| File | Purpose |
|------|---------|
| `streamlit/pages/0_dash-test_plot.py` | Quick signal plot test |
| `streamlit/pages/1_interactive_plot.py` | Full interactive signal viewer |
| `streamlit/pages/2_interactive_plot_altair.py` | Altair-based signal viewer |
| `streamlit/pages/2_zero_offset_correction.py` | ZOC review UI |
| `streamlit/pages/6_peak_detect.py` | Peak detection review |
| `streamlit/pages/7_hr_classifier_detect.py` | HR classifier UI |
| `streamlit/pages/8_ecg_library_peaks.py` | ECG library peak review |
| `streamlit/pages/9_state_event_review.py` | State event annotation UI |

## Check suite

```bash
# From pyologger/
pytest tests/
```

## Terminology

- **`organism_id`** — canonical identifier in pipeline code and segmentation output Parquets.
- **`animal_id`** — legacy field kept in `data.pkl` attributes and NetCDF files for DiveDB upload compatibility. Use `organism_id` in all new code; translate at upload boundaries only.

## Do not

- Commit `data.pkl` files — they contain private biologging data.
- Commit raw tag data files to the repo.
- Modify `config.yaml` `paths` entries without confirming it will not break other team members' environments.
- Commit `segmentation_runs.yaml` changes that reference paths or deployment IDs not yet available on shared storage.
- Run production Nautilus deploy from an automated agent session.
- Import pyologger from `venv/` paths — use the repo's `pip install -e .` install.
- Touch `venv/` directory contents.
- Use `animal_id` terminology in new pipeline code — use `organism_id`.
