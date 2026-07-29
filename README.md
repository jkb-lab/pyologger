# Pyologger

[![PyPI version](https://badge.fury.io/py/pyologger.svg)](https://pypi.org/project/pyologger/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jmkendallbar.github.io/pyologger/)

Pyologger is a Python library designed for analyzing multi-logger, multi-signal biologging data. It provides tools for data loading, processing, visualization, and feature generation, making it easier to analyze data from various signals, including accelerometers, gyroscopes, and depth signals.

The minimal Dash app also includes an experimental water-tank viewer adapted from Evan Wallace's MIT-licensed [WebGL Water](https://github.com/evanw/webgl-water) project. Credit for the original water simulation/tank demo belongs to Evan Wallace.

For segmentation reruns, use:

```bash
scripts/reset_segmentation_run.sh <run_name>
```

Then rerun a single segmentation run by marker target (from `pyologger/`):

```bash
RUN_NAME=<run_name>
MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['paths']['local_private_meta_analysis_data'])")/segmentation
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/14_summary.done
```


## Segmentation Workflow

See [Segmentation_Workflow.md](Segmentation_Workflow.md) for the current segmentation pipeline: run commands, marker path configuration, reset instructions, output structure, QA checklist, and cross-dataset QC. Use [notebooks/SEGMENTATION_REVIEW.ipynb](notebooks/SEGMENTATION_REVIEW.ipynb) for artifact inspection.

## Features

- **Data reading**: Efficiently read and organize biologging data from multiple formats- see datareader.
- **Data processing**: Calibrate signal data, perform zero-offset corrections, and generate features.
- **Visualization**: Interactive plotting and exploration of signal and derived data.
- **Pipeline support**: Compatibility with [DiveDB](https://github.com/ecophysviz-lab/DiveDB) to store and compare ecophysiological data across species.

### Data model: `data_pkl`

Each loaded deployment is a `data_pkl` object with three main fields:

#### Signal data

**`data_pkl.signal_data[signal_name]`** — pandas DataFrame with a timezone-aware datetime index and one or more signal channels (e.g. `'ecg'`, `'ax'`/`'ay'`/`'az'` for accelerometer).

**`data_pkl.signal_info[signal_name]`** — JSON metadata per signal including channel names, units, sampling frequency, logger ID, and processing log.

<details>
<summary>Example signal metadata</summary>

```json
{
  "channels": ["gx", "gy", "gz"],
  "metadata": {
    "gx": { "original_name": "Gyroscope X [mrad/s]", "unit": "mrad/s", "parent_signal": "gyroscope" },
    "gy": { "original_name": "Gyroscope Y [mrad/s]", "unit": "mrad/s", "parent_signal": "gyroscope" },
    "gz": { "original_name": "Gyroscope Z [mrad/s]", "unit": "mrad/s", "parent_signal": "gyroscope" }
  },
  "original_sampling_frequency": 100,
  "sampling_frequency": 50,
  "logger_id": "CC-96",
  "logger_manufacturer": "CATS"
}
```

</details>

**`data_pkl.signal_data[derived_signal_name]`** — Derived signals (e.g. `'heart_rate'`, `'prh'`, `'depth'`) share the same DataFrame structure. Use [`data_manager.py`](pyologger/utils/data_manager.py) `clear_intermediate_signals()` to drop signals no longer needed.

**`data_pkl.derived_info[signal_name]`** — Metadata for derived signals including `derived_from_signals` provenance and transformation log.

**`data_pkl.signal_data[signal_name + '_2']`** — Signals imported from a logger whose montage
carries products derived in an earlier analysis (montage id containing `derived`). Channels
that pyologger recomputes later — `stroke_rate`, `heart_rate`, `depth`, `prh`, `velocity`,
`position`, `location` — are stored with a `_2` suffix so the canonical name stays free for
pyologger's own version. Sleep-scoring labels and analysis products keep their names. See
[EDF Import and Decimation](docs/source/edf_import.rst).

### EDF import

EDF signals are grouped into one DataFrame per **resulting** sampling rate and stored as
`float32` (lossless relative to the `int16` EDF source). Signal types are decimated at import
with an anti-aliasing filter — by default `eeg`/`eog`/`emg` to 100 Hz and `ecg` to 250 Hz —
configurable via `edf_import.target_frequencies` in `config.yaml` or
`settings.edf_target_frequencies` in `parameter_log.json`. Motion channels keep their native
rate. Full details in [docs/source/edf_import.rst](docs/source/edf_import.rst).

#### Event data

**`data_pkl.event_data`** — pandas DataFrame of manually or automatically detected events.

| Column | Description |
| --- | --- |
| `type` | `"point"` (instantaneous) or `"state"` (with duration) |
| `key` | Standardized event identifier, e.g. `heartbeat_manual_ok` |
| `value` | Optional numeric value (e.g. heart rate at detection) |
| `short_description` | Brief label |
| `datetime` | Timezone-aware start time |
| `duration` | Duration in seconds (0 for point events) |

<details>
<summary>Example event_data rows</summary>

```plaintext
   datetime                          type   key                  value  short_description        duration
0  2024-06-13 11:22:00.750000-07:00  point  heartbeat_manual_ok  65.93  heartbeat detection  0
1  2024-06-13 11:22:02.720000-07:00  point  heartbeat_manual_ok  30.46  heartbeat detection  0
```

```plaintext
# state event example:
datetime:   2024-06-13 23:00:00-07:00
type:       state
key:        sleep_state_auto
duration:   5400   # 1.5 hours
```

</details>


**`output.nc`** — netCDF tag data format for saving raw or processed data at each pipeline step, suitable for upload to [DiveDB](https://github.com/ecophysviz-lab/DiveDB).


## Installation

Eventually:
```bash
pip install pyologger
```

Currently:

1. Create virtual environment from which to run your code:

```bash
python3 -m venv venv
``` 

This creates a folder `venv`, ignored by git by default, that contains all library-related files.

2. Activate your virtual environment:

On Windows:
```bash
source venv/Scripts/activate
``` 
On a Mac:
```bash
source venv/bin/activate
```

3. Install the package `pyologger` localling using: 

```bash
pip install -e .
``` 

Use the `pyologger/` repo directory as your working directory for Snakemake and direct workflow runs. The workflow scripts are written to resolve the local repo copy of `pyologger`, so they should be run from this checkout rather than relying on a separately installed package.

4. As you develop, please remember to add any new packages used into the `pyproject.toml` file and add documentation; see [Instructions for Contributing](CONTRIBUTING.md).
---

## Snakemake Usage

### Re-run only Step 01 (Calibrate Pressure) for one deployment

From the `pyologger/` directory:

```bash
snakemake -s Snakefile /Volumes/WORK-SSD/Datasets/Unpublished/mian-juv-nese_sleep_lml-ano_JKB/2019-10-25_mian-001/outputs/2019-10-25_mian-001_step01.nc --cores 1
```

This runs only the rule that builds `{deployment}_step01.nc` for that deployment.

### Optional overwrite from `{deployment}_00_processed.nc`

Step 01 supports a workflow-level overwrite mode that reloads pressure from:

`outputs/{deployment}_00_processed.nc`

before applying calibration. To enable it, set this in `config.yaml`:

```yaml
overwrite_step01_from_nc: true
```

When enabled, Snakemake passes `--overwrite` to `workflows/01_calibrate_pressure.py` for Step 01.

### Dataset-Level Segmentation Pipeline

The segmentation DAG is separate from the step00-step06 deployment pipeline.
Current stages are marker-driven:

- `09_qc.done`
- `10_algorithmic.done`
- `11_features.done`
- `12_unsupervised.done`
- `13_supervised.done`
- `14_summary.done`

Configure one or more runs under `segmentation_runs.yaml`, then run:

```bash
RUN_NAME=<run_name>
snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
    segmentation_markers/${RUN_NAME}/14_summary.done
```

To run up to specific stages:

```bash
snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
    segmentation_markers/${RUN_NAME}/10_algorithmic.done
```

```bash
snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
    segmentation_markers/${RUN_NAME}/11_features.done
```

```bash
snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
    segmentation_markers/${RUN_NAME}/12_unsupervised.done
```

```bash
snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
    segmentation_markers/${RUN_NAME}/13_supervised.done
```

The stage scripts live in:

- `workflows/09_cross_dataset_qc.py`
- `workflows/10_algorithmic_segmentation.py`
- `workflows/11_feature_generation.py`
- `workflows/12_unsupervised_segmentation.py`
- `workflows/13_supervised_segmentation.py`
- `workflows/14_summary.py`

The canonical feature outputs are:

- `<run_output_root>/features/features_raw.parquet`
- `<run_output_root>/features/feature_index.parquet`
- `<run_output_root>/features/feature_correlation_matrix.parquet`
- `<run_output_root>/features/dropped_correlated_features.csv`
- `<run_output_root>/features/features_filtered.parquet`

The canonical supervised outputs are written under:

- `<run_output_root>/supervised/`

These include holdout predictions, threshold diagnostics, feature importances,
deployment-wide predictions, and the supervised summary report. When
`export_rf_events` is enabled, the supervised stage also writes merged `rf_*`
state events back into each deployment `data.pkl`.

Use `notebooks/SEGMENTATION_REVIEW.ipynb` as the canonical artifact-review notebook.

For multi-dataset runs, the output root is:

`/Volumes/WORK-SSD/Datasets/Unpublished/00_Meta-Analysis/segmentation/<analysis_id>/`

## Folder Structure

```plaintext
pyologger/
├── pyologger/
│   ├── calibrate_data/         # Tools for signal calibration (e.g., accelerometer, magnetometer).
│   ├── interactive_pyologger/  # Interactive Streamlit apps and utilities.
│   ├── load_data/              # Modules for loading and organizing data.
│   ├── plot_data/              # Visualization tools for biologging data.
│   ├── process_data/           # Data processing methods (e.g., cropping, feature extraction).
│   ├── utils/                  # General utilities (e.g., configuration, data management).
│   └── __init__.py             # Library initialization.
├── docs/                       # Documentation source files and built docs.
├── notebooks/                  # Jupyter notebooks for example workflows and tutorials.
├── dash/                       # Interactive dash app to view 3D rotations alongside video.
├── data/                       # Biologging data for testing and demonstration.
└── pyproject.toml              # Project configuration for Python build tools.
```

---

## Folder Descriptions

View the in-progress documentation here: [Documentation](docs/build/html/index.html)

### `pyologger/`

The main library directory, containing modularized tools:

- **`calibrate_data/`**: Calibrate and align data to the animal reference frame.
- **`interactive_pyologger/`**: Interactive Streamlit applications for data exploration and annotation.
- **`load_data/`**: Functions for loading data into structured formats like pandas or xarray.
- **`plot_data/`**: Interactive and static plotting utilities.
- **`process_data/`**: Methods for preprocessing biologging data, including feature extraction and resampling.
- **`utils/`**: Helper functions for managing configurations, events, and I/O operations.

### `docs/`

Source files for documentation built using Sphinx. The `build/` directory contains compiled HTML documentation.

### `notebooks/`

Jupyter notebooks demonstrating typical workflows:

- Loading data.
- Calibrating signals.
- Feature extraction and event detection.
- Visualizing processed data.

### `sample_netcdf_files/`

Example biologging data files in NetCDF format. Use these for testing and exploring the library's features.

---

## Documentation

Comprehensive documentation is available at [pyologger.readthedocs.io](https://pyologger.readthedocs.io/en/latest/).

---

## Contributing

Contributions are welcome! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on submitting issues and pull requests.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by the growing field of biologging and wildlife telemetry- integrating functions developed by: 
    - **TagTools** by Mark Johnson & Stacey DeRuiter of tagtools by animaltags:
        - Github: https://github.com/animaltags 
        - Website: https://animaltags.org/
    - **CATS Toolbox** by Dave Cade & Will Gough of the Goldbogen Lab:
        - Github: https://github.com/cadede/CATS-Methods-Materials
        - Wiki tutorial: https://github.com/cadede/CATS-Methods-Materials/wiki
- Thank you to the biologging community for their support and feedback.

---

## Quick Start Example

```python
import os
import pickle
from pyologger.utils.param_manager import ParamManager
from pyologger.load_data.datareader import DataReader
from pyologger.load_data.metadata import Metadata
from pyologger.plot_data.plotter import plot_tag_data_interactive5
from pyologger.process_data.sampling import *
from pyologger.calibrate_data.tag2animal import *
from pyologger.calibrate_data.zoc import *

# Setup paths
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, "data")
color_mapping_path = os.path.join(root_dir, "color_mappings.json")

# Load metadata
metadata = Metadata()
metadata.fetch_databases(verbose=False)
metadata.find_relations(verbose=False)

# Fetch deployment data
deployment_db = metadata.get_metadata("deployment_DB")

# Initialize DataReader
montage_path = os.path.join(root_dir, 'montage_log.json')
datareader = DataReader(deployment_folder_path=data_dir)

# Process deployment data
deployment_folder, deployment_id = datareader.check_deployment_folder(deployment_db, data_dir)
if deployment_folder:
    datareader.read_files(
        metadata, save_csv=False, save_parq=False, save_edf=False,
        montage_path=montage_path, save_netcdf=True
    )

# Load processed data
pkl_path = os.path.join(deployment_folder, 'outputs', 'data.pkl')
with open(pkl_path, 'rb') as file:
    data_pkl = pickle.load(file)

# Plot signal data
fig = plot_tag_data_interactive5(
    data_pkl=data_pkl,
    signals=['ecg', 'accelerometer', 'magnetometer','depth', 'corrected_acc', 'corrected_mag', 'prh'],
    channels={},
    time_range=("2023-01-01 00:00:00", "2023-01-01 23:59:59"),  # Example time range
    note_annotations={},
    color_mapping_path=color_mapping_path,
    target_sampling_rate=1,
    zoom_start_time="2023-01-01 12:00:00",
    zoom_end_time="2023-01-01 12:30:00",
    zoom_range_selector_channel='depth',
    plot_event_values=[]
)
fig.show()
```

---

## Support

For issues or questions, please visit the [GitHub repository](https://github.com/yourusername/pyologger) or open a ticket in the [issue tracker](https://github.com/yourusername/pyologger/issues).
