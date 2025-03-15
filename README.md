# Pyologger

[![PyPI version](https://badge.fury.io/py/pyologger.svg)](https://pypi.org/project/pyologger/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pyologger/badge/?version=latest)](https://pyologger.readthedocs.io/en/latest/)

Pyologger is a Python library designed for analyzing multi-logger, multi-sensor biologging data. It provides tools for data loading, processing, visualization, and feature generation, making it easier to analyze data from various sensors, including accelerometers, gyroscopes, and depth sensors.

---

## Features

- **Data reading**: Efficiently read and organize biologging data from multiple formats- see datareader.
- **Data processing**: Calibrate sensor data, perform zero-offset corrections, and generate features.
- **Visualization**: Interactive plotting and exploration of sensor and derived data.
- **Pipeline support**: Compatibility with a custom database, [DiveDB](https://github.com/ecophysviz-lab/DiveDB), to store and compare ecophysiological data across species.

---

## Installation

Eventually:
```bash
pip install pyologger
```

Currently:

1. Create virtual environment from which to run your code:

```bash
python -m venv venv
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
pip install .
``` 

4. As you develop, please remember to add any new packages used into the `pyproject.toml` file and add documentation; see [Instructions for Contributing](CONTRIBUTING.md).
---

## Folder Structure

```plaintext
pyologger/
├── pyologger/
│   ├── calibrate_data/         # Tools for sensor calibration (e.g., accelerometer, magnetometer).
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
- Calibrating sensors.
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
from pyologger.utils.config_manager import ConfigManager
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
montage_path = os.path.join(root_dir, 'channel_mapping.json')
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

# Plot sensor data
fig = plot_tag_data_interactive5(
    data_pkl=data_pkl,
    sensors=['ecg', 'accelerometer', 'magnetometer'],
    derived_data_signals=['depth', 'corrected_acc', 'corrected_mag', 'prh'],
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

