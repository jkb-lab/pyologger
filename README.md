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

## Data formats

After importing diverse logger data types, pyologger leverages two primary data structures to store all of the data for a deployment in a single file: 

1. ### `data_pkl` - In-memory Pickle data format 
    Flexible in-memory data format for saving intermediate signals. Import functions will create this structure for you, but this information should help you understand how the data is stored and can be retrieved: 
    
    - ### Easy-access filepaths: 
        - **Deployment Folder**: `data_pkl.deployment_folder`: Filepath to the folder that stores subfolders of data associated with the deployment. Root directories can be adjusted in [config.yaml](config.yaml).
        - **Data Folder**: `data_pkl.data_folder`: Filepath to the folder with raw data from the deployment (input data for pyologger).
        - **Output Folder**: `data_pkl.output_folder`: Filepath to the folder with output data from the deployment (output data for pyologger); this folder can be empty or not exist before processing.

    - ### Essential metadata:
        - **Deployment ID**: `data_pkl.deployment_id`: Deployment ID is the combination of the date of logger attachment to animal and that animal's animal ID. For multiple loggers, deployment begins with the attachment of the first logger and ends with the removal of the last logger.

            🐳 Example deployment ID: 2023-06-13_oror-002

        - **Deployment Info**: `data_pkl.deployment_info`: Dictionary with deployment metadata fields

            <details>
            <summary>Example deployment info</summary>
            
            ```{json}
            {'Deployment Date': '2023-06-13',
                'Deployment Latitude': 32.764655,
                'Deployment Longitude': -117.228585,
                'Time Zone': 'America/Los_Angeles'}
            ```

            </details>

            - **Deployment Date**: Date of first logger attachment (local time); type=string; format `YYYY-MM-DD`; e.g. `2023-06-13`
            - **Deployment Latitude**: Deployment latitude in decimal degrees; type=float; e.g. `32.764000`
            - **Deployment Longitude**: Deployment longitude in decimal degrees; type=float; e.g. `-117.228000`
            - **Deployment Time Zone**: Time zone of deployment start in pandas pytz format (use `pytz.all_timezones` to find relevant time zone or use python package `timezonefinder`).

        - **Logger Info**: `data_pkl.logger_info`: *Nested* dictionary with logger metadata fields. Logger metadata for each logger is *nested* under the logger ID.

            <details>
            <summary>Example logger info</summary>
            
            ```{json}
            {'CC-96': {'ID': 'CC-96',
                'Manufacturer': 'CATS',
                'Montage ID': 'cats-video-montage_V1',
                'datetime_created_from': 'date and time',
                'fs': 400},
            'UF-01': {'ID': 'UF-01',
                'Manufacturer': 'UFI',
                'Montage ID': 'hr-montage_V1',
                'datetime_created_from': 'datetime',
                'fs': 100}}
            ```
            </details>

            - **Logger ID**: Items in dictionary are nested within unique logger ID combining descriptive letter string and numerical code, e.g. `CC-96` for Cats Camera with serial number ending in 96. More detailed logger metadata (like serial number) should be stored separately.
            Access by using: `data_pkl.logger_info[logger_id]`
                - **Manufacturer**: Logger manufacturer name to use relevant methods to read in raw data. Ensure format matches known manufacturers to match known methods. Access key by using: `data_pkl.logger_info[logger_id]['Manufacturer']` Example values: `CATS`, `UFI`, `Wildlife Computers`, `Evolocus`, etc.
                - **Montage ID**: Unique montage ID used for deployment. A montage is a dictionary that maps each original channel name/ID to a standardized channel name/ID. Montages are stored in a `montage_log.json` in the repository root directory and managed using the [`MontageManager`](pyologger/utils/montage_manager.py) class. Access key by using: `data_pkl.logger_info[logger_id]['Montage ID']` Example values: `hr-montage_V1`
                - **fs**: Sampling frequencies found in the data for the logger. Can be a single sampling frequency or multiple.
                - **datetime_created_from**: A record of how the date time field was created (to help debug in cases where there are time issues).
          
        - **Animal Info**: `data_pkl.animal_info`: Dictionary with animal metadata including required Animal ID as well as other optional animal metadata such as age at deployment, flipper tag IDs, lab-specific ID, project-specific ID, domain-specific ID, etc.
            - **Animal ID**: We use the animal ID structure of DiveDB that has the genus-species two-letter codes followed by a unique numerical ID. E.g. `oror-002` for the second *Orcinus orca* entered into DiveDB.

        - **Dataset Info**: `data_pkl.dataset_info`: We use the term dataset to represent a unified data collection effort. This could take place over multiple years or be associated with multiple funding sources, but is typically unified in the types of loggers that are deployed, the types of metrics measured, and the species studied. However, this is flexible as long as a study can be mapped one or more deployments that are part of the data collection effort. A deployment should not normally be associated with multiple datasets. At minimum, a dataset must be defined by an ID, but once made public should ideally contain information on funding, citation, collaborators, PIs, etc.
            - **Dataset ID**: Dataset ID - flexible but unique string identifier that contains some key words related to the unique purpose of the data collection effort, such as the metric studied (e.g. `hr`, `sleep`) as well as the study system (common names are OK here; e.g. `ep` for emperor penguin, `nese` for northern elephant seal), and followed by the key data curators/PIs' initials (e.g. `JKB`). Example for sleep study on northern elephant seals: `mian-juv-nese_sleep_lml-ano_JKB`
            
            ***Note***: There are other formalized fields where each data contributor can be more formally recognized, this is a good place for the initials of a "corresponding PI". 

    - ### Key data types:
        
        - #### Signal data:
            - **Sensor Data**: `data_pkl.sensor_data[sensor_name]`
            
            Contains a pandas dataframe that is accessed using the sensor_name (e.g. 'ecg', 'accelerometer', 'pressure', etc.). Dataframes have a first column with pandas datetime values localized with a pytz timezone (e.g. "America/Los Angeles") and contain one or more columns that have standardized names for the channels associated with that sensor (e.g. 'ecg' for ecg sensor, 'ax', ay', 'az' for accelerometer sensor, etc.). 
                - **Sensor Metadata**: `data_pkl.sensor_info[sensor_name]`

                Holds JSON dictionary with metadata about the sensor and each of its channels. This receives metadata from the montage mapping process on the original channel name and units for each channel. It also holds information on the min and max value of the sensor, the sampling frequency, the logger ID it's associated with, the logger manufacturer, the data type, and processing metadata.

                *Note*: Should at some point clean this metadata up and add things like precision, or other useful info about the sensor. Perhaps hard to standardize across sensor types.

            - **Derived Data**: `data_pkl.derived_data[signal_name]`
            
            Contains a pandas dataframe that is accessed using the derived signal_name (e.g. 'heart_rate', 'prh', 'depth', etc.). When deciding which channels to assign to separate sensors, consider which you would want to appear on a single subplot. For example, x y and z values for accelerometry are often more clearly visualized when superimposed; this could also hold for EEG signals, or versions of a smoothed signal that you want to easily compare. By using the [`data_manager.py`](pyologger/utils/data_manager.py) module, you can easily remove derived signals that are no longer necessary using the `clear_intermediate_signals()` function.
            
            Dataframes are structured exactly like the sensor dataframes with a first column of pandas datetime values localized with a pytz timezone (e.g. "America/Los Angeles") and one or more columns that have standardized names for the channels associated with that derived signal type (e.g. 'pitch', 'roll', and 'heading' for derived signal 'prh', 'ax', ay', 'az' for calibrated/corrected accelerometer data, etc.). 
                - **Derived Signal Metadata**: `data_pkl.derived_info[signal_name]`

                Holds JSON dictionary with metadata about the derived signal and each of its channels. Critically, this stores information about what sensors informed the derived signal (e.g. 'heart_rate' would point to 'ecg' in the 'derived_from_sensors' field), as well as the transformations which have occurred.

        - #### Event data:`data_pkl.event_data`

        A pandas dataframe that stores **manually or automatically detected events** during the deployment. Events can be used to mark discrete points in time (e.g. detected heartbeats, user-added notes) or to define state intervals (e.g. sleep periods, active bouts). This table allows both fine-grained and high-level annotation across the time series and supports downstream processing and visualization.

        Each row in `event_data` represents a single event and includes:

        - **type**: Event type — either:
            - `"point"`: An instantaneous event with no duration (e.g. `heartbeat_manual_ok`)
            - `"state"`: A continuous event with a start and duration (e.g. `sleep_state`)
        - **key**: A standardized string identifier for the event (e.g. `heartbeat_manual_ok`, `sleep_state_auto`)
        - **value**: A numeric value associated with the event (optional; e.g. heart rate at time of detection)
        - **short_description**: A brief text description of the event (e.g. `heartbeat detection`)
        - **long_description**: A longer, optional field for detailed annotation notes
        - **datetime**: Localized datetime (`pytz` timezone-aware) when the event occurs or begins
        - **datetime_utc**: UTC timestamp corresponding to the same event time
        - **time_unix_ms**: Unix timestamp in milliseconds
        - **duration**: Duration of the event in seconds (0 for `"point"` events, nonzero for `"state"` events)

        This structure is designed to be **searchable, sortable, and filterable**, supporting integration with visualization tools (e.g. annotations in timeseries plots) and analysis pipelines (e.g. labeling windows for supervised learning).

        *Tip*: Events can be appended programmatically during processing steps (e.g. automated beat detection) or manually annotated using interactive interfaces. Custom symbols and colors can be defined for visual overlays in plots using a note-style dictionary.

        <details>
        <summary>Example `event_data` rows</summary>

        ```plaintext
            datetime                       type    key                 value   short_description         duration
        0    2024-06-13 11:22:00.750000-07:00  point   heartbeat_manual_ok  65.93   heartbeat detection        0
        1    2024-06-13 11:22:02.720000-07:00  point   heartbeat_manual_ok  30.46   heartbeat detection        0
        2    2024-06-13 11:22:04.240000-07:00  point   heartbeat_manual_ok  39.47   heartbeat detection        0
        ```

        ```plaintext
        # example of a "state" event row:
        datetime:       2024-06-13 23:00:00.000000-07:00  
        type:           state  
        key:            sleep_state_auto  
        value:          1  
        short_description:  auto sleep scoring  
        duration:       5400  # seconds = 1.5 hours
        ```
        </details>


2. `output.nc`: netCDF tag data format for saving raw or processed data at different steps of the pipeline. This netCDF format allows upload and intake into [DiveDB](https://github.com/ecophysviz-lab/DiveDB).


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

