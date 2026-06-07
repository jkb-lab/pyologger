AI Agent Reference (AI_DOCS)
=============================

Token-efficient reference for AI agents and developers working on the pyologger biologging processing and segmentation pipeline.

Quick Reference
---------------

- **Project**: Multi-logger biologging data processing, segmentation, and DiveDB upload pipeline
- **Entry Point**: ``select_and_load_deployment(data_dir, dataset_id, deployment_id)`` (``utils/folder_manager.py``)
- **Tech Stack**: Python, Snakemake, pandas, numpy, scikit-learn, LightGBM, pycatch22, UMAP, xarray, PyArrow, Streamlit, Plotly
- **Key Dependencies**: ``ParamManager``, ``DataReader``, ``SegmentationPipeline``, ``find_segments``
- **Config**: ``config.yaml`` (paths, datasets) + ``segmentation_runs.yaml`` (named segmentation runs); loaded via ``load_combined_config()``

Architecture Overview
---------------------

Data Flow
~~~~~~~~~

.. code-block:: text

   raw tag files (CATS/WC/StarOddi/CSV/…)
       → DataReader (load_data/datareader.py)
       → data.pkl (DataPkl object) + _00_processed.nc
       → workflows/01–06_*.py  (calibrate, stroke, HR, export)
       → _step01.nc … output.nc
       → DiveDB DataUploader (notebooks/00_data_to_diveDB.ipynb)
       → Apache Iceberg lake

Segmentation Flow
~~~~~~~~~~~~~~~~~

.. code-block:: text

   config.yaml + segmentation_runs.yaml
       → RunContext (run_name, scope, features, unsupervised, supervised, summary cfg)
       → Stage 09: cross-dataset QC  → qc/qc_channels.csv
       → Stage 10: algorithmic segs  → segments/algorithmic_segments.parquet
       → Stage 11: feature gen       → features/features_filtered.parquet
       → Stage 12: unsupervised      → clustering/clustered_windows.parquet
       → Stage 13: supervised        → supervised/supervised_predictions.parquet
       → Stage 14: summary           → summary/method_budget_daily.parquet
                                        summary/method_budget_hourly.parquet

File Map
--------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - File
     - Purpose
     - Key Exports
   * - ``utils/folder_manager.py``
     - Deployment selection and config loading
     - ``select_and_load_deployment``, ``resolve_deployment_context``, ``load_configuration``, ``load_combined_config``, ``match_to_metadata``, ``create_dataset_structure``, ``map_deployment_to_dataset``, ``trim_data_pkl_to_accelerometer_window``
   * - ``load_data/datareader.py``
     - Raw tag file reader; produces data.pkl
     - ``DataReader``
   * - ``load_data/metadata.py``
     - Notion metadata fetcher
     - ``Metadata``
   * - ``analyze_data/segmentation_pipeline.py``
     - Full segmentation pipeline orchestrator
     - ``RunContext``, stage runner functions
   * - ``analyze_data/find_segments.py``
     - Segment detection from labeled columns
     - ``find_segments``, ``find_adjusted_segments``
   * - ``analyze_data/segmentation_run_summaries.py``
     - Parquet summary writers
     - ``write_segmentation_run_summary_parquets``
   * - ``utils/param_manager.py``
     - Per-deployment parameter persistence
     - ``ParamManager``
   * - ``utils/time_manager.py``
     - Datetime and sampling utilities
     - ``calculate_sampling_frequency``
   * - ``utils/event_manager.py``
     - Event dataframe helpers
     - ``create_state_event``
   * - ``utils/deployment_source.py``
     - Load data.pkl or NetCDF, resolve signals
     - ``DeploymentSource``
   * - ``utils/data_manager.py``
     - data.pkl manipulation helpers
     - ``clear_intermediate_signals``
   * - ``process_data/peak_detect.py``
     - Stroke and heartbeat peak detection
     - ``detect_strokes``, ``detect_heartbeats``
   * - ``process_data/feature_generation_utils.py``
     - Catch22 + stats feature generation
     - —
   * - ``plot_data/plotter.py``
     - Interactive Plotly signal viewer
     - ``plot_tag_data_interactive5``
   * - ``io_operations/datareader.py``
     - Format-specific importers (base + CATS, WC, etc.)
     - ``BaseImporter``, format subclasses

Data Structures
---------------

DataPkl Object Fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Field
     - Type
     - Contents
   * - ``signal_data``
     - ``dict[str, pd.DataFrame]``
     - Signal name → DataFrame with ``datetime`` + channel columns
   * - ``signal_info``
     - ``dict[str, dict]``
     - Signal name → metadata (channels, units, freq, logger_id, stats, processing_log)
   * - ``event_data``
     - ``pd.DataFrame``
     - Event table: ``datetime``, ``type``, ``key``, ``value``, ``duration``, ``short_description``, ``long_description``
   * - ``derived_data``
     - ``dict[str, pd.DataFrame]``
     - Derived signal name → DataFrame (same structure as signal_data)
   * - ``derived_info``
     - ``dict[str, dict]``
     - Derived signal metadata (mirrors signal_info; includes ``derived_from_signals``)
   * - ``param_manager``
     - ``ParamManager``
     - Per-deployment parameter store (reads/writes JSON sidecar)
   * - ``deployment_info``
     - ``dict``
     - Deployment metadata from Notion (deployment ID, dates, time zone, etc.)
   * - ``animal_info``
     - ``dict``
     - Animal metadata from Notion (Animal_ID, species, etc.)

``signal_data`` DataFrame columns: first column always ``datetime`` (tz-aware pandas Timestamp); remaining columns are standardized channel names (e.g. ``ax``, ``ay``, ``az`` for accelerometer; ``ecg`` for ECG; ``depth`` for pressure).

event_data Schema
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Column
     - Type
     - Notes
   * - ``datetime``
     - tz-aware Timestamp
     - Event start
   * - ``type``
     - str
     - ``"point"`` or ``"state"``
   * - ``key``
     - str
     - Standardized event key (e.g. ``heartbeat_manual_ok``, ``sleep_state_auto``)
   * - ``value``
     - float
     - Optional numeric value
   * - ``duration``
     - float
     - Seconds; 0 for point events
   * - ``short_description``
     - str
     - Brief label
   * - ``long_description``
     - str
     - Optional extended annotation

Segmentation Output Parquet Schemas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**clustering/clustered_windows.parquet**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Notes
   * - ``window_start``
     - tz-aware Timestamp
   * - ``window_end``
     - tz-aware Timestamp
   * - ``deployment_id``
     - str
   * - ``dataset_id``
     - str
   * - ``organism_id``
     - str (canonical; replaces ``animal_id`` in new outputs)
   * - ``cluster``
     - int — KMeans cluster label
   * - ``umap_x``, ``umap_y``
     - float — UMAP embedding coordinates
   * - feature columns
     - float — normalized feature values

**supervised/supervised_predictions.parquet**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Notes
   * - ``window_start``, ``window_end``
     - tz-aware Timestamps
   * - ``deployment_id``, ``dataset_id``, ``organism_id``
     - str
   * - ``predicted_label``
     - str — classifier output
   * - ``confidence``
     - float — calibrated probability of predicted class
   * - ``true_label``
     - str or NaN — ground truth (holdout rows only)

segmentation_runs.yaml Structure
---------------------------------

.. code-block:: yaml

   segmentation_runs:
     <run_name>:
       scope:
         analysis_id: <str>           # output subdirectory name
         dataset_ids: [<str>, ...]    # datasets to include
         deployment_ids: [<str>, ...] # specific deployments (subset of datasets)
         expected_signals: [<str>, ...] # signals required to be present
         groups:                      # optional grouping for cross-group analysis
           <group_label>: [<deployment_id>, ...]

       features:
         normalization_level: deployment | global
         cluster_length_mode: fixed | dive | event
         duration_s: <int>            # window size for fixed mode
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
             required: <bool>         # default true

       unsupervised:
         n_clusters: <int>            # KMeans k
         umap_n_neighbors: <int>
         umap_min_dist: <float>
         umap_sample_frac: <float>

       supervised:
         label_source: <event_key>    # event key to use as ground truth
         classifier: rf | gbm | svm | knn
         export_rf_events: <bool>     # write rf_* state events back to data.pkl

       summary:
         primary_method_kind: unsupervised | supervised
         primary_method_id: <str>
         budget_label_mode: canonical | raw
         context_filter_pass_mode: none | measured_only | full
         actograms:
           daily: <bool>
           continuous: <bool>

Common Patterns
---------------

Loading a Deployment
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyologger.utils.folder_manager import load_configuration, select_and_load_deployment

   config, data_dir, color_mapping_path, montage_path = load_configuration()

   animal_id, dataset_id, deployment_id, dataset_folder, deployment_folder, data_pkl, param_manager = \
       select_and_load_deployment(
           data_dir,
           dataset_id="oror-adult-orca_hr-sr-vid_sw_JKB-PP",
           deployment_id="2023-06-13_oror-002",
       )

Accessing Signals from data_pkl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get a signal DataFrame
   depth_df = data_pkl.signal_data["depth"]
   # depth_df.columns: ["datetime", "depth"]

   acc_df = data_pkl.signal_data["accelerometer"]
   # acc_df.columns: ["datetime", "ax", "ay", "az"]

   # Signal metadata
   freq = data_pkl.signal_info["depth"]["sampling_frequency"]  # Hz

   # Event data
   sleep_events = data_pkl.event_data[data_pkl.event_data["key"] == "sleep_state_auto"]

Running Segmentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # From pyologger/
   MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); \
     print(c['paths']['local_private_meta_analysis_data'])")/segmentation

   RUN_NAME=finescale_hr
   snakemake -s Snakefile --configfile config.yaml --cores 4 \
     ${MARKERS_BASE}/${RUN_NAME}/14_summary.done

Finding Segments Programmatically
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyologger.analyze_data.find_segments import find_segments

   dive_segments = find_segments(
       data=depth_df,
       column="depth",
       criteria=lambda v: v > 10.0,  # depth > 10 m
       min_duration=30.0,             # at least 30 seconds
   )
   # Returns DataFrame: window_start, window_end, duration, ...

Loading Config Directly
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyologger.utils.folder_manager import load_combined_config

   config, config_path, runs_path = load_combined_config()
   data_root = config["paths"]["local_private_data"]
   seg_runs = config.get("segmentation_runs", {})

Terminology
-----------

- **organism_id** — canonical identifier in pipeline code and segmentation output Parquets.
- **animal_id** — legacy field kept in ``data.pkl`` attributes and NetCDF files for DiveDB upload compatibility. Use ``organism_id`` in all new code; translate at upload boundaries only.
