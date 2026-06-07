Workflows And Dash App
======================

Snakemake Deployment Pipeline (Step00-Step06)
---------------------------------------------

Run commands from ``pyologger/``.

Run one workflow target for one deployment:

.. code-block:: bash

   snakemake -s Snakefile /path/to/<dataset>/<deployment>/outputs/<deployment>_step01.nc --cores 1

Run the full workflow for one deployment:

.. code-block:: bash

   snakemake -s Snakefile /path/to/<dataset>/<deployment>/outputs/<deployment>_final.nc --cores 4

Use ``-n -p`` to preview and print commands.

Segmentation / Clustering / Supervised Workflow (Stages 09–14)
--------------------------------------------------------------

The segmentation workflow is a separate Snakemake DAG driven by ``segmentation_runs.yaml``.
Marker files are written outside the repo under ``paths.local_private_meta_analysis_data/segmentation/<run_name>/``.

Stage overview
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 35 30 27

   * - Stage
     - Script
     - Inputs
     - Key Outputs
   * - 09 QC
     - ``09_cross_dataset_qc.py``
     - ``data.pkl`` files in scope
     - ``qc/qc_channels.csv``
   * - 10 Algorithmic
     - ``10_algorithmic_segmentation.py``
     - ``data.pkl``, config
     - ``segments/algorithmic_segments.parquet``
   * - 11 Features
     - ``11_feature_generation.py``
     - ``data.pkl``, algorithmic segments
     - ``features/features_raw.parquet``, ``features_filtered.parquet``, ``feature_index.parquet``, ``feature_correlation_matrix.parquet``, ``dropped_correlated_features.csv``
   * - 12 Unsupervised
     - ``12_unsupervised_segmentation.py``
     - ``features_filtered.parquet``
     - ``clustering/clustered_windows.parquet``
   * - 13 Supervised
     - ``13_supervised_segmentation.py``
     - features, event labels
     - ``supervised/supervised_predictions.parquet``, holdout diagnostics, feature importances
   * - 14 Summary
     - ``14_summary.py``
     - all prior outputs
     - ``summary/method_budget_daily.parquet``, ``method_budget_hourly.parquet``

Running the segmentation pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Set convenience vars (run from pyologger/)
   MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); \
     print(c['paths']['local_private_meta_analysis_data'])")/segmentation
   RUN_NAME=<run_name>

   # Full run
   snakemake -s Snakefile --configfile config.yaml --cores 4 \
     ${MARKERS_BASE}/${RUN_NAME}/14_summary.done

   # Partial run (up to features stage)
   snakemake -s Snakefile --configfile config.yaml --cores 4 \
     ${MARKERS_BASE}/${RUN_NAME}/11_features.done

   # Reset and rerun from a stage
   scripts/reset_segmentation_run.sh <run_name> --scope supervised
   snakemake -s Snakefile --configfile config.yaml --cores 4 \
     ${MARKERS_BASE}/${RUN_NAME}/14_summary.done

Key outputs
~~~~~~~~~~~

- ``features/features_filtered.parquet``
- ``clustering/clustered_windows.parquet``
- ``supervised/supervised_predictions.parquet``
- ``summary/method_budget_daily.parquet``
- ``summary/method_budget_hourly.parquet``

Cross-dataset QC (stage 09) can also be run directly:

.. code-block:: bash

   python3 scripts/run_cross_dataset_qc.py --config config.yaml --help

Review entrypoint:

- ``notebooks/SEGMENTATION_REVIEW.ipynb``

See :doc:`segmentation_workflow_review` for a concise stage reference.

Dash App
--------

Current interactive app entrypoint:

- ``pyologger/dash/minimal_interactive/app.py``

Run from repository root:

.. code-block:: bash

   python pyologger/dash/minimal_interactive/app.py --dataset <dataset_id> --deployment <deployment_id> --port 8061

Open:

- ``http://127.0.0.1:8061``
- ``http://127.0.0.1:8061/segmentation``
