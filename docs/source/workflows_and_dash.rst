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

Segmentation / Clustering / Supervised Workflow
------------------------------------------------

The segmentation workflow is configured by ``segmentation_runs.yaml`` and executed via:

- ``pyologger/workflows/Snakefile``

Each run writes markers to:

- ``segmentation_markers/<run_name>/``

Current marker stages:

- ``09_qc.done``
- ``10_algorithmic.done``
- ``11_features.done``
- ``12_unsupervised.done``
- ``13_supervised.done``
- ``14_summary.done``

Run one full segmentation pipeline:

.. code-block:: bash

   RUN_NAME=mian_mile_sleep_transfer_rf
   snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
     segmentation_markers/${RUN_NAME}/14_summary.done

Run a partial stage:

.. code-block:: bash

   snakemake -s workflows/Snakefile --configfile config.yaml --cores 4 \
     segmentation_markers/${RUN_NAME}/11_features.done

Output roots are resolved by the run context and typically land at:

- ``00_Meta-Analysis/segmentation/<analysis_id>/`` (multi-dataset)
- ``<dataset_id>/00_Meta-Analysis/segmentation/<analysis_id>/`` (single-dataset)

Key run outputs:

- ``features/features_filtered.parquet``
- ``clustering/clustered_windows.parquet``
- ``supervised/supervised_predictions.parquet``
- ``summary/method_budget_daily.parquet``
- ``summary/method_budget_hourly.parquet``

Cross-dataset channel/unit QC is stage ``09_qc`` and can also be run directly:

.. code-block:: bash

   python3 scripts/run_cross_dataset_qc.py --config config.yaml --help

Review entrypoint:

- ``notebooks/SEGMENTATION_REVIEW.ipynb``

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
