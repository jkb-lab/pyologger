Workflows And Dash App
======================

Snakemake
---------

Run commands from the ``pyologger/`` directory.

Run one workflow target for one deployment:

.. code-block:: bash

   snakemake -s Snakefile /path/to/<dataset>/<deployment>/outputs/<deployment>_step01.nc --cores 1

This builds only the rule needed for that output file.

Run the full workflow for a deployment output:

.. code-block:: bash

   snakemake -s Snakefile /path/to/<dataset>/<deployment>/outputs/<deployment>_final.nc --cores 4

Tip: Use ``-n`` for a dry run and ``-p`` to print shell commands.

.. code-block:: bash

   snakemake -s Snakefile /path/to/<dataset>/<deployment>/outputs/<deployment>_step01.nc -n -p --cores 1

Optional Step 01 overwrite mode:

Set this in ``config.yaml``:

.. code-block:: yaml

   overwrite_step01_from_nc: true

When enabled, Step 01 reads pressure from ``outputs/<deployment>_00_processed.nc`` before calibration.

Dash App
--------

The current interactive app entrypoint is:

``pyologger/dash/minimal_interactive/app.py``

Run it from the repository root:

.. code-block:: bash

   python pyologger/dash/minimal_interactive/app.py --dataset <dataset_id> --deployment <deployment_id> --port 8061

Then open:

``http://127.0.0.1:8061``

Notes:

- The app reads paths from ``CONFIG_PATH`` (environment variable) via ``pyologger/utils/folder_manager.py``.
- ``--dataset`` and ``--deployment`` are required by this entrypoint.
