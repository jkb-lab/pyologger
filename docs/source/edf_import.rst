EDF Import and Decimation
=========================

Purpose
-------
This document describes how pyologger reads EDF files, how signals are grouped into
per-frequency matrices, and how per-signal-type decimation is configured.

It applies to the EDF-capable importers (``EvolocusImporter``, ``ManittyImporter``), which
share their implementation through ``BaseImporter``.

Why This Exists
---------------
High-rate electrophysiology EDFs are large. A 191-hour juvenile elephant seal sleep
deployment stores 16 channels at 500 Hz, which is roughly 25 GB of ``float64`` for the
500 Hz group alone.

Two properties of the EDF format and its Python reader drive the cost:

- ``edfio.EdfSignal.data`` is a property that re-computes a full ``float64`` array on every
  access. It is not cached.
- EDF stores samples interleaved by data record, so reading one signal means a strided pass
  over the whole file. Reading N signals separately means N passes.

The import path is built around those two facts.

Grouping Model
--------------
Signals are grouped into one DataFrame per **resulting** sampling rate, not per source rate.

A single source group can therefore split into several output frames. A 500 Hz group holding
both ECG and EEG becomes a 250 Hz frame (ECG) and a 100 Hz frame (EEG/EOG/EMG) under the
default targets. Channels that end up at the same rate stay together in one matrix, so
related signals remain aligned and share a datetime column.

Each group is accumulated into a preallocated ``float32`` matrix, one signal at a time. EDF
physical values derive from ``int16`` digital samples, so ``float32`` is lossless relative to
the source while halving the stored footprint.

Decimation
----------
Decimation is configured per ``parent_signal``, not per source rate.

Defaults are declared in ``BaseImporter.DEFAULT_EDF_TARGET_FREQUENCIES``:

.. code-block:: python

   DEFAULT_EDF_TARGET_FREQUENCIES = {
       "eeg": 100,
       "eog": 100,
       "emg": 100,
       "ecg": 250,
   }

A signal type absent from the mapping keeps its native rate. Motion channels
(accelerometer, gyroscope, magnetometer, pressure) are not decimated by default.

Resolution order, later winning:

1. ``BaseImporter.DEFAULT_EDF_TARGET_FREQUENCIES``
2. ``edf_import.target_frequencies`` in ``config.yaml`` (dataset-wide)
3. ``settings.edf_target_frequencies`` in ``parameter_log.json`` (per deployment, itself
   merging dataset defaults under the deployment entry)

Set a signal type to ``null`` to disable decimation for it.

Config Shape
------------
Dataset-wide, in ``config.yaml``:

.. code-block:: yaml

   edf_import:
     target_frequencies:
       eeg: 100
       ecg: 250
       emg: null   # keep native rate

Per deployment, in the dataset's ``parameter_log.json`` under the deployment's
``settings`` section:

.. code-block:: json

   {
     "deployment_id": "2021-04-17_mian-011",
     "settings": {
       "edf_target_frequencies": { "eeg": 200 }
     }
   }

Anti-Aliasing
-------------
Decimation uses ``scipy.signal.decimate(..., ftype="fir", zero_phase=True)``, which
low-pass filters before discarding samples.

Plain slicing (``data[::step]``) is **not** used for decimation. Slicing folds any content
above the new Nyquist frequency back into the retained band, which on EEG produces
plausible-looking but incorrect spectra. Slicing appears only as a fallback if the
anti-alias filter fails on a very short signal, and it logs a warning when it does.

Only integer decimation factors are applied. A target that would produce a non-integer
resulting rate is skipped with a warning, because downstream code stores sampling frequency
as an integer (``int(logger_info['fs'])``).

Verifying a decimation
~~~~~~~~~~~~~~~~~~~~~~
Sample counts divided by their rates should give the same recording duration across groups.
For ``2021-04-17_mian-011``:

.. code-block:: text

   ecg     172,183,480 samples / 250 Hz = 191.31 h
   eeg      68,873,392 samples / 100 Hz = 191.31 h

Matching durations confirm the factors were applied correctly.

Bulk Reads
----------
``BaseImporter._prefetch_edf_signals`` populates ``_digital`` for lazily-loaded signals
before grouping begins.

Signals recorded at the same rate occupy adjacent column spans in edfio's shared
data-record buffer. The prefetch splits the pending signals into maximal contiguous runs and
copies each run in one pass, in row chunks. Row chunking rather than a single strided
``ascontiguousarray`` gives better page and cache locality; measured at roughly 1.6x faster
on a 6.8 GB EDF.

If the buffer layout is not as expected the prefetch is skipped and edfio's per-signal path
is used, so behaviour is unchanged where the optimization does not apply.

Previously-Derived Signals
--------------------------
A deployment may include a logger carrying products derived in an earlier analysis — sleep
scoring, heart rate, corrected depth — alongside the raw sensor logger.

When a logger's montage id contains ``derived``, signals that pyologger recomputes later in
the pipeline are stored under a ``_2`` suffix:

.. code-block:: text

   stroke_rate  ->  stroke_rate_2
   heart_rate   ->  heart_rate_2
   depth        ->  depth_2
   prh          ->  prh_2
   velocity     ->  velocity_2
   position     ->  position_2
   location     ->  location_2

The canonical names stay free for pyologger's own, more precise versions written by
``workflows/04_stroke_detect.py`` and ``workflows/05_heartbeat_detect.py``.

Signal types that pyologger does not recompute keep their names — notably ``label``
(sleep-scoring codes), ``eeg_analysis``, and ``heart_rate_analysis``. Label channels are
converted into events by
``DataReader._append_label_state_events_from_dataframe``, so sleep-scoring states from a
derived logger still appear in ``event_data``.

The signal set is declared in ``BaseImporter.RECOMPUTED_SIGNALS`` and resolved by
``BaseImporter.resolve_signal_name``.

Mixed EDF and Tabular Sources
-----------------------------
A single logger may ship an EDF, a CSV/Parquet export, or both. When both are present the
EDF is imported first and the tabular file is imported afterward into the same
``DataReader``; tabular handling delegates to ``CSVImporter``.

Memory Profile
--------------
Measured on ``2021-04-17_mian-011`` (12.3 GB EDF, 20 retained channels):

.. list-table::
   :header-rows: 1

   * -
     - Before
     - After
   * - Peak resident memory
     - >40 GB
     - ~10 GB
   * - Frequency-group frames
     - 9.59 GB
     - 3.58 GB

Peak usage during grouping is approximately one output matrix plus one signal's ``float64``
temporary, rather than every channel's ``float64`` array at once.
