# Changelog

All notable changes to pyologger are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Per-signal-type EDF decimation.** EDF signals can now be anti-alias filtered and
  decimated at import, configured per `parent_signal` rather than per source rate. Defaults
  live in `BaseImporter.DEFAULT_EDF_TARGET_FREQUENCIES` (`eeg`/`eog`/`emg` → 100 Hz,
  `ecg` → 250 Hz) and are overridable dataset-wide via `edf_import.target_frequencies` in
  `config.yaml` or per deployment via `settings.edf_target_frequencies` in
  `parameter_log.json`. Set a signal type to `null` to keep its native rate.
  Decimation uses `scipy.signal.decimate(ftype="fir", zero_phase=True)`; only integer
  factors are applied, and a target that would produce a non-integer rate is skipped with a
  warning (downstream code stores sampling frequency as an integer).
- **Bulk EDF reads.** `BaseImporter._prefetch_edf_signals` fills `_digital` for all signals
  sharing a contiguous column span in edfio's data-record buffer using one chunked pass,
  replacing one strided pass per signal.
- **Dual EDF + tabular import.** `EvolocusImporter` and `ManittyImporter` now ingest an EDF
  and a CSV/Parquet export from the same logger when both are present, instead of returning
  empty when no EDF is found. Tabular handling delegates to `CSVImporter`.
- **`_2` suffix for previously-derived signals.** When a logger's montage id contains
  `derived`, signals that pyologger recomputes later in the pipeline
  (`stroke_rate`, `heart_rate`, `depth`, `prh`, `velocity`, `position`, `location`) are
  stored as `<signal>_2`, reserving the canonical name for pyologger's own version. See
  `BaseImporter.RECOMPUTED_SIGNALS` and `resolve_signal_name`.
- `DataReader` accepts a `config` argument and builds a `param_manager`, so importers can
  read pipeline settings and per-deployment overrides.
- Progress output during EDF import: bulk-read size before the read begins, and a
  `[n/total] <channel>` line per signal during conversion.

### Changed

- **EDF frequency groups are keyed by resulting rate, not source rate.** A single source
  group may now split into several output frames — e.g. a 500 Hz group carrying both ECG and
  EEG becomes a 250 Hz frame and a 100 Hz frame. Channels that end up at the same rate stay
  together in one matrix.
- **EDF signal matrices are built as `float32`.** EDF physical values derive from `int16`
  digital samples, so `float32` is lossless relative to the source while halving the
  footprint of the stored frame.
- Frequency grouping is now shared in `BaseImporter.group_edf_signals_by_frequency` instead
  of duplicated as a private function in each EDF importer. `resample_mean_dataframe` moved
  to `BaseImporter` for the same reason.

### Fixed

- `EdfSignal.data` re-materializes a full `float64` array on every access. It is now read
  exactly once per signal; group length comes from `len(signal.digital)` (`int16`) and the
  preview path materializes only the samples it needs. Previously each signal was fully
  expanded three times.
- Frequency groups are no longer assembled via `pd.DataFrame({label: signal.data ...})`,
  which held every channel's `float64` array simultaneously and then copied them again into a
  consolidated block.
- `EvolocusImporter` recorded `original_name` in channel metadata after overwriting
  `signal.label`, storing the standardized channel id instead of the original label.
- `resample_mean_dataframe` no longer takes a full `df.copy()` before resampling.

### Performance

Measured on `2021-04-17_mian-011` (12.3 GB EDF, 191 h recording, 20 retained channels):

| | Before | After |
|---|---|---|
| Peak resident memory | >40 GB | ~10 GB |
| Frequency-group frames | 9.59 GB | 3.58 GB |

The memory reduction comes from single-access reads, `float32` storage, and preallocation;
decimation accounts for the further drop in stored frame size. Chunked bulk reads cut the
raw gather time roughly 1.6x (57.5 s → 35.0 s per contiguous run on a 6.8 GB EDF).
