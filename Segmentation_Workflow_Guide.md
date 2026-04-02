# Segmentation Workflow

Current segmentation pipeline using `Snakefile` and `segmentation_runs.yaml`.

## Marker Path

Stage markers are written outside the repo to avoid inflating the Docker build context.
The root is derived from `paths.local_private_meta_analysis_data` in `config.yaml`:

```text
<local_private_meta_analysis_data>/segmentation/<run_name>/
```

Set a shell variable for convenience:

```bash
MARKERS_BASE=$(python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['paths']['local_private_meta_analysis_data'])")/segmentation
```

## Stages

- `09_qc.done`
- `10_algorithmic.done`
- `11_features.done`
- `12_unsupervised.done`
- `13_supervised.done`
- `14_summary.done`

## Run Commands

All commands from `pyologger/`. Set `RUN_NAME` and `MARKERS_BASE` first (see above).

**Full run:**

```bash
RUN_NAME=mian_mile_sleep_transfer_rf
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/14_summary.done
```

**Up to a specific stage:**

```bash
# QC + algorithmic only
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/10_algorithmic.done

# Features only
snakemake -s Snakefile --configfile config.yaml --cores 4 \
  ${MARKERS_BASE}/${RUN_NAME}/11_features.done
```

## Reset

Reset a run's markers to re-trigger stages from Snakemake:

```bash
# Reset all stages for a run
scripts/reset_segmentation_run.sh <run_name>

# Reset from a specific stage onward
scripts/reset_segmentation_run.sh <run_name> --scope algorithmic
scripts/reset_segmentation_run.sh <run_name> --scope features
scripts/reset_segmentation_run.sh <run_name> --scope unsupervised
scripts/reset_segmentation_run.sh <run_name> --scope supervised
scripts/reset_segmentation_run.sh <run_name> --scope summary

# Reset all runs
scripts/reset_segmentation_run.sh --all

# Preview without deleting
scripts/reset_segmentation_run.sh <run_name> --scope summary --dry-run
```

## Output Structure

Run data is written to `paths.local_private_meta_analysis_data` (or `paths.local_private_data/<dataset_id>/00_Meta-Analysis/` for single-dataset runs):

```text
segmentation/<analysis_id>/
  qc/
    qc_channels.csv
  segments/
    algorithmic_segments.parquet
  features/
    features_filtered.parquet
  clustering/
    clustered_windows.parquet
  supervised/
    supervised_predictions.parquet
    supervised_variant_predictions.parquet   # if ablation enabled
  summary/
    method_budget_daily.parquet
    method_budget_hourly.parquet
```

## Cross-Dataset QC

Stage `09_qc` runs cross-dataset channel congruency checks automatically. To run standalone:

```bash
python3 scripts/run_cross_dataset_qc.py --config config.yaml --help
```

Or via the shared utility in Python:

```python
from pyologger.utils.cross_dataset_qc import run_cross_dataset_qc
```

## QA Checklist

After a run completes, verify:

- **Scope** — expected datasets/deployments present in `clustered_windows.parquet` or `supervised_predictions.parquet`
- **Features** — `features_filtered.parquet` non-empty with full deployment coverage
- **Clustering** — cluster distribution not collapsed to a single cluster
- **Supervised** — holdout diagnostics present; confidence distribution is reasonable
- **Summary** — daily/hourly budget tables exist and contain expected method rows
- **Cross-dataset QC** — channel presence and units are congruent across run scope

## Review Notebook

`notebooks/SEGMENTATION_REVIEW.ipynb` is the canonical entrypoint for reviewing segmentation, clustering, and supervised outputs.
