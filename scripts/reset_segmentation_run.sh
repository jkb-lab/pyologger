#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Reset segmentation Snakemake markers for one run or all runs.

Usage:
  pyologger/scripts/reset_segmentation_run.sh <run_name>
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope qc
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope algorithmic
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope features
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope unsupervised
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope supervised
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope summary
  pyologger/scripts/reset_segmentation_run.sh <run_name> --scope <stage>
  pyologger/scripts/reset_segmentation_run.sh --all

Examples:
  pyologger/scripts/reset_segmentation_run.sh mian_sleep_rf_test
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope qc
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope algorithmic
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope features
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope unsupervised
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope supervised
  pyologger/scripts/reset_segmentation_run.sh mian_mile_sleep_transfer_rf --scope summary
  pyologger/scripts/reset_segmentation_run.sh --all

Scopes:
  all          Remove the run marker directory/directories (default for a run).
  qc           New workflow: remove 09_qc.done through 14_summary.done.
  algorithmic  New workflow: remove 10_algorithmic.done through 14_summary.done.
  features     New workflow: remove 11_features.done through 14_summary.done.
  unsupervised New workflow: remove 12_unsupervised.done through 14_summary.done.
  supervised   New workflow: remove 13_supervised.done and 14_summary.done.
  summary      New workflow: remove only 14_summary.done.
  last         Alias for summary.
  cluster      Legacy workflow alias for algorithmic.
  learning     Legacy workflow alias for features.
  export       Legacy workflow alias for unsupervised.
  supervised+  Legacy workflow alias for supervised.
  plots        Legacy workflow alias for summary.

Flags:
  --dry-run    Print what would be removed without deleting anything.
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pyologger_root="${repo_root}/pyologger"
config_file="${pyologger_root}/config.yaml"

meta_analysis_data="$(python3 -c "import yaml; cfg=yaml.safe_load(open('${config_file}')); print(cfg['paths']['local_private_meta_analysis_data'])")"

legacy_markers_root="${pyologger_root}/.snakemake_segmentation"
new_markers_root="${meta_analysis_data}/segmentation"

target="${1:-}"
scope="all"
dry_run=false

if [[ "${target}" == "--help" || "${target}" == "-h" ]]; then
  usage
  exit 0
fi

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --scope" >&2
        exit 1
      fi
      scope="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${target}" == "--all" ]]; then
  echo "Removing all segmentation marker directories..."
  if [[ "${dry_run}" == "true" ]]; then
    echo "DRY RUN: rm -rf ${legacy_markers_root}"
    for d in "${new_markers_root}"/*/; do
      echo "DRY RUN: rm -rf ${d}"
    done
    exit 0
  fi
  rm -rf "${legacy_markers_root}"
  for d in "${new_markers_root}"/*/; do
    [[ -d "${d}" ]] && rm -rf "${d}" && echo "Removed: ${d}"
  done
  echo "Done."
  exit 0
fi

run_name="${target}"
legacy_marker_dir="${legacy_markers_root}/${run_name}"
new_marker_dir="${new_markers_root}/${run_name}"
scope_lc="$(echo "${scope}" | tr '[:upper:]' '[:lower:]')"

remove_files=()
case "${scope_lc}" in
  all)
    echo "Removing segmentation marker directory for run: ${run_name}"
    if [[ "${dry_run}" == "true" ]]; then
      echo "DRY RUN: rm -rf ${legacy_marker_dir}"
      echo "DRY RUN: rm -rf ${new_marker_dir}"
      exit 0
    fi
    rm -rf "${legacy_marker_dir}"
    rm -rf "${new_marker_dir}"
    echo "Removed:"
    echo "  ${legacy_marker_dir}"
    echo "  ${new_marker_dir}"
    exit 0
    ;;
  qc)
    remove_files+=(
      "${new_marker_dir}/09_qc.done"
      "${new_marker_dir}/10_algorithmic.done"
      "${new_marker_dir}/11_features.done"
      "${new_marker_dir}/12_unsupervised.done"
      "${new_marker_dir}/13_supervised.done"
      "${new_marker_dir}/14_summary.done"
    )
    ;;
  algorithmic|cluster)
    remove_files+=(
      "${new_marker_dir}/10_algorithmic.done"
      "${new_marker_dir}/11_features.done"
      "${new_marker_dir}/12_unsupervised.done"
      "${new_marker_dir}/13_supervised.done"
      "${new_marker_dir}/14_summary.done"
    )
    ;;
  features|learning)
    remove_files+=(
      "${new_marker_dir}/11_features.done"
      "${new_marker_dir}/12_unsupervised.done"
      "${new_marker_dir}/13_supervised.done"
      "${new_marker_dir}/14_summary.done"
    )
    ;;
  unsupervised|export)
    remove_files+=(
      "${new_marker_dir}/12_unsupervised.done"
      "${new_marker_dir}/13_supervised.done"
      "${new_marker_dir}/14_summary.done"
    )
    ;;
  supervised|supervised+|supervised_plus)
    remove_files+=(
      "${new_marker_dir}/13_supervised.done"
      "${new_marker_dir}/14_summary.done"
    )
    ;;
  summary|last|plots)
    remove_files+=("${new_marker_dir}/14_summary.done")
    ;;
  *)
    echo "Unknown scope: ${scope}" >&2
    usage
    exit 1
    ;;
esac

echo "Removing segmentation markers for run: ${run_name} (scope=${scope_lc})"
for f in "${remove_files[@]}"; do
  if [[ "${dry_run}" == "true" ]]; then
    echo "DRY RUN: rm -f ${f}"
  else
    rm -f "${f}"
    echo "Removed: ${f}"
  fi
done
