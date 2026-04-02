#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one configured segmentation run by name via its Snakemake marker target.

Usage:
  ./scripts/run_segmentation_run.sh <run_name> [cores]
  ./scripts/run_segmentation_run.sh <run_name> [cores] [stage]

Arguments:
  run_name   Name from config.yaml segmentation_runs
  cores      Optional core count (default: 8)
  stage      Optional marker stage (default: 08_plots.done)

Examples:
  ./scripts/run_segmentation_run.sh mian_mile_depth_standardized_30s
  ./scripts/run_segmentation_run.sh mian_mile_depth_standardized_30s 4
  ./scripts/run_segmentation_run.sh mian_mile_depth_standardized_30s 8 04_correlate.done

Typical reset + rerun flow:
  ./scripts/reset_segmentation_run.sh mian_mile_depth_standardized_30s
  ./scripts/run_segmentation_run.sh mian_mile_depth_standardized_30s 8
EOF
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit 1
fi

if [[ "${1}" == "--help" || "${1}" == "-h" ]]; then
  usage
  exit 0
fi

run_name="${1}"
cores="${2:-8}"
stage="${3:-08_plots.done}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

target=".snakemake_segmentation/${run_name}/${stage}"

echo "Running segmentation target:"
echo "  ${target}"
echo "Using ${cores} core(s)"

snakemake -s Snakefile "${target}" --cores "${cores}"
