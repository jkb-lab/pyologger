#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run cross-dataset QC for one configured segmentation run.

Usage:
  ./scripts/run_segmentation_qc.sh <run_name> [cores]

Examples:
  ./scripts/run_segmentation_qc.sh mian_mile_sleep_transfer_rf
  ./scripts/run_segmentation_qc.sh mian_mile_sleep_transfer_rf 4
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

if [[ "${1}" == "--help" || "${1}" == "-h" ]]; then
  usage
  exit 0
fi

run_name="${1}"
cores="${2:-4}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

target="segmentation_markers/${run_name}/09_qc.done"
echo "🔎 Running segmentation QC target:"
echo "   ${target}"
echo "⚙️  Using ${cores} core(s)"

if [[ -x "${repo_root}/venv/bin/snakemake" ]]; then
  SNAKEMAKE_BIN="${repo_root}/venv/bin/snakemake"
else
  SNAKEMAKE_BIN="snakemake"
fi

"${SNAKEMAKE_BIN}" -s workflows/Snakefile "${target}" --configfile config.yaml --cores "${cores}"
echo "✅ Segmentation run-scope QC finished."
