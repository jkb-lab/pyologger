#!/usr/bin/env bash
set -euo pipefail

# Bootstrap .snakemake_markers for deployments that already have *_output.nc
# and clear stale Snakemake provenance metadata after DAG input changes.

ROOT="${1:-/Volumes/WORK-SSD/Datasets/Unpublished}"
SNAKEFILE="${2:-Snakefile}"

echo "Using data root: ${ROOT}"
echo "Using snakefile: ${SNAKEFILE}"

if ! command -v snakemake >/dev/null 2>&1; then
  echo "snakemake not found in PATH."
  exit 1
fi

count=0
while IFS= read -r -d '' output_nc; do
  outdir="$(dirname "${output_nc}")"
  dep="$(basename "${output_nc}" _output.nc)"
  marker_dir="${outdir}/.snakemake_markers"
  mkdir -p "${marker_dir}"
  for s in 01 02 03 04 05; do
    touch "${marker_dir}/${dep}_step${s}.done"
  done
  count=$((count + 1))
done < <(find "${ROOT}" -type f -path "*/outputs/*_output.nc" -print0)

echo "Bootstrapped marker sets for ${count} deployment(s)."

# Clear stale metadata on final outputs so Snakemake ignores old input signature.
# Some outputs may not have metadata (archived folders, external runs). Skip those.
cleaned=0
skipped=0
while IFS= read -r -d '' output_nc; do
  if snakemake -s "${SNAKEFILE}" --cleanup-metadata "${output_nc}" >/dev/null 2>&1; then
    cleaned=$((cleaned + 1))
  else
    skipped=$((skipped + 1))
    echo "Skipping metadata cleanup (not present/in DAG): ${output_nc}"
  fi
done < <(find "${ROOT}" -type f -path "*/outputs/*_output.nc" -print0)

echo "Metadata cleanup done: cleaned=${cleaned}, skipped=${skipped}."
echo "Done. Next run suggestion:"
echo "  snakemake -s ${SNAKEFILE} -n --rerun-triggers mtime"
