#!/bin/bash
# Complete processing of remaining mile deployments

set -e

DATASET="mile-adult-sese_vdr_argentina_RD-KM"
REPO_ROOT="/Users/jessiekb/Documents/GitHub/EcoViz_DiveDB"

# Remaining deployments
DEPLOYMENTS=(
    "2013-11-10_mile-006"
    "2015-11-05_mile-011"
    "2015-11-05_mile-013"
    "2015-11-06_mile-014"
)

echo "Processing remaining ${#DEPLOYMENTS[@]} mile deployments..."

cd "${REPO_ROOT}"
source venv/bin/activate

for deployment in "${DEPLOYMENTS[@]}"; do
    echo ""
    echo "▶▶▶ Processing: ${deployment}"
    echo "────────────────────────────────────────────────────────────"
    
    python "${REPO_ROOT}/pyologger/workflows/00_load_data.py" \
        --dataset "${DATASET}" \
        --deployment "${deployment}" 2>&1 | tail -2 && echo "  ✅ Step 00 complete"
    
    python "${REPO_ROOT}/pyologger/workflows/01_calibrate_pressure.py" \
        --dataset "${DATASET}" \
        --deployment "${deployment}" 2>&1 | grep -E "(conversion_factor|dives detected)" && echo "  ✅ Step 01 complete"
done

echo ""
echo "✅ All remaining deployments processed!"
