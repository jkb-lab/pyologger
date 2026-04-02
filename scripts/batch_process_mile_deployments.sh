#!/bin/bash
# Batch process all mile deployments with depth sign fix
# The conversion_factor: -1.0 is already configured in parameter_log.json dataset defaults

set -e

DATASET="mile-adult-sese_vdr_argentina_RD-KM"
REPO_ROOT="/Users/jessiekb/Documents/GitHub/EcoViz_DiveDB"
WORKFLOW_SCRIPT="${REPO_ROOT}/pyologger/workflows/01_calibrate_pressure.py"

# Deployments to process (mile-008 already completed)
DEPLOYMENTS=(
    "2012-11-02_mile-001"
    "2012-11-02_mile-002"
    "2012-11-01_mile-003"
    "2013-11-10_mile-006"
    "2015-11-05_mile-011"
    "2015-11-05_mile-013"
    "2015-11-06_mile-014"
)

echo "================================================================================"
echo "BATCH PROCESSING MILE DEPLOYMENTS - DEPTH SIGN FIX"
echo "================================================================================"
echo "Dataset: ${DATASET}"
echo "Number of deployments to process: ${#DEPLOYMENTS[@]}"
echo ""
echo "Deployments:"
for deployment in "${DEPLOYMENTS[@]}"; do
    echo "  - ${deployment}"
done
echo ""
echo "Configuration: conversion_factor: -1.0 (from parameter_log.json dataset defaults)"
echo "================================================================================"
echo ""

# Activate virtual environment
cd "${REPO_ROOT}"
source venv/bin/activate

# Process each deployment
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_DEPLOYMENTS=()

for deployment in "${DEPLOYMENTS[@]}"; do
    echo ""
    echo "▶▶▶ Processing: ${deployment}"
    echo "────────────────────────────────────────────────────────────────────────────"
    
    # Run Step 00 first to create/update NetCDF
    echo "  Step 00: Loading data..."
    if python "${REPO_ROOT}/pyologger/workflows/00_load_data.py" \
        --dataset "${DATASET}" \
        --deployment "${deployment}" 2>&1 | tail -3; then
        echo "  ✅ Step 00 complete"
    else
        echo "  ⚠️  Step 00 had issues (may be normal if already processed)"
    fi
    
    # Run Step 01 to calibrate pressure/depth with conversion_factor
    echo "  Step 01: Calibrating pressure/depth..."
    if python "${WORKFLOW_SCRIPT}" \
        --dataset "${DATASET}" \
        --deployment "${deployment}" 2>&1 | grep -E "(conversion_factor|Baseline-adjusted|dives detected)"; then
        echo "  ✅ Step 01 complete"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ❌ Step 01 failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_DEPLOYMENTS+=("${deployment}")
    fi
    
    echo "────────────────────────────────────────────────────────────────────────────"
done

echo ""
echo "================================================================================"
echo "BATCH PROCESSING COMPLETE"
echo "================================================================================"
echo "✅ Successful: ${SUCCESS_COUNT}/${#DEPLOYMENTS[@]}"
if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "❌ Failed: ${FAIL_COUNT}/${#DEPLOYMENTS[@]}"
    echo ""
    echo "Failed deployments:"
    for failed in "${FAILED_DEPLOYMENTS[@]}"; do
        echo "  - ${failed}"
    done
else
    echo "🎉 All deployments processed successfully!"
fi
echo "================================================================================"
echo ""
echo "Next steps:"
echo "1. Verify a couple deployments by loading them in the visualization dashboard"
echo "2. Check that depth values are positive and dives are detected correctly"
echo ""
echo "Note: The conversion_factor: -1.0 is saved in parameter_log.json dataset"
echo "defaults, so future processing of mile deployments will automatically apply"
echo "the sign fix without any additional configuration."
