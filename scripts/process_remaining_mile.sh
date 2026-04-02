#!/bin/bash
# Process remaining mile deployments

cd /Users/jessiekb/Documents/GitHub/EcoViz_DiveDB/pyologger
source ../venv/bin/activate

DATASET="mile-adult-sese_vdr_argentina_RD-KM"
REMAINING=("2013-11-10_mile-006" "2015-11-05_mile-011" "2015-11-05_mile-013" "2015-11-06_mile-014")

echo "========================================="
echo "Processing Remaining Mile Deployments"
echo "========================================="

for DEP in "${REMAINING[@]}"; do
    echo ""
    echo "Processing ${DEP}..."
    echo "-----------------------------------------"
    
    # Step 00: Load data
    echo "Running 00_load_data.py..."
    python workflows/00_load_data.py --dataset "$DATASET" --deployment "$DEP"
    
    # Step 01: Calibrate pressure
    echo "Running 01_calibrate_pressure.py..."
    python workflows/01_calibrate_pressure.py --dataset "$DATASET" --deployment "$DEP"
    
    echo "✅ Completed ${DEP}"
    echo ""
done

echo "========================================="
echo "All remaining deployments processed!"
echo "========================================="
