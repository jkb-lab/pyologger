#!/usr/bin/env python3
import netCDF4 as nc
import numpy as np

deploys = [
    ("2012-11-02_mile-001", "001"), ("2012-11-02_mile-002", "002"),
    ("2012-11-01_mile-003", "003"), ("2013-11-10_mile-006", "006"),
    ("2013-11-09_mile-008", "008"), ("2015-11-05_mile-011", "011"),
    ("2015-11-05_mile-013", "013"), ("2015-11-06_mile-014", "014")
]

base_path = "/Volumes/WORK-SSD/Datasets/Unpublished/mile-adult-sese_vdr_argentina_RD-KM"

print("\n" + "="*80)
print("MILE DEPLOYMENT DEPTH VERIFICATION")
print("="*80)

for dep_id, seal_num in deploys:
    nc_file = f"{base_path}/{dep_id}/outputs/{dep_id}_step01.nc"
    
    try:
        ds = nc.Dataset(nc_file, 'r')
        depth_data = ds.variables['signal_data_depth'][:]
        ds.close()
        
        # Calculate statistics
        depth_data = depth_data[~np.isnan(depth_data)]
        total = len(depth_data)
        positive = (depth_data > 0).sum()
        negative = (depth_data < 0).sum()
        pct_positive = 100 * positive / total if total > 0 else 0
        
        status = "✅" if pct_positive > 50 else "❌"
        print(f"{status} SEAL {seal_num}: {pct_positive:.1f}% positive (min={depth_data.min():.1f}, max={depth_data.max():.1f})")
        
    except Exception as e:
        print(f"❌ SEAL {seal_num}: ERROR - {str(e)[:50]}")

print("="*80)
