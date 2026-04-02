#!/usr/bin/env python3
"""
Inspect the NetCDF file to see what signals and channels are available.
"""

import xarray as xr
import sys

nc_path = "/Volumes/WORK-SSD/Datasets/Unpublished/mile-adult-sese_vdr_argentina_RD-KM/2013-11-09_mile-008/outputs/2013-11-09_mile-008_step00.nc"

print("="*80)
print("INSPECTING NETCDF FILE")
print("="*80)
print(f"File: {nc_path}\n")

try:
    with xr.open_dataset(nc_path) as ds:
        print("Variables in NetCDF:")
        for var in ds.variables:
            print(f"  - {var}")
            if var.startswith("signal_data_"):
                signal_name = var.replace("signal_data_", "")
                print(f"      Signal: {signal_name}")
                print(f"      Shape: {ds[var].shape}")
                print(f"      Dims: {ds[var].dims}")
                
        print("\n" + "="*80)
        print("CHECKING FOR PRESSURE/DEPTH")
        print("="*80)
        
        # Check for pressure signal
        if "signal_data_pressure" in ds.variables:
            print("✅ signal_data_pressure EXISTS")
            print(f"   Shape: {ds['signal_data_pressure'].shape}")
        else:
            print("❌ signal_data_pressure NOT FOUND")
        
        # Check for depth signal  
        if "signal_data_depth" in ds.variables:
            print("✅ signal_data_depth EXISTS")
            print(f"   Shape: {ds['signal_data_depth'].shape}")
            # Check attributes to find channel names
            depth_var = ds['signal_data_depth']
            attrs = dict(depth_var.attrs)
            print(f"   Attributes: {attrs}")
            channels_attr = depth_var.attrs.get("variables") or depth_var.attrs.get("variable")
            print(f"   Channels attribute: {channels_attr}")
            
            # Check for specific channel names in global attributes
            print("\n   Checking global attributes for depth channels:")
            for attr_name in ds.attrs:
                if "depth" in attr_name.lower() and "channel" in attr_name.lower():
                    print(f"      {attr_name}: {ds.attrs[attr_name]}")
        else:
            print("❌ signal_data_depth NOT FOUND")
            
        print("\n" + "="*80)
        print("DATASET INFO")
        print("="*80)
        print(ds)
        
except Exception as e:
    print(f"❌ Error opening NetCDF: {e}")
    sys.exit(1)
