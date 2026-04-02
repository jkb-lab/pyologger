#!/usr/bin/env python3
import pickle
import sys

deployment_path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/WORK-SSD/Datasets/Unpublished/mile-adult-sese_vdr_argentina_RD-KM/2013-11-10_mile-006"
pkl_path = f"{deployment_path}/outputs/data.pkl"

print(f"\nChecking: {deployment_path.split('/')[-1]}")
print("="*70)

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get depth column
    df = data['depth']['data']
    depth_col = 'depth' if 'depth' in df.columns else 'corrected_depth'
    depth_values = df[depth_col].dropna()
    
    # Calculate statistics
    total = len(depth_values)
    negative = (depth_values < 0).sum()
    zero = (depth_values == 0).sum()
    positive = (depth_values > 0).sum()
    
    print(f"Total depth values: {total}")
    print(f"Min: {depth_values.min():.2f} m")
    print(f"Max: {depth_values.max():.2f} m")
    print(f"Mean: {depth_values.mean():.2f} m")
    print(f"\nSign distribution:")
    print(f"  Negative: {negative} ({100*negative/total:.1f}%)")
    print(f"  Zero: {zero} ({100*zero/total:.1f}%)")
    print(f"  Positive: {positive} ({100*positive/total:.1f}%)")
    
    # Check transformation log
    if 'transformation_log' in data['depth']:
        tf_log = data['depth']['transformation_log']
        has_conversion = any('conversion_factor' in t for t in tf_log)
        print(f"\nTransformation log: {tf_log}")
        if has_conversion:print(f"✅ Conversion factor was applied!")
    
    # Verdict
    print("\n" + "="*70)
    if positive > total * 0.5:
        print("✅ SUCCESSFUL: Depth values are mostly POSITIVE")
    else:
        print("❌ FAILED: Depth values are still mostly negative")
    print("="*70)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
