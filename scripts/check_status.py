#!/usr/bin/env python3
import os

data_root = '/Volumes/WORK-SSD/Datasets/Unpublished/mile-adult-sese_vdr_argentina_RD-KM'
deployments = [
    ('2012-11-02_mile-001', '001'), ('2012-11-02_mile-002', '002'),
    ('2012-11-01_mile-003', '003'), ('2013-11-10_mile-006', '006'),
    ('2013-11-09_mile-008', '008'), ('2015-11-05_mile-011', '011'),
    ('2015-11-05_mile-013', '013'), ('2015-11-06_mile-014', '014')
]

print('\n' + '='*70)
print('MILE DEPLOYMENT PROCESSING STATUS')
print('='*70)

completed = []
pending = []

for dep, num in deployments:
    nc_file = os.path.join(data_root, dep, 'outputs', f'{dep}_step01.nc')
    if os.path.exists(nc_file):
        print(f'✅ SEAL {num}: {dep}')
        completed.append(dep)
    else:
        print(f'❌ SEAL {num}: {dep} - NOT YET PROCESSED')
        pending.append(dep)

print('='*70)
print(f'Completed: {len(completed)}/8')
if pending:
    print(f'Pending: {len(pending)} deployments')
    print(f'  {", ".join([p.split("_")[-1] for p in pending])}')
print('='*70)
