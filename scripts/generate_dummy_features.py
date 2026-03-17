import csv
import os
import random

out_dir = os.path.join('data', 'processed')
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'features.csv')

n_samples = 200
n_features = 10

headers = ['code', 'label'] + [f'feat_{i}' for i in range(n_features)]

with open(out_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(n_samples):
        label = 0 if i < n_samples // 2 else 1
        code = f"sample_code_{i}"
        feats = [f"{random.uniform(-1,1):.6f}" for _ in range(n_features)]
        writer.writerow([code, label] + feats)

print(f"Wrote {out_file} with {n_samples} samples and {n_features} features")
