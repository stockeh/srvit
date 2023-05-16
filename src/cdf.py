import os
import numpy as np

from tqdm import tqdm

def zero_cdf(data_dir, dataset):
    xt_samples = []
    v = os.path.join(data_dir, dataset)
    for f in os.listdir(v):
        if f.endswith('.npz'):
            xt_samples.append(os.path.join(v, f))
    xt_samples.sort()

    distribution_sample_map = np.zeros((768, 1536))
    output_sample_counts = np.zeros((len(xt_samples)))

    for i, f in enumerate(tqdm(xt_samples)):
        with np.load(f) as data:
            t = data['ydata']
            t[t > 0] = 1
            distribution_sample_map += t
            output_sample_counts[i] = int(t.sum())

    return distribution_sample_map, output_sample_counts

def main():
    data_dir = '/home/jstock/data/conus3/A/'
    for dataset in ['train', 'test', 'val']:
        d, o = zero_cdf(data_dir, dataset)
        np.save(os.path.join(data_dir, 'out', f'distribution_sample_map_{dataset}.npy'), d)
        np.save(os.path.join(data_dir, 'out', f'output_sample_counts_{dataset}.npy'), o)

if __name__ == '__main__':
    main()
