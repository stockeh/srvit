import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from multiprocessing.pool import ThreadPool

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    default='/mnt/conus3/jason_conus3/test',
                    help='data directory')
parser.add_argument('--results', type=str,
                    default='/mnt/mlnas01/stock/',
                    help='results directory')
parser.add_argument('--model', type=str,
                    default='complete01-vit',
                    help='model directory')

################################################################

refthrs_default = np.arange(5, 55, 5)
ymax = 60.0  # value from Hilburn et al. (2020)

################################################################

def load_ty(args):
    xtf, yf = args
    with np.load(xtf) as data:  # C x H x W
        # x = np.flip(np.moveaxis(data['xdata'], -1, 0), axis=1)
        t = np.flip(data['ydata'][np.newaxis, ...], axis=1) * ymax
    y = np.flip(np.load(yf), axis=1) * ymax
    return t, y

def get_refc_stats(goes_filenames, mrms_filenames, batch_size=100, refthrs=refthrs_default):
    '''
    inputs:
        goes_filenames: list of str, filenames of GOES data (npy files)
        mrms_filenames: list of str, filenames of MRMS data (npy files)
        batch_size: int, batch size for processing
        refthrs: np array, thresholds for statistics computation
    outputs:
        stats: dictionary of stats
    '''

    goes_sample = np.load(goes_filenames[0])
    goes_shape = goes_sample.shape
    num_samples = len(goes_filenames)
    n_batches = int(np.ceil(num_samples / batch_size))

    print('=> starting...')
    stats = {}
    stats['ref'] = refthrs
    stats['pod'] = []
    stats['far'] = []
    stats['csi'] = []
    stats['bias'] = []

    for i, rthr in tqdm(enumerate(refthrs), total=len(refthrs)):

        nhit_total = 0
        nmis_total = 0
        nfal_total = 0
        if i == 0:
            diff_mean_total = 0
            diff_sumsq_total = 0

        with ThreadPool(32) as pool:
            for batch_start in tqdm(range(0, num_samples, batch_size), total=n_batches, leave=False):
                batch_end = min(batch_start + batch_size, num_samples)

                goes_batch, mrms_batch = zip(*pool.map(load_ty, [(mrms_filenames[j], goes_filenames[j])
                                                                for j in range(batch_start, batch_end)]))

                goes_batch = np.array(goes_batch)
                mrms_batch = np.array(mrms_batch)

                goes_batch[goes_batch < 0] = 0.
                mrms_batch[mrms_batch < 0] = 0.

                hasrad = mrms_batch > rthr
                hassat = goes_batch > rthr

                nhit = np.sum(hasrad & hassat)
                nmis = np.sum(hasrad & ~hassat)
                nfal = np.sum(~hasrad & hassat)

                nhit_total += nhit
                nmis_total += nmis
                nfal_total += nfal

                if i == 0:
                    diff = goes_batch - mrms_batch
                    diff_mean = np.mean(diff)
                    diff_mean_total += diff_mean * diff.size
                    diff_sumsq_total += np.sum(np.square(diff))

        if nhit_total == 0:
            stats['pod'].append(np.nan)
            stats['far'].append(np.nan)
            stats['csi'].append(np.nan)
            stats['bias'].append(np.nan)
        else:
            csi = float(nhit_total) / float(nhit_total + nmis_total + nfal_total)
            pod = float(nhit_total) / float(nhit_total + nmis_total)
            far = float(nfal_total) / float(nhit_total + nfal_total)
            bias = float(nhit_total + nfal_total) / float(nhit_total + nmis_total)
            stats['pod'].append(pod)
            stats['far'].append(far)
            stats['csi'].append(csi)
            stats['bias'].append(bias)

    diff_mean = diff_mean_total / (num_samples * goes_shape[1] * goes_shape[2])
    rmse = np.sqrt(diff_sumsq_total / (num_samples * goes_shape[1] * goes_shape[2]))

    stats['diff_mean'] = diff_mean
    stats['rmse'] = rmse

    return stats



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
    # 1) get target and predicted data files
    xt_samples = []
    v = args.input
    for f in os.listdir(v):
        if 'regA' in f and f.endswith('.npz'):
            xt_samples.append(os.path.join(v, f))
    xt_samples.sort()

    y_samples = []
    v = os.path.join(args.results, args.model)
    for f in os.listdir(v):
        if f.endswith('.npy'):
            y_samples.append(os.path.join(v, f))
    y_samples.sort()

    # xt_samples = xt_samples[:2000]
    # y_samples = y_samples[:2000]

    # 2) compute statistics
    stats = get_refc_stats(y_samples, xt_samples)

    dumped = json.dumps(stats, cls=NumpyEncoder)
    output_file = os.path.join(args.results, args.model, 'stats.json') 
    with open(output_file, 'w') as f:
        json.dump(dumped, f)
    print(f'=> results saved to {output_file}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
