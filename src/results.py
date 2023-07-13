import os
import json
import argparse
import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map  # or process_map

from scipy.stats import pearsonr


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    default='/home/jstock/data/conus3/A/',
                    help='data directory')
parser.add_argument('--name', type=str, default='unet01-unet',
                    help='experiment-model name')

################################################################

refthrs_default = np.arange(5, 55, 5)
ymax = 60.0  # value from Hilburn et al. (2020)

################################################################


def get_refc_stats(goes, mrms, refthrs=refthrs_default):

    # inputs, required:
    #   goes = goes refc
    #   mrms = mrms refc

    # inputs, optional:
    #   refthrs = refc thresholds to evaluate statistics

    # outputs:
    #   stats = dictionary of stats
    
    print('=> starting...')
    good = (goes > -999) & (mrms > -999)

    # note: remove sub-zero variability
    goes[goes < 0] = 0.
    mrms[mrms < 0] = 0.

    stats = {}
    stats['ref'] = refthrs
    stats['pod'] = []
    stats['far'] = []
    stats['csi'] = []
    stats['bias'] = []
    stats['nrad'] = []
    stats['nsat'] = []
    diff = goes[good] - mrms[good]
    print('=> diff (goes-mrms)')
    stats['mean(goes-mrms)'] = np.mean(diff)
    print('=> mean(goes-mrms)')
    stats['std(goes-mrms)'] = np.std(diff)
    print('=> std(goes-mrms)')
    stats['rmsd'] = np.sqrt(np.mean(diff**2))
    print('=> rmsd')
    del diff
    stats['rsq'] = pearsonr(goes[good], mrms[good])[0]**2
    print('=> rsq')

    for rthr in tqdm(refthrs):

        hasrad = mrms > rthr
        nrad = np.sum(hasrad)

        hassat = goes > rthr
        nsat = np.sum(hassat)

        if nrad == 0:
            stats['pod'].append(np.nan)
            stats['far'].append(np.nan)
            stats['csi'].append(np.nan)
            stats['bias'].append(np.nan)
            stats['nrad'].append(nrad)
            stats['nsat'].append(nsat)
            continue

        nhit = np.sum(hasrad & hassat & good)
        nmis = np.sum(hasrad & ~hassat & good)
        nfal = np.sum(~hasrad & hassat & good)
        # nrej = np.sum( ~hasrad & ~hassat & good )

        try:
            csi = float(nhit) / float(nhit + nmis + nfal)
        except ZeroDivisionError:
            csi = np.nan
        try:
            pod = float(nhit) / float(nhit + nmis)
        except ZeroDivisionError:
            pod = np.nan
        try:
            far = float(nfal) / float(nhit + nfal)  # FA ratio
        except ZeroDivisionError:
            far = np.nan
        try:
            bias = float(nhit + nfal) / float(nhit + nmis)
        except ZeroDivisionError:
            bias = np.nan

        stats['pod'].append(pod)
        stats['far'].append(far)
        stats['csi'].append(csi)
        stats['bias'].append(bias)
        stats['nrad'].append(nrad)
        stats['nsat'].append(nsat)

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


def load_ty(args):
    xtf, yf = args
    with np.load(xtf) as data:  # C x H x W
        # x = np.flip(np.moveaxis(data['xdata'], -1, 0), axis=1)
        t = np.flip(data['ydata'][np.newaxis, ...], axis=1) * ymax
    y = np.flip(np.load(yf), axis=1) * ymax
    return t, y

def main(args):
    # 1) get target and predicted data files
    xt_samples = []
    v = os.path.join(args.data_dir, 'test')
    for f in os.listdir(v):
        if f.endswith('.npz'):
            xt_samples.append(os.path.join(v, f))
    xt_samples.sort()

    y_samples = []
    v = os.path.join(args.data_dir, 'out', args.name)
    for f in os.listdir(v):
        if f.endswith('.npy'):
            y_samples.append(os.path.join(v, f))
    y_samples.sort()

    # 2) load data from disk
    # 17344
    # Ttest = np.zeros((1734,1,768,1536))
    # Ytest = np.zeros((1734,1,768,1536))
    # print('=> storage arrays created')
    # for i in tqdm(range(Ttest.shape[0])):
    #     t, y = load_ty(xt_samples, y_samples, i)
    #     Ttest[i] = t
    #     Ytest[i] = y

    iterable = [(xt_samples[i], y_samples[i]) for i in range(1734)]
    Ttest, Ytest = zip(*thread_map(load_ty, iterable, max_workers=32))
    Ttest, Ytest = np.array(Ttest), np.array(Ytest)
    print('=> finished loading...')

    # 3) compute statistics
    stats = get_refc_stats(Ytest, Ttest)

    dumped = json.dumps(stats, cls=NumpyEncoder)
    output_file = os.path.join(args.data_dir, 'out', args.name, 'stats.json') 
    with open(output_file, 'w') as f:
        json.dump(dumped, f)
    print(f'=> results saved to {output_file}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
