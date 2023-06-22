import os
import json
import argparse
import numpy as np

from tqdm import tqdm
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
    stats['mean(goes-mrms)'] = np.mean(goes[good]-mrms[good])
    stats['std(goes-mrms)'] = np.std(goes[good]-mrms[good])
    stats['rmsd'] = np.sqrt(np.mean((goes[good]-mrms[good])**2))
    stats['rsq'] = pearsonr(goes[good], mrms[good])[0]**2

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


def load_ty(xtf, yf, idx):
    with np.load(xtf[idx]) as data:  # C x H x W
        # x = np.flip(np.moveaxis(data['xdata'], -1, 0), axis=1)
        t = np.flip(data['ydata'][np.newaxis, ...], axis=1) * ymax
    with np.load(yf[idx]) as data:
        y = np.flip(data, axis=1) * ymax
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
    v = os.path.join(args.data_dir, 'out', args.model_name)
    for f in os.listdir(v):
        if f.endswith('.npy'):
            y_samples.append(os.path.join(v, f))
    y_samples.sort()

    # 2) load data from disk
    Ttest, Ytest = [], []
    for i in tqdm(range(len(xt_samples))):
        t, y = load_ty(xt_samples, y_samples, i)
        Ttest.append(t)
        Ytest.append(y)
        if i == 10:
            break
    Ttest = np.concatenate(Ttest, axis=0)
    Ytest = np.concatenate(Ytest, axis=0)

    # 3) compute statistics
    stats = get_refc_stats(Ytest, Ttest)

    with open(os.path.join(args.data_dir, 'out',
                           args.model_name, 'stats.json'), 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
