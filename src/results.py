import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import pearsonr

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

    for rthr in refthrs:

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
        #nrej = np.sum( ~hasrad & ~hassat & good )

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


if __name__ == '__main__':
    # original stats from Hilburn et al. (2020) Table 3
    og_stats = dict()
    og_stats['ref'] = refthrs_default
    og_stats['pod'] = [0.92, 0.85, 0.80, 0.71,
                       0.63, 0.55, 0.51, 0.52, 0.43, 0.37]
    og_stats['far'] = [0.23, 0.18, 0.22, 0.31,
                       0.40, 0.46, 0.57, 0.57, 0.65, 0.77]
    og_stats['csi'] = [0.72, 0.72, 0.65, 0.54,
                       0.45, 0.38, 0.33, 0.31, 0.24, 0.14]
    og_stats['bias'] = [1.19, 1.04, 1.03, 1.03,
                        1.05, 1.01, 1.06, 1.23, 1.24, 1.17]

    # TODO: load target and prediction data
    Ytest *= ymax
    Ttest *= ymax
    stats = get_refc_stats(Ytest, Ttest)

    # rest is for plotting
    media_path = '../media'

    cmap = cm.get_cmap('Spectral', 4)

    fontsize = 11
    lw = 2.5
    metrics = ['pod', 'far', 'csi', 'bias']

    # differences == True  : plot difference from original stats
    #                False : plot statistics as they are
    for difference in [False, True]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for i, m in enumerate(metrics):
            ax.plot(stats['ref'], np.array(stats[m])-(og_stats[m] if difference else 0),
                    label=m.upper(), ls='-', lw=lw, color=cmap(i))
        ax.set_xticks(stats['ref'])

        ax.set_xlabel('REFC Threshold (dBZ)', fontsize=fontsize)
        ax.set_ylabel('Score', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(fontsize=fontsize-1)
        ax.grid(alpha=0.5)
        ax.set_title(f'diff: {str(difference).lower()}',
                     loc='left', style='italic')

        # fig.savefig(os.path.join(media_path, f'metrics_diff_{str(difference).lower()}.png'),
        #             dpi=300, bbox_inches='tight')
