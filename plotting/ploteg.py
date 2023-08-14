import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from tqdm import tqdm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Composite Reflectivity Plotter')

# Config arguments
parser.add_argument('-c', '--config', required=True, type=str,
                    help='path to config file (.json)')

parser.add_argument('-m', '--models', nargs='+', default=['mrms', 'complete01-vit', 'unet01-unet'],
                    help='list of models to plot')

parser.add_argument('--horizontal', action='store_true',  # default false
                    help='make plots horizontal instead of vertical')

parser.add_argument('--stats', action='store_true',  # default false
                    help='include stats in the output')

parser.add_argument('--mode', default='panel', const='panel', type=str, nargs='?',
                    choices=['panel', 'composite', 'case'],
                    help='plot mode (default: %(default)s)')

# Directory arguments
parser.add_argument('--hrrr-grid-file', default='/mnt/conus3/jason_conus3/code/hrrr_grid.bin', type=str,
                    help='path to HRRR Grid file')

parser.add_argument('--number-date-file', default='/mnt/conus3/jason_conus3/sample_numbers_dates_regA_sclA.txt',
                    type=str, help='path to number date file')

parser.add_argument('--xt-dir', default='/mnt/conus3/jason_conus3/test', type=str,
                    help='path to xt npz directory')

parser.add_argument('--results-dir', default='/mnt/mlnas01/stock/', type=str,
                    help='path to precomputed results directory')

parser.add_argument('--media-dir', default='../media', type=str,
                    help='path to media directory')

ymax = 60.0


def load_y(y_dir, idx):
    """
    Assumes data formated as: `test_predictions_000001.npy`
    """
    global ymax
    return np.clip(np.load(os.path.join(y_dir, f'test_predictions_{idx:06d}.npy')), 0, 1) * ymax


def load_xt(xt_dir, idx):
    """
    Assumes data formated as: `conus3_regA_sclA_000001.npz`
    """
    global ymax
    # C x H x W
    with np.load(os.path.join(xt_dir, f'conus3_regA_sclA_{idx:06d}.npz')) as data:
        x = np.moveaxis(data['xdata'], -1, 0)
        t = data['ydata'][np.newaxis, ...] * ymax
    return x, t


def load_xty(xt_dir, y_dir, idx):
    x, t = load_xt(xt_dir, idx)
    y = load_y(y_dir, idx)
    return x, t, y


def get_colors():
    rgb_colors = []  # new: CVD accessible
    rgb_colors.append((231, 231, 231))  # 0
    rgb_colors.append((111, 239, 255))  # 5
    rgb_colors.append((95, 207, 239))  # 10
    rgb_colors.append((79, 175, 223))  # 15
    rgb_colors.append((47,  95, 191))  # 20
    rgb_colors.append((31,  63, 175))  # 25
    rgb_colors.append((15,  31, 159))  # 30
    rgb_colors.append((247, 239,  63))  # 35
    rgb_colors.append((239, 191,  55))  # 40
    rgb_colors.append((231, 143,  47))  # 45
    rgb_colors.append((207,  15,  23))  # 50
    rgb_colors.append((183,   7,  15))  # 55
    rgb_colors.append((159,   0,   8))  # 60

    colors = []
    for atup in rgb_colors:
        colors.append('#%02x%02x%02x' % atup)

    cmap = ListedColormap(colors, 'radar')
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])

    bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    ticklabels = [str(a) for a in bounds]

    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds, ticklabels


def init_params(config, args, full_domain=False):
    nlon = 1799
    nlat = 1059
    count = nlat*nlon
    shape = (nlat, nlon)
    with open(args.hrrr_grid_file, 'rb') as f:
        lon = np.fromfile(f, dtype=np.float64, count=count).reshape(shape)
        lat = np.fromfile(f, dtype=np.float64, count=count).reshape(shape)

    # TODO: close :)
    lon = lon[147:nlat-144, 153:nlon-110]
    lat = lat[147:nlat-144, 153:nlon-110]

    if full_domain:
        config['minlon'] = lon.min()
        config['maxlon'] = lon.max()
        config['minlat'] = lat.min()
        config['maxlat'] = lat.max()

    basemap = {}
    basemap['projection'] = 'cyl'
    basemap['resolution'] = 'i'
    basemap['fix_aspect'] = False
    basemap['llcrnrlon'] = config['minlon']
    basemap['urcrnrlon'] = config['maxlon']
    basemap['llcrnrlat'] = config['minlat']
    basemap['urcrnrlat'] = config['maxlat']
    basemap = Basemap(**basemap)

    x, y = basemap(lon, lat)

    number_date_df = pd.read_csv(args.number_date_file, sep=' ',
                                 header=None, names=['kind', 'number', 'date'])

    ids = number_date_df[number_date_df.date.isin(
        config['dates'])]['number'].values

    return basemap, x, y, ids


def plot_pannel(config, args):
    models = args.models
    xt_dir, results_dir = args.xt_dir, args.results_dir

    basemap, x, y, ids = init_params(config, args, full_domain=True)

    output_dir = os.path.join(args.media_dir, f'{config["name"]}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'=> plotting {len(ids)} images')

    fig = plt.figure(figsize=(5 * len(ids), 2.7 * len(models)),
                     constrained_layout=True)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    cmap, norm, bounds, ticklabels = get_colors()
    fontsize = 12

    axes = []
    for i, model in enumerate(tqdm(models)):
        for j, ide in enumerate(tqdm(ids, leave=False, total=len(ids))):

            ax = plt.subplot(len(models), len(ids), i*len(ids)+j+1)
            axes.append(ax)

            y_dir = os.path.join(results_dir, model)
            if model.lower() == 'mrms':
                data = load_xt(xt_dir, ide)[1][0]
            else:
                data = load_y(y_dir, ide)[0]

            covmask = np.zeros(data.shape)
            nocover = (data == -999)
            covmask[nocover] = 1
            covmask = np.ma.masked_where(covmask == 0, covmask)

            pcm = basemap.pcolormesh(x, y, data, cmap=cmap, norm=norm)

            covpcm = basemap.pcolormesh(
                x, y, covmask, cmap='Greys', vmin=0, vmax=2, alpha=0.5)

            basemap.drawcoastlines()
            basemap.drawcountries()
            basemap.drawstates()
            basemap.drawcounties()

            if i == 0:
                plt.title(config['dates'][j].replace(
                    '_', ' '), fontsize=fontsize)
            if j == 0:
                label = model.upper()
                if '-' in label:
                    label = label.split('-')[1]
                plt.ylabel(label, loc='top', fontsize=fontsize,
                           fontweight='bold')

            if model.lower() != 'mrms' and args.stats:
                ti, yi = load_xty(xt_dir, y_dir, ide)[1:]
                rmse = np.sqrt(np.mean((yi-ti)**2))
                r2 = r2_score(yi.flatten(), ti.flatten())
                # plt.xlabel(f'RMSE: {rmse:.2f}, R2: {r2:.2f}', fontsize=fontsize, color='gray')
                t = plt.text(0.98, 0.035, f'RMSE: {rmse:.2f}\nR2: {r2:.2f}', transform=ax.transAxes,
                             fontsize=fontsize, va='bottom', ha='right', color='black')
                t.set_bbox(dict(facecolor='lightgray',
                           alpha=0.3, edgecolor='gray'))

            # pannel text labels in bottom left corner
            plt.text(0.01, 0.01, f'{chr(97+(i*len(ids)+j))})', transform=ax.transAxes, fontsize=fontsize+1,
                     fontweight='bold', va='bottom', ha='left', color='black')

    cb = plt.colorbar(pcm, ticks=bounds, orientation='vertical',
                      ax=axes, fraction=0.05, pad=0.005)
    # cb.ax.set_xticklabels(ticklabels)
    cb.set_label('Composite Reflectivity (dBZ)', fontsize=fontsize)

    filename = os.path.join(output_dir, f'{config["name"]}.png')
    fig.savefig(filename, dpi=300)

    print(f'=> file saved to {filename}')


def plot_composite(config, args):
    """!convert -delay 30 -loop 0 2022-06-1*.png loop_squall.gif
    """
    models = args.models
    xt_dir, results_dir = args.xt_dir, args.results_dir

    basemap, x, y, ids = init_params(config, args)

    output_dir = os.path.join(args.media_dir, f'{config["name"]}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'=> plotting {len(ids)} images')

    fontsize = 12
    if args.horizontal:
        fig = plt.figure(figsize=(6 * len(models), 5), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(5, len(models)*3.5), constrained_layout=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    cmap, norm, bounds, ticklabels = get_colors()

    for i, ide in enumerate(tqdm(ids)):
        for j, model in enumerate(tqdm(models, leave=False, total=len(models))):
            time = config['dates'][i]
            if args.horizontal or len(models) == 1:
                s = model.upper()
            else:
                s = ''
            title = f"{s}{time.replace('_', ' ')}"

            if args.horizontal:
                ax = plt.subplot(1, len(models), 1+j)
            else:
                ax = plt.subplot(len(models), 1, 1+j)

            if model.lower() == 'mrms':
                data = load_xt(xt_dir, ide)[1][0]
            else:
                y_dir = os.path.join(results_dir, model)
                data = load_y(y_dir, ide)[0]

            covmask = np.zeros(data.shape)
            nocover = (data == -999)
            covmask[nocover] = 1
            covmask = np.ma.masked_where(covmask == 0, covmask)

            pcm = basemap.pcolormesh(x, y, data, cmap=cmap, norm=norm)

            covpcm = basemap.pcolormesh(
                x, y, covmask, cmap='Greys', vmin=0, vmax=2, alpha=0.5)

            basemap.drawcoastlines()
            basemap.drawcountries()
            basemap.drawstates()
            basemap.drawcounties()

            if not args.horizontal and j == 0:
                plt.title(title, fontsize=fontsize)
            if len(models) > 1 and not args.horizontal and i == 0:
                label = model.upper()
                if '-' in label:
                    label = label.split('-')[1]
                plt.ylabel(label, loc='top', fontsize=fontsize,
                           fontweight='bold')

            colorbar = False
            if not args.horizontal and j == len(models)-1:
                colorbar = True
            elif args.horizontal:
                colorbar = True
            if colorbar:
                cb = plt.colorbar(
                    pcm, ticks=bounds, orientation='horizontal', fraction=0.1, pad=0.02)
                cb.ax.set_xticklabels(ticklabels)
                cb.set_label('Composite Reflectivity (dBZ)', fontsize=fontsize)

            if model.lower() != 'mrms' and args.stats:
                ti, yi = load_xty(xt_dir, y_dir, ide)[1:]
                rmse = np.sqrt(np.mean((yi-ti)**2))
                r2 = r2_score(yi.flatten(), ti.flatten())
                # 0.03
                # plt.xlabel(f'RMSE: {rmse:.2f}, R2: {r2:.2f}', fontsize=fontsize, color='gray')
                t = plt.text(0.98, 0.85, f'RMSE: {rmse:.2f}\nR2: {r2:.2f}', transform=ax.transAxes,
                             fontsize=fontsize, va='bottom', ha='right', color='black')
                t.set_bbox(dict(facecolor='lightgray',
                           alpha=0.3, edgecolor='gray'))

            # pannel text labels in bottom left corner
            plt.text(0.01, 0.01, f'{chr(97+(i*len(ids)+j))})', transform=ax.transAxes, fontsize=fontsize+1,
                     fontweight='bold', va='bottom', ha='left', color='black')

        filename = os.path.join(output_dir, f'{time}.png')
        fig.savefig(filename, dpi=300)
        plt.clf()

    print(f'=> files saved to {output_dir}')


def plot_cases(config, args):
    """!convert -delay 30 -loop 0 2022-06-1*.png loop_squall.gif
    """
    models = args.models
    xt_dir, results_dir = args.xt_dir, args.results_dir

    if 'multi' not in config:
        basemap, x, y, ids = init_params(config, args)
    else:
        ids = []
        for c in config['multi']:
            _, _, _, localids = init_params(c, args)
            ids.extend(localids)
    print(ids)

    output_dir = os.path.join(args.media_dir, f'{config["name"]}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'=> plotting {len(ids)} images')

    fontsize = 12
    fig = plt.figure(figsize=(5 * len(models), 2.7 * len(ids)),
                     constrained_layout=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    cmap, norm, bounds, ticklabels = get_colors()
    axes = []
    for i, ide in enumerate(tqdm(ids)):
        for j, model in enumerate(tqdm(models, leave=False, total=len(models))):
            if 'multi' in config:
                basemap, x, y, _ = init_params(config['multi'][i], args)
            ax = plt.subplot(len(ids), len(models), i*len(models)+j+1)
            axes.append(ax)

            if model.lower() == 'mrms':
                data = load_xt(xt_dir, ide)[1][0]
            else:
                y_dir = os.path.join(results_dir, model)
                data = load_y(y_dir, ide)[0]

            covmask = np.zeros(data.shape)
            nocover = (data == -999)
            covmask[nocover] = 1
            covmask = np.ma.masked_where(covmask == 0, covmask)

            pcm = basemap.pcolormesh(x, y, data, cmap=cmap, norm=norm)

            covpcm = basemap.pcolormesh(
                x, y, covmask, cmap='Greys', vmin=0, vmax=2, alpha=0.5)

            basemap.drawcoastlines()
            basemap.drawcountries()
            basemap.drawstates()
            basemap.drawcounties()

            if i == 0:
                title = model.upper()
                if '-' in title:
                    title = title.split('-')[1]
                plt.title(title, fontsize=fontsize, fontweight='bold')
            if j == 0:
                if 'multi' not in config:
                    time = config['dates'][i]
                else:
                    time = config['multi'][i]['dates'][0]
                plt.ylabel(f"{time.replace('_', ' ')}", fontsize=fontsize)

            if model.lower() != 'mrms' and args.stats:
                ti, yi = load_xty(xt_dir, y_dir, ide)[1:]
                rmse = np.sqrt(np.mean((yi-ti)**2))
                r2 = r2_score(yi.flatten(), ti.flatten())
                # 0.03
                # plt.xlabel(f'RMSE: {rmse:.2f}, R2: {r2:.2f}', fontsize=fontsize, color='gray')
                t = plt.text(0.98, 0.82, f'RMSE: {rmse:.2f}\nR2: {r2:.2f}', transform=ax.transAxes,
                             fontsize=fontsize, va='bottom', ha='right', color='black')
                t.set_bbox(dict(facecolor='lightgray',
                           alpha=0.3, edgecolor='gray'))

            # pannel text labels in bottom left corner
            plt.text(0.01, 0.01, f'{chr(97+(i*len(ids)+j))})', transform=ax.transAxes, fontsize=fontsize+1,
                     fontweight='bold', va='bottom', ha='left', color='black')

    fraction = 0.05 if args.horizontal else 0.1
    pad = 0.005 if args.horizontal else 0.02
    cb = plt.colorbar(
        pcm, ticks=bounds, ax=axes,
        orientation='vertical' if args.horizontal else 'horizontal',
        fraction=fraction, pad=pad)
    # cb.ax.set_xticklabels(ticklabels)
    cb.set_label('Composite Reflectivity (dBZ)', fontsize=fontsize)

    filename = os.path.join(output_dir, 'cases.png')
    fig.savefig(filename, dpi=300)

    print(f'=> files saved to {output_dir}')


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    # TODO: remove space from dates, look into fixing this
    if 'multi' not in config:
        config['dates'] = [d.replace(' ', '') for d in config['dates']]
    else:
        for c in config['multi']:
            c['dates'] = [d.replace(' ', '') for d in c['dates']]

    if args.mode == 'panel':
        plot_pannel(config, args)
    elif args.mode == 'composite':
        plot_composite(config, args)
    elif args.mode == 'case':
        plot_cases(config, args)
    else:
        raise ValueError(f'invalid mode {args.mode}')


if __name__ == '__main__':
    """
    Example usage:
        python ploteg.py -c /home/stock/conus3-attention/plotting/casestudies/derecho.json -m mrms complete01-vit unet01-unet --horizontal --mode composite
        python ploteg.py -c /home/stock/conus3-attention/plotting/casestudies/conus_colin.json -m mrms complete01-vit
    Create a gif:
        convert -delay 30 -loop 0 *.png loop.gif
    """
    args = parser.parse_args()
    main(args)
