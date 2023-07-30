import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def token2token_scores(scores_mat, title='Head',
                       xlabel='x', ylabel='x', filename=None):
    # fig = plt.figure(figsize=(9, 12), )
    fig, axs = plt.subplots(4, 1, figsize=(9, 12),
                            sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0.0, 'wspace': 0.25})
    #  constrained_layout=True)
    axs = axs.ravel()
    for idx, scores in enumerate(scores_mat):
        # ax = fig.add_subplot(4, 3, idx+1)
        ax = axs[idx]
        im = ax.imshow(scores, cmap='bone_r')

        # ax.set_xticks(range(len(all_tokens)))
        # ax.set_yticks(range(len(all_tokens)))

        # ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
        # ax.set_yticklabels(all_tokens, fontdict=fontdict)
        ax.set_title('{} {}'.format(title, idx+1), fontsize=11)
        if idx > 8:
            ax.set_xlabel(xlabel)
        if idx % 3 == 0:
            ax.set_ylabel(ylabel)

        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
    fig.tight_layout()

    plt.show()

    if filename is not None:
        fig.savefig(f'../media/{filename}.png',
                    dpi=300, bbox_inches='tight')


def individual_token_scores(scores_mat, block=0, xxyy=(90, 110, 75, 95),
                            title='Layer', filename=None):

    x1, x2, y1, y2 = xxyy
    cmap = 'bone_r'
    o = torch.linalg.norm(scores_mat[block], dim=0).numpy()[1:, 1:]
    extent = (0, o.shape[1], 0, o.shape[0])

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(o, extent=extent, cmap=cmap, interpolation='none')
    step = 10
    max_val = int(o.shape[0] * 0.25)
    xticks = np.arange(0, max_val+step, step=step)
    ax.xaxis.tick_top()
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([])
    yticks = np.arange(o.shape[0]-max_val, o.shape[0]+step, step=step)[::-1]
    ax.set_yticks(yticks - 0.5)
    ax.set_yticklabels([])

    axins = zoomed_inset_axes(ax, 4, loc=1)  # zoom = 2
    axins.imshow(o, extent=extent, cmap=cmap, interpolation='none')

    # sub region of the original image
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xticks([])
    axins.set_yticks([])

    mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5')
    ax.set_title('{} {}'.format(title, block+1))

    fig.colorbar(im, fraction=0.046, pad=0.04)

    plt.draw()
    plt.show()

    if filename is not None:
        fig.savefig(f'../media/{filename}.png',
                    dpi=300, bbox_inches='tight')
