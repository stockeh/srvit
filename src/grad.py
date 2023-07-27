import os
import argparse
import numpy as np

from tqdm import tqdm
from skimage import filters
from tqdm.contrib.concurrent import thread_map


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    default='/mnt/conus3/jason_conus3/test',
                    help='data directory')
parser.add_argument('--results', type=str,
                    default='/mnt/mlnas01/stock/',
                    help='results directory')
parser.add_argument('--model', type=str,
                    choices=['mrms', 'complete01-vit', 'unet01-unet'],
                    help='model name')
parser.add_argument('--kind', type=str, default='test',
                    choices=['train', 'val', 'test'],
                    help='model name')
parser.add_argument('-b', '--batch-size', type=int, default=100,
                    help='batch size for processing')
parser.add_argument('--num-workers', type=int, default=8,
                    help='number of workers for processing')

ymax = 60.0  # value from Hilburn et al. (2020)

################################################################

def load_t(filename):
    global ymax
    with np.load(filename) as data:  # C x H x W
        # x = np.moveaxis(data['xdata'], -1, 0)
        t = data['ydata'][np.newaxis, ...] * ymax
    return t

def load_y(filename):
    global ymax
    return np.load(filename) * ymax
    

def grad(args):
    filename, mrms = args
    data = load_t(filename) if mrms else load_y(filename)
    # Average Magnitude of the Gradient
    # Edge magnitude is computed as:
    #     sqrt(Gx^2 + Gy^2)
    return np.mean(filters.sobel(data))


def main(args):
    samples = []
    mrms = True if args.model == 'mrms' else False
    if mrms:
        v = args.input
        for f in os.listdir(v):
            if 'regA' in f and f.endswith('.npz'):
                samples.append(os.path.join(v, f))
    else:
        if args.kind != 'test':
            raise ValueError('kind must be test (for now)')
        v = os.path.join(args.results, args.model)
        for f in os.listdir(v):
            if f.endswith('.npy'):
                samples.append(os.path.join(v, f))
    samples.sort()

    grads = np.zeros((len(samples)))
    batch_size = args.batch_size
    n_batches = int(np.ceil(len(samples) / batch_size))
    for i in tqdm(range(n_batches)):
        batch = samples[i*batch_size:(i+1)*batch_size]
        grads[i*batch_size:(i+1)*batch_size] = thread_map(grad, zip(batch, [mrms]*len(batch)), 
                                                          max_workers=args.num_workers, 
                                                          leave=False, total=len(batch))

    output_file = os.path.join(args.results,
                         f'gradient_magnitude_{args.model}_{args.kind}.npy')
    np.save(output_file, grads)

    print(f'=> results saved to {output_file}')

if __name__ == '__main__':
    """
    Usage:
        python grad.py --model mrms
        python grad.py --input /mnt/conus3/jason_conus3/test --results /mnt/mlnas01/stock/ --model mrms
    """
    args = parser.parse_args()
    main(args)
