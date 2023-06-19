import os
import sys
import argparse
import warnings
import random
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from trainer import Trainer
import dataloader

parser = argparse.ArgumentParser(description='PyTorch GREMLIN Attention')

parser.add_argument('-e', '--experiment', required=True, type=str,
                    help='Experiment name [must be unique]')

# Model configuration
parser.add_argument('-m', '--model_name', required=True, type=str,
                    help='Name of model to use')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--shuffle', action='store_false',  # default true
                    help='shuffle training data')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training')
parser.add_argument('--train_patience', type=int, default=50,
                    help='Patience for early stopping')
parser.add_argument('-l', '--loss', type=str,
                    help='Loss function to use [genexp|mse]')

# Hyperparameter configuration
parser.add_argument('--h-patch', default=16, type=int, metavar='N',
                    help='square patch size, e.g. 16 (16 x 16)')
parser.add_argument('--h-dim', default=256, type=int, metavar='N',
                    help='model dimension d, e.g., n x d')
parser.add_argument('--h-depth', default=6, type=int, metavar='N',
                    help='number of transformer blocks')
parser.add_argument('--h-heads', default=12, type=int, metavar='N',
                    help='number of heads in each block')
parser.add_argument('--h-mlp-dim', default=512, type=int, metavar='N',
                    help='mlp dimension m, e.g., n x d -> n x m -> n x d')
parser.add_argument('--h-dim-head', default=64, type=int, metavar='N',
                    help='inner dimension of q,k,v b, e.g. n x d -> n x b -> n x d')

# Finetuning configuration
parser.add_argument('--complete', action='store_true',  # default false
                    help='Train complete model')
parser.add_argument('--finetune', action='store_true',  # default false
                    help='Train with finetuning')
parser.add_argument('--backbone', type=str,
                    default='Name of model, e.g., model.pth.tar',
                    help='PyTorch model to load as backbone to finetuning')
parser.add_argument('--finetune-hiddens', nargs='+',
                    help='Convolutional filters to add to fine-tuning '
                    'model as a list of integers, e.g. 32 64 128. '
                    'If not specified, no additional layers will be added')

# UNet configuration
parser.add_argument('--unet-hiddens', nargs='+',
                    help='Convolutional filters to add to the unet '
                    'model as a list of integers, e.g. 32 64 128. '
                    'If not specified, no additional layers will be added')
parser.add_argument('--unet-skip', action='store_true',  # default false
                    help='use skip connections for unet')

# Distributed configuration
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:30002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Directory configuration
parser.add_argument('--data-name', type=str, default='gremlin',
                    help='Name of dataset to load')
parser.add_argument('--data-dir', type=str,
                    default='/s/chopin/l/grad/stock/mlai2es/data/conus3/preprocessed',
                    help='Directory in which data is stored')
parser.add_argument('--ckpt-dir', type=str, default='../ckpt',
                    help='Directory in which to save model checkpoints')

# Training configuration
parser.add_argument('--cuda', action='store_true',  # default false
                    help='use CUDA')
parser.add_argument('--resume', action='store_true',  # default false
                    help='resume from last checkpoint')
parser.add_argument('--best', action='store_false',  # default true
                    help='Load best model or most recent for testing')

# Testing configuration
parser.add_argument('--test', action='store_true',  # default false
                    help='Test the model on the test set')
parser.add_argument('--save', action='store_true',  # default false
                    help='Save the results of the test set')


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs of the current node.
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int(
            (args.workers + ngpus_per_node - 1) / ngpus_per_node)

    train_loader, train_sampler, \
        val_loader, val_sampler = dataloader.get_dataset(args)
    print(
        f"=> Finished loading data: {args.data_name} "
        f"for {'training' if not args.test else 'testing'} "
        f"with {len(train_loader.dataset) if not args.test else len(val_loader.dataset)} samples "
        f"X.shape={val_loader.dataset.xshape}, T.shape={val_loader.dataset.tshape}."
    )

    trainer = Trainer(args, val_loader.dataset.xshape,
                      val_loader.dataset.tshape)

    trainer.summary()

    if not args.test:
        trainer.train(train_loader, train_sampler, val_loader)
    else:
        trainer.eval(val_loader, args.test, args.save)


def main(args):
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        args.device = 'cuda' if args.cuda else 'cpu'
        args.ngpus_per_node = torch.cuda.device_count()
        print(f'{args.ngpus_per_node} GPU(s) available')
    else:
        args.ngpus_per_node = 1
        args.device = 'cpu'
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.gpu, args.ngpus_per_node, args)


if __name__ == '__main__':
    """
    Usage:
    Dummy data:
        python main.py -m vit --cuda --epochs 50 -b 8 --data-name dummy

    Single node, multiple GPUs:
        python main.py -m vit --cuda --epochs 50 -b 8 --data-name dummy --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
    """
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main(args=args)
