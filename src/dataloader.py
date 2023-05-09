import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset


class FakeData(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.xshape = (4, 224, 224)
        self.tshape = (1, 224, 224)
        self.X = torch.rand((3000, *self.xshape))
        self.T = torch.rand((3000, *self.tshape))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        T = self.T[idx]
        if self.transform is not None:
            X = self.transform(X)
            T = self.transform(T)
        return X, T


class RawGremlin(Dataset):
    def __init__(self, data_dir, data_type, transform=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform

        self.nlon = 1799
        self.nlat = 1059
        self.xdim = 1024
        self.ydim = 1536
        self.xvars = ['ABIC07', 'ABIC09', 'ABIC13', 'GLMGED']
        self.tvar = 'MRMSREFC'

        self.xshape = (len(self.xvars), self.xdim, self.ydim)
        self.tshape = (1, self.xdim, self.ydim)

        self.samples = []
        with open(os.path.join(data_dir, f'{data_type}_samples.txt'), 'r') as f:
            for l in f:
                l = l.rstrip().rpartition('/')
                self.samples.append(l)

        self.XMIN = np.array([[[197.30528259]],
                              [[194.23445129]],
                              [[194.2412262]],
                              [[0.]]])
        self.XMAX = np.array([[[385.55102539]],
                              [[271.90548706]],
                              [[314.81765747]],
                              [[804.44445801]]])
        self.TMIN = np.array([[[-99.]]])
        self.TMAX = np.array([[[81.72222137]]])

    def _read_data(self, filename):
        dtype = np.float32
        count = self.nlat * self.nlon
        shape = (self.nlat, self.nlon)

        with open(filename, 'rb') as f:
            return np.fromfile(f, dtype=dtype, count=count).reshape(shape)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        l = self.samples[idx]
        date = l[0]
        num = l[2]

        # TODO: parallelize this for each variable?
        x = np.zeros((self.xshape[0], self.nlat, self.nlon))
        for i, var in enumerate(self.xvars):
            file = f'{num}_{var}.bin'
            filename = os.path.join(self.data_dir, date, file)
            data = self._read_data(filename)
            data[data < 0] = np.nan
            x[i] = data

        # np.nanmin(x, axis=(1, 2), keepdims=True)
        x -= self.XMIN
        # np.nanmax(x, axis=(1, 2), keepdims=True) + np.finfo(float).eps
        x /= self.XMAX
        # TODO: use median of each channel as nan value
        x = np.nan_to_num(x, nan=0.0, copy=False)

        t = np.expand_dims(self._read_data(os.path.join(
            self.data_dir, date, f'{num}_{self.tvar}.bin')), axis=0)
        t[t == -999.0] = np.nan
        # np.nanmin(t, axis=(1, 2), keepdims=True)
        t -= self.TMIN
        # np.nanmax(t, axis=(1, 2), keepdims=True) + np.finfo(float).eps
        t /= self.TMAX
        t = np.nan_to_num(t, nan=0.0, copy=False)

        # crop to be size of xdim and ydim
        xdiff = (self.xshape[-2] - self.xdim) // 2
        ydiff = (self.xshape[-1] - self.ydim) // 2
        x = x[:, xdiff:xdiff+self.xdim, ydiff:ydiff+self.ydim]
        t = t[:, xdiff:xdiff+self.xdim, ydiff:ydiff+self.ydim]

        x = torch.from_numpy(x).float()
        t = torch.from_numpy(t).float()

        if self.transform is not None:
            x = self.transform(x)
            t = self.transform(t)

        return x, t


class Gremlin(Dataset):
    def __init__(self, data_dir, data_type, transform=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform

        self.samples = []
        v = os.path.join(data_dir, data_type)
        for f in os.listdir(v):
            if f.endswith('.npz'):
                self.samples.append(os.path.join(v, f))
        self.samples.sort()

        self.xshape, self.tshape = self._get_shape()

    def _get_shape(self):
        with np.load(self.samples[0]) as data:
            xshape = np.moveaxis(data['xdata'], -1, 0).shape
            tshape = data['ydata'][np.newaxis, ...].shape
        return xshape, tshape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self.samples[idx]
        with np.load(f) as data:  # C x H x W
            x = np.moveaxis(data['xdata'], -1, 0)
            t = data['ydata'][np.newaxis, ...]

        if self.transform is not None:
            x = self.transform(x)
            t = self.transform(t)

        return x, t


def get_dataset(args):
    """
    return:
        train_dataset: torch.utils.data.Dataset OR None if args.test
        train_sampler: torch.utils.data.distributed.DistributedSampler OR None if args.test
        val_dataset: torch.utils.data.Dataset
        val_sampler: torch.utils.data.distributed.DistributedSampler
    """
    data_name = args.data_name
    data_dir = args.data_dir

    if data_name == 'dummy':
        print("=> Dummy data is used!")
        train_dataset = FakeData()
        val_dataset = FakeData()
    elif data_name == 'gremlin':
        if args.test:
            val_dataset = Gremlin(data_dir, 'test')
        else:
            train_dataset = Gremlin(data_dir, 'train')
            val_dataset = Gremlin(data_dir, 'val')
    else:
        raise ValueError(f'Unknown dataset name: {data_name=}')

    if args.distributed:
        if args.test:
            train_sampler = None
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    if args.test:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(
                train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, train_sampler, val_loader, val_sampler


if __name__ == "__main__":

    # Dummy Data
    args = argparse.Namespace()
    args.test = False
    args.data_name = 'dummy'
    args.data_dir = None
    args.distributed = False
    args.batch_size = 4
    args.workers = 4
    args.seed = 0

    train_loader, train_sampler, val_loader, val_sampler = get_dataset(args)

    print(len(train_loader), train_loader.dataset.xshape)

    for i, (X, T) in enumerate(train_loader):
        print(i, X.shape, T.shape)
        if i > 3:
            break

    # Gremlin Data
    args = argparse.Namespace()
    args.test = False
    args.data_name = 'gremlin'
    args.data_dir = '/s/chopin/l/grad/stock/mlai2es/data/conus3/preprocessed'
    args.distributed = False
    args.batch_size = 4
    args.workers = 4
    args.seed = 0

    train_loader, train_sampler, val_loader, val_sampler = get_dataset(args)

    print(len(train_loader), train_loader.dataset.xshape)

    for i, (X, T) in enumerate(train_loader):
        print(i, X.shape, T.shape)
        if i > 10:
            break
