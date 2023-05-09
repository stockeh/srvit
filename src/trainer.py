import os
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from vit.model import ViT

from utils import AverageMeter, Summary


class Trainer:

    def __init__(self, args, xshape, tshape):

        if args.model_name == 'vit':
            self.model = ViT(image_size=xshape[1:], patch_size=args.h_patch,
                             in_chans=xshape[0], out_chans=tshape[0],
                             dim=args.h_dim, depth=args.h_depth, heads=args.h_heads,
                             mlp_dim=args.h_mlp_dim, dim_head=args.h_dim_head)
        else:
            raise NotImplementedError

        self.multiprocessing_distributed = args.multiprocessing_distributed
        self.ngpus_per_node = args.ngpus_per_node
        self.rank = args.rank
        self.distributed = args.distributed
        self.device = torch.device(args.device)
        self.gpu = args.gpu

        if self.distributed and args.device == 'cuda':
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                self.model.cuda(args.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[args.gpu])
                print(
                    f'=> DistributedDataParallel initialization on GPU:{args.gpu}')
            else:
                self.model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model)
                print('=> DistributedDataParallel initialization on GPU(s)')
        elif args.gpu is not None and args.device == 'cuda':
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
            self.device = torch.device(f'cuda:{args.gpu}')
            print('=> Standard initialization on GPU')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model = self.model.to(self.device)
        elif args.device == 'cuda':
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = torch.nn.DataParallel(self.model).cuda()
            print('=> DataParallel initialization on GPU(s)')

        # training params
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.shuffle = args.shuffle
        self.seed = args.seed

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.loss_fn = self.loss_fn.to(self.device)

        # bookkeeping
        self.start_epoch = 0
        self.new_epoch_counter = 0
        self.best_val_metric = float('inf')
        self.train_patience = args.train_patience
        self.resume = args.resume
        self.model_name = args.model_name
        self.best = args.best
        self.ckpt_dir = args.ckpt_dir
        self.data_dir = args.data_dir

    def summary(self):
        print(self.model)
        print(
            f'=> Trainable Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(
            f'[DEBUG] {self.device=} {next(self.model.parameters()).is_cuda=}')

    def _train_one_epoch(self, train_loader, epoch):
        batch_time = AverageMeter(Summary.NONE)
        data_time = AverageMeter(Summary.NONE)
        losses = AverageMeter(Summary.NONE)
        metrics = AverageMeter(Summary.AVERAGE)

        self.model.train()
        end = time.time()
        with tqdm(total=len(train_loader) * self.batch_size, position=0, leave=True) as pbar:
            for i, (X, T) in enumerate(train_loader):
                data_time.update(time.time() - end)

                X = X.to(self.device, non_blocking=True)
                T = T.to(self.device, non_blocking=True)

                # forward
                Y = self.model(X)
                loss = self.loss_fn(Y, T)

                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # metrics
                rmse = torch.sqrt(torch.mean((Y - T)**2))

                losses.update(loss.item(), X.size()[0])
                metrics.update(rmse.item(), X.size()[0])
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_description(
                    (
                        f"Epoch: {epoch + 1} [{i + 1}/{self.n_train_batches}] "
                        f"Time {batch_time.val:.5f} ({batch_time.avg:.5f}) "
                        f"Data {data_time.val:.5f} ({data_time.avg:.5f}) "
                        f"Loss {losses.val:.5f} ({losses.avg:.5f}) "
                        f"Met {metrics.val:.5f} ({metrics.avg:.5f})"
                    )
                )
                pbar.update(X.shape[0])

            return losses.avg, metrics.avg

    def train(self, train_loader, train_sampler, val_loader):

        # load the most recent checkpoint
        if self.resume:
            self._load_checkpoint(best=False)

        self.n_train_batches = len(train_loader)

        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:  # shuffle training data
                train_sampler.set_epoch(epoch)

            train_loss, train_metric = self._train_one_epoch(
                train_loader, epoch)
            val_loss, val_metric = self.eval(val_loader)

            # min < float(inf), max > 0
            is_best = val_metric < self.best_val_metric
            msg1 = "train loss: {:.3f} - train met: {:.3f} "
            msg2 = "- val loss: {:.3f} - val met: {:.3f}"
            if is_best:
                self.new_epoch_counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_metric, val_loss, val_metric
                )
            )
            if not is_best:
                self.new_epoch_counter += 1
            if self.new_epoch_counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return self.best_val_metric

            # check for improvement
            self.best_val_metric = min(val_metric, self.best_val_metric)
            if not self.multiprocessing_distributed or (self.multiprocessing_distributed
                                                        and self.rank % self.ngpus_per_node == 0):
                self._save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "best_val_metric": self.best_val_metric,
                    },
                    is_best,
                )
        print('=> Finished Training.')

        return self.best_val_metric

    @torch.no_grad()
    def eval(self, val_loader, test=False, save=False):

        if test:
            self._load_checkpoint(best=self.best)

        losses = AverageMeter(Summary.NONE)
        metrics = AverageMeter(Summary.AVERAGE)

        if save:
            Ys = []

        self.model.eval()
        for i, (X, T) in enumerate(val_loader):
            X, T = X.to(self.device), T.to(self.device)

            Y = self.model(X)
            loss = self.loss_fn(Y, T)

            # metrics
            rmse = torch.sqrt(torch.mean((Y - T)**2))

            losses.update(loss.item(), X.size()[0])
            metrics.update(rmse.item(), X.size()[0])

            if save:
                Ys.append(Y.detach().cpu())

        if self.distributed:
            losses.all_reduce()
            metrics.all_reduce()

        if test:
            print(
                f"[*] test loss: {losses.avg:.3f} - "
                f"test met: {metrics.avg:.3f}"
            )

        if save:
            Ys = torch.cat(Ys).numpy()
            # save predictions as numpy array
            f = os.path.join(self.data_dir, 'out', self.model_name)
            os.makedirs(f, exist_ok=True)
            np.save(os.path.join(f, f'test_predictions.npy'), Ys)

        return losses.avg, metrics.avg

    def _save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.
        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def _load_checkpoint(self, best=False):
        """Load the best copy of a model.
        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.
        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        else:
            filename = self.model_name + "_ckpt.pth.tar"

        ckpt_path = os.path.join(self.ckpt_dir, filename)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                "[!] No checkpoint found at '{}'".format(ckpt_path)
            )
        print("[*] Loading model from {}".format(ckpt_path))

        if self.gpu is None:
            ckpt = torch.load(ckpt_path)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(self.gpu)
            ckpt = torch.load(ckpt_path, map_location=loc)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_val_metric = ckpt["best_val_metric"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best metric of {:.5f}".format(
                    filename, ckpt["epoch"], ckpt["best_val_metric"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
