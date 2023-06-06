from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

# from pytorch_msssim import MS_SSIM, SSIM


class Lion(Optimizer):
    """ EvoLved Sign Momentum
    Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., ... 
    & Le, Q. V. (2023). Symbolic Discovery of Optimization Algorithms. 
    arXiv preprint arXiv:2302.06675.

    https://github.com/lucidrains/lion-pytorch
    https://github.com/google/automl/blob/master/lion/lion_pytorch.py
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

    def exists(self, val):
        return val is not None

    def update_fn(self, p, grad, exp_avg, lr, wd, beta1, beta2):
        # stepweight decay
        p.data.mul_(1 - lr * wd)

        # weight update
        update = exp_avg.clone().lerp_(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if self.exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: self.exists(p.grad), group['params']):

                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group[
                    'weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.update_fn(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss


class MSEGenexp(nn.Module):
    def __init__(self, weight=(1.0, 0.0, 0.0)):
        super().__init__()
        self.weight = weight

    def forward(self, Y, T):
        return torch.mean(torch.multiply(
            torch.multiply(self.weight[0], torch.exp(torch.multiply(
                self.weight[1], torch.pow(T, self.weight[2])))),
            torch.square(torch.subtract(Y, T))))


# class SSIMMSE(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)

#     def forward(self, Y, T):
#         mse = self.mse(Y, T)
#         ssim = 1-self.ssim(Y, T)
#         return mse + 0.1*ssim


# # SSIM_Loss(data_range=1.0, size_average=True, channel=1)
# class MS_SSIM_Loss(MS_SSIM):
#     def forward(self, Y, T):
#         return nn.MSELoss.forward(Y, T) + 0.01*(1 - super(MS_SSIM_Loss, self).forward(Y, T))


# class SSIM_Loss(SSIM):
#     def forward(self, Y, T):
#         return 100*(1 - super(SSIM_Loss, self).forward(Y, T))
