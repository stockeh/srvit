import torch
import torch.nn as nn
import torch.nn.functional as F

class Smoother(nn.Module):
    def __init__(self, in_chans, hiddens, out_chans, backbone):
        super().__init__()
        self.net = nn.Sequential(
            backbone,
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=0, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    pass