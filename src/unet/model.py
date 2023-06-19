import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.out_chans = out_chans
        self.conv = nn.Conv2d(in_chans, out_chans, 3, padding='same')

    def forward(self, x):
        return F.relu(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, in_chans, channels, skip):
        super().__init__()
        self.skip = skip
        self.channels = [in_chans] + channels
        self.convs = nn.ModuleList(
            [Block(self.channels[i], self.channels[i+1]) for i in range(len(self.channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = []
        for block in self.convs:
            x = block(x)
            if self.skip:
                out.append(x)
            x = self.pool(x)
        return x, out

class Decoder(nn.Module):
    def __init__(self, channels, skip):
        super().__init__()
        self.skip = skip
        self.channels = [channels[0]] + channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.convs = nn.ModuleList(
            [Block(self.channels[i] * (2 if i != 0 and skip else 1), 
            self.channels[i+1]) for i in range(len(self.channels)-1)])

    def forward(self, x, o):
        if not self.skip:
            o = [None] * len(self.convs)
        for block, z in zip(self.convs, o):
            x = block(x)
            x = self.upsample(x)
            if self.skip:
                x = torch.cat([x, z], dim=1)
        return x

class UNet(nn.Module):
    def __init__(self, in_chans, out_chans,
                 channels=[64, 128, 256], skip=True, **kwargs):
        super().__init__()
        if not isinstance(channels[0], int):
            channels = [int(h) for h in channels]
        self.encoder = Encoder(in_chans, channels, skip)
        self.decoder = Decoder(channels[::-1], skip)
        self.head = nn.Conv2d(
            channels[::-1][-1] * (2 if skip else 1), out_chans, 1, padding='same')

    def forward(self, x):
        x, o = self.encoder(x)
        x = self.decoder(x, o[::-1])
        x = self.head(x)
        return x


if __name__ == '__main__':
    print('-------- BLOCK TEST --------')
    b = Block(1, 64)
    x = torch.randn(1, 1, 256, 256)
    print(b(x).shape)

    print('-------- ENCODER TEST --------')

    channels = [32, 64, 128]

    x = torch.randn(1, 4, 256, 256)
    e = Encoder(x.shape[1], channels, skip=True)
    z, o = e(x)
    print(z.shape)

    print('-------- DECODER TEST --------')
    d = Decoder(channels[::-1], skip=True)
    print(d(z, o[::-1]).shape)

    print('-------- UNET TEST --------')
    x = torch.randn(1, 4, 256, 256)
    u = UNet(x.shape[1], 1, channels=[32, 32, 32], skip=False)
    y = u(x)
    print(y.shape)
    assert y.shape[2:] == x.shape[2:], 'input and output shapes must match'
    print(u)
    print('Number of parameters: {}'.format(sum(p.numel() for p in u.parameters() if p.requires_grad)))