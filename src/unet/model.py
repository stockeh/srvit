import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.out_chans = out_chans
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, padding='same')
        self.conv2 = nn.Conv2d(out_chans, out_chans, 3, padding='same')

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, in_chans, channels=[64, 128, 256]):
        super().__init__()
        self.channels = [in_chans] + channels
        self.enc_blocks = nn.ModuleList(
            [Block(self.channels[i], self.channels[i+1]) for i in range(len(self.channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, channels=[256, 128, 64]):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels)-1)])
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chans, out_chans,
                 channels=[64, 128, 256], **kwargs):
        super().__init__()
        self.encoder = Encoder(in_chans, channels)
        self.decoder = Decoder(channels[::-1])
        self.head = nn.Conv2d(
            channels[::-1][-1], out_chans, 1, padding='same')

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z[::-1][0], z[::-1][1:])
        out = self.head(out)
        return out


if __name__ == '__main__':
    print('-------- BLOCK TEST --------')
    b = Block(1, 64)
    x = torch.randn(1, 1, 256, 256)
    print(b(x).shape)

    print('-------- ENCODER TEST --------')

    channels = [64, 128, 256]

    x = torch.randn(1, 3, 256, 256)
    e = Encoder(x.shape[1], channels)
    z = e(x)
    for f in z:
        print(f.shape)

    print('-------- DECODER TEST --------')
    d = Decoder(channels[::-1])
    x = torch.randn(1, channels[-1],
                    x.shape[2] // (2**(len(channels) - 1)),
                    x.shape[3] // (2**(len(channels) - 1)))
    print(d(x, z[::-1][1:]).shape)

    print('-------- UNET TEST --------')
    x = torch.randn(1, 3, 256, 256)
    u = UNet(x.shape[1], 1, channels=[64, 128, 256])
    y = u(x)
    print(y.shape)
    assert y.shape[2:] == x.shape[2:], 'input and output shapes must match'
    print(u)
