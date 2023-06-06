import torch.nn as nn


class Smoother(nn.Module):
    def __init__(self, in_chans, out_chans, backbone, freezeweights=False, hiddens=None):
        super().__init__()
        self.backbone = backbone
        if freezeweights:
            self._freeze_weights()

        self.hiddens = nn.Identity()
        ni = in_chans
        if hiddens is not None:
            assert isinstance(hiddens, list) and len(
                hiddens) > 0, 'Smoother: hiddens was found to be empty, but expected not to be'
            if not isinstance(hiddens[0], int):
                hiddens = [int(h) for h in hiddens]
            layers = []
            for c in hiddens:
                layers.append(nn.Conv2d(
                    ni, c, kernel_size=3, stride=1, padding='same', padding_mode='zeros'))
                layers.append(nn.ReLU())
                ni = c
            self.hiddens = nn.Sequential(*layers)

        self.head = nn.Conv2d(ni, out_chans, kernel_size=3,
                              stride=1, padding='same', padding_mode='zeros')

    def _freeze_weights(self):
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.hiddens(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    s = Smoother(1, 1, None, hiddens=['32', '16'])
    print(s)
