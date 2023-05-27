import torch.nn as nn


class Smoother(nn.Module):
    def __init__(self, in_chans, out_chans, backbone, hiddens=None):
        super().__init__()
        self.backbone = backbone
        self._freeze_weights()

        self.hiddens = nn.Identity()
        ni = in_chans
        if hiddens is not None:
            assert isinstance(hiddens, list) and len(
                hiddens) > 0, 'Smoother: hiddens was found to be empty, but expected not to be'
            self.hiddens = nn.ModuleList([])
            for c in hiddens:
                self.model.append(nn.Conv2d(
                    ni, c, kernel_size=3, stride=1, padding='same', padding_mode='zeros'))
                self.model.append(nn.ReLU())
                ni = c

        self.head = nn.Conv2d(ni, out_chans, kernel_size=3,
                              stride=1, padding='same', padding_mode='zeros')

    def _freeze_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.hiddens(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    pass
