import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

######### helpers #########


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device),
                          torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

######### classes #########


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """
    Inspired by: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
        Better plain ViT baselines for ImageNet-1k: https://arxiv.org/pdf/2205.01580.pdf

    """

    def __init__(self, *, image_size, patch_size=(16, 16),
                 in_chans=4, out_chans=1, dim=512, depth=6, heads=12,
                 mlp_dim=512, dim_head=64):
        super().__init__()
        self.image_size = image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.num_patches = (image_height // patch_height) * \
            (image_width // patch_width)

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=self.patch_size,
                      stride=self.patch_size),
            Rearrange('b d p1 p2 -> b p1 p2 d'),
            nn.LayerNorm(dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_chans*patch_height*patch_width, bias=False),
            Rearrange("b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                      p1=patch_height,
                      p2=patch_width,
                      h=image_height // patch_height,
                      w=image_width // patch_width)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # b, p1, p2, d
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe  # b, p1 * p2, d
        x = self.transformer(x)  # b, p1 * p2, d
        x = self.head(x)  # b, c, h, w
        return x


if __name__ == '__main__':
    x = torch.randn(1, 4, 256, 256)
    model = ViT(image_size=x.shape[2:], patch_size=8,
                in_chans=x.shape[1], out_chans=1, dim=512,
                depth=6, heads=12, mlp_dim=512, dim_head=64)
    print(model)
    print(
        f'=> Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    y = model(x)
    print(y.shape)
