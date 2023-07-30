import torch
import numpy as np

from einops import rearrange
from tqdm import tqdm


def fill_patches(indices, weights=None, img_size=224, p=16, z=None):
    """
    indices = [1, 2, 16, 17, 50, 51, 90]
    z = fill_patches(indices, img_size=224, p=16)
    z = np.ma.masked_where(z == 1, z)

    plt.imshow(unnorm_x)
    plt.imshow(z, alpha=0.8, cmap='gray_r')
    plt.axis('off');
    """
    p = (p, p) if isinstance(p, int) else p
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    if z is None:
        z = np.zeros((img_size))

    z = rearrange(z, '(h p1) (w p2) -> (h w) (p1 p2)',
                  p1=p[0], p2=p[1])

    for i in indices:
        weight = 1 if weights is None else weights[i]
        z[i] = weight

    z = rearrange(z, '(h w) (p1 p2) -> (h p1) (w p2)',
                  p1=p[0], p2=p[1],
                  h=img_size[0]//p[0],
                  w=img_size[1]//p[1])
    return z


def mean_norm_inds(model, x, normalize=True):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    layer_out_vec = []
    for b in range(len(model.blocks)):
        model.blocks[b].attn.proj.register_forward_hook(
            get_activation(f'proj_{b}'))

    _ = model(x)

    for b in range(len(model.blocks)):
        layer_out_vec.append(activation[f'proj_{b}'].cpu().detach())

    layer_out_vec = torch.stack(layer_out_vec).permute(
        1, 0, 2, 3)  # B x L x N + 1 x D

    o = torch.linalg.norm(layer_out_vec, dim=3)  # B x L x N + 1
    om = torch.mean(o, dim=1)[:, 1:].numpy()  # B, N (w/o CLS)

    # normlaize between 0 and 1
    if normalize:
        om = (om - np.expand_dims(om.min(axis=1), axis=1)) / \
            (np.expand_dims(om.max(axis=1) - om.min(axis=1), axis=1))

    indices = np.arange(om.shape[1])
    zs = np.zeros((x.shape[0], *x.shape[2:]))
    for i in range(zs.shape[0]):
        zs[i] = fill_patches(indices, weights=om[i],
                             img_size=x.shape[-1], p=16, z=zs[i])
    # (B, N) (B, W, H)
    return om, zs


def token_distribution(model, x, remove_cls=True,
                       zero_self=True, norm=True):
    all_token_scores = []

    for b in range(len(model.module.backbone.transformer.layers)):
        activation = dict()

        def get_activation(name):
            def hook(model, input, output):
                output.retain_grad()
                activation[name] = output
            return hook

        h1 = model.module.backbone.to_patch_embedding.register_forward_hook(get_activation('patch_embed'))
        h2 = model.module.backbone.transformer.layers[b][1].register_forward_hook(get_activation(f'block_{b}'))
        y = model(x)
        
        del y; torch.cuda.empty_cache(); h1.remove(); h2.remove()

        patch_embed = activation['patch_embed']

        token_norms = []
        for i in tqdm(range(activation[f'block_{b}'].shape[1])):  # N
            block = activation[f'block_{b}'][:, i]
            # (1 x D), (1 x N x D) -> (1 x N x D)
            # Actually (for conv patches): (1 x D), (1 x P1 x P2 x D) -> (1 x P1 x P2 x D)
            grad = torch.autograd.grad(block, patch_embed,
                                       grad_outputs=torch.ones_like(block),
                                       retain_graph=True)[0]
            grad = rearrange(grad, 'b ... d -> b (...) d') # (1 x P1 x P2 x D) -> (1 x N x D)
            if zero_self: # zero gradient of self w.r.t. self
                grad[:, i] = 0
            # (1 x N x D) -> (1 x N)
            if norm:
                norms = torch.linalg.vector_norm(
                    grad, dim=2).cpu().detach().numpy()
            else:
                norms = abs(torch.sum(
                    grad, dim=2).cpu().detach().numpy())
            token_norms.append(norms)
        
        # (1 x N x N)
        token_norms = np.stack(token_norms, axis=1)
        all_token_scores.append(token_norms)

    del activation; grad; patch_embed; block; torch.cuda.empty_cache()

    # 1 x B x N x N
    all_token_scores = np.stack(all_token_scores, axis=1)

    if remove_cls: # if N + 1 tokens and remove CLS token 
        # 1 x B x N + 1 x N -> 1 x B x N x N (remove CLS token)
        all_token_scores = all_token_scores[:, :, 1:, :]
    # dim 2 is token (z) and dim 3 is input token (x)
    return all_token_scores