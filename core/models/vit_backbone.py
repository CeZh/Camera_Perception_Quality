import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., **kwargs):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        if kwargs:
            self.super_pixel = kwargs['super_pixel']
            size_table = int(self.super_pixel['original_height']*self.super_pixel['original_width'])
            self.size_encode = nn.Parameter(torch.randn(1, size_table, 1))# add size encoding
            self.posx_encode = nn.Parameter(torch.randn(1, self.super_pixel['original_width'], 1))
            self.posy_encode = nn.Parameter(torch.randn(1, self.super_pixel['original_height'], 1))
            self.to_super_embedding = nn.Sequential(nn.Linear(9, self.super_pixel['att_dim']),
                                                    nn.Dropout(emb_dropout),
                                                    nn.Linear(self.super_pixel['att_dim'], self.super_pixel['att_dim']))
            self.super_embedding = nn.Parameter(torch.randn(1, self.super_pixel['segments']+1, self.super_pixel['att_dim']))
            self.super_transformer = Transformer(self.super_pixel['att_dim'], depth, heads, dim_head, mlp_dim, dropout)
            self.super_dropout = nn.Dropout(emb_dropout)
            self.super_cls = nn.Parameter(torch.randn(1, 1, self.super_pixel['att_dim']))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img, **kwargs):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        if kwargs:
            super_img = kwargs['super_x']
            position = kwargs['super_pos'][:, :, :2]
            size = kwargs['super_pos'][:, :, 2].unsqueeze(2)
            size_encoder = self.size_encode.repeat(b, 1, 1)
            size_encoder = size_encoder[torch.arange(size_encoder.shape[0]), size.long()][:, :, 0]
            posx_encoder = self.posx_encode.repeat(b, 1, 1)
            posx_encoder = posx_encoder[torch.arange(posx_encoder.shape[0]), position[:, :, 0].unsqueeze(2).long()][:, :, 0]
            posy_encoder = self.posy_encode.repeat(b, 1, 1)
            posy_encoder = posy_encoder[torch.arange(posy_encoder.shape[0]), position[:, :, 1].unsqueeze(2).long()][:, :, 0]
            super_features = torch.cat([super_img, size_encoder, posx_encoder, posy_encoder], dim=2)

            _, super_n, _ = super_img.shape
            super_img = self.to_super_embedding(super_features)
            super_cls_token = repeat(self.super_cls, '() n d -> b n d', b = b)
            super_img = torch.cat((super_cls_token, super_img), dim=1)
            super_img += self.super_embedding[:, :(super_n+1)]
            super_img = self.super_dropout(super_img)
            super_img = self.super_transformer(super_img)
            super_img = super_img[:, 0]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = self.to_latent(x)
        # x = self.linear_downsize(x)
        # x = torch.cat((x, super_img), dim=1)
        if kwargs:
            output = {'img': x, 'super_img': super_img}
        else:
            output = self.to_latent(x)
        return output