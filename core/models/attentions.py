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
    def __init__(self, dim, heads = 16, dim_head = 128, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attention_dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = self.attention_dropout(dots)
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

class Attention_Module(nn.Module):
    def __init__(self, *, input_size, patch_size, channels, patch_mlp_dim, depth, heads, mlp_dim,  dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(input_size)
        patch_height, patch_width = pair(int(patch_size))

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # change this later
        self.patch_dim = channels * patch_height * patch_width
        out_channel = int((patch_size**2)*channels)
        self.to_patch_embedding = nn.Conv2d(in_channels = channels, out_channels = patch_mlp_dim,
                                            kernel_size = patch_size, stride = patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, self.patch_dim))
        self.pos_linear = nn.Linear(self.patch_dim, patch_mlp_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_mlp_dim))


        self.dropout = nn.Dropout(emb_dropout)
        # original
        self.transformer = Transformer(patch_mlp_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.layer_norm = nn.LayerNorm(out_channel)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x.flatten(2).transpose(-1, -2)

        b, n, _ = x.shape
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        pos_emb = self.pos_linear(self.pos_embedding)
        x += pos_emb[:, :(n+1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        return x

class Attention_Network(nn.Module):
    def __init__(self, *, scale, patch_size, patch_mlp_dim, depth, heads, mlp_dim, dropout_rate):
        super().__init__()
        self.attention_modules = nn.ModuleList()
        self.scale = scale
        self.patch_mlp_dim = patch_mlp_dim
        for i in range(self.scale):
            input_size = int((2**i)*16)
            channel = int(1024/(2**i))
            self.attention_modules.append(Attention_Module(input_size = input_size, patch_size = patch_size[i], channels = channel,
                                                           patch_mlp_dim = self.patch_mlp_dim, depth = depth, heads = heads, mlp_dim = mlp_dim,
                                                           dropout = dropout_rate, emb_dropout = dropout_rate))
        self.normal = nn.LayerNorm(patch_mlp_dim)
        self.fusion = nn.Linear(self.scale, 1)


    def forward(self, x):
        out = []
        for i in range(len(self.attention_modules)):
            out.append(self.attention_modules[i](x[i]))

        out = torch.stack(out, dim=1)
        out = self.normal(out)
        out = self.fusion(torch.permute(out, (0, 2, 1)))
        out = out.view(-1, self.patch_mlp_dim)
        return out

class Regressor(nn.Module):
    def __init__(self, *, patch_mlp_dim, dropout_rate_regressor, output):
        super().__init__()
        self.mlp_head = nn.Sequential(nn.Dropout(dropout_rate_regressor),
                                      nn.Linear(patch_mlp_dim, int(patch_mlp_dim * 0.5)),
                                      nn.GELU(),
                                      nn.Dropout(dropout_rate_regressor),
                                      nn.Linear(int(patch_mlp_dim * 0.5), output))

    def forward(self, x):
        out = self.mlp_head(x)
        return out


class Classifier(nn.Module):
    def __init__(self, *, patch_mlp_dim, dropout_rate_classifier, class_num):
        super().__init__()
        self.mlp_head_iou = nn.Sequential(nn.Dropout(dropout_rate_classifier),
                                          nn.GELU(),
                                          nn.Linear(patch_mlp_dim, class_num))

        self.mlp_head_prob = nn.Sequential(nn.Dropout(dropout_rate_classifier),
                                           nn.GELU(),
                                           nn.Linear(patch_mlp_dim, class_num))

    def forward(self, x):
        out_iou = self.mlp_head_iou(x)
        out_prob = self.mlp_head_prob(x)
        out = [out_iou, out_prob]
        return out