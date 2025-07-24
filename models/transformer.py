from timm.models.layers import Mlp
import torch.nn as nn
import torch
import math
from .irpe import build_rpe, get_rpe_config


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, use_irpe=True, drop_ratio=0., skip=0):
        super().__init__()
        assert embedding_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.drop_ratio = drop_ratio
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        rpe_config = get_rpe_config(
            ratio=1.9,
            method="product",
            mode='ctx',
            shared_head=True,
            skip=skip,
            rpe_on='qkv',
        )
        if use_irpe:
            self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=self.head_dim, num_heads=num_heads)
        else:
            self.rpe_q, self.rpe_k, self.rpe_v = None, None, None
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v):
        B, N_q, C = q.shape
        _, N_k, _ = k.shape
        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        if self.rpe_k is not None:
            attn += self.rpe_k(q, int(math.sqrt(N_q)), int(math.sqrt(N_q)))
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, int(math.sqrt(N_k)), int(math.sqrt(N_k))).transpose(2, 3)
        # Use DropKey 0.05~0.1
        if self.drop_ratio:
            d_r = torch.ones_like(attn) * self.drop_ratio
            attn = attn + torch.bernoulli(d_r) * -1e12
        attn = attn.softmax(dim=-1)
        # B H N_q N_k  B H N_k head_dim = B H N_q head_dim
        out = attn @ v
        if self.rpe_v is not None:
            out += self.rpe_v(attn, int(math.sqrt(N_q)), int(math.sqrt(N_k)))
        out = out.transpose(1, 2).contiguous().reshape(B, N_q, C)
        out = self.proj(out)
        return out, attn


class TransformerSelfEncoderLayer(nn.Module):

    def __init__(self, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=True, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, use_irpe=True, drop_ratio=0., skip=0):
        super().__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.norm2 = norm_layer(embedding_dim)
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.mlp = Mlp(in_features=embedding_dim,
                       hidden_features=int(embedding_dim * mlp_ratio),
                       act_layer=act_layer)
        self.attn = MultiheadAttention(embedding_dim=embedding_dim,
                                       num_heads=num_heads,
                                       use_irpe=use_irpe,
                                       drop_ratio=drop_ratio,
                                       skip=skip)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        x = x + self.attn(q, k, v)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(
            self, embedding_dim, num_heads, mlp_ratio=4., qkv_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            use_irpe=True, drop_ratio=0.):
        super().__init__()
        self.cross_attn = MultiheadAttention(embedding_dim=embedding_dim,
                                             num_heads=num_heads,
                                             use_irpe=use_irpe,
                                             drop_ratio=drop_ratio)
        self.norm0_0 = norm_layer(embedding_dim)
        self.norm0_1 = norm_layer(embedding_dim)
        self.norm1_0 = norm_layer(embedding_dim)
        self.norm1_1 = norm_layer(embedding_dim)
        self.mlp_0 = Mlp(in_features=embedding_dim, hidden_features=int(embedding_dim * mlp_ratio), act_layer=act_layer)
        self.mlp_1 = Mlp(in_features=embedding_dim, hidden_features=int(embedding_dim * mlp_ratio), act_layer=act_layer)
        self.qkv0 = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.qkv1 = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)

    def forward(self, x0, x1):
        B, N_0, C = x0.shape
        _, N_1, _ = x1.shape
        qkv0 = self.qkv0(self.norm0_0(x0)).reshape(B, N_0, 3, C).permute(2, 0, 1, 3)
        q0, k0, v0 = qkv0.unbind(0)
        qkv1 = self.qkv1(self.norm0_1(x1)).reshape(B, N_1, 3, C).permute(2, 0, 1, 3)
        q1, k1, v1 = qkv1.unbind(0)
        x0 = x0 + self.cross_attn(q0, k1, v1)[0]
        x1 = x1 + self.cross_attn(q1, k0, v0)[0]
        x0 = x0 + self.mlp_0(self.norm1_0(x0))
        x1 = x1 + self.mlp_1(self.norm1_1(x1))
        return x0, x1











