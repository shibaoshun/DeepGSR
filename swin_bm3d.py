# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from torch_dct import dct_2d, idct_2d, dct, idct
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import common
from thop import profile
import cv2
import numpy as np
import numbers
from einops import rearrange


try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B,N, H, W, C = x.shape
    x = x.view(B,N, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B,N,-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B, Nwav,Nwin, _, _, C = windows.shape
    x = windows.view(B, Nwav, H // window_size, W // window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1,2, 4, 3, 5, 6).contiguous().view(B, Nwav, H, W, -1)
    return x


class StructuralSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(StructuralSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        batch_size1,Nwav1, Nwin1, seq_len1, dim1, embed_dim1 = x.size()
        batch_size2,Nwav2, Nwin2, seq_len2, dim2, embed_dim2 = y.size()

        # Project to queries, keys, and values
        q = self.q_proj(x).view(batch_size1,Nwav1, Nwin1, seq_len1, dim1, self.num_heads, self.head_dim).permute(0, 1,2, 5, 3, 4,
                                                                                                       6)
        k = self.k_proj(y).view(batch_size2,Nwav2, Nwin2, seq_len2, dim2, self.num_heads, self.head_dim).permute(0, 1,2, 5, 3, 4,
                                                                                                       6)
        v = y.view(batch_size2,Nwav2, Nwin2, seq_len2, dim2, self.num_heads, self.head_dim).permute(0, 1,2, 5, 3, 4,
                                                                                          6)
        # Scaled attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(4, 5).reshape(batch_size1,Nwav1, Nwin1, seq_len1, embed_dim1)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out

class ChannelAttention(nn.Module):  # SE蓝色框里的内容
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, mid_ch=32, memory_blocks=128):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.subnet = nn.Sequential(
            nn.Linear(num_feat, mid_ch),
        )
        self.upnet = nn.Sequential(
            nn.Linear(mid_ch, num_feat),
            nn.Sigmoid())
        self.mb = torch.nn.Parameter(
            torch.randn(mid_ch, memory_blocks))  # 随机生成的可学习矩阵  传入的指定了大小memory_blocks应该和patch_size一样
        self.low_dim = mid_ch

    def forward(self, x):
        b,N,Nwin, p, n, c = x.shape
        t = x.view(-1,n,c)
        y = self.pool(t).squeeze(-1)

        low_rank_f = self.subnet(y).unsqueeze(2)

        mbg = self.mb.unsqueeze(0).repeat(b*N*Nwin*p, 1, 1)
        f1 = (low_rank_f.transpose(1, 2)) @ mbg
        f_dic_c = F.softmax(f1 * (int(self.low_dim) ** (-0.5)), dim=-1)  # get the similarity information
        y1 = f_dic_c @ mbg.transpose(1, 2)
        y2 = self.upnet(y1)
        y2v = y2.view(b,N,Nwin, p, n, -1)
        out = x * y2v
        return out

class group_change(nn.Module):
    def __init__(self, d=96):
        super(group_change, self).__init__()
        self.sim = 100
        self.gch = self.sim+1
        self.lch = 80
        self.Lowrank=ChannelAttention(num_feat=self.gch, mid_ch=self.lch , memory_blocks=1)
        self.SS_attn = StructuralSelfAttention(embed_dim=d, num_heads=1)

    def forward(self, attn, v):
        topk_values, topk_indices = torch.topk(attn, k=self.sim, dim=-1)
        binary_mask = torch.zeros_like(attn)
        binary_mask.scatter_(-1, topk_indices, 1)
        attn_transposed = binary_mask.permute(0,1,2, 4, 5, 3)
        v_expanded = v.expand(-1,-1, -1,attn_transposed.size(4), -1, -1)
        result = v_expanded * attn_transposed
        non_zero_mask = attn_transposed.sum(dim=-1) != 0

        #计算过滤后元素的数量
        filtered_elements = result[non_zero_mask]
        filtered_result = filtered_elements.view(result.size(0), result.size(1), result.size(2), result.size(3),topk_indices.size(-1), result.size(-1))
        #filtered_result
        group3D = torch.cat((v.permute(0, 1,2, 4, 3, 5), filtered_result), dim=4)

        #low rank filter
        group3D_ = self.Lowrank(group3D)
        x = group3D_[:, :, :, :, :1, :]
        y = group3D_[:, :, :, :, 1:, :]

        #aggregating group into patch
        output = self.SS_attn(x, y)

        return output

class WA_LGSR(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.group = group_change(d=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_,Nwav ,Nwin , N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_,Nwav,Nwin , N, 2, self.num_heads, C // self.num_heads).permute(0, 1,2,4, 5, 3, 6)
        q, k = qkv[:,:,:, 0], qkv[:, :,:,1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        xin = torch.unsqueeze(x, 3)

        x = self.group(attn, xin)  # 滤波加合成图像

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.permute(0,1,2, 5, 3, 4)
        x = x.unfold(4, self.patch_size, 2).unfold(5, self.patch_size, 2)

        x = x.permute(0,1,2,4,5,3,6,7).flatten(5).permute(0,1,2,5,3,4).contiguous()

        x = x.flatten(4).transpose(3, 4).contiguous()

        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default:1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, window_size=8,norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.window_size = window_size
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = [img_size[0] // 2, img_size[1] // 2]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        x = F.fold(x.transpose(1, 2).contiguous(),
                   output_size=(self.patch_size*self.window_size ,self.patch_size*self.window_size),
                   kernel_size=(self.patch_size, self.patch_size),
                   stride=2)  # B, C, W, H
        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False, img_size=512, patch_size=4, embed_dim=96):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.LGSR = WA_LGSR(
            self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, drop_path=drop_path)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.fused_window_process = fused_window_process

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,window_size=self.window_size,
            norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=0.)
        self.patch_size=patch_size

        self.ch = embed_dim//(self.patch_size*self.patch_size)
        self.norm1 = LayerNorm(self.ch, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(self.ch, LayerNorm_type='WithBias')

        self.convfin = nn.Sequential(nn.Conv2d(self.ch, self.ch*16, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                    nn.Conv2d(self.ch*16, self.ch*16, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.ch * 16, self.ch * 16, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.ch * 16, self.ch * 16, 3, 1, 1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.ch*16, self.ch, 3, 1, 1))

    def _padding(self, x, scale):
        delta_H = 0
        delta_W = 0
        if x.shape[2] % scale != 0:
            delta_H = scale - x.shape[2] % scale
            x = F.pad(x, (0, 0, 0, delta_H), 'reflect')
        if x.shape[3] % scale != 0:
            delta_W = scale - x.shape[3] % scale
            x = F.pad(x, (0, delta_W, 0, 0), 'reflect')
        return x, delta_H, delta_W

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)

        #shifting image
        Bo, _, Ho, Wo = x.shape
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size*self.patch_size, -self.shift_size*self.patch_size), dims=(2, 3))
        else:
            shifted_x = x

        #wavelet transform
        x, delta_H, delta_W = self._padding(shifted_x, 3)
        x = self.DWT(x)
        _, _, wH, wW = x.shape
        ch=self.dim//(self.patch_size*self.patch_size)
        x = x.view(Bo, -1,  wH, wW, ch)

        #split windows
        x_windows = window_partition(x, self.window_size*self.patch_size)

        # split image into non-overlapping patches
        x = self.patch_embed(x_windows)
        x = self.pos_drop(x)

        H, W= (wH-self.patch_size)//1+1, (wW-self.patch_size)//1+1
        B, Nwav, Nwin, Np, C =x.shape
        x_windows = x

        #grouping and aggregating
        attn_windows = self.LGSR(x_windows)

        attn_windows = attn_windows.view(-1, Np, C)

        # merge patches into image
        attn_windows = self.patch_unembed(attn_windows)
        attn_windows = attn_windows.view(B, Nwav, Nwin, self.window_size*self.patch_size,self.window_size*self.patch_size,ch)

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size*self.patch_size, wH, wW)

        #inverse wavelet transform
        x = shifted_x.reshape(Bo, -1, wH, wW)
        x = self.IWT(x)
        shifted_x = x[:, :, :Ho, :Wo]

        # reverse cyclic shift  图像偏移回原位置
        if self.shift_size > 0:
            if not self.fused_window_process:
                x = torch.roll(shifted_x, shifts=(self.shift_size*self.patch_size, self.shift_size*self.patch_size), dims=(2, 3))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        # FFN
        x = x +self.drop_path(self.convfin(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, img_size=512, patch_size=4, embed_dim=96):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        self.ch = embed_dim//(patch_size*patch_size)
        self.conv = nn.Conv2d(self.ch, self.ch, 3, 1, 1)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.conv(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape  # false
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(
                self.num_layers):  # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None  num_layers=4 #4个串联的
            layer = BasicLayer(dim=int(embed_dim),
                               input_resolution=(img_size[0]//patch_size,img_size[1]//patch_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
            self.layers.append(layer)

        self.ch = embed_dim // (patch_size * patch_size)
        self.conv_first = nn.Conv2d(num_in_ch, self.ch, 3, 1, 1)
        self.conv_after_body = nn.Conv2d(self.ch, self.ch, 3, 1, 1)
        self.conv_last = nn.Conv2d(self.ch, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        x_out = x + self.conv_last(res)
        return x_out


class BM3DTrans(nn.Module):
    def __init__(self, img_size=[], patch_size=3, in_chans=1, num_classes=1000, embed_dim=108,
                 depths=[2, 2, 2], num_heads=[1, 1, 1],  window_size=10, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                 fused_window_process=False, img_range=1.):
        """
        初始化 BM3DTrans 类并设置 Swin Transformer 模型参数。
        """
        super(BM3DTrans, self).__init__()
        self.model = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            fused_window_process=fused_window_process
        )
        self.img_range = img_range
        self.window_size = window_size*patch_size*2

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


    def forward(self, x):
        """
        前向传播函数，接收输入图像并返回模型输出。
        :param x: 输入图像张量，形状为 (batch_size, channels, height, width)
        :return: 输出张量
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x=self.model(x)

        return x[:, :, :H, :W]

# 示例使用
if __name__ == "__main__":
    x1 = torch.randn(1, 1, 180, 180).cuda()

    window_size = 30
    _, _, h, w = x1.size()
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size
    x = F.pad(x1, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    _, _, h, w = x.size()
    bm3d_model  = BM3DTrans(img_size=(h, w)).cuda()

    # 执行前向传播
    output = bm3d_model(x1)

    # 输出结果的形状
    print(f"Output shape: {output.shape}")
