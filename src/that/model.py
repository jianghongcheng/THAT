"""THAT: Token-wise High-frequency Augmentation Transformer
for Hyperspectral Pansharpening.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
from sklearn.cluster import KMeans


def pivotal_attention_mask(scores):
    """Independent 2-cluster fits per sample/head; keep higher-mean score cluster.

    Discrete assignments are detached. Retained scores remain differentiable.
    If a row has no pivotal entry, retain its maximum to keep softmax finite.
    """
    if scores.ndim != 4 or not scores.is_floating_point():
        raise ValueError("Expected floating [batch, heads, query, key] scores")
    if not torch.isfinite(scores).all():
        raise ValueError("Attention scores must be finite")
    keep = torch.empty_like(scores, dtype=torch.bool)
    values = scores.detach().float().cpu().numpy()
    for b in range(scores.shape[0]):
        for h in range(scores.shape[1]):
            a = values[b, h].reshape(-1, 1)
            if a.max() == a.min():
                keep[b, h] = True
            else:
                km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(a)
                mask = km.labels_ == km.cluster_centers_.reshape(-1).argmax()
                keep[b, h] = torch.as_tensor(mask, device=scores.device).reshape(
                    scores.shape[-2:]
                )
    empty = ~keep.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(keep).scatter_(
        -1, scores.argmax(-1, keepdim=True), True
    )
    return keep | (empty & fallback)


def kmeans_attn(scores):
    return scores.masked_fill(~pivotal_attention_mask(scores), float("-inf"))


class PTSA(nn.Module):
    """Pivotal Token Selective Attention (PTSA)
    Tips:
        Mainly borrows from DRSFormer (https://github.com/cschenxiang/DRSformer)
    """

    def __init__(self, dim, num_heads, bias):
        super(PTSA, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn1 = kmeans_attn(attn)

        attn1 = attn1.softmax(dim=-1)

        out = attn1 @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)

        return out


def get_local_weights(residual, ksize, padding):
    pad = padding
    # Pad the residual with reflection padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")

    # Unfold with a stride of 1 to maintain the original size
    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)

    # Compute variance across the local window
    pixel_level_weight = (
        torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
        .squeeze(-1)
        .squeeze(-1)
    )

    return pixel_level_weight


class MVFN(nn.Module):
    """Multi-level variance-aware feed-forward network (Figure 4)."""

    def __init__(self, num_feat=24):
        super(MVFN, self).__init__()
        if num_feat % 12:
            raise ValueError("MVFN width must be divisible by 12")
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1)

        self.dwconv_3x3 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, groups=f)
        self.dwconv_5x5 = nn.Conv2d(f, f, kernel_size=5, stride=1, padding=2, groups=f)
        self.dwconv_7x7 = nn.Conv2d(f, f, kernel_size=7, stride=1, padding=3, groups=f)

        self.silu = nn.SiLU(inplace=True)
        self.var = get_local_weights

        self.maxPooling_3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.maxPooling_5x5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxPooling_7x7 = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.project_out = nn.Conv2d(f * 3, num_feat, kernel_size=1)

    def forward(self, x):
        c1_ = self.conv1(x)  # channel squeeze

        x1_3, x2_3, x3_3 = self.dwconv_3x3(c1_).chunk(3, dim=1)
        x1_5, x2_5, x3_5 = self.dwconv_5x5(c1_).chunk(3, dim=1)
        x1_7, x2_7, x3_7 = self.dwconv_7x7(c1_).chunk(3, dim=1)

        x2_3_var = self.var(x2_3, 3, padding=1)
        x2_5_var = self.var(x2_5, 5, padding=2)
        x2_7_var = self.var(x2_7, 7, padding=3)

        x3_3_maxPooling = self.maxPooling_3x3(x3_3)
        x3_5_maxPooling = self.maxPooling_5x5(x3_5)
        x3_7_maxPooling = self.maxPooling_7x7(x3_7)

        x1 = self.silu(torch.cat([x1_3, x1_5, x1_7], dim=1))
        x2 = self.silu(torch.cat([x2_3_var, x2_5_var, x2_7_var], dim=1))
        x3 = self.silu(
            torch.cat([x3_3_maxPooling, x3_5_maxPooling, x3_7_maxPooling], dim=1)
        )

        x_cat = torch.cat([x1, x2, x3], dim=1)

        output = self.project_out(x_cat) + x

        return output


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    )
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b, h // window_size, w // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
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

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b_, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowSelfAttention(nn.Module):
    def __init__(self, dim, heads=6, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attn = WindowAttention(dim, (window_size, window_size), heads)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)
        relative = (
            (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        )
        relative += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rpi", relative.sum(-1))

    def forward(self, x):
        b, c, h, w = x.shape
        if h % self.window_size or w % self.window_size:
            raise ValueError("Spatial dimensions must be multiples of window size")
        windows = window_partition(x.permute(0, 2, 3, 1), self.window_size)
        windows = windows.reshape(-1, self.window_size**2, c)
        out = self.attn(windows, self.rpi).reshape(
            -1, self.window_size, self.window_size, c
        )
        return (
            window_reverse(out, self.window_size, h, w).permute(0, 3, 1, 2).contiguous()
        )


class PTSG(nn.Module):
    """Figure 3: LayerNorm -> parallel PTSA/WSA -> MVFN -> residual addition.

    Additive branch fusion is an explicit interpretation of the unspecified merge.
    """

    def __init__(self, dim=180, heads=6, window_size=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ptsa = PTSA(dim, heads, True)
        self.wsa = WindowSelfAttention(dim, heads, window_size)
        self.mvfn = MVFN(dim)

    def forward(self, x):
        z = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return x + self.mvfn(self.ptsa(z) + self.wsa(z))


class THATBackbone(nn.Module):
    def __init__(
        self, in_chans=65, embed_dim=180, num_groups=3, num_heads=6, window_size=8
    ):
        super().__init__()
        if embed_dim % 12 or embed_dim % num_heads:
            raise ValueError("Width must be divisible by 12 and attention head count")
        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        self.embed_dim = embed_dim
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList(
            [PTSG(embed_dim, num_heads, window_size) for _ in range(num_groups)]
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(embed_dim, in_chans, 3, padding=1), nn.PixelShuffle(1)
        )

    def forward(self, x):
        shallow = self.conv_first(x)
        deep = shallow
        for group in self.layers:
            deep = group(deep)
        return self.reconstruction(deep + shallow)


class THAT(nn.Module):
    def __init__(self, in_ch, factor, **backbone_kwargs):
        super().__init__()
        if in_ch < 1 or factor not in (2, 4, 8):
            raise ValueError("Positive band count and factor 2/4/8 required")
        self.in_channels, self.factor = in_ch, factor
        self.config = dict(in_ch=in_ch, factor=factor, **backbone_kwargs)
        self.conv1 = nn.Conv2d(in_ch, 64, 1)
        self.backbone = THATBackbone(**backbone_kwargs)
        self.conv7 = nn.Conv2d(65, in_ch, 1)

    def forward(self, ms, pan):
        if ms.ndim != 4 or pan.ndim != 4 or ms.shape[1] != self.in_channels:
            raise ValueError("Expected MS BCHW and PAN B1HW with matching band count")
        if pan.shape[:2] != (ms.shape[0], 1) or pan.shape[-2:] != (
            ms.shape[-2] * self.factor,
            ms.shape[-1] * self.factor,
        ):
            raise ValueError("PAN and MS sizes do not match the configured scale")
        up = F.interpolate(ms, size=pan.shape[-2:], mode="bicubic", align_corners=False)
        x = torch.cat([F.relu(self.conv1(up)), pan], dim=1)
        return up + self.conv7(self.backbone(x))
