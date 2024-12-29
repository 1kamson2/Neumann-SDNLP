from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PosEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        """
        Positional Embedding (Time embedding) for UNet
        """
        self.n_channels = n_channels
        self.swish = Swish()
        self.linear1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.linear2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, x):
        half_dim = self.n_channels // 8
        emb = np.log(10000) // (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=_device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.swish(self.linear1(emb))
        emb = self.linear2(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_channels, n_groups=32, dropout=0.1
    ):
        super().__init__()
        self.gnorm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gnorm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        self.time_embedding = nn.Linear(time_channels, out_channels)

    def forward(self, x, t):
        out = self.conv1(self.swish(self.gnorm1(x)))
        out += self.time_embedding(self.swish(t))[:, :, None, None]
        out = self.conv2(self.dropout(self.swish(self.gnorm2(out))))
        return out + self.shortcut(x)


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x, t):
        _ = t
        out = self.conv(x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x, t):
        # Do not use this value. It is just to match Residual Block.
        _ = t
        out = self.conv(x)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k=None, n_groups: int = 32):
        super().__init__()
        if d_k is None:
            d_k = n_channels

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        batch_sz, n_channels, height, width = x.shape
        x = x.view(batch_sz, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_sz, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=2)
        out = torch.einsum("bijh,bjhd->bihd", attn, v)
        out = out.view(batch_sz, -1, self.n_heads * self.d_k)
        out = self.output(out)
        out += x
        out = out.permute(0, 2, 1).view(batch_sz, n_channels, height, width)
        return out


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
        n_blocks: int = 2,
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()
        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        self.time_emb = PosEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(
                    DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i])
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(DownSample(in_channels))

        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(
            out_channels,
            n_channels * 4,
        )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i])
                )
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            up.append(UpSample(in_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)
        for m in self.up:
            if isinstance(m, UpSample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))


class DenoiseModel(nn.Module):
    def __init__(self, noise_model: nn.Module, steps: int, batch_sz=32):
        super().__init__()
        """
        For this model the recommended model for noise is UNet
        """
        self.batch_sz = batch_sz
        self.noise_model = noise_model
        self.beta = torch.linspace(0.0001, 0.02, steps)
        self.n_steps = steps
        self.alpha = 1 - self.beta
        self.alpha_b = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def usq_n_gather(self, constants, t):
        return constants.gather(-1, t).to(_device).view(-1, 1, 1, 1) 

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        c_r = self.usq_n_gather(self.alpha_b, t)
        mean = c_r**0.5 * x0
        var = 1 - c_r
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        if eps is None:
            eps = torch.randn_like(x0)
        mean, var = self.q_xt_x0(x0, t) 
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.noise_model(xt, t)
        alpha_b = self.usq_n_gather(self.alpha_b, t)
        alpha = self.usq_n_gather(self.alpha, t)
        eps_coefficients = ((1 - alpha) / (1 - alpha_b)) ** 0.5
        mean = (1 / alpha**0.5) * (xt - eps_coefficients * eps_theta)
        var = self.usq_n_gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=_device)
        return mean + (var**0.5) * eps

    def loss(self, x0: torch.Tensor, noise=None):
        t = torch.randint(
            0, self.n_steps, (self.batch_sz,), device=_device, dtype=torch.long
        )
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.noise_model(xt, t)
        if noise is None:
            noise = torch.randn_like(eps_theta)
        return F.mse_loss(noise, eps_theta)
