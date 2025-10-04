"""JannerUNet - U-Net architecture for flow matching policies.
Based on Janner et al.'s diffusion policy implementation.
"""

import einops
import numpy as np
import torch
import torch.nn as nn

from mip.embeddings import SUPPORTED_TIMESTEP_EMBEDDING
from mip.network_utils import GroupNorm1d
from mip.networks.base import BaseNetwork


def get_norm(dim: int, norm_type: str = "groupnorm"):
    if norm_type == "groupnorm":
        return GroupNorm1d(dim, 8, 4)
    elif norm_type == "layernorm":
        return LayerNorm(dim)
    else:
        return nn.Identity()


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        emb_dim: int,
        kernel_size: int = 3,
        norm_type: str = "groupnorm",
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            get_norm(out_dim, norm_type),
            nn.Mish(),
        )
        self.emb_mlp = nn.Sequential(nn.Mish(), nn.Linear(emb_dim, out_dim))
        self.residual_conv = (
            nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, emb):
        out = self.conv1(x) + self.emb_mlp(emb).unsqueeze(-1)
        out = self.conv2(out)
        return out + self.residual_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = (
            einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads) for t in qkv
        )
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        out = self.to_out(out)
        return out + x


class JannerUNet(BaseNetwork):
    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        model_dim: int = 32,
        emb_dim: int = 32,
        kernel_size: int = 3,
        dim_mult: list[int] | None = None,
        norm_type: str = "groupnorm",
        attention: bool = False,
        timestep_emb_type: str = "positional",
        timestep_emb_params: dict | None = None,
        disable_time_embedding: bool = False,
    ):
        # Default dim_mult if not provided
        if dim_mult is None:
            dim_mult = [1, 2, 2, 2]

        # BaseNetwork expects: act_dim, Ta, obs_dim, To, emb_dim, n_layers
        num_layers = len(dim_mult) * 2 + 2  # downs + ups + mids
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, num_layers)

        self.Ta = Ta
        self.model_dim = model_dim
        self.emb_dim = emb_dim
        self.disable_time_embedding = disable_time_embedding

        # Use act_dim as in_dim for the U-Net
        in_dim = act_dim
        dims = [in_dim] + [model_dim * m for m in np.cumprod(dim_mult)]
        in_out = list(zip(dims[:-1], dims[1:], strict=False))

        # Time embedding mappings for s and t
        timestep_emb_params = timestep_emb_params or {}
        if not disable_time_embedding:
            self.map_s = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                emb_dim // 2, **timestep_emb_params
            )
            self.map_t = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](
                emb_dim // 2, **timestep_emb_params
            )
        else:
            self.map_s = None
            self.map_t = None

        # Map combined embeddings to model dimension
        self.map_emb = nn.Sequential(
            nn.Linear(emb_dim * 2, model_dim * 4),
            nn.Mish(),
            nn.Linear(model_dim * 4, model_dim),
        )

        # Condition encoder
        self.cond_encoder = nn.Linear(obs_dim * To, emb_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            dim_in, dim_out, model_dim, kernel_size, norm_type
                        ),
                        ResidualBlock(
                            dim_out, dim_out, model_dim, kernel_size, norm_type
                        ),
                        LinearAttention(dim_out) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, norm_type
        )
        self.mid_attn = LinearAttention(mid_dim) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(
            mid_dim, mid_dim, model_dim, kernel_size, norm_type
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            dim_out * 2, dim_in, model_dim, kernel_size, norm_type
                        ),
                        ResidualBlock(
                            dim_in, dim_in, model_dim, kernel_size, norm_type
                        ),
                        LinearAttention(dim_in) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 5, padding=2),
            get_norm(model_dim, norm_type),
            nn.Mish(),
            nn.Conv1d(model_dim, in_dim, 1),
        )

        # Scalar output head
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(mid_dim, 128),
            nn.Mish(),
            nn.Linear(128, 1),
        )

        # Zero-out scalar head
        nn.init.constant_(self.scalar_head[-1].weight, 0)
        nn.init.constant_(self.scalar_head[-1].bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        """Input:
            x:          (b, Ta, act_dim)
            s:          (b, ) - source time parameter
            t:          (b, ) - target time parameter
            condition:  (b, To, obs_dim) or None

        Output:
            y:          (b, Ta, act_dim) - predicted action sequence
            scalar:     (b, 1) - predicted scalar value
        """
        # check Ta dimension
        assert x.shape[1] & (x.shape[1] - 1) == 0, "Ta dimension must be 2^n"

        batch_size = x.shape[0]
        device = x.device

        x = x.permute(0, 2, 1)  # (b, act_dim, Ta)

        # Process time embeddings
        if not self.disable_time_embedding:
            emb_s = self.map_s(s)  # (b, emb_dim // 2)
            emb_t = self.map_t(t)  # (b, emb_dim // 2)
            time_emb = torch.cat([emb_s, emb_t], dim=-1)  # (b, emb_dim)
        else:
            time_emb = torch.zeros((batch_size, self.emb_dim), device=device)

        # Process condition
        if condition is not None:
            cond_flat = torch.flatten(condition, 1)  # (b, To * obs_dim)
            cond_emb = self.cond_encoder(cond_flat)  # (b, emb_dim)
        else:
            cond_emb = torch.zeros((batch_size, self.emb_dim), device=device)

        # Combine time and condition embeddings
        combined_emb = torch.cat([time_emb, cond_emb], dim=-1)  # (b, emb_dim * 2)
        emb = self.map_emb(combined_emb)  # (b, model_dim)

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # Get scalar output from bottleneck
        scalar_out = self.scalar_head(x)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet1(x, emb)
            x = resnet2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = x.permute(0, 2, 1)  # (b, Ta, act_dim)
        return x, scalar_out


def test_jannerunet():
    """Test JannerUNet network"""
    print("=" * 50)
    print("Testing JannerUNet")
    print("=" * 50)

    act_dim = 2
    Ta = 8  # Must be power of 2
    obs_dim = 3
    To = 2
    batch_size = 4

    # Test with default parameters
    model = JannerUNet(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        model_dim=32,
        emb_dim=32,
        dim_mult=[1, 2],
        attention=True,
        timestep_emb_type="positional",
    )

    x = torch.randn(batch_size, Ta, act_dim)
    s = torch.randn(batch_size)
    t = torch.randn(batch_size)
    condition = torch.randn(batch_size, To, obs_dim)

    y, scalar_out = model(x, s, t, condition)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Scalar output shape: {scalar_out.shape}")

    # Compute number of parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of parameters: {num_params:.2f}M")

    # Test without condition
    y_no_cond, scalar_no_cond = model(x, s, t, None)
    print(
        f"Works without condition: y={y_no_cond.shape}, scalar={scalar_no_cond.shape}"
    )

    # Test with disable_time_embedding=True
    print("\nTesting with disable_time_embedding=True:")
    model_no_time = JannerUNet(
        act_dim=act_dim,
        Ta=Ta,
        obs_dim=obs_dim,
        To=To,
        model_dim=32,
        emb_dim=32,
        disable_time_embedding=True,
    )

    y1, s1 = model_no_time(x, s, t, condition)
    # Test with different time values - should give same output
    y2, s2 = model_no_time(
        x, torch.randn(batch_size), torch.randn(batch_size), condition
    )

    print(
        f"Time invariant: {torch.allclose(y1, y2, atol=1e-6) and torch.allclose(s1, s2, atol=1e-6)}"
    )

    print("=" * 50)
    print("JannerUNet test completed!")
    print("=" * 50)


if __name__ == "__main__":
    test_jannerunet()
