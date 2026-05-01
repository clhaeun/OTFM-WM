"""
U-Net based Conditional Flow Matching World Model for Crafter.

Architecture:
  - Conditioning: CNN state encoder + action embedding -> fused condition
  - Flow Network: U-Net with time embedding (sinusoidal) and condition injection
  - Input: noisy x_t (3,64,64), time t (scalar), condition c
  - Output: vector field v_theta (3,64,64)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """ResNet block with GroupNorm + SiLU, conditioned via AdaGN (scale & shift)."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # AdaGN: project condition to scale & shift
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # AdaGN conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Conditioning Network
# ---------------------------------------------------------------------------

class ConditioningNetwork(nn.Module):
    """
    Encodes (s_t, a_t) into a conditioning vector c.
    - State: lightweight CNN encoder
    - Action: nn.Embedding
    - Fusion: concatenate and project
    """

    def __init__(self, num_actions: int = 17, action_dim: int = 32, cond_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_dim)

        # Simple CNN to encode s_t (3, 64, 64) -> feature vector
        self.state_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # -> 32x32
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 16x16
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> 8x8
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # -> 4x4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),       # -> 256x1x1
            nn.Flatten(),                  # -> 256
        )

        self.fuse = nn.Sequential(
            nn.Linear(256 + action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        state_feat = self.state_encoder(s_t)        # (B, 256)
        action_feat = self.action_embed(a_t)         # (B, action_dim)
        return self.fuse(torch.cat([state_feat, action_feat], dim=-1))  # (B, cond_dim)


# ---------------------------------------------------------------------------
# U-Net Flow Network
# ---------------------------------------------------------------------------

class FlowMatchingUNet(nn.Module):
    """
    U-Net for predicting vector field v_theta(x_t, t, c).

    Architecture (64x64 input):
      Encoder: 64 -> 32 -> 16 -> 8
      Bottleneck: 8x8
      Decoder: 8 -> 16 -> 32 -> 64

    Channel progression: 64 -> 128 -> 256 -> 512
    """

    def __init__(self, in_ch: int = 3, cond_dim: int = 256, time_dim: int = 256,
                 ch_mult=(1, 2, 4, 8), base_ch: int = 64):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Combined condition: time + condition
        combined_dim = time_dim + cond_dim

        channels = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]

        # Input projection
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_c = channels[0]
        for out_c in channels:
            self.enc_blocks.append(ResBlock(in_c, out_c, combined_dim))
            self.downsamples.append(Downsample(out_c))
            in_c = out_c

        # Bottleneck
        self.mid_block1 = ResBlock(channels[-1], channels[-1], combined_dim)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], combined_dim)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i, out_c in enumerate(reversed(channels)):
            self.upsamples.append(Upsample(in_c))
            # skip connection doubles channels
            self.dec_blocks.append(ResBlock(in_c + out_c, out_c, combined_dim))
            in_c = out_c

        # Output projection
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_ch, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Time + condition fusion
        t_emb = self.time_embed(t)                          # (B, time_dim)
        combined = torch.cat([t_emb, cond], dim=-1)         # (B, time_dim + cond_dim)

        # Encoder
        h = self.input_conv(x_t)
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            h = enc(h, combined)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, combined)
        h = self.mid_block2(h, combined)

        # Decoder
        for up, dec, skip in zip(self.upsamples, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h, combined)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Full World Model
# ---------------------------------------------------------------------------

class FlowMatchingWorldModel(nn.Module):
    """
    Complete world model combining conditioning + flow network.

    Usage:
        model = FlowMatchingWorldModel()
        v_pred = model(x_t, t, s_t, a_t)  # predict vector field
    """

    def __init__(self, num_actions: int = 17, cond_dim: int = 256, time_dim: int = 256,
                 base_ch: int = 64):
        super().__init__()
        self.cond_net = ConditioningNetwork(num_actions=num_actions, cond_dim=cond_dim)
        self.flow_net = FlowMatchingUNet(in_ch=3, cond_dim=cond_dim, time_dim=time_dim,
                                         base_ch=base_ch)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        cond = self.cond_net(s_t, a_t)
        return self.flow_net(x_t, t, cond)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowMatchingWorldModel().to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Test forward pass
    B = 4
    x_t = torch.randn(B, 3, 64, 64, device=device)
    t = torch.rand(B, device=device)
    s_t = torch.rand(B, 3, 64, 64, device=device)
    a_t = torch.randint(0, 17, (B,), device=device)

    v_pred = model(x_t, t, s_t, a_t)
    print(f"Input:  x_t={x_t.shape}, t={t.shape}, s_t={s_t.shape}, a_t={a_t.shape}")
    print(f"Output: v_pred={v_pred.shape}")
