"""
V2: Spatially-conditioned Flow Matching World Model for Crafter.

Key changes from v1:
  1. Spatial conditioning: s_t is concatenated with x_t at input (6ch)
     AND a multi-scale encoder injects s_t features at every U-Net level.
  2. Designed for conditional OT: x_0 = s_t, x_1 = s_{t+1}
     so the model only learns the small residual between frames.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (same as v1)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """ResNet block with GroupNorm + SiLU, conditioned via AdaGN."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Multi-scale Spatial State Encoder
# ---------------------------------------------------------------------------

class SpatialStateEncoder(nn.Module):
    """
    Encodes s_t into multi-scale feature maps matching U-Net resolutions.
    Output: list of feature maps at [64x64, 32x32, 16x16, 8x8].
    These get added to U-Net encoder features at corresponding levels.
    """

    def __init__(self, base_ch: int = 64, ch_mult=(1, 2, 4, 8)):
        super().__init__()
        channels = [base_ch * m for m in ch_mult]

        # First: produce base features at full resolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[0], channels[0], 3, padding=1),
        )

        # Progressive downsampling with feature extraction
        self.down_blocks = nn.ModuleList()
        in_c = channels[0]
        for out_c in channels[1:]:
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.SiLU(),
            ))
            in_c = out_c

    def forward(self, s_t):
        features = []
        h = self.input_conv(s_t)   # 64x64, base_ch
        features.append(h)
        for block in self.down_blocks:
            h = block(h)            # 32x32, 16x16, 8x8
            features.append(h)
        return features  # [64x64, 32x32, 16x16, 8x8]


# ---------------------------------------------------------------------------
# U-Net with Spatial Conditioning
# ---------------------------------------------------------------------------

class FlowMatchingUNetV2(nn.Module):
    """
    U-Net for v_theta(x_t, t, s_t, a_t).

    Spatial conditioning:
      - Input: concat(x_t, s_t) = 6 channels
      - Multi-scale s_t features added at each encoder level
    Global conditioning:
      - Time (sinusoidal) + action embedding -> AdaGN
    """

    def __init__(self, cond_dim: int = 256, time_dim: int = 256,
                 ch_mult=(1, 2, 4, 8), base_ch: int = 64):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        combined_dim = time_dim + cond_dim  # time + action
        channels = [base_ch * m for m in ch_mult]  # [64, 128, 256, 512]

        # Input: concat(x_t, s_t) = 6 channels
        self.input_conv = nn.Conv2d(6, channels[0], 3, padding=1)

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
            self.dec_blocks.append(ResBlock(in_c + out_c, out_c, combined_dim))
            in_c = out_c

        # Output
        self.out_norm = nn.GroupNorm(min(32, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, x_t, t, s_t, action_emb, spatial_features):
        """
        Args:
            x_t: (B, 3, 64, 64) noisy interpolation
            t: (B,) time
            s_t: (B, 3, 64, 64) current state (for input concat)
            action_emb: (B, cond_dim) action embedding vector
            spatial_features: list of (B, C, H, W) from SpatialStateEncoder
        """
        t_emb = self.time_embed(t)
        combined = torch.cat([t_emb, action_emb], dim=-1)

        # Input: concat x_t with s_t
        h = self.input_conv(torch.cat([x_t, s_t], dim=1))

        # Encoder with spatial feature injection (after ResBlock, matching channels)
        skips = []
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downsamples)):
            h = enc(h, combined)
            h = h + spatial_features[i]  # add after ResBlock so channels match
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
# Full World Model V2
# ---------------------------------------------------------------------------

class FlowMatchingWorldModelV2(nn.Module):
    """
    V2 world model with spatial conditioning + conditional OT.

    Usage:
        model = FlowMatchingWorldModelV2()
        v_pred = model(x_t, t, s_t, a_t)
    """

    def __init__(self, num_actions: int = 17, action_dim: int = 64,
                 cond_dim: int = 256, time_dim: int = 256, base_ch: int = 64):
        super().__init__()
        # Action: embedding -> projection to cond_dim
        self.action_embed = nn.Embedding(num_actions, action_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Spatial state encoder
        self.spatial_encoder = SpatialStateEncoder(base_ch=base_ch)

        # Flow network
        self.flow_net = FlowMatchingUNetV2(
            cond_dim=cond_dim, time_dim=time_dim, base_ch=base_ch,
        )

    def forward(self, x_t, t, s_t, a_t):
        action_emb = self.action_proj(self.action_embed(a_t))  # (B, cond_dim)
        spatial_feats = self.spatial_encoder(s_t)               # multi-scale features
        return self.flow_net(x_t, t, s_t, action_emb, spatial_feats)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowMatchingWorldModelV2().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    B = 4
    x_t = torch.randn(B, 3, 64, 64, device=device)
    t = torch.rand(B, device=device)
    s_t = torch.rand(B, 3, 64, 64, device=device)
    a_t = torch.randint(0, 17, (B,), device=device)

    v_pred = model(x_t, t, s_t, a_t)
    print(f"Input:  x_t={x_t.shape}, t={t.shape}, s_t={s_t.shape}, a_t={a_t.shape}")
    print(f"Output: v_pred={v_pred.shape}")
