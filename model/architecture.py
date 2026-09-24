"""
architecture.py
---------------
ForecastBustUNet: A dual-head multi-task U-Net for gridded weather forecast
bust detection and error regression.

Architecture:
  - Encoder: 4-level convolutional encoder (64→128→256→512 channels)
  - Bottleneck: DoubleConv(512)
  - Decoder: U-Net skip connections with ConvTranspose2d upsampling
  - Head A (Regression): Predicts absolute forecast error per grid cell
  - Head B (Classification): Predicts bust probability per grid cell [0, 1]
  - CoordConv: lat/lon channels can be pre-appended to input

Input:  [B, C, H, W]  (B=batch, C=channels, H=lat, W=lon)
Output: (error_map [B, 1, H, W], bust_prob [B, 1, H, W])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections (Oktay et al. 2018, Attention U-Net).
    Allows the decoder to focus on spatially relevant features.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        # Align spatial dims: after odd-sized inputs and repeated pooling,
        # g1 (from decoder) may be 1 pixel smaller than x1 (encoder skip).
        # Upsample g1 to x1's size so element-wise add is valid.
        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Align psi back to x's spatial size for element-wise multiply
        if psi.shape[2:] != x.shape[2:]:
            psi = torch.nn.functional.interpolate(
                psi, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return x * psi


class ForecastBustUNet(nn.Module):
    """
    Dual-head Attention U-Net for forecast bust detection.

    Args:
        in_channels: Number of input channels (forecast vars + static + lead encoding)
        encoder_channels: Feature map sizes at each encoder level
        dropout: Dropout rate in encoder bottleneck

    Outputs:
        error_map: Predicted absolute forecast error [B, 1, H, W] (Softplus ≥ 0)
        bust_prob: Probability of forecast bust [B, 1, H, W] (Sigmoid ∈ [0, 1])
    """

    def __init__(
        self,
        in_channels: int = 14,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.2,
    ):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels

        # ── Encoder ─────────────────────────────────────────
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3, dropout=dropout / 2)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────
        self.bottleneck = DoubleConv(c4, c4 * 2, dropout=dropout)

        # ── Attention Gates (skip connections) ────────────────
        # g = decoder output AFTER upsampling (so c4, c3, c2, c1 channels)
        # x = encoder skip connection at matching resolution
        self.att4 = AttentionGate(f_g=c4,   f_l=c4, f_int=c4 // 2)
        self.att3 = AttentionGate(f_g=c3,   f_l=c3, f_int=c3 // 2)
        self.att2 = AttentionGate(f_g=c2,   f_l=c2, f_int=c2 // 2)
        self.att1 = AttentionGate(f_g=c1,   f_l=c1, f_int=c1 // 2)


        # ── Decoder ─────────────────────────────────────────
        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)

        # ── Output Heads ───────────────────────────────────
        self.head_error = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
            nn.Softplus(),  # Ensures non-negative error predictions
        )
        self.head_bust = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
            nn.Sigmoid(),  # Bust probability in [0, 1]
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal initialisation for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, in_channels, H, W]

        Returns:
            error_map: Predicted error magnitude [B, 1, H, W] (same H,W as input)
            bust_prob: Bust probability map [B, 1, H, W] (same H,W as input)
        """
        H_in, W_in = x.shape[2], x.shape[3]   # remember original spatial size

        # Encoder
        e1 = self.enc1(x)                    # [B, c1, H, W]
        e2 = self.enc2(self.pool1(e1))       # [B, c2, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))       # [B, c3, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))       # [B, c4, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # [B, c4*2, H/16, W/16]

        # Decoder with attention gates
        # NOTE: _pad_and_cat pads the decoder tensor up to match the encoder skip.
        # For odd H or W, this adds 1 pixel — the final crop below corrects it.
        d4 = self.up4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = self._pad_and_cat(d4, e4_att)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self._pad_and_cat(d3, e3_att)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self._pad_and_cat(d2, e2_att)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self._pad_and_cat(d1, e1_att)
        d1 = self.dec1(d1)

        error_map = self.head_error(d1)   # [B, 1, H', W']
        bust_prob = self.head_bust(d1)    # [B, 1, H', W']

        # Crop to original input spatial size (handles odd H/W padding artefacts)
        error_map = error_map[:, :, :H_in, :W_in]
        bust_prob = bust_prob[:, :, :H_in, :W_in]

        return error_map, bust_prob


    @staticmethod
    def _pad_and_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pad 'a' to match 'b' spatial dims if needed, then concatenate."""
        dh = b.shape[2] - a.shape[2]
        dw = b.shape[3] - a.shape[3]
        if dh != 0 or dw != 0:
            a = F.pad(a, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([a, b], dim=1)


def compute_confidence_map(
    error_map: torch.Tensor,
    bust_prob: torch.Tensor,
    clim_sigma: float = 10.0,
) -> torch.Tensor:
    """
    Compute the composite forecast confidence indicator:
        C(x,y) = 100 * exp(-E(x,y) / sigma_clim) * (1 - P_bust(x,y))

    Args:
        error_map: Predicted error magnitude [B, 1, H, W]
        bust_prob: Predicted bust probability [B, 1, H, W]
        clim_sigma: Climatological error scale factor (default: 10 mm/day)

    Returns:
        confidence: Forecast confidence map [B, 1, H, W] in range [0, 100]
    """
    confidence = 100.0 * torch.exp(-error_map / clim_sigma) * (1.0 - bust_prob)
    return confidence.clamp(0.0, 100.0)


if __name__ == "__main__":
    # Quick smoke test
    model = ForecastBustUNet(in_channels=6, encoder_channels=(32, 64, 128, 256))
    model.eval()
    dummy = torch.randn(2, 6, 64, 64)  # [B=2, C=6, H=64, W=64]
    with torch.no_grad():
        err, prob = model(dummy)
    print(f"Error map: {err.shape}, Bust prob: {prob.shape}")
    conf = compute_confidence_map(err, prob)
    print(f"Confidence map: {conf.shape} | range [{conf.min():.1f}, {conf.max():.1f}]")
    print("Architecture test PASSED")
