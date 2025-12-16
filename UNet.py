import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal


# =====================
# U-Net Building Blocks
# =====================

class ConvBlock(nn.Module):
    """
    (Conv -> BN -> ReLU) x 2 + optional attention
    attention: "none" | "se" | "cbam"
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        attention: Literal["none", "se", "cbam"] = "none"
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        if attention == "se":
            self.attn = SEBlock(out_ch)
        elif attention == "cbam":
            self.attn = CBAM(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return self.attn(x)


class Down(nn.Module):
    """Downscale with maxpool then double conv"""
    def __init__(self, in_ch: int, out_ch: int, attention: Literal["none", "se", "cbam"] = "none") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, attention=attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscale then double conv. Uses transposed conv for upsampling."""
    def __init__(self, in_ch: int, out_ch: int, attention: Literal["none", "se", "cbam"] = "none") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, attention=attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block:
    - Channel-wise attention via global average pooling
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x)
        return x * w


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM):
    - Channel attention (avg/max pool -> MLP)
    - Spatial attention (avg/max across channels -> conv)
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)

        # Channel attention
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        # Spatial attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx, _ = torch.max(x, dim=2, keepdim=True)
        mx, _ = torch.max(mx, dim=3, keepdim=True)
        ch_att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ch_att

        # Spatial attention
        avg_c = torch.mean(x, dim=1, keepdim=True)
        max_c, _ = torch.max(x, dim=1, keepdim=True)
        sp = torch.cat([avg_c, max_c], dim=1)
        sp_att = torch.sigmoid(self.spatial(sp))
        return x * sp_att


class DilatedConvBlock(nn.Module):
    """
    Slightly larger receptive field in the bottleneck via dilation.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    """
    Lightweight U-Net for binary segmentation (1 class: plate vs background):
    - CBAM attention in encoder/decoder blocks
    - SE attention in encoder/decoder blocks
    - Dilated bottleneck (larger receptive field)
    - One FPN-style lateral fusion at the 1/4 scale (x2 -> y2)
    
    Input: 3xHxW, Output: 1xHxW logits (use with BCEWithLogits).
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        attention: Literal["none", "se", "cbam"] = "none",
    ) -> None:
        super().__init__()

        # Encoder
        self.inc = ConvBlock(in_ch, base_ch, attention=attention)
        self.down1 = Down(base_ch, base_ch * 2, attention=attention)
        self.down2 = Down(base_ch * 2, base_ch * 4, attention=attention)
        self.down3 = Down(base_ch * 4, base_ch * 8, attention=attention)

        # Dilated bottleneck
        self.bottleneck = DilatedConvBlock(base_ch * 8, base_ch * 16)

        # Decoder
        self.up3 = Up(base_ch * 16, base_ch * 8, attention=attention)
        self.up2 = Up(base_ch * 8, base_ch * 4, attention=attention)
        self.up1 = Up(base_ch * 4, base_ch * 2, attention=attention)
        self.up0 = Up(base_ch * 2, base_ch, attention=attention)

        # FPN-style lateral fusion (1/4 scale)
        self.lat_x2 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=1, bias=False)

        # Head
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)         # H
        x1 = self.down1(x0)      # H/2
        x2 = self.down2(x1)      # H/4
        x3 = self.down3(x2)      # H/8

        xb = self.bottleneck(x3) # H/8

        y3 = self.up3(xb, x3)    # H/8
        y2 = self.up2(y3, x2)    # H/4

        # FPN-style lateral fusion: inject refined encoder features at same scale
        y2 = y2 + self.lat_x2(x2)

        y1 = self.up1(y2, x1)    # H/2
        y0 = self.up0(y1, x0)    # H

        return self.outc(y0)
    