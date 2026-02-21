"""Semantic segmentation heads."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation import HEADS


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


@HEADS.register("fpn_seg_head")
class FPNSegHead(nn.Module):
    """FPN-style segmentation head.

    Fuses multi-scale features and produces per-pixel class predictions.

    Args:
        in_channels_list: Channel dimensions for each input scale [s2, s3, s4, s5].
        mid_channels: Intermediate channel dimension after lateral convolutions.
        num_classes: Number of segmentation classes.
        input_keys: Which feature keys to consume (default: all 4 scales).
    """

    def __init__(
        self,
        in_channels_list: List[int],
        mid_channels: int = 256,
        num_classes: int = 19,
        input_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys or ["s2", "s3", "s4", "s5"]
        self.num_classes = num_classes

        # Lateral 1x1 convolutions to unify channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, mid_channels, 1) for in_ch in in_channels_list
        ])

        # Smooth 3x3 convolutions after feature fusion
        self.smooth_convs = nn.ModuleList([
            ConvBNReLU(mid_channels, mid_channels) for _ in in_channels_list
        ])

        # Final segmentation conv
        self.seg_conv = nn.Sequential(
            ConvBNReLU(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, num_classes, 1),
        )

    def forward(
        self, features: Dict[str, torch.Tensor], target_size: Optional[tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dict of multi-scale feature tensors.
            target_size: (H, W) to upsample the final prediction to.

        Returns:
            Dict with "seg_logits" and optionally intermediate features.
        """
        feats = [features[k] for k in self.input_keys]

        # Lateral projections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # Smooth
        outs = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]

        # Use the highest resolution feature for prediction
        seg_feat = outs[0]
        seg_logits = self.seg_conv(seg_feat)

        if target_size is not None:
            seg_logits = F.interpolate(
                seg_logits, size=target_size, mode="bilinear", align_corners=False
            )

        return {"seg_logits": seg_logits, "seg_feat": seg_feat}


@HEADS.register("aspp_seg_head")
class ASPPSegHead(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling) segmentation head.

    Uses dilated convolutions at multiple rates to capture multi-scale context.

    Args:
        in_channels: Input channel dimension (from a single feature level).
        mid_channels: Intermediate channel dimension.
        num_classes: Number of segmentation classes.
        atrous_rates: Dilation rates for ASPP branches.
        input_key: Which feature key to consume (default: "s5").
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 19,
        atrous_rates: tuple = (6, 12, 18),
        input_key: str = "s4",
    ):
        super().__init__()
        self.input_key = input_key
        self.num_classes = num_classes

        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ))
        # Global average pooling branch
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ))
        self.aspp_branches = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * (len(atrous_rates) + 2), mid_channels, 1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1)

    def forward(
        self, features: Dict[str, torch.Tensor], target_size: Optional[tuple] = None
    ) -> Dict[str, torch.Tensor]:
        x = features[self.input_key]
        h, w = x.shape[2:]

        branch_outs = []
        for i, branch in enumerate(self.aspp_branches):
            out = branch(x)
            if out.shape[2:] != (h, w):
                out = F.interpolate(out, size=(h, w), mode="bilinear",
                                    align_corners=False)
            branch_outs.append(out)

        fused = torch.cat(branch_outs, dim=1)
        seg_feat = self.project(fused)
        seg_logits = self.classifier(seg_feat)

        if target_size is not None:
            seg_logits = F.interpolate(
                seg_logits, size=target_size, mode="bilinear", align_corners=False
            )

        return {"seg_logits": seg_logits, "seg_feat": seg_feat}
