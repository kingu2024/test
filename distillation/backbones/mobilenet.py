"""MobileNetV2 backbone with multi-scale feature extraction."""

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from distillation import BACKBONES


@BACKBONES.register("mobilenetv2")
class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone outputting multi-scale features.

    Extracts features at 4 scale levels to match the ResNet interface:
        - "s2": stride 4   (24 channels)
        - "s3": stride 8   (32 channels)
        - "s4": stride 16  (96 channels)
        - "s5": stride 32  (1280 channels)

    Args:
        pretrained: Whether to load ImageNet pretrained weights.
        frozen_stages: Number of stages (1-4) to freeze.
    """

    # MobileNetV2 inverted residual block indices for each scale
    # features[0..1]  -> stride 2, 4  (layer indices 0-3)
    # features[2..3]  -> stride 8     (layer indices 4-6)
    # features[4..6]  -> stride 16    (layer indices 7-13)
    # features[7..17] -> stride 32    (layer indices 14-18)
    STAGE_INDICES = [
        (0, 4),    # s2: up to inverted_residual_3
        (4, 7),    # s3: inverted_residual_4 to 6
        (7, 14),   # s4: inverted_residual_7 to 13
        (14, 19),  # s5: inverted_residual_14 to conv_1x1 + final
    ]
    OUT_CHANNELS = [24, 32, 96, 1280]

    def __init__(self, pretrained: bool = True, frozen_stages: int = 0):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        features = list(mobilenet.features)

        self.stage1 = nn.Sequential(*features[0:4])
        self.stage2 = nn.Sequential(*features[4:7])
        self.stage3 = nn.Sequential(*features[7:14])
        self.stage4 = nn.Sequential(*features[14:19])

        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int):
        stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        for i in range(min(num_stages, 4)):
            for p in stages[i].parameters():
                p.requires_grad = False

    @property
    def out_channels(self) -> List[int]:
        return self.OUT_CHANNELS

    @property
    def feature_keys(self) -> List[str]:
        return ["s2", "s3", "s4", "s5"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s2 = self.stage1(x)
        s3 = self.stage2(s2)
        s4 = self.stage3(s3)
        s5 = self.stage4(s4)
        return {"s2": s2, "s3": s3, "s4": s4, "s5": s5}
