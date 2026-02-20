"""ResNet backbone with multi-scale feature extraction."""

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from distillation import BACKBONES


class ResNetBackbone(nn.Module):
    """ResNet backbone that outputs multi-scale features.

    Returns a dict of feature maps at different stages:
        - "s2": stride 4,  channels depend on variant
        - "s3": stride 8
        - "s4": stride 16
        - "s5": stride 32

    Args:
        variant: One of "resnet18", "resnet34", "resnet50", "resnet101".
        pretrained: Whether to load ImageNet pretrained weights.
        frozen_stages: Number of stages (1-4) to freeze. 0 means no freezing.
    """

    VARIANT_CHANNELS = {
        "resnet18":  [64, 128, 256, 512],
        "resnet34":  [64, 128, 256, 512],
        "resnet50":  [256, 512, 1024, 2048],
        "resnet101": [256, 512, 1024, 2048],
    }

    def __init__(
        self,
        variant: str = "resnet50",
        pretrained: bool = True,
        frozen_stages: int = 0,
    ):
        super().__init__()
        self.variant = variant
        self.out_channels_list = self.VARIANT_CHANNELS[variant]

        builder = getattr(models, variant)
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = builder(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # stride 4
        self.layer2 = resnet.layer2  # stride 8
        self.layer3 = resnet.layer3  # stride 16
        self.layer4 = resnet.layer4  # stride 32

        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int):
        if num_stages >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(num_stages, 4)):
            for p in layers[i].parameters():
                p.requires_grad = False

    @property
    def out_channels(self) -> List[int]:
        return self.out_channels_list

    @property
    def feature_keys(self) -> List[str]:
        return ["s2", "s3", "s4", "s5"]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return {"s2": s2, "s3": s3, "s4": s4, "s5": s5}


@BACKBONES.register("resnet18")
class ResNet18Backbone(ResNetBackbone):
    def __init__(self, pretrained=True, frozen_stages=0):
        super().__init__("resnet18", pretrained, frozen_stages)


@BACKBONES.register("resnet34")
class ResNet34Backbone(ResNetBackbone):
    def __init__(self, pretrained=True, frozen_stages=0):
        super().__init__("resnet34", pretrained, frozen_stages)


@BACKBONES.register("resnet50")
class ResNet50Backbone(ResNetBackbone):
    def __init__(self, pretrained=True, frozen_stages=0):
        super().__init__("resnet50", pretrained, frozen_stages)


@BACKBONES.register("resnet101")
class ResNet101Backbone(ResNetBackbone):
    def __init__(self, pretrained=True, frozen_stages=0):
        super().__init__("resnet101", pretrained, frozen_stages)
