"""Object detection heads."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation import HEADS


@HEADS.register("anchor_free_det_head")
class AnchorFreeDetHead(nn.Module):
    """FCOS-style anchor-free detection head.

    Produces per-level classification, regression, and centerness predictions.

    Args:
        in_channels_list: Channel dimensions for each input scale.
        mid_channels: Intermediate channels in the conv stacks.
        num_classes: Number of object classes.
        num_convs: Number of stacked convolutions per branch.
        input_keys: Which feature keys to consume.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        mid_channels: int = 256,
        num_classes: int = 80,
        num_convs: int = 4,
        input_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys or ["s3", "s4", "s5"]
        self.num_classes = num_classes

        # Shared lateral 1x1 to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, mid_channels, 1)
            for in_ch in in_channels_list
        ])

        # Classification branch
        cls_layers = []
        for i in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, mid_channels),
                nn.ReLU(inplace=True),
            ])
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(mid_channels, num_classes, 3, padding=1)

        # Regression branch
        reg_layers = []
        for i in range(num_convs):
            reg_layers.extend([
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, mid_channels),
                nn.ReLU(inplace=True),
            ])
        self.reg_tower = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(mid_channels, 4, 3, padding=1)

        # Centerness branch (shared with cls tower)
        self.centerness = nn.Conv2d(mid_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.cls_tower, self.reg_tower]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize cls logits bias for focal loss
        nn.init.constant_(self.cls_logits.bias, -2.0)

    def forward_single(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        return {
            "cls_logits": self.cls_logits(cls_feat),
            "reg_pred": F.relu(self.reg_pred(reg_feat)),
            "centerness": self.centerness(cls_feat),
            "cls_feat": cls_feat,
            "reg_feat": reg_feat,
        }

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with per-level predictions stacked into lists:
              - "det_cls_logits": list of [B, num_classes, H, W]
              - "det_reg_preds":  list of [B, 4, H, W]
              - "det_centerness": list of [B, 1, H, W]
              - "det_cls_feats":  list of intermediate cls features
              - "det_reg_feats":  list of intermediate reg features
        """
        feats = [features[k] for k in self.input_keys]
        projected = [conv(f) for conv, f in zip(self.lateral_convs, feats)]

        all_cls, all_reg, all_ctr = [], [], []
        all_cls_feats, all_reg_feats = [], []

        for feat in projected:
            out = self.forward_single(feat)
            all_cls.append(out["cls_logits"])
            all_reg.append(out["reg_pred"])
            all_ctr.append(out["centerness"])
            all_cls_feats.append(out["cls_feat"])
            all_reg_feats.append(out["reg_feat"])

        return {
            "det_cls_logits": all_cls,
            "det_reg_preds": all_reg,
            "det_centerness": all_ctr,
            "det_cls_feats": all_cls_feats,
            "det_reg_feats": all_reg_feats,
        }


@HEADS.register("ssd_det_head")
class SSDDetHead(nn.Module):
    """SSD-style multi-box detection head.

    Args:
        in_channels_list: Channel dims for each input scale.
        num_classes: Number of object classes.
        num_anchors_per_level: Number of anchor boxes per spatial location.
        input_keys: Which feature keys to consume.
    """

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = 80,
        num_anchors_per_level: List[int] = None,
        input_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_keys = input_keys or ["s3", "s4", "s5"]
        self.num_classes = num_classes

        if num_anchors_per_level is None:
            num_anchors_per_level = [6] * len(in_channels_list)

        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for in_ch, n_anchors in zip(in_channels_list, num_anchors_per_level):
            self.cls_heads.append(
                nn.Conv2d(in_ch, n_anchors * num_classes, 3, padding=1)
            )
            self.reg_heads.append(
                nn.Conv2d(in_ch, n_anchors * 4, 3, padding=1)
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = [features[k] for k in self.input_keys]
        all_cls, all_reg = [], []

        for feat, cls_head, reg_head in zip(feats, self.cls_heads, self.reg_heads):
            all_cls.append(cls_head(feat))
            all_reg.append(reg_head(feat))

        return {
            "det_cls_logits": all_cls,
            "det_reg_preds": all_reg,
        }
