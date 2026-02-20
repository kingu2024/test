"""Multi-task model with switchable heads and feature extraction hooks."""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from distillation import BACKBONES, HEADS

# Trigger registration of all components
import distillation.backbones  # noqa: F401
import distillation.heads  # noqa: F401


class MultiTaskModel(nn.Module):
    """A model that pairs a backbone with multiple switchable task heads.

    Supports:
      - Multiple heads (segmentation, detection, etc.) attached simultaneously.
      - Selectively running a subset of heads per forward pass.
      - Swapping heads between teacher/student for cross-prediction.
      - Exposing intermediate backbone features for distillation.

    Args:
        backbone_name: Registered backbone name.
        backbone_cfg: Kwargs for backbone construction.
        heads_cfg: Dict of {head_name: {"type": registered_name, **kwargs}}.
            The in_channels / in_channels_list will be auto-filled from the backbone
            if not explicitly provided.

    Example:
        model = MultiTaskModel(
            backbone_name="resnet50",
            backbone_cfg={"pretrained": True},
            heads_cfg={
                "seg": {"type": "fpn_seg_head", "num_classes": 19},
                "det": {"type": "anchor_free_det_head", "num_classes": 80},
            },
        )
        out = model(images, active_heads=["seg"])
        out = model(images, active_heads=["det"])
        out = model(images)  # runs all heads
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_cfg: Optional[dict] = None,
        heads_cfg: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()

        backbone_cfg = backbone_cfg or {}
        self.backbone = BACKBONES.build(backbone_name, **backbone_cfg)

        self.heads = nn.ModuleDict()
        self._head_types: Dict[str, str] = {}

        if heads_cfg:
            for head_name, cfg in heads_cfg.items():
                self.add_head(head_name, cfg)

        # Feature hooks for distillation
        self._feature_hooks: Dict[str, torch.Tensor] = {}
        self._hooked_layers: Dict[str, torch.utils.hooks.RemovableHook] = {}

    def add_head(self, name: str, cfg: dict):
        """Add a new task head to the model.

        The config should include a "type" key for the registered head name.
        in_channels_list / in_channels will be inferred from the backbone
        if not explicitly provided.
        """
        cfg = dict(cfg)  # copy to avoid mutation
        head_type = cfg.pop("type")
        self._head_types[name] = head_type

        # Auto-fill channel info based on head type requirements
        head_cls = HEADS.get(head_type)
        init_params = head_cls.__init__.__code__.co_varnames

        if "in_channels_list" in init_params and "in_channels_list" not in cfg:
            input_keys = cfg.get("input_keys", None)
            if input_keys:
                key_to_ch = dict(zip(self.backbone.feature_keys,
                                     self.backbone.out_channels))
                cfg["in_channels_list"] = [key_to_ch[k] for k in input_keys]
            else:
                cfg["in_channels_list"] = self.backbone.out_channels

        if "in_channels" in init_params and "in_channels" not in cfg:
            input_key = cfg.get("input_key", "s4")
            key_to_ch = dict(zip(self.backbone.feature_keys,
                                 self.backbone.out_channels))
            cfg["in_channels"] = key_to_ch[input_key]

        self.heads[name] = HEADS.build(head_type, **cfg)

    def remove_head(self, name: str):
        """Remove a task head by name."""
        if name in self.heads:
            del self.heads[name]
            del self._head_types[name]

    def get_head(self, name: str) -> nn.Module:
        """Get a head module by name."""
        return self.heads[name]

    def swap_head(self, name: str, new_head: nn.Module):
        """Replace a head with a new module (e.g., from another model)."""
        self.heads[name] = new_head

    def attach_head_from(self, source_model: "MultiTaskModel", head_name: str,
                         alias: Optional[str] = None):
        """Borrow a head from another model (shares parameters).

        Args:
            source_model: The model to borrow the head from.
            head_name: Name of the head in the source model.
            alias: Name to use in this model (defaults to head_name).
        """
        target_name = alias or head_name
        self.heads[target_name] = source_model.heads[head_name]
        self._head_types[target_name] = source_model._head_types[head_name]

    def register_feature_hook(self, layer_name: str):
        """Register a forward hook on a named backbone sub-module.

        This enables extracting intermediate features for distillation.

        Args:
            layer_name: Dot-separated path to the sub-module
                        (e.g., "backbone.layer3" or "layer3").
        """
        # Try to resolve relative to backbone first, then model root
        try:
            module = dict(self.backbone.named_modules())[layer_name]
        except KeyError:
            module = dict(self.named_modules())[layer_name]

        def hook_fn(module, input, output):
            self._feature_hooks[layer_name] = output

        handle = module.register_forward_hook(hook_fn)
        self._hooked_layers[layer_name] = handle

    def remove_feature_hooks(self):
        """Remove all registered feature hooks."""
        for handle in self._hooked_layers.values():
            handle.remove()
        self._hooked_layers.clear()
        self._feature_hooks.clear()

    def get_hooked_features(self) -> Dict[str, torch.Tensor]:
        """Return features captured by registered hooks."""
        return dict(self._feature_hooks)

    @property
    def head_names(self) -> List[str]:
        return list(self.heads.keys())

    def forward(
        self,
        x: torch.Tensor,
        active_heads: Optional[Set[str]] = None,
        target_size: Optional[tuple] = None,
    ) -> Dict[str, dict]:
        """Forward pass.

        Args:
            x: Input images [B, C, H, W].
            active_heads: Set of head names to run. None = all heads.
            target_size: Optional (H, W) passed to heads that support it.

        Returns:
            Dict of {head_name: head_output_dict}, plus a "features" entry
            containing the backbone multi-scale features.
        """
        self._feature_hooks.clear()

        features = self.backbone(x)

        outputs: Dict[str, dict] = {"features": features}

        heads_to_run = active_heads or set(self.heads.keys())
        for name in heads_to_run:
            if name not in self.heads:
                raise KeyError(
                    f"Head '{name}' not found. Available: {self.head_names}"
                )
            head = self.heads[name]
            # Pass target_size if the head's forward accepts it
            sig_params = head.forward.__code__.co_varnames
            if "target_size" in sig_params:
                outputs[name] = head(features, target_size=target_size)
            else:
                outputs[name] = head(features)

        return outputs

    def forward_backbone_only(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run only the backbone, returning multi-scale features."""
        self._feature_hooks.clear()
        return self.backbone(x)
