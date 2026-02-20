"""Component registry for flexible module lookup and instantiation."""

from typing import Any, Dict, Optional, Type


class Registry:
    """A registry that maps string names to classes.

    Usage:
        BACKBONES = Registry("backbones")

        @BACKBONES.register("resnet18")
        class ResNet18(nn.Module):
            ...

        model = BACKBONES.build("resnet18", pretrained=True)
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None):
        """Decorator to register a class under a given name."""
        def wrapper(cls):
            key = name or cls.__name__
            if key in self._registry:
                raise KeyError(
                    f"'{key}' is already registered in {self._name}"
                )
            self._registry[key] = cls
            return cls
        return wrapper

    def get(self, name: str) -> Type:
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """Instantiate a registered class with given kwargs."""
        cls = self.get(name)
        return cls(**kwargs)

    def list_available(self):
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"
