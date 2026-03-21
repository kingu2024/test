from .utils.registry import Registry

BACKBONES = Registry("backbones")
HEADS = Registry("heads")
LOSSES = Registry("losses")
DISTILLERS = Registry("distillers")
ALIGNERS = Registry("aligners")
