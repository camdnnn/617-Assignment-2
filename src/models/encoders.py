from collections.abc import Iterator
from typing import Callable, Dict, Protocol

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from transformers import DistilBertModel


class ImageEncoderModule(Protocol):
    out_dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


class TextEncoderModule(Protocol):
    out_dim: int

    def __call__(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...


ImageEncoderBuilder = Callable[[], ImageEncoderModule]
TextEncoderBuilder = Callable[[str], TextEncoderModule]

IMAGE_REGISTRY: Dict[str, ImageEncoderBuilder] = {}
TEXT_REGISTRY: Dict[str, TextEncoderBuilder] = {}


def _register(registry: Dict, name: str):
    def dec(fn):
        registry[name] = fn
        return fn

    return dec


def build_image_encoder(name: str) -> ImageEncoderModule:
    if name not in IMAGE_REGISTRY:
        raise ValueError(
            f"Unknown image encoder '{name}'. Available: {list(IMAGE_REGISTRY)}"
        )
    return IMAGE_REGISTRY[name]()


def build_text_encoder(name: str, model_name: str) -> TextEncoderModule:
    if name not in TEXT_REGISTRY:
        raise ValueError(
            f"Unknown text encoder '{name}'. Available: {list(TEXT_REGISTRY)}"
        )
    return TEXT_REGISTRY[name](model_name)


@_register(IMAGE_REGISTRY, "convnext_tiny")
class ConvNeXtEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features, self.avgpool = base.features, base.avgpool
        self.out_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        return torch.flatten(x, 1)


@_register(TEXT_REGISTRY, "distilbert")
class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
