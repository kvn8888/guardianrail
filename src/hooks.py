from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def find_transformer_layer(model: nn.Module, layer: int) -> tuple[str, nn.Module]:
    candidates = (
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        module: nn.Module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
            return path, module[layer]
        except (AttributeError, IndexError, TypeError):
            continue

    for name, module in model.named_modules():
        if not name.endswith(("layers", "h")):
            continue
        try:
            candidate = module[layer]
        except (IndexError, TypeError, KeyError):
            continue
        if isinstance(candidate, nn.Module):
            return name, candidate

    raise ValueError("Could not locate transformer layers for this model architecture.")


def get_transformer_layer(model: nn.Module, layer: int) -> nn.Module:
    _path, module = find_transformer_layer(model, layer)
    return module


@contextmanager
def capture_layer_output(model: nn.Module, layer: int) -> Iterator[list[torch.Tensor]]:
    activations: list[torch.Tensor] = []
    target = get_transformer_layer(model, layer)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        activations.append(hidden.detach().cpu())

    handle = target.register_forward_hook(hook)
    try:
        yield activations
    finally:
        handle.remove()
