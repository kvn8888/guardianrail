from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


def get_transformer_layer(model: nn.Module, layer: int) -> nn.Module:
    candidates = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        module: nn.Module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
            return module[layer]
        except (AttributeError, IndexError, TypeError):
            continue
    raise ValueError("Could not locate transformer layers for this model architecture.")


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

