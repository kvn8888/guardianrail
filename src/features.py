from __future__ import annotations

import torch


def flatten_token_activations(acts: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if acts.ndim != 3:
        raise ValueError(f"Expected activations shaped [batch, seq, dim], got {tuple(acts.shape)}")
    mask = attention_mask.bool()
    flat_acts = acts[mask]
    token_positions = torch.nonzero(mask, as_tuple=False)
    return flat_acts, token_positions


def top_feature_examples(
    features: torch.Tensor,
    token_positions: torch.Tensor,
    feature_idx: int,
    top_k: int,
) -> list[tuple[float, int, int]]:
    values = features[:, feature_idx]
    scores, rows = torch.topk(values, k=min(top_k, values.numel()))
    out: list[tuple[float, int, int]] = []
    for score, row in zip(scores.tolist(), rows.tolist()):
        batch_idx, seq_idx = token_positions[row].tolist()
        out.append((score, batch_idx, seq_idx))
    return out

