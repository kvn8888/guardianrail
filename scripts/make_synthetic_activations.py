from __future__ import annotations

import argparse
import random

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument("--output", default="artifacts/synthetic_acts.pt")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    base_tokens = [
        "math",
        "GPU",
        "recipe",
        "city",
        "poem",
        "Python",
        "matrix",
        "music",
        "robot",
        "coffee",
        "cloud",
        "tensor",
    ]
    directions = torch.randn(args.features, args.dim)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    activations = torch.randn(args.batch, args.seq_len, args.dim) * 0.25
    token_strings: list[list[str]] = []
    texts: list[str] = []

    for row in range(args.batch):
        row_tokens: list[str] = []
        active_topic = row % args.features
        trigger = base_tokens[active_topic]
        for col in range(args.seq_len):
            token = random.choice(base_tokens)
            if col % 7 == active_topic:
                token = trigger
                activations[row, col] += directions[active_topic] * 4.0
            row_tokens.append(token)
        token_strings.append(row_tokens)
        texts.append(" ".join(row_tokens))

    torch.save(
        {
            "model": "synthetic",
            "layer": 0,
            "activations": activations,
            "attention_mask": torch.ones(args.batch, args.seq_len, dtype=torch.long),
            "texts": texts,
            "token_strings": token_strings,
            "description": "Synthetic activations with repeated topic-like directions.",
        },
        args.output,
    )
    print(f"Saved synthetic activations to {args.output}")


if __name__ == "__main__":
    main()

