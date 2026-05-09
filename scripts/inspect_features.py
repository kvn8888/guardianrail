from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import flatten_token_activations, top_feature_examples
from src.sae import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="artifacts/activations.pt")
    parser.add_argument("--sae", default="artifacts/sae.pt")
    parser.add_argument("--feature", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acts_payload = torch.load(args.activations, map_location="cpu")
    sae_payload = torch.load(args.sae, map_location="cpu")

    sae = SparseAutoencoder(sae_payload["input_dim"], sae_payload["hidden_dim"])
    sae.load_state_dict(sae_payload["state_dict"])
    sae.eval()

    x, positions = flatten_token_activations(acts_payload["activations"], acts_payload["attention_mask"])
    with torch.no_grad():
        features = sae.encode(x.float())

    feature_idx = args.feature
    if feature_idx is None:
        feature_idx = int(features.max(dim=0).values.argmax())

    examples = top_feature_examples(features, positions, feature_idx, args.top_k)
    print(f"Feature {feature_idx} top examples")

    token_strings = acts_payload.get("token_strings")
    if token_strings is not None:
        for score, batch_idx, seq_idx in examples:
            row_tokens = token_strings[batch_idx]
            token = row_tokens[seq_idx]
            context = " ".join(row_tokens[max(0, seq_idx - 8) : seq_idx + 9])
            print(f"{score:8.3f} token={token!r} context={context!r}")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(acts_payload["model"])
    for score, batch_idx, seq_idx in examples:
        ids = tokenizer(
            acts_payload["texts"][batch_idx],
            return_tensors="pt",
            truncation=True,
            max_length=acts_payload["attention_mask"].shape[1],
        )["input_ids"][0]
        token = tokenizer.decode(ids[seq_idx : seq_idx + 1])
        context = tokenizer.decode(ids[max(0, seq_idx - 12) : seq_idx + 12])
        print(f"{score:8.3f} token={token!r} context={context!r}")


if __name__ == "__main__":
    main()
