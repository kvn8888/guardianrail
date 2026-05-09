from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hooks import get_transformer_layer
from src.sae import SparseAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--sae", default="artifacts/sae.pt")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    return parser.parse_args()


def next_token_distribution(model, tokenizer, prompt: str) -> list[tuple[str, float]]:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**encoded).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    values, ids = torch.topk(probs, k=10)
    return [(tokenizer.decode([idx]), float(prob)) for idx, prob in zip(ids.tolist(), values.tolist())]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    sae_payload = torch.load(args.sae, map_location=device)
    sae = SparseAutoencoder(sae_payload["input_dim"], sae_payload["hidden_dim"]).to(device)
    sae.load_state_dict(sae_payload["state_dict"])
    direction = sae.decoder.weight[:, args.feature].detach()

    print("Baseline")
    for token, prob in next_token_distribution(model, tokenizer, args.prompt):
        print(f"{token!r}: {prob:.4f}")

    target = get_transformer_layer(model, args.layer)

    def steering_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[:, -1, :] += args.strength * direction.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = target.register_forward_hook(steering_hook)
    try:
        print("\nSteered")
        for token, prob in next_token_distribution(model, tokenizer, args.prompt):
            print(f"{token!r}: {prob:.4f}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
