from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm, trange

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import flatten_token_activations
from src.sae import SparseAutoencoder, sae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="artifacts/activations.pt")
    parser.add_argument("--hidden-mult", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--output", default="artifacts/sae.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.activations, map_location="cpu")
    x, _positions = flatten_token_activations(payload["activations"], payload["attention_mask"])
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = x.shape[-1]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=input_dim * args.hidden_mult).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    for step in trange(args.steps):
        idx = torch.tensor(random.choices(range(x.shape[0]), k=args.batch_size))
        batch = x[idx].to(device)
        recon, features = sae(batch)
        loss, metrics = sae_loss(batch, recon, features, args.l1_coeff)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            tqdm.write(
                f"step={step} loss={metrics['loss']:.4f} mse={metrics['mse']:.4f} l1={metrics['l1']:.4f}"
            )

    torch.save(
        {
            "state_dict": sae.cpu().state_dict(),
            "input_dim": input_dim,
            "hidden_dim": input_dim * args.hidden_mult,
            "source": args.activations,
        },
        args.output,
    )
    print(f"Saved SAE to {args.output}")


if __name__ == "__main__":
    main()
