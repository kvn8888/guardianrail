from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.check_gemma_hook import load_model, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find SAE features that separate adversarial prompts from benign prompts.")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--sae-repo", default="google/gemma-scope-2-12b-it")
    parser.add_argument("--sae-path", default="resid_post/layer_12_width_16k_l0_small")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--benign", default="data/prompts/benign.txt")
    parser.add_argument("--adversarial", default="data/prompts/adversarial.txt")
    parser.add_argument("--output-json", default="artifacts/contrastive_features_layer12.json")
    parser.add_argument("--output-csv", default="artifacts/contrastive_features_layer12.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--last-n", type=int, default=4)
    parser.add_argument(
        "--aggregation",
        choices=("last-n-mean", "max-token"),
        default="last-n-mean",
        help="How to pool token-level SAE features into one vector per prompt.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--rank-by",
        choices=("candidate-score", "z-score", "diff"),
        default="candidate-score",
    )
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def read_prompts(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def first_parameter_device(model) -> torch.device:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def prompt_activation(hidden: torch.Tensor, attention_mask: torch.Tensor, last_n: int) -> torch.Tensor:
    import torch

    rows: list[torch.Tensor] = []
    for row_idx in range(hidden.shape[0]):
        token_count = int(attention_mask[row_idx].sum().item())
        start = max(0, token_count - last_n)
        rows.append(hidden[row_idx, start:token_count, :].float().mean(dim=0))
    return torch.stack(rows, dim=0)


def prompt_feature_max(hidden: torch.Tensor, attention_mask: torch.Tensor, sae) -> torch.Tensor:
    import torch

    rows: list[torch.Tensor] = []
    for row_idx in range(hidden.shape[0]):
        token_count = int(attention_mask[row_idx].sum().item())
        token_acts = hidden[row_idx, :token_count, :].float()
        with torch.no_grad():
            token_features = sae.encode(token_acts)
        rows.append(token_features.max(dim=0).values)
    return torch.stack(rows, dim=0)


def encode_prompts(
    model,
    tokenizer,
    sae,
    layer: int,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    last_n: int,
    aggregation: str,
    device: torch.device,
) -> torch.Tensor:
    import torch
    from tqdm import tqdm

    from src.hooks import find_transformer_layer

    _layer_path, layer_module = find_transformer_layer(model, layer)
    encoded_batches: list[torch.Tensor] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts"):
        batch_prompts = prompts[start : start + batch_size]
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                captured.append(hidden.detach())

        handle = layer_module.register_forward_hook(hook)
        try:
            batch = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                model(**batch)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Hook did not capture activations.")

        if aggregation == "last-n-mean":
            acts = prompt_activation(captured[-1], batch["attention_mask"], last_n=last_n)
            with torch.no_grad():
                features = sae.encode(acts)
        elif aggregation == "max-token":
            features = prompt_feature_max(captured[-1], batch["attention_mask"], sae)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        encoded_batches.append(features.detach().cpu())

    return torch.cat(encoded_batches, dim=0)


def rank_features(
    benign: torch.Tensor,
    adversarial: torch.Tensor,
    benign_prompts: list[str],
    adversarial_prompts: list[str],
    top_k: int,
    rank_by: str,
) -> list[dict[str, float | int | str]]:
    import torch

    benign_mean = benign.mean(dim=0)
    adversarial_mean = adversarial.mean(dim=0)
    benign_std = benign.std(dim=0, unbiased=False)
    adversarial_std = adversarial.std(dim=0, unbiased=False)
    pooled = torch.sqrt((benign_std.square() + adversarial_std.square()) / 2).clamp_min(1e-6)
    diff = adversarial_mean - benign_mean
    score = diff / pooled
    benign_active = (benign > 0).float().mean(dim=0)
    adversarial_active = (adversarial > 0).float().mean(dim=0)
    active_delta = adversarial_active - benign_active
    candidate_score = diff.clamp_min(0) * active_delta.clamp_min(0)

    if rank_by == "candidate-score":
        ranking = candidate_score
    elif rank_by == "z-score":
        ranking = score
    elif rank_by == "diff":
        ranking = diff
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    values, indices = torch.topk(ranking, k=min(top_k, ranking.numel()))
    rows: list[dict[str, float | int | str]] = []
    for idx, value in zip(indices.tolist(), values.tolist()):
        top_adv_idx = int(torch.argmax(adversarial[:, idx]).item())
        top_benign_idx = int(torch.argmax(benign[:, idx]).item())
        rows.append(
            {
                "feature_id": int(idx),
                "candidate_score": float(candidate_score[idx]),
                "z_score": float(score[idx]),
                "adv_mean": float(adversarial_mean[idx]),
                "benign_mean": float(benign_mean[idx]),
                "diff": float(diff[idx]),
                "adv_active_frac": float(adversarial_active[idx]),
                "benign_active_frac": float(benign_active[idx]),
                "active_delta": float(active_delta[idx]),
                "top_adv_value": float(adversarial[top_adv_idx, idx]),
                "top_adv_prompt": adversarial_prompts[top_adv_idx],
                "top_benign_value": float(benign[top_benign_idx, idx]),
                "top_benign_prompt": benign_prompts[top_benign_idx],
                "rank_value": float(value),
            }
        )
    return rows


def write_outputs(
    rows: list[dict[str, float | int | str]],
    output_json: str | Path,
    output_csv: str | Path,
    metadata: dict[str, object],
) -> None:
    json_path = Path(output_json)
    csv_path = Path(output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps({"metadata": metadata, "features": rows}, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoTokenizer

    from src.gemma_scope import load_gemma_scope_jumprelu_sae
    from src.hooks import find_transformer_layer

    benign_prompts = read_prompts(args.benign)
    adversarial_prompts = read_prompts(args.adversarial)
    dtype = resolve_dtype(args.dtype)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    model = load_model(args.model, dtype=dtype, device_map=args.device_map)
    model.eval()
    device = first_parameter_device(model)

    print(f"Loading SAE: {args.sae_repo}/{args.sae_path}")
    sae = load_gemma_scope_jumprelu_sae(args.sae_repo, args.sae_path, device=device, dtype=torch.float32)
    sae.eval()

    layer_path, layer_module = find_transformer_layer(model, args.layer)
    print(f"Hook target: {layer_path}[{args.layer}] -> {layer_module.__class__.__name__}")
    print(f"Benign prompts: {len(benign_prompts)}")
    print(f"Adversarial prompts: {len(adversarial_prompts)}")

    benign_features = encode_prompts(
        model,
        tokenizer,
        sae,
        args.layer,
        benign_prompts,
        args.batch_size,
        args.max_length,
        args.last_n,
        args.aggregation,
        device,
    )
    adversarial_features = encode_prompts(
        model,
        tokenizer,
        sae,
        args.layer,
        adversarial_prompts,
        args.batch_size,
        args.max_length,
        args.last_n,
        args.aggregation,
        device,
    )

    rows = rank_features(
        benign_features,
        adversarial_features,
        benign_prompts=benign_prompts,
        adversarial_prompts=adversarial_prompts,
        top_k=args.top_k,
        rank_by=args.rank_by,
    )
    metadata = {
        "model": args.model,
        "sae_repo": args.sae_repo,
        "sae_path": args.sae_path,
        "layer": args.layer,
        "benign_count": len(benign_prompts),
        "adversarial_count": len(adversarial_prompts),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "last_n": args.last_n,
        "aggregation": args.aggregation,
        "rank_by": args.rank_by,
    }
    write_outputs(rows, args.output_json, args.output_csv, metadata)

    print("\nTop contrastive features:")
    for rank, row in enumerate(rows[: min(15, len(rows))], start=1):
        print(
            f"{rank:02d}. feature={row['feature_id']} candidate_score={row['candidate_score']:.3f} "
            f"z={row['z_score']:.3f} "
            f"diff={row['diff']:.3f} adv_mean={row['adv_mean']:.3f} "
            f"benign_mean={row['benign_mean']:.3f} "
            f"active={row['adv_active_frac']:.2f}/{row['benign_active_frac']:.2f}"
        )
    print(f"\nWrote {args.output_json}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
