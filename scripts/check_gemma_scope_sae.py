from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.check_gemma_hook import load_model, resolve_dtype
from src.gemma_scope import load_gemma_scope_jumprelu_sae
from src.hooks import find_transformer_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode a real Gemma activation with a Gemma Scope 2 SAE.")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--sae-repo", default="google/gemma-scope-2-12b-it")
    parser.add_argument("--sae-path", default="resid_post/layer_12_width_16k_l0_small")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--prompt", default="You are a regulated support agent. Say hello in one sentence.")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def first_parameter_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

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
    sae = load_gemma_scope_jumprelu_sae(
        repo_id=args.sae_repo,
        sae_path=args.sae_path,
        device=device,
        dtype=torch.float32,
    )
    sae.eval()

    layer_path, layer_module = find_transformer_layer(model, args.layer)
    print(f"Hook target: {layer_path}[{args.layer}] -> {layer_module.__class__.__name__}")
    print(f"SAE hook point: {sae.config.hook_point_in}")
    print(f"SAE width: {sae.config.width}, target l0: {sae.config.l0}")

    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(hidden):
            captured.append(hidden.detach())

    handle = layer_module.register_forward_hook(hook)
    try:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model(**encoded)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Hook did not capture an activation.")

    activation = captured[-1][:, -1, :].float()
    with torch.no_grad():
        features = sae.encode(activation)

    nonzero = int((features > 0).sum().item())
    values, indices = torch.topk(features[0], k=min(args.top_k, features.shape[-1]))

    print("\nActivation summary:")
    print(f"captured shape: {tuple(captured[-1].shape)}")
    print(f"encoded shape: {tuple(features.shape)}")
    print(f"nonzero features: {nonzero}")
    print("\nTop features:")
    for rank, (feature_id, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f"{rank:02d}. feature={feature_id} activation={value:.6f}")


if __name__ == "__main__":
    main()
