from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Forward hooks intentionally mutate Python state. Eager mode avoids TorchDynamo
# recompiling on every generated token while we are validating hook access.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Gemma, generate once, and print hooked activation shapes.")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--prompt", default="You are Meridian Bank's support agent. Say hello in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def resolve_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model(model_id: str, dtype, device_map: str):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device_map:
        kwargs["device_map"] = device_map

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as causal_error:
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as import_error:
            raise RuntimeError(
                "AutoModelForCausalLM failed and this Transformers version does not provide "
                "AutoModelForImageTextToText. Install transformers>=4.50.0."
            ) from import_error

        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except Exception as image_text_error:
            raise RuntimeError(
                "Could not load the model with AutoModelForCausalLM or AutoModelForImageTextToText."
            ) from image_text_error


def first_parameter_device(model) -> torch.device:
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)

    import torch
    from transformers import AutoTokenizer

    from src.hooks import find_transformer_layer

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    model = load_model(args.model, dtype=dtype, device_map=args.device_map)
    model.eval()

    layer_path, layer_module = find_transformer_layer(model, args.layer)
    print(f"Hook target: {layer_path}[{args.layer}] -> {layer_module.__class__.__name__}")

    seen: list[dict[str, str | tuple[int, ...]]] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            seen.append({"shape": ("non_tensor",), "dtype": type(hidden).__name__, "device": "unknown"})
            return
        seen.append(
            {
                "shape": tuple(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": str(hidden.device),
            }
        )

    handle = layer_module.register_forward_hook(hook)
    try:
        encoded = tokenizer(args.prompt, return_tensors="pt")
        input_device = first_parameter_device(model)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(generated)
    print("\nHook summary:")
    print(f"forward calls captured: {len(seen)}")
    if seen:
        for idx, item in enumerate(seen[:4], start=1):
            print(
                f"activation[{idx}] shape={item['shape']} "
                f"dtype={item['dtype']} device={item['device']}"
            )
        print(f"first activation shape: {seen[0]['shape']}")
        print(f"last activation shape: {seen[-1]['shape']}")
    else:
        raise RuntimeError("Hook did not fire. Try another layer path or model class.")


if __name__ == "__main__":
    main()
