from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import load_texts, tokenize_texts
from src.hooks import capture_layer_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train[:256]")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="artifacts/activations.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    texts = load_texts(args.dataset, args.dataset_config, args.split, args.text_column)
    collected: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    kept_texts: list[str] = []

    with torch.no_grad():
        for batch in tqdm(tokenize_texts(tokenizer, texts, args.max_length, args.batch_size)):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with capture_layer_output(model, args.layer) as activations:
                model(input_ids=input_ids, attention_mask=attention_mask)
            collected.append(activations[-1].float())
            masks.append(batch.attention_mask.cpu())
            kept_texts.extend(batch.texts)

    torch.save(
        {
            "model": args.model,
            "layer": args.layer,
            "activations": torch.cat(collected, dim=0),
            "attention_mask": torch.cat(masks, dim=0),
            "texts": kept_texts,
        },
        args.output,
    )
    print(f"Saved activations to {args.output}")


if __name__ == "__main__":
    main()
