# ROCm Mechanistic Interpretability Workbench

Hackathon project for learning sparse features in a small open language model and demonstrating one causal intervention on AMD GPUs.

Current hackathon direction: [GuardianRail MVP](docs/guardianrail-mvp.md).

First GPU checkpoint: [AMD first checkpoint](docs/amd-first-checkpoint.md).

The intended scope is intentionally narrow:

1. Load one causal language model.
2. Capture one activation stream from one transformer layer.
3. Train one sparse autoencoder.
4. Inspect top activating text examples for learned features.
5. Ablate or steer one feature and compare next-token/logit behavior.

## Quick Start

Your MacBook does not need to run an LLM for local development. Locally, only run a synthetic test that pretends we already captured model activations. The real model run belongs on AMD Cloud, Colab, or another GPU machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Run the no-LLM local smoke test:

```bash
python scripts/make_synthetic_activations.py \
  --output artifacts/synthetic_acts.pt

python scripts/train_sae.py \
  --activations artifacts/synthetic_acts.pt \
  --hidden-mult 4 \
  --steps 100 \
  --batch-size 256 \
  --output artifacts/synthetic_sae.pt

python scripts/inspect_features.py \
  --activations artifacts/synthetic_acts.pt \
  --sae artifacts/synthetic_sae.pt \
  --top-k 5
```

If that prints repeated topic-like tokens such as `GPU`, `math`, or `tensor`, the local plumbing works.

## Real Model Runs

Do this only on a GPU machine:

```bash
pip install -r requirements.txt

python scripts/collect_activations.py \
  --model Qwen/Qwen3-4B-Base \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split train[:2048] \
  --layer 16 \
  --max-length 128 \
  --batch-size 2 \
  --output artifacts/qwen_layer16_acts.pt

python scripts/train_sae.py \
  --activations artifacts/qwen_layer16_acts.pt \
  --hidden-mult 8 \
  --steps 2000 \
  --batch-size 4096 \
  --output artifacts/qwen_layer16_sae.pt

python scripts/inspect_features.py \
  --activations artifacts/qwen_layer16_acts.pt \
  --sae artifacts/qwen_layer16_sae.pt \
  --top-k 10

python scripts/steer_feature.py \
  --model Qwen/Qwen3-4B-Base \
  --sae artifacts/qwen_layer16_sae.pt \
  --layer 16 \
  --feature 0 \
  --prompt "The future of AI interpretability is" \
  --strength 3.0
```

## AMD Cloud Run Shape

Use the same scripts on an AMD Developer Cloud PyTorch image. Start with one MI300X, not an 8-GPU droplet.

Candidate model flags:

```bash
--model Qwen/Qwen3-4B-Base
--model HuggingFaceTB/SmolLM3-3B-Base
--model allenai/OLMo-2-0425-1B
```

Keep the first credible run small: one middle layer, 2k-10k short sequences, residual stream activations, and a modest SAE expansion factor.
