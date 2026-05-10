# AMD First Checkpoint

This doc is the practical runbook for proving the real GuardianRail backend works on AMD MI300X.

## Goal

```text
Load Gemma 3 12B IT on AMD, hook layer 12, encode the activation with Gemma Scope 2, and run GuardianRail's real backend.
```

The first checkpoint is intentionally narrow. Do not debug frontend polish until this passes.

## Prerequisites

- AMD Developer Cloud GPU droplet with an MI300X.
- PyTorch/ROCm image or container.
- GitHub repo access:

```text
https://github.com/kvn8888/guardianrail
```

- Hugging Face token with accepted access for:

```text
google/gemma-3-12b-it
google/gemma-scope-2-12b-it
```

## Clone And Install On AMD

In the AMD Developer Cloud image used for this project, the ROCm container is named `rocm`.

```bash
ssh root@<amd-droplet-ip>
docker exec -it rocm bash
cd /workspace

git clone https://github.com/kvn8888/guardianrail.git
cd guardianrail

pip install -r requirements.txt
huggingface-cli login
```

## Checkpoint 1: Gemma Hook

```bash
python scripts/check_gemma_hook.py \
  --model google/gemma-3-12b-it \
  --layer 12 \
  --max-new-tokens 32
```

Expected output:

```text
Hook target: ...
activation[1] shape=(...)
Generated text:
...
Hook summary:
forward calls captured: ...
```

This proves:

- Gemma 3 loads.
- ROCm/PyTorch can run a forward pass.
- The target layer hook fires.

## Checkpoint 2: Gemma Scope 2 SAE Encode

```bash
python scripts/check_gemma_scope_sae.py \
  --model google/gemma-3-12b-it \
  --sae-repo google/gemma-scope-2-12b-it \
  --sae-path resid_post/layer_12_width_16k_l0_small \
  --layer 12 \
  --top-k 10
```

Expected output:

```text
SAE width: 16384
captured shape: (1, ..., 3840)
encoded shape: (1, 16384)
nonzero features: ...
Top features:
```

This proves GuardianRail can turn Gemma layer-12 activations into Gemma Scope feature values.

## Checkpoint 3: Real GuardianRail Terminal Demo

```bash
python scripts/run_real_guardian_demo.py --all-demo
```

Expected behavior:

```text
normal -> allow
prompt_injection -> refuse
social_engineering -> escalate
features print with activation / threshold values
```

This also writes audit events to:

```text
artifacts/guardianrail.sqlite3
```

## Checkpoint 4: Real Streamlit App

Run inside the AMD container:

```bash
GUARDIAN_BACKEND=real streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
```

Tunnel from the laptop:

```bash
ssh -N -L 127.0.0.1:8501:172.17.0.2:8501 root@<amd-droplet-ip>
```

Open:

```text
http://127.0.0.1:8501
```

## GPU Cost Discipline

Use one MI300X. The current AMD Cloud price seen during the hackathon was `$1.99/GPU-hour`, so `$100` is roughly 50 single-GPU hours.

Stop or destroy the droplet when not using it. Do not leave the GPU idle overnight.

## Troubleshooting

License or 403 failure:

```text
Accept the Gemma terms in Hugging Face, then rerun huggingface-cli login inside the AMD container.
```

Model class or tokenizer failure:

```bash
pip install --upgrade "transformers>=4.50.0" accelerate safetensors sentencepiece protobuf
```

Hook does not fire:

```text
Try --layer 10 or --layer 14.
Do not continue to SAE integration until one layer hook fires.
```

Streamlit unreachable:

```text
Confirm Streamlit is running inside Docker.
Confirm the SSH tunnel is active.
Use http://127.0.0.1:8501 locally.
```

First real prompt is slow:

```text
The real backend may load Gemma and SAE assets on first use.
Keep a pre-recorded demo video as backup.
```
