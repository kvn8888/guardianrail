# AMD First Checkpoint

Goal:

```text
Load Gemma 3 IT on the AMD GPU, generate one short response, hook layer 12, and print activation shapes.
```

This is the first thing to do on the GPU VM. Do not start SAE loading, feature hunting, or frontend integration until this passes.

## On The AMD VM

Use the ROCm/PyTorch image if AMD Developer Cloud offers one. After SSH:

```bash
git clone https://github.com/<your-username>/guardianrail.git
cd guardianrail

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

Make sure the Hugging Face account tied to the token has accepted access for:

```text
google/gemma-3-12b-it
google/gemma-scope-2-12b-it
```

Run:

```bash
python scripts/check_gemma_hook.py \
  --model google/gemma-3-12b-it \
  --layer 12 \
  --max-new-tokens 32
```

Expected output includes:

```text
Hook target: ...
activation[1] shape=(...)
Generated text:
...
Hook summary:
forward calls captured: ...
```

If this works, Person A's hour-4 checkpoint is complete.

The checkpoint script disables Torch compile/Dynamo by default because forward hooks mutate Python state while recording activation shapes. Leave that behavior in place for hook validation.

## Next Checkpoint: Gemma Scope 2 SAE Encode

After the hook checkpoint passes, encode one captured activation through the matching layer-12 residual SAE:

```bash
python scripts/check_gemma_scope_sae.py \
  --model google/gemma-3-12b-it \
  --sae-repo google/gemma-scope-2-12b-it \
  --sae-path resid_post/layer_12_width_16k_l0_small \
  --layer 12 \
  --top-k 10
```

Expected output includes:

```text
SAE width: 16384
captured shape: (1, ..., 3840)
encoded shape: (1, 16384)
nonzero features: ...
Top features:
```

If this works, we have the core GuardianRail technical path: Gemma activation hook plus Gemma Scope 2 feature encoding.

## If 12B Fails

Switch to the smaller fallback:

```bash
python scripts/check_gemma_hook.py \
  --model google/gemma-3-4b-it \
  --layer 12 \
  --max-new-tokens 32
```

If that works, continue the MVP with the 4B model and matching Gemma Scope 2 4B IT SAE.

## Troubleshooting

If loading fails with a license or 403 error:

```text
Accept the Gemma model terms in Hugging Face, then rerun huggingface-cli login.
```

If loading fails because the model class is unknown:

```bash
pip install --upgrade "transformers>=4.50.0" accelerate safetensors sentencepiece protobuf
```

If the hook does not fire:

```text
Try --layer 10 or --layer 14.
Copy the printed error into the team chat.
Do not continue to SAE integration until a hook fires.
```
