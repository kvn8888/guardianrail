# GuardianRail

GuardianRail is an interpretable safety layer for open-weight customer support agents in regulated domains. The demo agent is a fictional Meridian Bank support assistant. GuardianRail monitors selected Gemma Scope SAE features, applies configurable policy-layer interventions when safety features cross threshold, and writes an auditable SQLite trail for every decision.

The hackathon pitch:

> Regulated teams may need open-weight models on their own infrastructure for cost, compliance, and data sovereignty. Those models do not ship with frontier-lab safety observability. GuardianRail adds an inspectable, tunable safety layer around an open-weight agent.

## Current MVP

What works:

- Streamlit demo app for the bank support agent.
- Mock backend for MacBook/local development.
- Real backend for AMD MI300X using `google/gemma-3-12b-it` and `google/gemma-scope-2-12b-it`.
- Live feature activation panel with calibrated layer-12 Guardian features.
- GPU use visualizer for MI300X VRAM/utilization/session burn.
- Feature Clamp Rail showing monitor, clamp, boost, and pause interventions.
- Custom Guardian Features panel for adding feature IDs, thresholds, actions, and intervention types.
- Text to Feature Finder that maps phrases like `system prompt override` to locally discovered candidate SAE features.
- SQLite audit log with prompt, response, feature ID, activation, threshold, action, intervention, and metadata.

Important limitation:

- The current MVP performs **real SAE feature monitoring** plus **policy-layer feature clamping and audit control**.
- It does **not yet perform true activation replacement inside the model forward pass**. The UI/control path is designed so true activation steering can be added next.

## Quick Start: Local MacBook Dev

Use this path for UI work, demo rehearsal, README/docs work, and mock backend behavior. Your MacBook does not need to run the LLM.

```bash
git clone https://github.com/kvn8888/guardianrail.git
cd guardianrail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt

streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

By default, the app uses the mock backend. You can still test:

- normal / prompt-injection / social-engineering demo prompts
- feature activation bars
- custom feature rules
- text-to-feature lookup
- feature clamp rail
- audit log UI

## Quick Start: AMD GPU Real Backend

Use this path when you need the actual Gemma 3 + Gemma Scope backend.

Prerequisites:

- AMD Developer Cloud GPU droplet, preferably one MI300X.
- PyTorch/ROCm image or container.
- Hugging Face token.
- Accepted Hugging Face access for:
  - `google/gemma-3-12b-it`
  - `google/gemma-scope-2-12b-it`

In the AMD Developer Cloud PyTorch image used for this project, the ROCm container is named `rocm`.

```bash
ssh root@<amd-droplet-ip>
docker exec -it rocm bash
cd /workspace

git clone https://github.com/kvn8888/guardianrail.git
cd guardianrail

pip install -r requirements.txt
huggingface-cli login
```

Run a backend smoke test:

```bash
python scripts/check_gemma_hook.py \
  --model google/gemma-3-12b-it \
  --layer 12 \
  --max-new-tokens 32
```

Run the real GuardianRail demo in the terminal:

```bash
python scripts/run_real_guardian_demo.py --all-demo
```

Run the Streamlit app with the real backend:

```bash
GUARDIAN_BACKEND=real streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
```

From your laptop, tunnel the app:

```bash
ssh -N -L 127.0.0.1:8501:172.17.0.2:8501 root@<amd-droplet-ip>
```

Open:

```text
http://127.0.0.1:8501
```

Stop the GPU VM/droplet when you are done. Powered-off or idle GPU resources may still cost money depending on the provider configuration.

## Demo Script

Use this for the live judge demo.

1. Open **Custom Guardian Features**.
2. In **Text to Feature Finder**, search:

```text
system prompt override
```

3. Confirm the finder suggests:

```text
feat_166 · hidden/system instruction request
```

4. Click **Clamp feat_166**.
5. Run **Prompt Injection**.
6. Show:
   - feature activation spike
   - clamp rail firing
   - refusal response
   - audit log row with intervention metadata
7. Run **Social Engineering**.
8. Show escalation instead of a simple refusal.

Narration:

> We describe a risk in plain English. GuardianRail maps it to a candidate SAE feature. We add it as a clamp rule. When a jailbreak prompt arrives, that feature crosses threshold, the intervention rail fires, and the audit log records the feature, threshold, action, and response path.

## Architecture

```text
Streamlit UI
  ├── chat/demo prompt panel
  ├── GPU visualizer
  ├── feature activation panel
  ├── feature clamp rail
  ├── custom feature rule editor
  ├── text-to-feature finder
  └── audit log table

Guardian controller
  ├── mock backend for local dev
  └── real backend on AMD MI300X
        ├── Gemma 3 12B IT
        ├── layer-12 residual hook
        ├── Gemma Scope 2 SAE encoder
        ├── guardian rule evaluation
        └── SQLite audit write
```

## Key Files

```text
frontend/app.py                 Streamlit app
src/real_guardian.py            Real Gemma + Gemma Scope backend
src/mock_guardian.py            Fast local mock backend
src/interventions.py            Clamp / boost / pause intervention construction
src/rules.py                    Custom rule normalization and rule matching
src/feature_search.py           Text-to-feature finder over local feature scan
src/audit.py                    SQLite audit log
src/gpu_monitor.py              MI300X telemetry helper
artifacts/guardian_rules_layer12.json
artifacts/guardian_candidates_layer12.json
scripts/run_real_guardian_demo.py
scripts/check_gemma_hook.py
```

## Text to Feature Finder

The finder is intentionally modest. It searches the local GuardianRail feature catalog and contrastive scan, not every SAE feature in Gemma Scope.

Good queries:

```text
system prompt override
jailbreak hidden rules
unauthorized transfer
verification bypass
social engineering distress
```

Click **Clamp feat_X** to add a refusal/clamp custom rule, or **Escalate feat_X** to add a pause/escalation rule.

## Audit Log

Audit events are stored in:

```text
artifacts/guardianrail.sqlite3
```

Each event includes:

- prompt
- response
- model ID
- SAE release/path
- layer
- feature ID and label
- activation and threshold
- action
- intervention summary
- custom rule metadata

SQLite DB files are git-ignored.

## Troubleshooting

If the MacBook is slow or cannot load the model:

- Use local mock mode only: `streamlit run frontend/app.py`.
- Do not use `GUARDIAN_BACKEND=real` locally.

If the AMD backend fails to load Gemma:

- Confirm the Hugging Face token is logged in inside the AMD container.
- Confirm both Gemma access gates were accepted on Hugging Face.
- Re-run `python scripts/check_gemma_hook.py --model google/gemma-3-12b-it --layer 12 --max-new-tokens 32`.

If the browser cannot reach the remote Streamlit app:

- Confirm Streamlit is running inside the container.
- Confirm the SSH tunnel is active.
- Use `http://127.0.0.1:8501`, not the raw container URL.

If the real backend is slow:

- The first request may load model weights and SAE assets.
- Use a pre-recorded demo video for judging backup.
- The mock backend is acceptable for UI rehearsals, but real backend evidence should be shown in screenshots/video.

## Related Docs

- [GuardianRail MVP](docs/guardianrail-mvp.md)
- [AMD first checkpoint](docs/amd-first-checkpoint.md)
- [Guardian feature candidates](docs/guardian-feature-candidates.md)
- [Guardian rules](docs/guardian-rules.md)
