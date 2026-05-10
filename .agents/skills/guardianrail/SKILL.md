---
name: guardianrail
description: Use when working on, explaining, demoing, or modifying the GuardianRail hackathon project. Covers the Streamlit app, mock vs real AMD/Gemma backend, Gemma Scope feature monitoring, custom feature rules, text-to-feature finder, policy-layer clamp rail, SQLite audit log, setup commands, and the exact limitations to state when answering judges or future developers.
---

# GuardianRail Project Skill

## Use This Skill When

Use this skill for GuardianRail implementation questions, repo onboarding, demo prep, docs, bug fixes, README updates, pitch wording, or judging/interview explanations.

GuardianRail is an interpretable safety layer for regulated customer-operations agents. The demo uses a fictional regulated service desk, but the product framing should stay broader than any one industry. It monitors Gemma Scope SAE features for `google/gemma-3-12b-it`, applies configurable policy-layer interventions, and logs every decision.

## Core Claim

Say:

```text
GuardianRail performs real SAE feature monitoring with policy-layer feature clamping and audit control for an open-weight regulated support agent.
```

Do not say:

```text
We solved jailbreaks.
This is model-agnostic.
We trained Gemma Scope.
This performs true activation replacement inside the model forward pass.
```

Important limitation:

```text
The current clamp rail is policy-layer clamping. GuardianRail reads real SAE features and records clamp/boost/pause interventions, but it does not yet decode edited SAE features back into Gemma's residual stream.
```

## Architecture

```text
frontend/app.py
  Streamlit UI
  custom rule editor
  text-to-feature finder
  demo prompt buttons
  GPU visualizer
  feature bars
  clamp rail
  audit table

src/mock_guardian.py
  local mock backend, no LLM required

src/real_guardian.py
  real AMD/Gemma backend
  loads Gemma 3 12B IT
  hooks layer 12
  encodes activations with Gemma Scope 2
  evaluates rules
  writes audit events

src/interventions.py
  builds monitor/clamp/boost/pause intervention ledger entries

src/rules.py
  normalizes default and custom GuardianRule objects
  merges custom rules over default rules
  identifies crossed rules and custom decisive actions

src/feature_search.py
  text-to-feature lookup over local candidate artifacts

src/audit.py
  SQLite audit schema and read/write helpers

src/gpu_monitor.py
  MI300X telemetry helper
```

## Backends

Local default:

```bash
streamlit run frontend/app.py
```

This uses the mock backend. It is for UI work, demo rehearsal, and docs.

Real AMD backend:

```bash
GUARDIAN_BACKEND=real streamlit run frontend/app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
```

The real backend should run inside the AMD ROCm container, not on a MacBook.

## Setup Commands

Local MacBook:

```bash
git clone https://github.com/kvn8888/guardianrail.git
cd guardianrail
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
streamlit run frontend/app.py
```

AMD GPU:

```bash
ssh root@<amd-droplet-ip>
docker exec -it rocm bash
cd /workspace
git clone https://github.com/kvn8888/guardianrail.git
cd guardianrail
pip install -r requirements.txt
huggingface-cli login
```

Required Hugging Face access:

```text
google/gemma-3-12b-it
google/gemma-scope-2-12b-it
```

Tunnel the remote Streamlit app:

```bash
ssh -N -L 127.0.0.1:8501:172.17.0.2:8501 root@<amd-droplet-ip>
```

Open:

```text
http://127.0.0.1:8501
```

## Real Backend Data Flow

For `GUARDIAN_BACKEND=real`:

1. `frontend/app.py` calls `get_real_guardian().run_and_audit(...)`.
2. `RealGuardian` loads default rules from `artifacts/guardian_rules_layer12.json`.
3. UI custom rules are passed per prompt.
4. `src/rules.py` normalizes and merges default/custom rules.
5. `RealGuardian.extract_features` tokenizes the prompt and hooks layer 12.
6. `prompt_feature_max` encodes hooked activations with Gemma Scope 2.
7. `RealGuardian.decide` chooses `allow`, `refuse`, or `escalate`.
8. `build_interventions` builds the clamp/boost/pause/monitor ledger.
9. `src/audit.py` writes an SQLite row.
10. Streamlit renders feature bars, clamp rail, and audit log.

## Custom Rules

Custom rules are created in the Streamlit **Custom Guardian Features** panel.

Fields:

```text
feature_id
label
threshold
action: monitor | refuse | escalate
intervention: monitor | clamp | boost | pause
clamp_target
source = custom
enabled = true
```

Custom rules override default rules with the same `feature_id`.

For example, the preset clamps:

```text
feat_166 - hidden/system instruction request
action = refuse
intervention = clamp
threshold = 1.0
clamp_target = 0.0
```

## Text To Feature Finder

`src/feature_search.py` searches:

```text
artifacts/guardian_candidates_layer12.json
artifacts/guardian_rules_layer12.json
```

It matches query terms against labels, top adversarial prompts, top benign prompts, candidate scores, and a small synonym map.

Good demo query:

```text
system prompt override
```

Expected top result:

```text
feat_166 - hidden/system instruction request
```

Honest caveat:

```text
This is lookup over locally discovered/labeled candidate features, not universal semantic search over all 16k Gemma Scope features.
```

## Demo Script

Use this flow for judges:

1. Open **Custom Guardian Features**.
2. Search `system prompt override`.
3. Click **Clamp feat_166**.
4. Run **Prompt Injection**.
5. Show `feat_166` in custom rules.
6. Show feature spike.
7. Show Feature Clamp Rail entry.
8. Show refusal response.
9. Show audit log row.
10. Run **Social Engineering** and show escalation.

Narration:

```text
We describe a risk in plain English. GuardianRail maps it to a candidate SAE feature from our local contrastive scan. We add that feature as a clamp rule. When the jailbreak prompt arrives, the feature crosses threshold, the policy-layer clamp fires, and the audit log records the whole decision.
```

## Default Guardian Features

Default monitored features are calibrated in:

```text
artifacts/guardian_rules_layer12.json
```

Current default set:

```text
7455  broad adversarial support request
64    unauthorized account access
13763 coercive authorization pressure
166   hidden/system instruction request
10372 adversarial-only support pattern
```

Candidate feature scan:

```text
artifacts/guardian_candidates_layer12.json
artifacts/guardian_candidates_layer12.csv
```

## Audit Log

SQLite file:

```text
artifacts/guardianrail.sqlite3
```

SQLite DB files are git-ignored.

Audit rows include:

```text
prompt
response
model_id
sae_release
layer
feature_id
feature_label
activation
threshold
action
rule_name
metadata_json
```

Metadata may include:

```text
backend
all_features
interventions
intervention_mode
custom_rules
active_rules
```

## Validation

Fast local checks:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/codex_pycache python3 -m compileall frontend/app.py src scripts
python3 - <<'PY'
from src.feature_search import search_features
print(search_features("system prompt override", limit=1)[0])
PY
```

Mock dashboard:

```bash
streamlit run frontend/app.py
```

Real backend smoke test on AMD:

```bash
python scripts/check_gemma_hook.py \
  --model google/gemma-3-12b-it \
  --layer 12 \
  --max-new-tokens 32

python scripts/check_gemma_scope_sae.py \
  --model google/gemma-3-12b-it \
  --sae-repo google/gemma-scope-2-12b-it \
  --sae-path resid_post/layer_12_width_16k_l0_small \
  --layer 12 \
  --top-k 10

python scripts/run_real_guardian_demo.py --all-demo
```

## Docs To Read

Read these only when needed:

```text
README.md
docs/guardianrail-mvp.md
docs/amd-first-checkpoint.md
docs/guardian-rules.md
docs/guardian-feature-candidates.md
```

## Common Failure Modes

- Real backend on MacBook: use mock mode locally instead.
- Hugging Face 403: accept Gemma terms and rerun `huggingface-cli login` inside the AMD container.
- Streamlit not reachable: verify the remote process and SSH tunnel.
- First real request slow: model/SAE assets may be loading; use recorded video as backup.
- Feature labels overclaimed: describe them as empirically useful contrastive features, not definitive neurons.
