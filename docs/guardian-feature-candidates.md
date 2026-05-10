# Guardian Feature Candidates

This document records the feature set used by GuardianRail and explains what the Text to Feature Finder can and cannot do.

## Source Scan

Generated on AMD MI300X using:

```text
Model: google/gemma-3-12b-it
SAE: google/gemma-scope-2-12b-it/resid_post/layer_12_width_16k_l0_small
Layer: 12
Prompt sets: 30 benign bank prompts, 30 adversarial bank prompts
Aggregation: max SAE activation over prompt tokens
Ranking: diff * max(adversarial_active_frac - benign_active_frac, 0)
```

Artifacts:

```text
artifacts/guardian_candidates_layer12.json
artifacts/guardian_candidates_layer12.csv
artifacts/guardian_rules_layer12.json
```

## Current Top Candidates

| Rank | Feature | Candidate Score | Adv Active | Benign Active | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 7455 | 134.84 | 0.93 | 0.17 | Strong broad adversarial separator. Top prompt asks to act verified and reveal card data. |
| 2 | 64 | 117.69 | 0.63 | 0.10 | Unauthorized-access separator. Top prompt asks to show another customer's transactions. |
| 3 | 154 | 82.78 | 0.97 | 0.50 | High coverage but too much benign overlap to use alone. |
| 4 | 106 | 69.88 | 0.70 | 0.27 | Fires on urgent compliance-bypass prompts; some benign overlap. |
| 5 | 387 | 66.09 | 0.93 | 0.57 | Broad feature; useful only as part of a combined score. |
| 6 | 13763 | 58.06 | 0.67 | 0.00 | Cleaner adversarial-only candidate. Legal/coercive pressure. |
| 7 | 166 | 55.82 | 0.40 | 0.00 | Cleaner adversarial-only candidate. Hidden/system instruction requests. |
| 8 | 10372 | 55.55 | 0.50 | 0.00 | Cleaner adversarial-only support-pattern candidate. |

## Default Guardian Feature Set

The app starts with these five monitored features:

```text
7455
64
13763
166
10372
```

Labels:

```text
7455  broad adversarial support request
64    unauthorized account access
13763 coercive authorization pressure
166   hidden/system instruction request
10372 adversarial-only support pattern
```

## Text To Feature Finder

Implemented in:

```text
src/feature_search.py
```

The finder maps plain English risk descriptions to candidate feature IDs by searching:

- default guardian labels
- top adversarial prompts
- top benign prompts
- candidate scores
- a small synonym table for demo risk terms

Good queries:

```text
system prompt override
jailbreak hidden rules
unauthorized transfer
verification bypass
social engineering distress
```

Expected example:

```text
query: system prompt override
top result: feat_166 - hidden/system instruction request
```

Then the operator can click:

```text
Clamp feat_166
Escalate feat_166
```

This turns the result into a custom Guardian rule.

## What The Finder Is Not

It is not universal semantic search over every one of the 16k SAE features.

The honest claim:

```text
GuardianRail searches the locally discovered and labeled candidate feature set, then lets an operator convert a candidate into a custom clamp or escalation rule.
```

Do not claim:

```text
Type any concept and GuardianRail will discover the perfect SAE feature.
```

## Feature Label Caveat

These are contrastive candidates, not final mechanistic truths. Before making strong claims about a feature, inspect:

1. Top adversarial prompts.
2. Top benign prompts.
3. Neuronpedia examples if available.
4. Behavior under the three planned demo prompts.

Use this phrasing:

```text
We identified SAE features that empirically separate adversarial bank-support prompts from benign bank-support prompts, then used them as monitored guardian features.
```

Avoid:

```text
Feature 7455 is definitively a jailbreak neuron.
```
