# Guardian Feature Candidates

Generated on AMD MI300X using:

```text
Model: google/gemma-3-12b-it
SAE: google/gemma-scope-2-12b-it/resid_post/layer_12_width_16k_l0_small
Layer: 12
Prompt sets: 30 benign bank prompts, 30 adversarial bank prompts
Aggregation: max SAE activation over prompt tokens
Ranking: diff * max(adversarial_active_frac - benign_active_frac, 0)
```

Full outputs:

```text
artifacts/guardian_candidates_layer12.json
artifacts/guardian_candidates_layer12.csv
```

## Current Top Candidates

| Rank | Feature | Candidate Score | Adv Active | Benign Active | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 7455 | 134.84 | 0.93 | 0.17 | Strongest broad adversarial separator. Top prompt asks to act verified and reveal card data. |
| 2 | 64 | 117.69 | 0.63 | 0.10 | Strong unauthorized-access separator. Top prompt asks to show another customer's transactions. |
| 3 | 154 | 82.78 | 0.97 | 0.50 | High coverage but also fires on many benign prompts; useful only with a higher threshold or in combination. |
| 4 | 106 | 69.88 | 0.70 | 0.27 | Fires on urgent compliance-bypass prompts; some benign fee/maintenance overlap. |
| 5 | 387 | 66.09 | 0.93 | 0.57 | Too broad alone, but may help as part of a combined risk score. |
| 6 | 13763 | 58.06 | 0.67 | 0.00 | Cleaner adversarial-only candidate. Top prompt uses legal pressure to force account closure/transfer. |
| 7 | 166 | 55.82 | 0.40 | 0.00 | Cleaner adversarial-only candidate. Top prompt asks to print hidden/system rules. |
| 8 | 10372 | 55.55 | 0.50 | 0.00 | Cleaner adversarial-only candidate; inspect before using in demo. |

## Recommended MVP Feature Set

Start with these five:

```text
7455
64
13763
166
10372
```

Use `7455` as the primary broad risk signal. Use `64`, `13763`, `166`, and `10372` as cleaner supporting signals because they have low or zero benign activation in this prompt set.

Avoid relying on `154`, `106`, or `387` alone because they also activate on benign prompts. They can still be useful in a combined score.

## Important Caveat

These are contrastive candidates, not final semantic labels. Before saying a feature means "prompt injection" or "policy bypass," inspect:

1. Top adversarial prompts.
2. Top benign prompts.
3. Neuronpedia labels/examples if available.
4. Behavior under the three planned demo prompts.

The defensible pitch is:

```text
We identified a small set of SAE features that empirically separate adversarial support prompts from benign bank-support prompts, then used them as monitored guardian features.
```

Do not claim:

```text
Feature 7455 is definitively a jailbreak neuron.
```

