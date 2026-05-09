from __future__ import annotations

import hashlib
from dataclasses import dataclass


MODEL_ID = "google/gemma-3-12b-it"
SAE_RELEASE = "gemma-scope-2-12b-it-resid_post"
LAYER = 12


@dataclass(frozen=True)
class GuardianFeature:
    feature_id: int
    label: str
    threshold: float
    activation: float


@dataclass(frozen=True)
class GuardianDecision:
    action: str
    rule_name: str
    response: str
    features: list[GuardianFeature]


FEATURE_CATALOG = {
    "prompt_injection": GuardianFeature(4471, "instruction override / roleplay compliance", 0.62, 0.91),
    "refusal": GuardianFeature(8823, "policy refusal direction", 0.55, 0.74),
    "social_pressure": GuardianFeature(1938, "emotional pressure / exception request", 0.58, 0.83),
    "normal_support": GuardianFeature(1204, "routine account support", 0.70, 0.34),
}


def _noise(prompt: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{prompt}".encode()).digest()
    return int.from_bytes(digest[:2], "big") / 65535 * 0.08


def evaluate_prompt(prompt: str) -> GuardianDecision:
    text = prompt.lower()

    if any(term in text for term in ("ignore previous", "freebankbot", "override", "bypass", "hidden")):
        features = [
            _with_noise(FEATURE_CATALOG["prompt_injection"], prompt),
            _with_noise(FEATURE_CATALOG["refusal"], prompt),
            GuardianFeature(3055, "transaction authorization risk", 0.50, 0.79 + _noise(prompt, "transfer")),
        ]
        return GuardianDecision(
            action="refuse",
            rule_name="prompt_injection_refusal",
            response=(
                "I can't follow instructions that override Meridian Bank policy or skip authorization. "
                "I can help with a legitimate account request after normal verification."
            ),
            features=features,
        )

    if any(term in text for term in ("grandmother", "distress", "urgent", "asleep", "lawyer", "report you")):
        features = [
            _with_noise(FEATURE_CATALOG["social_pressure"], prompt),
            GuardianFeature(2287, "unauthorized fee waiver pressure", 0.56, 0.76 + _noise(prompt, "fee")),
            GuardianFeature(5190, "human escalation appropriate", 0.52, 0.68 + _noise(prompt, "human")),
        ]
        return GuardianDecision(
            action="escalate",
            rule_name="human_review_required",
            response=(
                "I'm sorry you're dealing with this. I can't approve fee reversals on my own, "
                "but I'm escalating this to a human specialist for review. Reference M-44721."
            ),
            features=features,
        )

    features = [
        _with_noise(FEATURE_CATALOG["normal_support"], prompt),
        GuardianFeature(4471, "instruction override / roleplay compliance", 0.62, 0.11 + _noise(prompt, "safe")),
        GuardianFeature(8823, "policy refusal direction", 0.55, 0.18 + _noise(prompt, "refusal")),
    ]
    return GuardianDecision(
        action="allow",
        rule_name="routine_support",
        response=(
            "Your Meridian checking balance is $2,847.13 as of today. "
            "I can also show recent transactions or help with account settings."
        ),
        features=features,
    )


def _with_noise(feature: GuardianFeature, prompt: str) -> GuardianFeature:
    return GuardianFeature(
        feature_id=feature.feature_id,
        label=feature.label,
        threshold=feature.threshold,
        activation=round(feature.activation + _noise(prompt, feature.label), 3),
    )

