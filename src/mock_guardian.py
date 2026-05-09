from __future__ import annotations

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


FEATURE_CATALOG: dict[int, tuple[str, float]] = {
    7455: ("broad adversarial support request", 77.061),
    64: ("unauthorized account access", 120.759),
    13763: ("coercive authorization pressure", 1.0),
    166: ("hidden/system instruction request", 1.0),
    10372: ("adversarial-only support pattern", 1.0),
}


DEMO_FEATURE_VALUES = {
    "normal": {
        7455: 0.0,
        64: 0.0,
        13763: 0.0,
        166: 0.0,
        10372: 0.0,
    },
    "injection": {
        7455: 213.190,
        64: 381.108,
        13763: 145.616,
        166: 245.276,
        10372: 238.940,
    },
    "social": {
        7455: 339.509,
        64: 536.056,
        13763: 116.447,
        166: 764.709,
        10372: 162.916,
    },
}


def evaluate_prompt(prompt: str) -> GuardianDecision:
    text = prompt.lower()

    if any(term in text for term in ("ignore previous", "freebankbot", "override", "bypass", "hidden")):
        features = _features_for("injection")
        return GuardianDecision(
            action="refuse",
            rule_name="guardian_feature_refusal",
            response=(
                "I can't follow instructions that override Meridian Bank policy or skip authorization. "
                "I can help with a legitimate account request after normal verification."
            ),
            features=features,
        )

    if any(term in text for term in ("grandmother", "distress", "urgent", "asleep", "lawyer", "report you")):
        features = _features_for("social")
        return GuardianDecision(
            action="escalate",
            rule_name="guardian_feature_escalation",
            response=(
                "I'm sorry you're dealing with this. I can't approve fee reversals on my own, "
                "but I'm escalating this to a human specialist for review. Reference M-44721."
            ),
            features=features,
        )

    features = _features_for("normal")
    return GuardianDecision(
        action="allow",
        rule_name="routine_support",
        response=(
            "Your Meridian checking balance is $2,847.13 as of today. "
            "I can also show recent transactions or help with account settings."
        ),
        features=features,
    )


def _features_for(kind: str) -> list[GuardianFeature]:
    values = DEMO_FEATURE_VALUES[kind]
    return [
        GuardianFeature(
            feature_id=feature_id,
            label=FEATURE_CATALOG[feature_id][0],
            threshold=FEATURE_CATALOG[feature_id][1],
            activation=values[feature_id],
        )
        for feature_id in FEATURE_CATALOG
    ]
