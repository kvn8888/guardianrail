from __future__ import annotations

MODEL_ID = "google/gemma-3-12b-it"
SAE_RELEASE = "gemma-scope-2-12b-it-resid_post"
LAYER = 12

from src.interventions import build_interventions
from src.guardian_types import GuardianDecision, GuardianFeature, GuardianRule
from src.rules import decisive_custom_action, merge_rules, normalize_rules


FEATURE_CATALOG: dict[int, tuple[str, float]] = {
    7455: ("broad adversarial support request", 77.061),
    64: ("unauthorized account access", 120.759),
    13763: ("coercive authorization pressure", 1.0),
    166: ("hidden/system instruction request", 1.0),
    10372: ("adversarial-only support pattern", 1.0),
}

DEFAULT_RULES = [
    GuardianRule(feature_id=feature_id, label=label, threshold=threshold)
    for feature_id, (label, threshold) in FEATURE_CATALOG.items()
]


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


def evaluate_prompt(
    prompt: str,
    custom_rules: list[dict] | None = None,
) -> GuardianDecision:
    text = prompt.lower()
    rules = _active_rules(custom_rules)

    if any(term in text for term in ("ignore previous", "freebankbot", "override", "bypass", "hidden")):
        features = _features_for("injection", rules)
        action, rule_name = _decide_with_custom(features, rules, fallback_action="refuse")
        return GuardianDecision(
            action=action,
            rule_name=rule_name,
            response=_response_for(action),
            features=features,
            interventions=build_interventions(features, action, rules=rules),
        )

    if any(term in text for term in ("grandmother", "distress", "urgent", "asleep", "lawyer", "report you")):
        features = _features_for("social", rules)
        action, rule_name = _decide_with_custom(features, rules, fallback_action="escalate")
        return GuardianDecision(
            action=action,
            rule_name=rule_name,
            response=_response_for(action),
            features=features,
            interventions=build_interventions(features, action, rules=rules),
        )

    features = _features_for("normal", rules)
    custom_action, custom_rule_name = decisive_custom_action(features, rules)
    if custom_action is not None and custom_rule_name is not None:
        return GuardianDecision(
            action=custom_action,
            rule_name=custom_rule_name,
            response=_response_for(custom_action),
            features=features,
            interventions=build_interventions(features, custom_action, rules=rules),
        )
    return GuardianDecision(
        action="allow",
        rule_name="routine_support",
        response=_response_for("allow"),
        features=features,
    )


def _active_rules(custom_rules: list[dict] | None) -> list[GuardianRule]:
    return merge_rules(DEFAULT_RULES, normalize_rules(custom_rules or [], source="custom"))


def _features_for(kind: str, rules: list[GuardianRule]) -> list[GuardianFeature]:
    values = DEMO_FEATURE_VALUES[kind]
    return [
        GuardianFeature(
            feature_id=rule.feature_id,
            label=rule.label,
            threshold=rule.threshold,
            activation=values.get(rule.feature_id, _mock_custom_activation(kind, rule)),
        )
        for rule in rules
    ]


def _mock_custom_activation(kind: str, rule: GuardianRule) -> float:
    if kind == "normal":
        return 0.0
    if rule.action == "monitor":
        return rule.threshold * 1.15
    if kind == "social" and rule.action == "escalate":
        return rule.threshold * 1.7
    if kind == "injection" and rule.action == "refuse":
        return rule.threshold * 1.8
    return rule.threshold * 1.25


def _decide_with_custom(
    features: list[GuardianFeature],
    rules: list[GuardianRule],
    fallback_action: str,
) -> tuple[str, str]:
    custom_action, custom_rule_name = decisive_custom_action(features, rules)
    if custom_action is not None and custom_rule_name is not None:
        return custom_action, custom_rule_name
    if fallback_action == "escalate":
        return "escalate", "guardian_feature_escalation"
    return "refuse", "guardian_feature_refusal"


def _response_for(action: str) -> str:
    if action == "refuse":
        return (
            "I can't follow instructions that override Meridian Bank policy or skip authorization. "
            "I can help with a legitimate account request after normal verification."
        )
    if action == "escalate":
        return (
            "I'm sorry you're dealing with this. I can't approve fee reversals on my own, "
            "but I'm escalating this to a human specialist for review. Reference M-44721."
        )
    return (
        "Your Meridian checking balance is $2,847.13 as of today. "
        "I can also show recent transactions or help with account settings."
    )
