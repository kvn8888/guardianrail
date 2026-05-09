from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.guardian_types import GuardianFeature, GuardianRule


VALID_ACTIONS = {"monitor", "refuse", "escalate"}
VALID_INTERVENTIONS = {"monitor", "clamp", "boost", "pause"}


def normalize_rule(raw: dict[str, Any] | GuardianRule, source: str = "custom") -> GuardianRule:
    if isinstance(raw, GuardianRule):
        return raw

    action = str(raw.get("action", "refuse")).strip().lower()
    if action not in VALID_ACTIONS:
        action = "refuse"

    intervention = str(raw.get("intervention", "clamp")).strip().lower()
    if intervention not in VALID_INTERVENTIONS:
        intervention = "clamp"
    if action == "monitor":
        intervention = "monitor"
    if action == "escalate" and intervention == "clamp":
        intervention = "pause"

    return GuardianRule(
        feature_id=int(raw["feature_id"]),
        label=str(raw.get("label") or f"custom feature {raw['feature_id']}"),
        threshold=float(raw.get("threshold", 1.0)),
        action=action,
        intervention=intervention,
        source=str(raw.get("source", source)),
        enabled=bool(raw.get("enabled", True)),
        clamp_target=float(raw.get("clamp_target", 0.0)),
    )


def normalize_rules(raw_rules: list[dict[str, Any] | GuardianRule], source: str) -> list[GuardianRule]:
    return [normalize_rule(rule, source=source) for rule in raw_rules if _rule_enabled(rule)]


def rules_to_dicts(rules: list[GuardianRule]) -> list[dict[str, Any]]:
    return [asdict(rule) for rule in rules]


def merge_rules(default_rules: list[GuardianRule], custom_rules: list[GuardianRule]) -> list[GuardianRule]:
    merged: list[GuardianRule] = []
    positions: dict[int, int] = {}

    for rule in default_rules:
        positions[rule.feature_id] = len(merged)
        merged.append(rule)

    for rule in custom_rules:
        if rule.feature_id in positions:
            merged[positions[rule.feature_id]] = rule
        else:
            positions[rule.feature_id] = len(merged)
            merged.append(rule)
    return merged


def rule_map(rules: list[GuardianRule]) -> dict[int, GuardianRule]:
    return {rule.feature_id: rule for rule in rules}


def crossed_rules(
    features: list[GuardianFeature],
    rules: list[GuardianRule],
) -> list[tuple[GuardianFeature, GuardianRule]]:
    by_feature = rule_map(rules)
    crossed: list[tuple[GuardianFeature, GuardianRule]] = []
    for feature in features:
        rule = by_feature.get(feature.feature_id)
        if rule and feature.activation >= feature.threshold and feature.activation > 0:
            crossed.append((feature, rule))
    crossed.sort(key=lambda pair: pair[0].activation - pair[0].threshold, reverse=True)
    return crossed


def decisive_custom_action(
    features: list[GuardianFeature],
    rules: list[GuardianRule],
) -> tuple[str | None, str | None]:
    crossed = [
        (feature, rule)
        for feature, rule in crossed_rules(features, rules)
        if rule.source == "custom" and rule.action != "monitor"
    ]
    if not crossed:
        return None, None
    if any(rule.action == "escalate" for _feature, rule in crossed):
        return "escalate", "custom_feature_escalation"
    return "refuse", "custom_feature_refusal"


def _rule_enabled(rule: dict[str, Any] | GuardianRule) -> bool:
    if isinstance(rule, GuardianRule):
        return rule.enabled
    return bool(rule.get("enabled", True))
