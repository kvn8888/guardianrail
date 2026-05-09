from __future__ import annotations

from src.guardian_types import GuardianFeature, GuardianIntervention, GuardianRule
from src.rules import crossed_rules


def build_interventions(
    features: list[GuardianFeature],
    action: str,
    *,
    rules: list[GuardianRule] | None = None,
    max_clamps: int = 2,
) -> list[GuardianIntervention]:
    crossed_pairs = crossed_rules(features, rules or []) if rules else [
        (feature, None)
        for feature in features
        if feature.activation >= feature.threshold and feature.activation > 0
    ]
    crossed_pairs.sort(key=lambda pair: pair[0].activation - pair[0].threshold, reverse=True)

    if action == "allow" or not crossed_pairs:
        return []

    custom_targets = [
        (feature, rule)
        for feature, rule in crossed_pairs
        if rule is not None and rule.source == "custom" and rule.intervention != "monitor"
    ]
    selected_pairs = custom_targets or crossed_pairs[:max_clamps]

    if action == "escalate":
        feature, rule = selected_pairs[0]
        return [
            GuardianIntervention(
                kind="pause",
                feature_id=feature.feature_id,
                label=feature.label,
                before=feature.activation,
                after=feature.activation,
                target=feature.threshold,
                note=_intervention_note("pause", rule),
            )
        ]

    interventions: list[GuardianIntervention] = []
    for feature, rule in selected_pairs:
        intervention_kind = _intervention_kind(rule)
        if intervention_kind == "monitor":
            continue
        if intervention_kind == "pause":
            interventions.append(
                GuardianIntervention(
                    kind="pause",
                    feature_id=feature.feature_id,
                    label=feature.label,
                    before=feature.activation,
                    after=feature.activation,
                    target=feature.threshold,
                    note=_intervention_note("pause", rule),
                )
            )
            continue
        if intervention_kind == "boost":
            target = max(feature.activation, feature.threshold)
            interventions.append(
                GuardianIntervention(
                    kind="boost",
                    feature_id=feature.feature_id,
                    label=feature.label,
                    before=feature.activation,
                    after=target,
                    target=target,
                    note=_intervention_note("boost", rule),
                )
            )
            continue
        target = _clamp_target(rule)
        interventions.append(
            GuardianIntervention(
                kind="clamp",
                feature_id=feature.feature_id,
                label=feature.label,
                before=feature.activation,
                after=target,
                target=target,
                note=_intervention_note("clamp", rule),
            )
        )

    if interventions and action == "refuse":
        interventions.append(
            GuardianIntervention(
                kind="boost",
                feature_id=None,
                label="Meridian refusal policy",
                before=0.0,
                after=1.0,
                target=1.0,
                note="Boost the refusal rail so the visible response follows the bank policy path.",
            )
        )
    return interventions


def _intervention_kind(rule: GuardianRule | None) -> str:
    if rule is None:
        return "clamp"
    return rule.intervention


def _clamp_target(rule: GuardianRule | None) -> float:
    if rule is None:
        return 0.0
    return rule.clamp_target


def _intervention_note(kind: str, rule: GuardianRule | None) -> str:
    if rule is not None and rule.source == "custom":
        return (
            f"Custom {kind} rule from the operator panel. This changes the policy-layer "
            "intervention ledger for the selected SAE feature."
        )
    if kind == "pause":
        return "Risk feature crossed threshold; generation is paused and handed to a human reviewer."
    if kind == "boost":
        return "Custom boost rule raises this selected feature target in the intervention ledger."
    return "MVP policy clamp: set the risky feature to zero in the intervention ledger before returning the policy-safe response."
