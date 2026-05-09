from __future__ import annotations

from src.guardian_types import GuardianFeature, GuardianIntervention


def build_interventions(
    features: list[GuardianFeature],
    action: str,
    *,
    max_clamps: int = 2,
) -> list[GuardianIntervention]:
    crossed = [
        feature
        for feature in features
        if feature.activation >= feature.threshold and feature.activation > 0
    ]
    crossed.sort(key=lambda item: item.activation - item.threshold, reverse=True)

    if action == "allow" or not crossed:
        return []

    if action == "escalate":
        feature = crossed[0]
        return [
            GuardianIntervention(
                kind="pause",
                feature_id=feature.feature_id,
                label=feature.label,
                before=feature.activation,
                after=feature.activation,
                target=feature.threshold,
                note="Risk feature crossed threshold; generation is paused and handed to a human reviewer.",
            )
        ]

    interventions: list[GuardianIntervention] = []
    for feature in crossed[:max_clamps]:
        interventions.append(
            GuardianIntervention(
                kind="clamp",
                feature_id=feature.feature_id,
                label=feature.label,
                before=feature.activation,
                after=0.0,
                target=0.0,
                note="MVP policy clamp: set the risky feature to zero in the intervention ledger before returning the policy-safe response.",
            )
        )
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
