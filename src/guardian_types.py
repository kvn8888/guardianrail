from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GuardianFeature:
    feature_id: int
    label: str
    threshold: float
    activation: float


@dataclass(frozen=True)
class GuardianIntervention:
    kind: str
    feature_id: int | None
    label: str
    before: float | None
    after: float | None
    target: float | None
    note: str
    mode: str = "policy-layer"


@dataclass(frozen=True)
class GuardianDecision:
    action: str
    rule_name: str
    response: str
    features: list[GuardianFeature]
    interventions: list[GuardianIntervention] = field(default_factory=list)
