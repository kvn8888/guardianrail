from __future__ import annotations

from dataclasses import dataclass


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

