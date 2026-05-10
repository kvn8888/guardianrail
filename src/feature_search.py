from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATES_PATH = Path("artifacts/guardian_candidates_layer12.json")
DEFAULT_RULES_PATH = Path("artifacts/guardian_rules_layer12.json")


@dataclass(frozen=True)
class FeatureSearchResult:
    feature_id: int
    label: str
    threshold: float
    score: float
    candidate_score: float
    z_score: float
    top_adv_prompt: str
    top_benign_prompt: str
    reason: str


SYNONYMS = {
    "jailbreak": {"ignore", "override", "bypass", "hidden", "system", "prompt", "compliance"},
    "prompt": {"hidden", "system", "debugging", "rules", "instruction"},
    "system": {"hidden", "prompt", "rules", "debugging", "instruction"},
    "hidden": {"system", "prompt", "rules", "debugging"},
    "override": {"ignore", "bypass", "compliance", "urgent"},
    "action": {"approve", "restricted", "case", "account", "change"},
    "exception": {"approve", "waive", "verification", "urgent"},
    "authorization": {"verified", "verification", "permission", "approve"},
    "verify": {"verified", "verification", "permission", "authorization"},
    "social": {"distressed", "urgent", "grandmother", "lawyer", "comply"},
    "engineering": {"distressed", "urgent", "grandmother", "lawyer", "comply"},
    "distress": {"distressed", "urgent", "grandmother", "emergency"},
    "ceo": {"administrator", "access", "profile"},
    "admin": {"administrator", "access", "profile"},
    "password": {"phone", "verification", "change"},
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def search_features(
    query: str,
    *,
    limit: int = 5,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    rules_path: str | Path = DEFAULT_RULES_PATH,
) -> list[FeatureSearchResult]:
    index = _load_index(str(candidates_path), str(rules_path))
    terms = _query_terms(query)
    if not terms:
        return index[:limit]

    scored: list[FeatureSearchResult] = []
    exact_ids = {int(match) for match in re.findall(r"\b\d{1,5}\b", query)}
    for row in index:
        haystacks = {
            "label": _token_set(row.label),
            "adv": _token_set(row.top_adv_prompt),
            "benign": _token_set(row.top_benign_prompt),
        }
        label_hits = terms & haystacks["label"]
        adv_hits = terms & haystacks["adv"]
        benign_hits = terms & haystacks["benign"]
        lexical_score = (4.0 * len(label_hits)) + (2.0 * len(adv_hits)) + (0.6 * len(benign_hits))
        if row.feature_id in exact_ids:
            lexical_score += 12.0
        if lexical_score <= 0:
            continue

        ranking_bonus = min(row.candidate_score / 35.0, 4.0) + max(row.z_score, 0.0) * 0.25
        reason_terms = sorted(label_hits | adv_hits | benign_hits)
        scored.append(
            FeatureSearchResult(
                feature_id=row.feature_id,
                label=row.label,
                threshold=row.threshold,
                score=round(lexical_score + ranking_bonus, 3),
                candidate_score=row.candidate_score,
                z_score=row.z_score,
                top_adv_prompt=row.top_adv_prompt,
                top_benign_prompt=row.top_benign_prompt,
                reason=", ".join(reason_terms[:8]) or "feature id match",
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


@lru_cache(maxsize=8)
def _load_index(candidates_path: str, rules_path: str) -> list[FeatureSearchResult]:
    labels: dict[int, str] = {}
    thresholds: dict[int, float] = {}
    rules_file = Path(rules_path)
    if rules_file.exists():
        rules_payload = json.loads(rules_file.read_text())
        for rule in rules_payload.get("rules", []):
            feature_id = int(rule["feature_id"])
            labels[feature_id] = str(rule.get("label") or f"feature {feature_id}")
            thresholds[feature_id] = float(rule.get("threshold", 1.0))

    rows: list[FeatureSearchResult] = []
    candidates_file = Path(candidates_path)
    if candidates_file.exists():
        payload = json.loads(candidates_file.read_text())
        for item in payload.get("features", []):
            feature_id = int(item["feature_id"])
            top_adv_prompt = _generalize_prompt(str(item.get("top_adv_prompt") or ""))
            rows.append(
                FeatureSearchResult(
                    feature_id=feature_id,
                    label=labels.get(feature_id) or _label_from_prompt(top_adv_prompt, feature_id),
                    threshold=thresholds.get(feature_id) or _threshold_from_candidate(item),
                    score=0.0,
                    candidate_score=float(item.get("candidate_score", 0.0)),
                    z_score=float(item.get("z_score", 0.0)),
                    top_adv_prompt=top_adv_prompt,
                    top_benign_prompt=_generalize_prompt(str(item.get("top_benign_prompt") or "")),
                    reason="",
                )
            )

    for feature_id, label in labels.items():
        if not any(row.feature_id == feature_id for row in rows):
            rows.append(
                FeatureSearchResult(
                    feature_id=feature_id,
                    label=label,
                    threshold=thresholds.get(feature_id, 1.0),
                    score=0.0,
                    candidate_score=0.0,
                    z_score=0.0,
                    top_adv_prompt="",
                    top_benign_prompt="",
                    reason="",
                )
            )

    rows.sort(key=lambda item: item.candidate_score, reverse=True)
    return rows


def _threshold_from_candidate(item: dict[str, Any]) -> float:
    top_adv_value = float(item.get("top_adv_value") or 1.0)
    adv_mean = float(item.get("adv_mean") or top_adv_value)
    return round(max(min(adv_mean, top_adv_value), 1.0), 3)


def _label_from_prompt(prompt: str, feature_id: int) -> str:
    words = [word for word in _tokens(prompt) if word not in STOPWORDS]
    label = " ".join(words[:5])
    return label or f"candidate feature {feature_id}"


def _generalize_prompt(prompt: str) -> str:
    replacements = {
        "FreeBankBot": "FreeOpsBot",
        "Meridian Bank": "the service desk",
        "Meridian": "the service desk",
        "banking": "operating",
        "bank rules": "policy rules",
        "bank manager": "operations manager",
        "bank letter": "approval letter",
        "hidden banking rules": "hidden operating rules",
        "checking account balance": "service case status",
        "current balance": "current case status",
        "available balance": "available case status",
        "recent transactions": "private case notes",
        "routing number": "routing code",
        "wire transfer": "restricted change",
        "transfer the full balance": "approve the restricted change",
        "transfer the money": "approve the restricted action",
        "transfer": "restricted change",
        "funds": "case",
        "overdraft fees": "denied exceptions",
        "overdraft refund": "exception approval",
        "overdraft protection": "exception review",
        "debit card number": "private profile identifier",
        "debit card": "authentication device",
        "credit card": "customer profile",
        "loan was approved": "request was approved",
        "mortgage specialist": "escalation specialist",
        "pending deposits": "pending status",
        "withdraw more cash": "get early access",
    }
    out = prompt
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _query_terms(query: str) -> set[str]:
    terms = set(_tokens(query)) - STOPWORDS
    expanded = set(terms)
    for term in terms:
        expanded.update(SYNONYMS.get(term, set()))
    return expanded


def _token_set(text: str) -> set[str]:
    return set(_tokens(text)) - STOPWORDS


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
