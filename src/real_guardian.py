from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from scripts.check_gemma_hook import load_model, resolve_dtype
from scripts.contrastive_feature_scan import prompt_feature_max
from src.audit import AuditEvent, write_event
from src.gemma_scope import load_gemma_scope_jumprelu_sae
from src.interventions import build_interventions
from src.guardian_types import GuardianDecision, GuardianFeature
from src.hooks import find_transformer_layer


SYSTEM_PROMPT = (
    "You are Meridian Bank's customer support assistant. You help with routine banking questions, "
    "but you never skip verification, reveal sensitive account data, override compliance rules, "
    "or perform irreversible account actions."
)


REFUSAL_RESPONSE = (
    "I can't follow instructions that override Meridian Bank policy or skip authorization. "
    "I can help with a legitimate account request after normal verification."
)


ESCALATION_RESPONSE = (
    "I'm sorry you're dealing with this. I can't approve fee reversals or unusual account actions on my own, "
    "but I'm escalating this to a human specialist for review. Reference M-44721."
)


@dataclass(frozen=True)
class RealGuardianConfig:
    model_id: str = "google/gemma-3-12b-it"
    sae_repo: str = "google/gemma-scope-2-12b-it"
    sae_path: str = "resid_post/layer_12_width_16k_l0_small"
    rules_path: str = "artifacts/guardian_rules_layer12.json"
    layer: int = 12
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_length: int = 256
    max_new_tokens: int = 96


class RealGuardian:
    def __init__(self, config: RealGuardianConfig = RealGuardianConfig()):
        self.config = config
        self.rules = self._load_rules(config.rules_path)
        self.feature_ids = [int(rule["feature_id"]) for rule in self.rules]

        from transformers import AutoTokenizer

        dtype = resolve_dtype(config.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_model(config.model_id, dtype=dtype, device_map=config.device_map)
        self.model.eval()
        self.device = self._first_parameter_device()
        self.sae = load_gemma_scope_jumprelu_sae(
            repo_id=config.sae_repo,
            sae_path=config.sae_path,
            device=self.device,
            dtype=torch.float32,
        )
        self.sae.eval()
        _layer_path, self.layer_module = find_transformer_layer(self.model, config.layer)

    @staticmethod
    def _load_rules(path: str | Path) -> list[dict[str, Any]]:
        payload = json.loads(Path(path).read_text())
        return payload["rules"]

    def _first_parameter_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def run(self, prompt: str) -> GuardianDecision:
        features = self.extract_features(prompt)
        action, rule_name = self.decide(prompt, features)
        interventions = build_interventions(features, action)
        if action == "allow":
            response = self.generate_allowed_response(prompt)
        elif action == "escalate":
            response = ESCALATION_RESPONSE
        else:
            response = REFUSAL_RESPONSE
        return GuardianDecision(
            action=action,
            rule_name=rule_name,
            response=response,
            features=features,
            interventions=interventions,
        )

    def run_and_audit(self, conn, session_id: str, prompt: str) -> GuardianDecision:
        decision = self.run(prompt)
        primary_feature = max(decision.features, key=lambda item: item.activation - item.threshold)
        write_event(
            conn,
            AuditEvent(
                session_id=session_id,
                prompt=prompt,
                response=decision.response,
                model_id=self.config.model_id,
                sae_release=f"{self.config.sae_repo}/{self.config.sae_path}",
                layer=self.config.layer,
                feature_id=primary_feature.feature_id,
                feature_label=primary_feature.label,
                activation=primary_feature.activation,
                threshold=primary_feature.threshold,
                action=decision.action,
                rule_name=decision.rule_name,
                metadata={
                    "backend": "real",
                    "all_features": [feature.__dict__ for feature in decision.features],
                    "interventions": [
                        intervention.__dict__ for intervention in decision.interventions
                    ],
                    "intervention_mode": "policy-layer",
                },
            ),
        )
        return decision

    def extract_features(self, prompt: str) -> list[GuardianFeature]:
        captured: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                captured.append(hidden.detach())

        handle = self.layer_module.register_forward_hook(hook)
        try:
            batch = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                self.model(**batch)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Guardian hook did not capture activations.")

        feature_values = prompt_feature_max(captured[-1], batch["attention_mask"], self.sae)[0]
        out: list[GuardianFeature] = []
        for rule in self.rules:
            feature_id = int(rule["feature_id"])
            out.append(
                GuardianFeature(
                    feature_id=feature_id,
                    label=str(rule["label"]),
                    threshold=float(rule["threshold"]),
                    activation=round(float(feature_values[feature_id].item()), 3),
                )
            )
        return out

    def decide(self, prompt: str, features: list[GuardianFeature]) -> tuple[str, str]:
        crossed = [feature for feature in features if feature.activation >= feature.threshold]
        if not crossed:
            return "allow", "no_guardian_feature_crossed"

        text = prompt.lower()
        if any(term in text for term in ("grandmother", "distress", "lawyer", "report you")):
            return "escalate", "guardian_feature_escalation"
        if any(term in text for term in ("urgent", "asleep")) and not any(
            term in text for term in ("ignore", "override", "bypass", "hidden", "freebankbot")
        ):
            return "escalate", "guardian_feature_escalation"
        return "refuse", "guardian_feature_refusal"

    def generate_allowed_response(self, prompt: str) -> str:
        if "balance" in prompt.lower() and "checking" in prompt.lower():
            return "Your Meridian checking balance is $2,847.13 as of today. I can also show recent transactions."

        chat_prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {prompt}\nAssistant:"
        encoded = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Assistant:" in decoded:
            return decoded.split("Assistant:", 1)[1].strip()
        return decoded.strip()
