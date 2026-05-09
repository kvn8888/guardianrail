from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


@dataclass(frozen=True)
class GemmaScopeConfig:
    repo_id: str
    sae_path: str
    hook_point_in: str
    hook_point_out: str
    width: int
    model_name: str
    architecture: str
    l0: int


class JumpReluSae(torch.nn.Module):
    def __init__(
        self,
        w_enc: torch.Tensor,
        b_enc: torch.Tensor,
        threshold: torch.Tensor,
        config: GemmaScopeConfig,
    ):
        super().__init__()
        self.register_buffer("w_enc", w_enc)
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("threshold", threshold)
        self.config = config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = x @ self.w_enc + self.b_enc
        return pre_acts * (pre_acts > self.threshold)


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    sae_path: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> JumpReluSae:
    config_path = hf_hub_download(repo_id, f"{sae_path}/config.json")
    params_path = hf_hub_download(repo_id, f"{sae_path}/params.safetensors")

    with open(config_path) as handle:
        raw_config = json.load(handle)

    if raw_config.get("architecture") != "jump_relu":
        raise ValueError(f"Expected jump_relu SAE, got {raw_config.get('architecture')!r}")

    params = load_file(params_path, device="cpu")
    config = GemmaScopeConfig(
        repo_id=repo_id,
        sae_path=sae_path,
        hook_point_in=raw_config["hf_hook_point_in"],
        hook_point_out=raw_config["hf_hook_point_out"],
        width=int(raw_config["width"]),
        model_name=raw_config["model_name"],
        architecture=raw_config["architecture"],
        l0=int(raw_config["l0"]),
    )
    return JumpReluSae(
        w_enc=params["w_enc"].to(device=device, dtype=dtype),
        b_enc=params["b_enc"].to(device=device, dtype=dtype),
        threshold=params["threshold"].to(device=device, dtype=dtype),
        config=config,
    )

