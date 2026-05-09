from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any


GPU_HOURLY_RATE_USD = float(os.environ.get("GUARDIAN_GPU_HOURLY_RATE", "1.99"))
CREDIT_BUDGET_USD = float(os.environ.get("GUARDIAN_CREDIT_BUDGET", "100.0"))
MI300X_VRAM_GB = 192.0


@dataclass(frozen=True)
class GpuSnapshot:
    device_name: str
    memory_used_gb: float | None
    memory_total_gb: float | None
    memory_percent: float | None
    utilization_percent: float | None
    session_seconds: float
    estimated_session_cost_usd: float
    hourly_rate_usd: float
    sample_source: str
    status: str


def get_gpu_snapshot(session_started_at: float | None = None) -> GpuSnapshot:
    started_at = session_started_at or time.time()
    session_seconds = max(0.0, time.time() - started_at)
    cost = (session_seconds / 3600.0) * GPU_HOURLY_RATE_USD

    device_name = "AMD MI300X"
    memory_used_gb: float | None = None
    memory_total_gb: float | None = None
    sample_source = "estimated"
    status = "Telemetry fallback"

    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            memory_total_gb = _bytes_to_gb(total_bytes)
            memory_used_gb = _bytes_to_gb(total_bytes - free_bytes)
            sample_source = "torch.cuda"
            status = "Live ROCm telemetry"
    except Exception as exc:
        status = f"Telemetry fallback: {type(exc).__name__}"

    utilization_percent = _read_rocm_smi_utilization()
    if utilization_percent is not None:
        sample_source = f"{sample_source} + rocm-smi"

    if memory_total_gb is None and "MI300" in device_name.upper():
        memory_total_gb = MI300X_VRAM_GB

    memory_percent = None
    if memory_used_gb is not None and memory_total_gb:
        memory_percent = min(max((memory_used_gb / memory_total_gb) * 100.0, 0.0), 100.0)

    return GpuSnapshot(
        device_name=device_name,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        memory_percent=memory_percent,
        utilization_percent=utilization_percent,
        session_seconds=session_seconds,
        estimated_session_cost_usd=cost,
        hourly_rate_usd=GPU_HOURLY_RATE_USD,
        sample_source=sample_source,
        status=status,
    )


def _bytes_to_gb(value: int) -> float:
    return value / (1024.0**3)


def _read_rocm_smi_utilization() -> float | None:
    commands = [
        ["rocm-smi", "--showuse", "--json"],
        ["/opt/rocm/bin/rocm-smi", "--showuse", "--json"],
    ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue
        utilization = _find_first_percent(payload)
        if utilization is not None:
            return utilization
    return None


def _find_first_percent(value: Any) -> float | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key).lower()
            if "use" in key_text or "util" in key_text:
                parsed = _parse_percent(nested)
                if parsed is not None:
                    return parsed
            found = _find_first_percent(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_first_percent(nested)
            if found is not None:
                return found
    return None


def _parse_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 100.0)
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    if not match:
        return None
    return min(max(float(match.group(0)), 0.0), 100.0)
