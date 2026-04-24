#!/usr/bin/env python3
"""Report whether the current host satisfies the FA3-only top-level policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from typing import Any


def _run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _probe_gpu() -> dict[str, Any]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {
            "available": False,
            "gpus": [],
        }

    gpus: list[dict[str, Any]] = []
    for row in output.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            continue
        name, capability, memory_gb = parts
        try:
            major_str, minor_str = capability.split(".")
            capability_major = int(major_str)
            capability_minor = int(minor_str)
        except ValueError:
            capability_major = -1
            capability_minor = -1
        gpus.append(
            {
                "name": name,
                "compute_capability": capability,
                "memory_gb": memory_gb,
                "capability_major": capability_major,
                "capability_minor": capability_minor,
            }
        )
    return {
        "available": bool(gpus),
        "gpus": gpus,
    }


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _choose_backend(gpu_info: dict[str, Any]) -> dict[str, Any]:
    forced_backend = os.environ.get("SAVER_V3_ATTN_BACKEND", "fa3").strip() or "fa3"
    require_fa3 = os.environ.get("SAVER_V3_REQUIRE_FA3", "1") == "1"
    disable_fa3 = os.environ.get("SAVER_V3_DISABLE_FA3", "0") == "1"

    fa3_modules = {
        "flash_attn_interface": _module_available("flash_attn_interface"),
        "flash_attn": _module_available("flash_attn"),
    }
    fa3_module_importable = any(fa3_modules.values())

    blockers: list[str] = []
    if forced_backend not in {"auto", "fa3"}:
        blockers.append(f"unsupported SAVER_V3_ATTN_BACKEND={forced_backend!r} for FA3-only layer")
    if disable_fa3:
        blockers.append("SAVER_V3_DISABLE_FA3=1 disables the only supported backend")
    if not gpu_info["available"]:
        blockers.append("no NVIDIA GPU detected")
    else:
        unsupported = [gpu for gpu in gpu_info["gpus"] if gpu["capability_major"] < 9]
        if unsupported:
            blockers.append("found GPU with compute capability < 9.0; Hopper-class GPUs are required")
    if not fa3_module_importable:
        blockers.append("no FA3-capable Python module importable (checked flash_attn_interface, flash_attn)")

    ready = not blockers
    notes: list[str] = []
    if ready:
        notes.append("Host satisfies the FA3-only top-level contract.")
    else:
        notes.append("This scaffold fails closed: it does not declare FA2 or SDPA fallback.")
    if require_fa3:
        notes.append("SAVER_V3_REQUIRE_FA3=1 keeps launchers in FA3-only mode.")

    return {
        "env_forced_backend": forced_backend,
        "require_fa3": require_fa3,
        "fa3_module_importable": fa3_module_importable,
        "fa3_modules": fa3_modules,
        "disabled": {
            "fa3": disable_fa3,
        },
        "recommended_order": ["fa3"] if not disable_fa3 else [],
        "ready": ready,
        "blockers": blockers,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    gpu_info = _probe_gpu()
    decision = _choose_backend(gpu_info)
    report = {
        "gpu": gpu_info,
        "decision": decision,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("FA3-only attention backend report")
    print("=================================")
    if not gpu_info["available"]:
        print("GPU: not detected")
    else:
        for index, gpu in enumerate(gpu_info["gpus"], start=1):
            print(
                f"GPU {index}: {gpu['name']} | cc={gpu['compute_capability']} | "
                f"memory_gb={gpu['memory_gb']}"
            )
    print(f"flash_attn_interface importable: {decision['fa3_modules']['flash_attn_interface']}")
    print(f"flash_attn importable: {decision['fa3_modules']['flash_attn']}")
    print(f"recommended order: {' -> '.join(decision['recommended_order']) or 'none'}")
    print(f"ready: {decision['ready']}")
    for blocker in decision["blockers"]:
        print(f"blocker: {blocker}")
    for note in decision["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
