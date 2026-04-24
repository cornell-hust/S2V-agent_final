from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RETIREMENT_MARKERS = (
    "retired",
    "run_full_pipeline.sh",
    "compact_trace_v5",
    "materialized_sft_messages_v5",
    "materialized_runtime_items_v5",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "rollout.py",
        "training.py",
        "saver_v3/sft_training.py",
        "saver_v3/common/training.py",
        "scripts/generate_hard_normals.py",
        "scripts/add_sample_weights.py",
    ],
)
def test_retired_python_entrypoints_fail_fast(relative_path: str):
    result = _run([sys.executable, str(REPO_ROOT / relative_path)])

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    for marker in RETIREMENT_MARKERS:
        assert marker in combined_output


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/launch_exp2.sh",
        "scripts/launch_exp3.sh",
    ],
)
def test_retired_shell_entrypoints_fail_fast(relative_path: str):
    result = _run(["bash", str(REPO_ROOT / relative_path)])

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    for marker in RETIREMENT_MARKERS:
        assert marker in combined_output


@pytest.mark.parametrize(
    "module_name",
    [
        "saver_v3.sft_training",
        "saver_v3.common.training",
    ],
)
def test_retired_module_imports_fail_fast(module_name: str):
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
        f"from {module_name} import run_standard_sft"
    )
    result = _run([sys.executable, "-c", script])

    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    assert "retired" in combined_output
    assert "saver_v3/sft/training.py" in combined_output or "saver_v3.sft.training.run_standard_sft" in combined_output
