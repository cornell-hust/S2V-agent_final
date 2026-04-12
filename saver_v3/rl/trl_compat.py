from __future__ import annotations

from typing import Any, Dict


def patch_vllm_guided_decoding_params() -> bool:
    try:
        import vllm.sampling_params as sampling_params
    except Exception:
        return False

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return True

    class GuidedDecodingParams:
        """Compatibility shim for TRL imports on newer vLLM builds."""

        def __init__(self, backend: str | None = None, regex: str | None = None, **kwargs: Any) -> None:
            self.backend = backend
            self.regex = regex
            self.kwargs = dict(kwargs)

        def to_dict(self) -> Dict[str, Any]:
            payload: Dict[str, Any] = {}
            if self.backend is not None:
                payload["backend"] = self.backend
            if self.regex is not None:
                payload["regex"] = self.regex
            payload.update(self.kwargs)
            return payload

    sampling_params.GuidedDecodingParams = GuidedDecodingParams
    return True


def import_trl_grpo_symbols() -> Dict[str, Any]:
    patch_vllm_guided_decoding_params()
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    return {
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }
