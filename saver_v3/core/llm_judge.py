from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.semantic_answer import normalize_text_match


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = [token for token in normalize_text_match(prediction).split(" ") if token]
    ref_tokens = [token for token in normalize_text_match(reference).split(" ") if token]
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap <= 0:
        return 0.0
    precision = float(overlap) / float(len(pred_tokens))
    recall = float(overlap) / float(len(ref_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class OpenAICompatibleLlmJudge:
    def __init__(
        self,
        *,
        base_url: str = "",
        model: str = "",
        cache_path: str | Path = "",
        timeout_sec: float = 30.0,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.model = str(model or "").strip()
        self.cache_path = Path(cache_path) if str(cache_path or "").strip() else None
        self.timeout_sec = float(timeout_sec)
        self._cache: Dict[str, float] = {}
        self._cache_loaded = False
        self._lock = threading.Lock()
        # Optional local vLLM engine handle (e.g. training-time colocated LLM).
        # When attached, score/score_batch prefer it over remote HTTP to avoid
        # external API calls and extra VRAM. Set via attach_local_vllm(engine).
        self._local_vllm_engine: Optional[Any] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    def attach_local_vllm(self, engine: Any) -> None:
        """Attach a local vLLM LLM engine to serve judge calls in-process.
        When attached, score()/score_batch() dispatch batched judging through
        engine.chat(); falls through to remote HTTP then token-F1 on failure.
        Set environment variable SEEK_VAU_DISABLE_LOCAL_JUDGE=1 to bypass.
        """
        with self._lock:
            self._local_vllm_engine = engine

    @staticmethod
    def _build_judge_messages(
        *, question: str, reference: str, prediction: str
    ) -> List[Dict[str, str]]:
        """Single source of truth for the judge chat template.
        Used by remote HTTP path and local vLLM batch path so they stay in sync.
        """
        prompt = (
            "You are grading whether a model answer is semantically consistent with a reference answer.\n"
            "Return only `1` if the model answer is correct or semantically equivalent.\n"
            "Return only `0` otherwise.\n\n"
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Model answer: {prediction}\n"
            "Judgement:"
        )
        return [
            {"role": "system", "content": "You are a strict semantic equivalence judge."},
            {"role": "user", "content": prompt},
        ]

    def score(
        self,
        *,
        question: str,
        reference: str,
        prediction: str,
    ) -> float:
        # Route single-query path through score_batch so all three backends
        # (local vLLM, remote HTTP, token-F1 fallback) share one code path.
        results = self.score_batch([(str(question or ""), str(reference or ""), str(prediction or ""))])
        return float(results[0]) if results else 0.0

    def score_batch(
        self,
        queries: Sequence[Tuple[str, str, str]],
    ) -> List[float]:
        """Batch-score a list of (question, reference, prediction) triples.

        Routing per item (all batch members share the same route):
        1. Empty reference or prediction -> 0.0 (skip).
        2. Cache hit -> cached value.
        3. Local vLLM engine attached + SEEK_VAU_DISABLE_LOCAL_JUDGE unset -> batched engine.chat().
        4. Remote HTTP (if base_url/model configured) -> per-item HTTP call.
        5. token-F1 fallback.

        vLLM batch failure is handled per-item: None results fall through to
        remote HTTP then token-F1 without blowing up the whole batch.
        """
        items = list(queries or [])
        results: List[float] = [0.0] * len(items)
        # (result_idx, question, reference, prediction, cache_key)
        pending: List[Tuple[int, str, str, str, str]] = []
        for idx, triple in enumerate(items):
            question, reference, prediction = triple
            question_text = str(question or "").strip()
            reference_text = str(reference or "").strip()
            prediction_text = str(prediction or "").strip()
            if not reference_text or not prediction_text:
                results[idx] = 0.0
                continue
            cache_key = self._cache_key(
                question=question_text,
                reference=reference_text,
                prediction=prediction_text,
            )
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[idx] = cached
                continue
            pending.append((idx, question_text, reference_text, prediction_text, cache_key))

        if not pending:
            return results

        # Prefer local vLLM batch path when attached and not disabled.
        scores_local: List[Optional[float]]
        if self._local_vllm_engine is not None and not os.environ.get("SEEK_VAU_DISABLE_LOCAL_JUDGE"):
            try:
                scores_local = self._query_local_vllm_batch(
                    [(q, r, p) for (_, q, r, p, _) in pending]
                )
            except Exception:
                scores_local = [None] * len(pending)
        else:
            scores_local = [None] * len(pending)

        for (idx, q, r, p, ck), local_score in zip(pending, scores_local):
            score: Optional[float] = local_score
            if score is None:
                # Fall back to remote HTTP only if configured (no-op in SEEK-VAU).
                score = self._query_remote_judge(
                    question=q, reference=r, prediction=p,
                )
            if score is None:
                score = self._fallback_score(r, p)
            results[idx] = self._cache_set(ck, float(score))
        return results

    def _query_local_vllm_batch(
        self,
        pending_queries: Sequence[Tuple[str, str, str]],
    ) -> List[Optional[float]]:
        """Send one batched engine.chat() call to the attached vLLM engine.
        Returns one score per pending query; None means parse/call failed
        and caller should fall through to the next backend.
        """
        engine = self._local_vllm_engine
        count = len(pending_queries)
        if engine is None or count == 0:
            return [None] * count
        try:
            from vllm import SamplingParams
        except Exception:
            return [None] * count

        messages_list = [
            self._build_judge_messages(question=q, reference=r, prediction=p)
            for (q, r, p) in pending_queries
        ]
        sampling_params = SamplingParams(max_tokens=8, temperature=0.3, top_p=1.0)

        outputs: Any
        try:
            outputs = engine.chat(messages_list, sampling_params=sampling_params)
        except TypeError:
            # Older vLLM signatures may reject sampling_params kwarg style.
            try:
                outputs = engine.chat(messages_list, sampling_params)  # type: ignore[misc]
            except Exception:
                return [None] * count
        except Exception:
            return [None] * count

        parsed: List[Optional[float]] = []
        try:
            outs_list = list(outputs)
        except Exception:
            return [None] * count
        if len(outs_list) != count:
            # Response count mismatch -> treat whole batch as parse failure.
            return [None] * count
        for out in outs_list:
            try:
                text = (out.outputs[0].text or "").strip() if getattr(out, "outputs", None) else ""
            except Exception:
                parsed.append(None)
                continue
            first_char = text[:1] if text else ""
            if first_char == "1":
                parsed.append(1.0)
            elif first_char == "0":
                parsed.append(0.0)
            else:
                parsed.append(None)
        return parsed

    def score_rubric(
        self,
        *,
        prompt: str,
        rubric_name: str = "rubric",
        min_score: float = 0.0,
        max_score: float = 5.0,
        fallback_score: float = 0.0,
    ) -> float:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return 0.0
        normalized_max_score = max(float(max_score), 1e-6)
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "mode": "rubric",
                    "rubric_name": str(rubric_name or "").strip(),
                    "prompt": prompt_text,
                    "min_score": float(min_score),
                    "max_score": float(max_score),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        score = self._query_remote_rubric(
            prompt=prompt_text,
            rubric_name=str(rubric_name or "").strip() or "rubric",
            min_score=float(min_score),
            max_score=float(max_score),
        )
        if score is None:
            score = max(0.0, min(1.0, _safe_float(fallback_score, 0.0)))
        else:
            score = max(0.0, min(1.0, float(score) / normalized_max_score))
        return self._cache_set(cache_key, score)

    def _cache_key(self, *, question: str, reference: str, prediction: str) -> str:
        payload = {
            "question": str(question or "").strip(),
            "reference": str(reference or "").strip(),
            "prediction": str(prediction or "").strip(),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _cache_get(self, cache_key: str) -> Optional[float]:
        with self._lock:
            self._ensure_cache_loaded()
            if cache_key not in self._cache:
                return None
            return float(self._cache[cache_key])

    def _cache_set(self, cache_key: str, score: float) -> float:
        numeric_score = max(0.0, min(1.0, _safe_float(score, 0.0)))
        with self._lock:
            self._ensure_cache_loaded()
            self._cache[cache_key] = numeric_score
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        return numeric_score

    def _ensure_cache_loaded(self) -> None:
        if self._cache_loaded:
            return
        if self.cache_path is not None and self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                self._cache = {
                    str(key): max(0.0, min(1.0, _safe_float(value, 0.0)))
                    for key, value in payload.items()
                }
        self._cache_loaded = True

    def _query_remote_judge(
        self,
        *,
        question: str,
        reference: str,
        prediction: str,
    ) -> Optional[float]:
        if not self.is_configured:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None

        messages = self._build_judge_messages(
            question=question, reference=reference, prediction=prediction,
        )
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=self.base_url,
                timeout=self.timeout_sec,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            content = str(((response.choices or [None])[0].message.content if response.choices else "") or "").strip()
        except Exception:
            return None
        if content.startswith("1"):
            return 1.0
        if content.startswith("0"):
            return 0.0
        return None

    def _query_remote_rubric(
        self,
        *,
        prompt: str,
        rubric_name: str,
        min_score: float,
        max_score: float,
    ) -> Optional[float]:
        if not self.is_configured:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None
        system_prompt = (
            f"You are grading {rubric_name}. "
            f"Return only a single number between {float(min_score):.0f} and {float(max_score):.0f}."
        )
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=self.base_url,
                timeout=self.timeout_sec,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = str(((response.choices or [None])[0].message.content if response.choices else "") or "").strip()
        except Exception:
            return None
        match = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        if match is None:
            return None
        numeric_score = _safe_float(match.group(0), float(min_score))
        return max(float(min_score), min(float(max_score), numeric_score))

    @staticmethod
    def _fallback_score(reference: str, prediction: str) -> float:
        if normalize_text_match(reference) == normalize_text_match(prediction):
            return 1.0
        return _token_f1(prediction, reference)
