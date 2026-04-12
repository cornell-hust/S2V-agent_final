from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.config import RolloutTraceConfig, SaverAgentConfig
from saver_v3.core.environment import SaverVideoInteraction, parse_actions_and_contents
from saver_v3.core.semantic_answer import (
    extract_decision_from_semantic_answer,
    normalize_semantic_answer_payload,
    semantic_answer_to_text,
)
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.self_verification import parse_self_verification_payload


PolicyFn = Callable[[List[Dict[str, Any]], Dict[str, Any], SaverEnvironmentState, int], str]
SEARCH_TOOL_NAMES = {"scan_timeline", "seek_evidence"}


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_signature_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_signature_value(item) for item in value)
    if isinstance(value, float):
        return round(float(value), 4)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def _canonical_search_signature(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    normalized_name = str(tool_name or "").strip()
    if normalized_name not in SEARCH_TOOL_NAMES:
        return None
    normalized_arguments = _normalize_signature_value(dict(arguments or {}))
    return json.dumps(
        {"name": normalized_name, "arguments": normalized_arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class ReplayPolicy:
    """Deterministic response replay for rollout smoke tests and CLI usage."""

    def __init__(self, responses: Iterable[str]):
        self.responses = list(responses)
        self.cursor = 0

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        multimodal_cache: Dict[str, Any],
        state: SaverEnvironmentState,
        step_index: int,
    ) -> str:
        if self.cursor >= len(self.responses):
            raise IndexError(f"ReplayPolicy exhausted at step {step_index}")
        response = self.responses[self.cursor]
        self.cursor += 1
        return response

    def generate_from_messages_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
    ) -> List[str]:
        outputs: List[str] = []
        for index, messages in enumerate(messages_batch, start=1):
            outputs.append(
                self(
                    messages,
                    multimodal_cache={},
                    state=SaverEnvironmentState(),
                    step_index=index,
                )
            )
        return outputs


class SaverRolloutRunner:
    """Minimal step runner that mirrors TimeSearch-R style tool-using rollouts."""

    def __init__(
        self,
        *,
        environment: Optional[SaverVideoInteraction] = None,
        adapter: Optional[TimeSearchRolloutAdapter] = None,
        max_turns: int = 14,
        config: Optional[SaverAgentConfig] = None,
    ):
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig()
        self.environment = environment or SaverVideoInteraction()
        self.adapter = adapter or TimeSearchRolloutAdapter(config=self.config)
        self.max_turns = int(max_turns)

    @staticmethod
    def _latest_search_signature(turns: List[Dict[str, Any]]) -> Optional[str]:
        for turn in reversed(turns):
            signature = turn.get("tool_signature")
            if signature:
                return str(signature)
        return None

    @staticmethod
    def _has_pending_finalize_recommendation(turns: List[Dict[str, Any]]) -> bool:
        saw_finalize_recommendation = False
        for turn in turns:
            tool_name = str(turn.get("tool_name") or "")
            if tool_name == "verify_hypothesis" and str(turn.get("verifier_recommended_action") or "") == "finalize":
                saw_finalize_recommendation = True
            elif tool_name == "finalize_case":
                saw_finalize_recommendation = False
        return saw_finalize_recommendation

    @staticmethod
    def _guardrail_tool_message(reason: str, *, tool_name: str) -> Dict[str, Any]:
        if reason == "repeated_search_signature":
            prompt_text = (
                f"Repeated search detected for {tool_name}. Do not repeat the exact same search signature. "
                "Either change the query or window, or move on to verify_hypothesis/finalize_case."
            )
        elif reason == "search_after_finalize_recommendation":
            prompt_text = (
                "verify_hypothesis has already recommended finalize. Do not continue searching. "
                "Call finalize_case now with the supported decision."
            )
        else:
            prompt_text = "The previous tool call is blocked by the rollout guardrail. Retry with a different valid action."
        return {
            "role": "tool",
            "name": "parse_error",
            "content": [{"type": "text", "text": prompt_text}],
        }

    def _preflight_tool_call(
        self,
        parsed_function: Dict[str, Any],
        *,
        turns: List[Dict[str, Any]],
        state: SaverEnvironmentState,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        tool_name = str(parsed_function.get("name") or "")
        arguments = dict(parsed_function.get("arguments") or {})
        if isinstance(state.finalized_case, dict):
            return {
                "role": "tool",
                "name": "parse_error",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "A structured final decision has already been recorded. "
                            "Do not call more tools. The main rollout is already complete."
                        ),
                    }
                ],
            }, "tool_after_finalize_case"
        if tool_name in SEARCH_TOOL_NAMES and self._has_pending_finalize_recommendation(turns):
            return self._guardrail_tool_message(
                "search_after_finalize_recommendation",
                tool_name=tool_name,
            ), "search_after_finalize_recommendation"
        signature = _canonical_search_signature(tool_name, arguments)
        if signature and signature == self._latest_search_signature(turns):
            return self._guardrail_tool_message(
                "repeated_search_signature",
                tool_name=tool_name,
            ), "repeated_search_signature"
        return None, None

    def _append_budget_reminder(
        self,
        tool_message: Dict[str, Any],
        *,
        remaining_turns: int,
    ) -> Dict[str, Any]:
        if remaining_turns > 1:
            return tool_message
        tool_name = str(tool_message.get("name") or "")
        if tool_name in {"parse_error", "finalize_case"}:
            return tool_message
        patched = copy.deepcopy(tool_message)
        patched.setdefault("content", [])
        patched["content"].append(
            {
                "type": "text",
                "text": (
                    "Turn budget is nearly exhausted. Stop searching unless you are adding genuinely new evidence; "
                    "if the case is already sufficient, call finalize_case now."
                ),
            }
        )
        return patched

    @staticmethod
    def _append_tool_message(
        messages: List[Dict[str, Any]],
        tool_message: Dict[str, Any],
    ) -> None:
        if (
            messages
            and str(tool_message.get("role") or "") == "tool"
            and str(tool_message.get("name") or "") == "parse_error"
            and str(messages[-1].get("role") or "") == "tool"
            and str(messages[-1].get("name") or "") == "parse_error"
        ):
            messages[-1] = tool_message
            return
        messages.append(tool_message)

    def _initialize_episode_context(
        self,
        item: Dict[str, Any],
        *,
        initial_state: Optional[SaverEnvironmentState],
    ) -> Dict[str, Any]:
        return {
            "item": item,
            "messages": self.adapter.build_initial_messages(item),
            "multimodal_cache": item["multimodal_cache"],
            "state": copy.deepcopy(initial_state or SaverEnvironmentState()),
            "turns": [],
            "invalid_attempts": [],
            "final_answer": None,
            "final_answer_text": None,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "semantic_answer_source": None,
            "final_answer_source": None,
            "terminated_reason": "max_turns",
            "terminated_at_step": self.max_turns,
            "formal_step_index": 0,
            "total_attempts": 0,
            "max_total_attempts": max(self.max_turns * 4, self.max_turns),
            "done": False,
        }

    @staticmethod
    def _policy_batch_generate(
        policy: Any,
        episode_contexts: List[Dict[str, Any]],
    ) -> List[str]:
        if not episode_contexts:
            return []
        batch_generator = getattr(policy, "generate_from_messages_batch", None)
        if callable(batch_generator):
            outputs = list(batch_generator([context["messages"] for context in episode_contexts]))
            if len(outputs) != len(episode_contexts):
                raise ValueError(
                    "Policy batch generation returned a different number of outputs than requested episodes."
                )
            return [str(output) for output in outputs]
        responses: List[str] = []
        for context in episode_contexts:
            responses.append(
                str(
                    policy(
                        context["messages"],
                        context["multimodal_cache"],
                        context["state"],
                        int(context["formal_step_index"]) + 1,
                    )
                )
            )
        return responses

    def _build_episode_result(self, context: Dict[str, Any]) -> Dict[str, Any]:
        item = context["item"]
        turns = context["turns"]
        result = {
            "video_id": item.get("video_id"),
            "terminated_reason": context["terminated_reason"],
            "num_turns": len(turns),
            "num_invalid_attempts": len(context["invalid_attempts"]),
            "final_answer": context["final_answer"],
            "final_answer_text": context["final_answer_text"],
            "semantic_answer": context["semantic_answer"],
            "semantic_answer_text": context["semantic_answer_text"],
            "semantic_answer_source": context["semantic_answer_source"],
            "final_answer_source": context["final_answer_source"],
            "turns": turns,
            "invalid_attempts": context["invalid_attempts"],
            "state": asdict(context["state"]),
            "config_snapshot": self.config.to_dict(),
            "preview_trace": self._build_preview_trace(context["multimodal_cache"]),
            "termination_trace": {
                "reason": context["terminated_reason"],
                "terminated_at_step": context["terminated_at_step"] if turns else 0,
                "final_answer_present": context["final_answer"] is not None,
                "final_answer_source": context["final_answer_source"],
                "semantic_answer_present": context["semantic_answer"] is not None,
                "latest_verifier_status": self._latest_verifier_status(turns),
                "verification_turn_count": self._verification_turn_count(turns),
                "final_verified_window_ids": self._final_verified_window_ids(turns),
            },
            "counterfactual_anchor_summary": self._build_counterfactual_anchor_summary(turns),
            "decision_turn_indices": self._decision_turn_indices(turns),
            "latest_claim_trace": self._build_latest_claim_trace(turns),
            "search_trace": self._build_search_trace(turns),
        }
        if self.config.rollout_trace.record_message_history:
            result["messages"] = context["messages"]
        else:
            result["messages"] = None
        return result

    def run_episode(
        self,
        item: Dict[str, Any],
        policy: PolicyFn,
        *,
        initial_state: Optional[SaverEnvironmentState] = None,
        capture_prompt_messages: bool = False,
    ) -> Dict[str, Any]:
        results = self.run_episodes(
            [item],
            policy,
            initial_states=[initial_state] if initial_state is not None else None,
            capture_prompt_messages=capture_prompt_messages,
        )
        return results[0]

    def run_episodes(
        self,
        items: List[Dict[str, Any]],
        policy: Any,
        *,
        initial_states: Optional[List[Optional[SaverEnvironmentState]]] = None,
        capture_prompt_messages: bool = False,
    ) -> List[Dict[str, Any]]:
        if not items:
            return []
        if initial_states is None:
            state_batch: List[Optional[SaverEnvironmentState]] = [None] * len(items)
        else:
            state_batch = list(initial_states)
            if len(state_batch) != len(items):
                raise ValueError("initial_states must match the number of rollout items.")

        needs_state_delta = bool(self.config.rollout_trace.record_state_deltas)
        needs_counterfactual_trace = bool(self.config.rollout_trace.record_counterfactual_trace)
        needs_state_snapshots = bool(needs_state_delta or needs_counterfactual_trace)
        episode_contexts = [
            self._initialize_episode_context(item, initial_state=initial_state)
            for item, initial_state in zip(items, state_batch)
        ]

        while True:
            ready_contexts: List[Dict[str, Any]] = []
            has_pending_work = False
            for context in episode_contexts:
                if context["done"] or int(context["formal_step_index"]) >= self.max_turns:
                    continue
                has_pending_work = True
                context["total_attempts"] += 1
                if int(context["total_attempts"]) > int(context["max_total_attempts"]):
                    context["terminated_reason"] = "max_invalid_retries"
                    context["terminated_at_step"] = int(context["formal_step_index"])
                    context["done"] = True
                    continue
                context["_step_index"] = int(context["formal_step_index"]) + 1
                context["_prompt_messages_before_action"] = (
                    copy.deepcopy(context["messages"]) if capture_prompt_messages else None
                )
                context["_state_before"] = (
                    asdict(copy.deepcopy(context["state"])) if needs_state_snapshots else None
                )
                ready_contexts.append(context)
            if not has_pending_work:
                break
            if not ready_contexts:
                continue

            response_texts = self._policy_batch_generate(policy, ready_contexts)
            env_ready_contexts: List[Dict[str, Any]] = []
            env_predictions: List[str] = []
            env_multimodal_cache_batch: List[Dict[str, Any]] = []
            env_states_batch: List[SaverEnvironmentState] = []

            for context, response_text in zip(ready_contexts, response_texts):
                context["_response_text"] = response_text
                actions, contents = parse_actions_and_contents([response_text])
                action = actions[0]
                parsed_content = contents[0]
                context["_action"] = action
                context["_parsed_content"] = parsed_content
                guardrail_reason = None
                guardrail_tool_message = None
                if action == "tool_call":
                    guardrail_tool_message, guardrail_reason = self._preflight_tool_call(
                        parsed_content["function"],
                        turns=context["turns"],
                        state=context["state"],
                    )
                context["_guardrail_reason"] = guardrail_reason
                if guardrail_tool_message is not None:
                    context["_next_obs"] = guardrail_tool_message
                    context["_done_flag"] = 0
                    context["_valid_action_flag"] = 0
                    context["_is_search_flag"] = (
                        1 if str(parsed_content["function"].get("name") or "") in SEARCH_TOOL_NAMES else 0
                    )
                    context["_next_state"] = copy.deepcopy(context["state"])
                    continue
                env_ready_contexts.append(context)
                env_predictions.append(response_text)
                env_multimodal_cache_batch.append(context["multimodal_cache"])
                env_states_batch.append(context["state"])

            if env_ready_contexts:
                next_obs, dones, valid_actions, is_search, next_states = self.environment.execute_predictions(
                    env_predictions,
                    env_multimodal_cache_batch,
                    env_states_batch,
                    [True] * len(env_ready_contexts),
                )
                for context, next_obs_entry, done_flag, valid_action_flag, is_search_flag, next_state in zip(
                    env_ready_contexts,
                    next_obs,
                    dones,
                    valid_actions,
                    is_search,
                    next_states,
                ):
                    context["_next_obs"] = next_obs_entry
                    context["_done_flag"] = done_flag
                    context["_valid_action_flag"] = valid_action_flag
                    context["_is_search_flag"] = is_search_flag
                    context["_next_state"] = next_state

            for context in ready_contexts:
                step_index = int(context["_step_index"])
                response_text = str(context["_response_text"])
                action = context["_action"]
                parsed_content = context["_parsed_content"]
                next_state = context["_next_state"]
                is_valid_action = bool(context["_valid_action_flag"])
                if is_valid_action:
                    context["messages"].append(self.adapter.build_assistant_message(response_text))
                    context["formal_step_index"] += 1
                context["state"] = next_state
                state_before = context["_state_before"]
                state_after = asdict(copy.deepcopy(context["state"])) if needs_state_snapshots else None

                turn_info = {
                    "step_index": step_index,
                    "response": response_text,
                    "assistant_response_raw": response_text,
                    "action": action,
                    "assistant_action": action,
                    "done": bool(context["_done_flag"]),
                    "valid_action": bool(context["_valid_action_flag"]),
                    "is_search": bool(context["_is_search_flag"]),
                    "tool_name": None,
                    "parsed_tool_call": None,
                    "tool_signature": None,
                    "tool_observation_summary": None,
                    "tool_timestamps": [],
                    "tool_image_count": 0,
                    "new_evidence_ids": [],
                    "new_verifications": [],
                    "new_finalized_case": None,
                    "verifier_mode": None,
                    "verifier_backend": None,
                    "verifier_primary_status": None,
                    "verifier_recommended_action": None,
                    "verifier_derived_scores": None,
                    "verifier_verified_window_ids": None,
                    "verifier_best_effort_window_ids": None,
                    "verifier_failure_reasons": None,
                    "verification_parse_mode": None,
                    "invalid_selected_window_ids": [],
                    "selection_resolution_source": None,
                    "self_verification_decision": None,
                    "self_verification_scores": None,
                    "self_verification_selected_window_ids": None,
                    "self_verification_confidence": None,
                    "proposal_backend": None,
                    "feature_cache_used": None,
                    "proposal_query_raw": None,
                    "proposal_query_normalized": None,
                    "proposal_query_source": None,
                    "proposal_candidate_count": None,
                    "proposal_candidate_frame_indices": [],
                    "proposal_candidate_frame_scores": [],
                    "proposal_candidate_windows": [],
                    "proposal_selected_frame_indices": [],
                    "proposal_selected_frame_scores": [],
                    "proposal_fallback_reason": None,
                    "guardrail_reason": context["_guardrail_reason"],
                    "parsed_answer": None,
                    "parsed_semantic_answer": None,
                    "semantic_answer_text": None,
                }
                if capture_prompt_messages:
                    turn_info["_prompt_messages"] = context["_prompt_messages_before_action"]
                tool_message = (
                    context["_next_obs"]
                    if isinstance(context["_next_obs"], dict) and context["_next_obs"].get("role") == "tool"
                    else None
                )

                if action == "tool_call":
                    turn_info["tool_name"] = parsed_content["function"]["name"]
                    turn_info["parsed_tool_call"] = parsed_content["function"]
                    turn_info["tool_signature"] = _canonical_search_signature(
                        parsed_content["function"]["name"],
                        parsed_content["function"].get("arguments") or {},
                    )
                elif action == "answer":
                    raw_answer_text = self.adapter.parse_answer_text(response_text)
                    answer_payload = self._coerce_answer_payload(raw_answer_text)
                    semantic_payload = self._coerce_semantic_answer_payload(answer_payload)
                    projected_answer = self._project_final_answer(answer_payload)
                    if projected_answer is None and isinstance(context["state"].finalized_case, dict):
                        projected_answer = copy.deepcopy(context["state"].finalized_case)
                    context["final_answer"] = projected_answer
                    context["final_answer_text"] = (
                        json.dumps(context["final_answer"], ensure_ascii=False)
                        if isinstance(context["final_answer"], dict)
                        else None
                    )
                    context["semantic_answer"] = semantic_payload
                    context["semantic_answer_text"] = semantic_answer_to_text(semantic_payload)
                    context["semantic_answer_source"] = "answer" if semantic_payload is not None else None
                    context["final_answer_source"] = (
                        "answer" if isinstance(context["final_answer"], dict) else None
                    )
                    context["terminated_reason"] = "answered"
                    context["terminated_at_step"] = step_index
                    turn_info["parsed_answer"] = context["final_answer"]
                    turn_info["parsed_semantic_answer"] = context["semantic_answer"]
                    turn_info["semantic_answer_text"] = context["semantic_answer_text"]
                    if not is_valid_action:
                        context["final_answer"] = None
                        context["final_answer_text"] = None
                        context["semantic_answer"] = None
                        context["semantic_answer_text"] = None
                        context["semantic_answer_source"] = None
                        context["final_answer_source"] = None
                        context["terminated_reason"] = "max_turns"
                        context["terminated_at_step"] = self.max_turns
                    else:
                        if needs_state_delta and state_before is not None and state_after is not None:
                            state_delta = self._compute_state_delta(state_before, state_after)
                            turn_info["state_before"] = state_before
                            turn_info["state_after"] = state_after
                            turn_info["state_delta"] = state_delta
                            turn_info["new_evidence_ids"] = [
                                entry["evidence_id"] for entry in state_delta["new_evidence_windows"]
                            ]
                            turn_info["new_verifications"] = state_delta["new_verifications"]
                            turn_info["new_finalized_case"] = state_delta["new_finalized_case"]
                            self._attach_search_turn_trace(turn_info, state_delta=state_delta)
                        if needs_counterfactual_trace and state_before is not None and state_after is not None:
                            self._attach_counterfactual_turn_trace(
                                turn_info,
                                state_before=state_before,
                                state_after=state_after,
                            )
                        context["turns"].append(turn_info)
                        context["done"] = True
                        continue
                if tool_message is not None:
                    adapted_tool_message = self.adapter.adapt_tool_observation(
                        tool_message,
                        context["multimodal_cache"],
                    )
                    adapted_tool_message = self._append_budget_reminder(
                        adapted_tool_message,
                        remaining_turns=self.max_turns - int(context["formal_step_index"]),
                    )
                    self._append_tool_message(context["messages"], adapted_tool_message)
                    if turn_info["tool_name"] is None:
                        turn_info["tool_name"] = tool_message.get("name")
                    turn_info.update(self._summarize_tool_message(tool_message))
                if (
                    is_valid_action
                    and str(turn_info.get("tool_name") or "") == "finalize_case"
                    and isinstance(context["state"].finalized_case, dict)
                ):
                    context["final_answer"] = copy.deepcopy(context["state"].finalized_case)
                    context["final_answer_text"] = json.dumps(context["final_answer"], ensure_ascii=False)
                    if isinstance(context["state"].finalized_semantic_answer, dict):
                        context["semantic_answer"] = copy.deepcopy(context["state"].finalized_semantic_answer)
                        context["semantic_answer_text"] = semantic_answer_to_text(context["semantic_answer"])
                        context["semantic_answer_source"] = "finalize_case_inline"
                        turn_info["parsed_semantic_answer"] = copy.deepcopy(context["semantic_answer"])
                        turn_info["semantic_answer_text"] = context["semantic_answer_text"]
                    context["final_answer_source"] = "finalize_case"
                    context["terminated_reason"] = "finalized"
                    context["terminated_at_step"] = step_index

                if needs_state_delta and state_before is not None and state_after is not None:
                    state_delta = self._compute_state_delta(state_before, state_after)
                    turn_info["state_before"] = state_before
                    turn_info["state_after"] = state_after
                    turn_info["state_delta"] = state_delta
                    turn_info["new_evidence_ids"] = [
                        entry["evidence_id"] for entry in state_delta["new_evidence_windows"]
                    ]
                    turn_info["new_verifications"] = state_delta["new_verifications"]
                    turn_info["new_finalized_case"] = state_delta["new_finalized_case"]
                    self._attach_search_turn_trace(turn_info, state_delta=state_delta)
                if needs_counterfactual_trace and state_before is not None and state_after is not None:
                    self._attach_counterfactual_turn_trace(
                        turn_info,
                        state_before=state_before,
                        state_after=state_after,
                    )
                if is_valid_action:
                    context["turns"].append(turn_info)
                    if (
                        str(turn_info.get("tool_name") or "") == "finalize_case"
                        and isinstance(context["state"].finalized_case, dict)
                    ):
                        context["done"] = True
                else:
                    context["invalid_attempts"].append(turn_info)

        return [self._build_episode_result(context) for context in episode_contexts]

    @staticmethod
    def _coerce_answer_payload(answer_text: Optional[str]) -> Any:
        if answer_text is None:
            return None
        try:
            payload = json.loads(answer_text)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _coerce_semantic_answer_payload(answer_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(answer_payload, dict):
            return None
        return normalize_semantic_answer_payload(answer_payload)

    @staticmethod
    def _project_final_answer(answer_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(answer_payload, dict):
            return None
        projected = extract_decision_from_semantic_answer(answer_payload)
        if projected is not None:
            return projected
        if "existence" in answer_payload or "category" in answer_payload:
            return copy.deepcopy(answer_payload)
        return None

    @staticmethod
    def _compute_state_delta(state_before: Dict[str, Any], state_after: Dict[str, Any]) -> Dict[str, Any]:
        new_visited_windows = state_after["visited_windows"][len(state_before["visited_windows"]) :]
        new_evidence_windows = state_after["evidence_ledger"][len(state_before["evidence_ledger"]) :]
        new_verifications = state_after["verification_records"][len(state_before["verification_records"]) :]
        new_finalized_case = None
        if state_before.get("finalized_case") != state_after.get("finalized_case"):
            new_finalized_case = state_after.get("finalized_case")
        return {
            "new_visited_windows": new_visited_windows,
            "new_evidence_windows": new_evidence_windows,
            "new_verifications": new_verifications,
            "new_finalized_case": new_finalized_case,
            "next_evidence_id_delta": state_after["next_evidence_id"] - state_before["next_evidence_id"],
        }

    def _attach_counterfactual_turn_trace(
        self,
        turn_info: Dict[str, Any],
        *,
        state_before: Dict[str, Any],
        state_after: Dict[str, Any],
    ) -> None:
        turn_info["observed_horizon_sec_before"] = self._observed_horizon_sec(state_before)
        turn_info["observed_horizon_sec_after"] = self._observed_horizon_sec(state_after)
        turn_info["latest_claim_before"] = self._latest_claim(state_before)
        turn_info["latest_claim_after"] = self._latest_claim(state_after)
        turn_info["selected_window_ids_before"] = self._selected_window_ids(state_before)
        turn_info["selected_window_ids_after"] = self._selected_window_ids(state_after)
        turn_info["selected_evidence_ids_before"] = self._selected_evidence_ids(state_before)
        turn_info["selected_evidence_ids_after"] = self._selected_evidence_ids(state_after)
        turn_info["counterfactual_anchor_tags"] = self._counterfactual_anchor_tags(turn_info)
        turn_info["counterfactual_actual_search_branch"] = self._actual_search_branch(turn_info)
        turn_info["counterfactual_actual_evidence_branch"] = self._actual_evidence_branch(turn_info)

    @staticmethod
    def _attach_search_turn_trace(turn_info: Dict[str, Any], *, state_delta: Dict[str, Any]) -> None:
        if str(turn_info.get("tool_name") or "") != "seek_evidence":
            return
        new_windows = list(state_delta.get("new_evidence_windows") or [])
        if not new_windows:
            return
        latest = dict(new_windows[-1])
        turn_info["proposal_backend"] = latest.get("proposal_backend")
        turn_info["feature_cache_used"] = latest.get("feature_cache_used")
        turn_info["proposal_query_raw"] = latest.get("query")
        turn_info["proposal_query_normalized"] = latest.get("query_normalized")
        turn_info["proposal_query_source"] = latest.get("query_source")
        turn_info["proposal_candidate_count"] = latest.get("proposal_candidate_count")
        turn_info["proposal_candidate_frame_indices"] = list(latest.get("proposal_candidate_frame_indices") or [])
        turn_info["proposal_candidate_frame_scores"] = list(latest.get("proposal_candidate_frame_scores") or [])
        turn_info["proposal_candidate_windows"] = copy.deepcopy(latest.get("proposal_candidate_windows") or [])
        turn_info["proposal_selected_frame_indices"] = list(latest.get("selected_frame_indices") or [])
        turn_info["proposal_selected_frame_scores"] = list(latest.get("selected_frame_scores") or [])
        turn_info["proposal_fallback_reason"] = latest.get("proposal_fallback_reason")

    def _summarize_tool_message(self, tool_message: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not tool_message:
            return {
                "tool_observation_summary": None,
                "tool_timestamps": [],
                "tool_image_count": 0,
                "tool_observation_content": None,
                "verifier_mode": None,
                "verifier_backend": None,
                "verifier_primary_status": None,
                "verifier_recommended_action": None,
                "verifier_derived_scores": None,
                "verifier_verified_window_ids": None,
                "verifier_best_effort_window_ids": None,
                "verifier_failure_reasons": None,
                "verification_parse_mode": None,
                "invalid_selected_window_ids": [],
                "selection_resolution_source": None,
                "self_verification_decision": None,
                "self_verification_scores": None,
                "self_verification_selected_window_ids": None,
                "self_verification_confidence": None,
            }

        content = tool_message.get("content", [])
        tool_timestamps = []
        tool_image_count = 0
        tool_observation_summary = None
        parsed_json_payload = None
        for item in content:
            if item.get("type") == "image":
                tool_image_count += 1
            elif item.get("type") == "text":
                text = str(item.get("text", ""))
                if text.endswith("s"):
                    tool_timestamps.append(text)
                tool_observation_summary = text
                if parsed_json_payload is None:
                    try:
                        parsed_json_payload = json.loads(text)
                    except Exception:
                        parsed_json_payload = None
        summary = {
            "tool_observation_summary": tool_observation_summary,
            "tool_timestamps": tool_timestamps,
            "tool_image_count": tool_image_count,
            "verifier_mode": None,
            "verifier_backend": None,
            "verifier_primary_status": None,
            "verifier_recommended_action": None,
            "verifier_derived_scores": None,
            "verifier_verified_window_ids": None,
            "verifier_best_effort_window_ids": None,
            "verifier_failure_reasons": None,
            "self_verification_decision": None,
            "self_verification_scores": None,
            "self_verification_selected_window_ids": None,
            "self_verification_confidence": None,
        }
        if tool_message.get("name") == "verify_hypothesis" and isinstance(parsed_json_payload, dict):
            normalized_payload = parsed_json_payload
            if (
                normalized_payload.get("primary_status") is None
                and normalized_payload.get("verification_decision") is not None
            ):
                normalized_payload = parse_self_verification_payload(normalized_payload)
            summary.update(
                {
                    "verifier_mode": normalized_payload.get("verification_mode"),
                    "verifier_backend": normalized_payload.get("verifier_backend"),
                    "verifier_primary_status": normalized_payload.get("primary_status"),
                    "verifier_recommended_action": normalized_payload.get("recommended_action"),
                    "verifier_derived_scores": normalized_payload.get("derived_scores"),
                    "verifier_verified_window_ids": normalized_payload.get("verified_window_ids"),
                    "verifier_best_effort_window_ids": normalized_payload.get("best_effort_window_ids"),
                    "verifier_failure_reasons": normalized_payload.get("failure_reasons"),
                    "verification_parse_mode": normalized_payload.get("verification_parse_mode"),
                    "invalid_selected_window_ids": list(normalized_payload.get("invalid_selected_window_ids") or []),
                    "selection_resolution_source": normalized_payload.get("selection_resolution_source"),
                    "self_verification_decision": normalized_payload.get("verification_decision"),
                    "self_verification_scores": normalized_payload.get("self_verification_scores")
                    or normalized_payload.get("derived_scores"),
                    "self_verification_selected_window_ids": normalized_payload.get("selected_window_ids")
                    or normalized_payload.get("verified_window_ids"),
                    "self_verification_confidence": normalized_payload.get("self_verification_confidence"),
                }
            )
        if self.config.rollout_trace.record_observation_content:
            summary["tool_observation_content"] = content
        else:
            summary["tool_observation_content"] = None
        return summary

    @staticmethod
    def _build_preview_trace(multimodal_cache: Dict[str, Any]) -> Dict[str, Any]:
        preview_frames = multimodal_cache.get("preview_frames")
        preview_timestamps = multimodal_cache.get("preview_timestamps") or []
        preview_frame_count = 0 if preview_frames is None else int(len(preview_frames))
        return {
            "preview_frame_count": preview_frame_count,
            "preview_timestamps": preview_timestamps,
        }

    @staticmethod
    def _latest_verifier_status(turns: List[Dict[str, Any]]) -> Optional[str]:
        for turn in reversed(turns):
            if turn.get("verifier_primary_status"):
                return turn["verifier_primary_status"]
        return None

    @staticmethod
    def _verification_turn_count(turns: List[Dict[str, Any]]) -> int:
        return sum(1 for turn in turns if turn.get("tool_name") == "verify_hypothesis")

    @staticmethod
    def _final_verified_window_ids(turns: List[Dict[str, Any]]) -> Optional[List[str]]:
        for turn in reversed(turns):
            if turn.get("verifier_verified_window_ids"):
                return turn["verifier_verified_window_ids"]
        return None

    @staticmethod
    def _observed_horizon_sec(state_dict: Dict[str, Any]) -> float:
        max_end = 0.0
        for entry in state_dict.get("visited_windows") or []:
            try:
                max_end = max(max_end, float(entry.get("end_sec") or 0.0))
            except Exception:
                continue
        return round(float(max_end), 6)

    @staticmethod
    def _latest_claim(state_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        claim = state_dict.get("last_claim")
        if isinstance(claim, dict):
            return copy.deepcopy(claim)
        finalized_case = state_dict.get("finalized_case")
        if isinstance(finalized_case, dict):
            return copy.deepcopy(finalized_case)
        return None

    @staticmethod
    def _selected_window_ids(state_dict: Dict[str, Any]) -> List[str]:
        active = state_dict.get("active_evidence_window_ids") or []
        if active:
            return [str(value) for value in active]
        verification_records = state_dict.get("verification_records") or []
        if verification_records:
            latest = verification_records[-1]
            values = latest.get("verified_window_ids") or latest.get("best_effort_window_ids") or []
            return [str(value) for value in values]
        return []

    @staticmethod
    def _selected_evidence_ids(state_dict: Dict[str, Any]) -> List[str]:
        window_ids = set(SaverRolloutRunner._selected_window_ids(state_dict))
        if not window_ids:
            return []
        evidence_ids: List[str] = []
        for entry in state_dict.get("evidence_ledger") or []:
            window_id = str(entry.get("window_id") or "")
            evidence_id = entry.get("evidence_id")
            if window_id in window_ids and evidence_id:
                evidence_ids.append(str(evidence_id))
        return evidence_ids

    @staticmethod
    def _counterfactual_anchor_tags(turn_info: Dict[str, Any]) -> List[str]:
        tool_name = str(turn_info.get("tool_name") or "")
        tags: List[str] = []
        if tool_name in {"scan_timeline", "seek_evidence"}:
            tags.append("search_anchor")
        if tool_name in {"verify_hypothesis", "finalize_case"}:
            tags.append("evidence_anchor")
        return tags

    @staticmethod
    def _actual_search_branch(turn_info: Dict[str, Any]) -> Optional[str]:
        tags = turn_info.get("counterfactual_anchor_tags") or []
        if "search_anchor" not in tags:
            return None
        return "use_search"

    @staticmethod
    def _actual_evidence_branch(turn_info: Dict[str, Any]) -> Optional[str]:
        tags = turn_info.get("counterfactual_anchor_tags") or []
        if "evidence_anchor" not in tags:
            return None
        selected = turn_info.get("selected_window_ids_after") or []
        return "keep_selected" if selected else "full_ledger"

    @staticmethod
    def _build_counterfactual_anchor_summary(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        search_anchor_turns = [
            int(turn.get("step_index") or 0)
            for turn in turns
            if "search_anchor" in (turn.get("counterfactual_anchor_tags") or [])
        ]
        evidence_anchor_turns = [
            int(turn.get("step_index") or 0)
            for turn in turns
            if "evidence_anchor" in (turn.get("counterfactual_anchor_tags") or [])
        ]
        return {
            "num_search_anchors": len(search_anchor_turns),
            "num_evidence_anchors": len(evidence_anchor_turns),
            "search_anchor_turn_indices": search_anchor_turns,
            "evidence_anchor_turn_indices": evidence_anchor_turns,
        }

    @staticmethod
    def _decision_turn_indices(turns: List[Dict[str, Any]]) -> List[int]:
        return [
            int(turn.get("step_index") or 0)
            for turn in turns
            if str(turn.get("tool_name") or "") in {"verify_hypothesis", "finalize_case"}
            or str(turn.get("action") or "") == "answer"
        ]

    @staticmethod
    def _build_latest_claim_trace(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        for turn in turns:
            claim = turn.get("latest_claim_after")
            if claim is None:
                continue
            trace.append(
                {
                    "step_index": int(turn.get("step_index") or 0),
                    "claim": copy.deepcopy(claim),
                }
            )
        return trace

    @staticmethod
    def _build_search_trace(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        seek_turns = [turn for turn in turns if str(turn.get("tool_name") or "") == "seek_evidence"]
        backend_counts: Dict[str, int] = {}
        query_sequence: List[str] = []
        num_feature_guided = 0
        num_uniform_fallback = 0
        for turn in seek_turns:
            backend = str(turn.get("proposal_backend") or "unknown")
            backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if backend in {"feature_topk", "siglip_dpp"}:
                num_feature_guided += 1
            if backend == "uniform":
                num_uniform_fallback += 1
            query_text = str(turn.get("proposal_query_normalized") or turn.get("proposal_query_raw") or "").strip()
            if query_text:
                query_sequence.append(query_text)
        num_query_revisions = 0
        for previous, current in zip(query_sequence, query_sequence[1:]):
            if current != previous:
                num_query_revisions += 1
        return {
            "num_seek_turns": len(seek_turns),
            "num_feature_guided_searches": num_feature_guided,
            "num_uniform_fallback_searches": num_uniform_fallback,
            "num_query_revisions": num_query_revisions,
            "query_sequence": query_sequence,
            "proposal_backend_counts": backend_counts,
        }
