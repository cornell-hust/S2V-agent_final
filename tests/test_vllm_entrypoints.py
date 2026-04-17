import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from saver_v3.inference.policy_rollout import PolicyRolloutConfig
from saver_v3.inference.rollout_eval import StepRolloutEvalConfig
from saver_v3.model import vllm_generation
from saver_v3.rl import trl_grpo_trainer


class VllmEntrypointConfigTests(unittest.TestCase):
    def test_single_rank_external_launcher_env_pins_visible_device_and_local_rank_zero(self) -> None:
        runtime = SimpleNamespace(local_rank=5)
        with mock.patch.dict(os.environ, {"MASTER_PORT": "29710"}, clear=False):
            env = vllm_generation._resolve_single_rank_external_launcher_env(runtime)
        self.assertEqual(env["RANK"], "0")
        self.assertEqual(env["WORLD_SIZE"], "1")
        self.assertEqual(env["LOCAL_RANK"], "0")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(env["MASTER_PORT"], "29815")

    def test_policy_rollout_config_requires_raw_data_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "io.data_path"):
            PolicyRolloutConfig.from_mapping(
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "io": {"input_manifest": "/tmp/legacy.jsonl", "output_path": "/tmp/out.jsonl"},
                }
            )

    def test_policy_rollout_config_parses_raw_data_contract(self) -> None:
        config = PolicyRolloutConfig.from_mapping(
            {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "server": {"mode": "colocate", "tensor_parallel_size": 1},
                "client": {"max_tokens": 384},
                "io": {
                    "data_path": "/tmp/raw_eval.jsonl",
                    "output_path": "/tmp/out.jsonl",
                    "include_splits": "val",
                    "count": 12,
                },
            }
        )

        self.assertEqual(config.data_path, "/tmp/raw_eval.jsonl")
        self.assertEqual(config.output_path, "/tmp/out.jsonl")
        self.assertEqual(config.include_splits, "val")
        self.assertEqual(config.count, 12)
        self.assertEqual(config.max_new_tokens, 384)

    def test_rollout_eval_config_requires_raw_data_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "io.data_path"):
            StepRolloutEvalConfig.from_mapping(
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "io": {"input_manifest": "/tmp/legacy.jsonl", "output_dir": "/tmp/eval"},
                }
            )

    def test_rollout_eval_config_parses_raw_data_contract(self) -> None:
        config = StepRolloutEvalConfig.from_mapping(
            {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "client": {"max_tokens": 256, "max_total_images": 24},
                "io": {
                    "data_path": "/tmp/raw_eval.jsonl",
                    "output_dir": "/tmp/eval",
                    "include_splits": "val",
                    "max_records": 8,
                },
                "evaluation": {"epoch_index": 3, "max_turns": 10},
            }
        )

        self.assertEqual(config.data_path, "/tmp/raw_eval.jsonl")
        self.assertEqual(config.output_dir, "/tmp/eval")
        self.assertEqual(config.max_records, 8)
        self.assertEqual(config.epoch_index, 3)
        self.assertEqual(config.max_turns, 10)
        self.assertEqual(config.policy_max_new_tokens, 256)
        self.assertEqual(config.max_total_images, 24)

    def test_per_rank_local_server_runtime_does_not_init_communicator_on_construct(self) -> None:
        fake_client = mock.Mock()
        runtime = SimpleNamespace(world_size=8, local_rank=3, is_main_process=False)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="server",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_server_host="127.0.0.1",
            vllm_server_port=8003,
            vllm_server_timeout=240.0,
            vllm_server_auto_launch=False,
            vllm_server_per_rank=True,
            vllm_server_max_lora_rank=64,
        )
        with mock.patch.object(vllm_generation, "_SAVERVLLMClient", return_value=fake_client), mock.patch(
            "torch.distributed.is_available", return_value=False
        ), mock.patch("torch.distributed.is_initialized", return_value=False):
            server_runtime = vllm_generation._VllmServerRuntime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )
        self.assertIs(server_runtime.client, fake_client)
        fake_client.init_communicator.assert_not_called()

    def test_per_rank_local_server_runtime_lazy_inits_communicator_on_weight_sync(self) -> None:
        fake_client = mock.Mock()
        runtime = SimpleNamespace(world_size=8, local_rank=2, is_main_process=False)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="server",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_server_host="127.0.0.1",
            vllm_server_port=8002,
            vllm_server_timeout=240.0,
            vllm_server_auto_launch=False,
            vllm_server_per_rank=True,
            vllm_server_max_lora_rank=64,
        )
        with mock.patch.object(vllm_generation, "_SAVERVLLMClient", return_value=fake_client), mock.patch.object(
            vllm_generation, "_iter_named_weights_for_vllm", return_value=[("weight", object())]
        ), mock.patch("torch.distributed.is_available", return_value=False), mock.patch(
            "torch.distributed.is_initialized", return_value=False
        ):
            server_runtime = vllm_generation._VllmServerRuntime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )
            server_runtime.ensure_weights_synced(source_model=object(), global_step=1)
        fake_client.init_communicator.assert_called_once()
        fake_client.update_named_param.assert_called_once()

    def test_create_vllm_runtime_prefers_external_launcher_for_static_distributed_eval(self) -> None:
        runtime = SimpleNamespace(world_size=8, rank=3, local_rank=3, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
        )
        fake_runtime = object()
        with mock.patch.object(
            vllm_generation,
            "_VllmExternalLauncherRuntime",
            return_value=fake_runtime,
        ) as external_runtime_cls, mock.patch.object(
            vllm_generation,
            "_VllmColocateRuntime",
        ) as colocate_runtime_cls:
            resolved_runtime = vllm_generation.create_vllm_runtime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
                prefer_direct_local_rank_runtime=True,
            )
        self.assertIs(resolved_runtime, fake_runtime)
        external_runtime_cls.assert_called_once()
        colocate_runtime_cls.assert_not_called()

    def test_external_launcher_runtime_preserves_torchrun_env(self) -> None:
        captured = {}

        class _FakeLLM:
            def __init__(self, **kwargs):
                captured["kwargs"] = dict(kwargs)
                captured["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                captured["rank"] = os.environ.get("RANK")
                captured["world_size"] = os.environ.get("WORLD_SIZE")
                captured["local_rank"] = os.environ.get("LOCAL_RANK")

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = _FakeLLM
        runtime = SimpleNamespace(world_size=8, rank=3, local_rank=3, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="external_launcher",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            torch_dtype="bfloat16",
            seed=0,
            max_total_images=28,
            vllm_server_max_lora_rank=64,
        )
        with mock.patch.dict(
            os.environ,
            {
                "RANK": "3",
                "WORLD_SIZE": "8",
                "LOCAL_RANK": "3",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29710",
            },
            clear=False,
        ), mock.patch.dict(sys.modules, {"vllm": fake_vllm}), mock.patch.object(
            vllm_generation, "patch_vllm_guided_decoding_params"
        ), mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch("torch.distributed.barrier"), mock.patch.object(
            vllm_generation, "runtime_log"
        ):
            runtime_impl = vllm_generation._VllmExternalLauncherRuntime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )
        self.assertEqual(captured["kwargs"]["distributed_executor_backend"], "external_launcher")
        self.assertEqual(captured["rank"], "3")
        self.assertEqual(captured["world_size"], "8")
        self.assertEqual(captured["local_rank"], "3")
        runtime_impl.close()

    def test_external_launcher_runtime_initializes_torch_distributed_before_vllm(self) -> None:
        class _FakeLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = _FakeLLM
        runtime = SimpleNamespace(world_size=8, rank=1, local_rank=1, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="external_launcher",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            torch_dtype="bfloat16",
            seed=0,
            max_total_images=28,
            vllm_server_max_lora_rank=64,
        )
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}), mock.patch.object(
            vllm_generation, "patch_vllm_guided_decoding_params"
        ), mock.patch.object(
            vllm_generation, "init_torch_distributed", return_value=True
        ) as init_dist_mock, mock.patch(
            "torch.distributed.is_available", return_value=True
        ), mock.patch(
            "torch.distributed.is_initialized", return_value=True
        ), mock.patch(
            "torch.distributed.barrier"
        ), mock.patch.object(
            vllm_generation, "runtime_log"
        ):
            runtime_impl = vllm_generation._VllmExternalLauncherRuntime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )
        init_dist_mock.assert_called_once_with(runtime)
        runtime_impl.close()

    def test_build_vllm_policy_from_model_path_uses_remote_lora_for_static_external_launcher_adapter(self) -> None:
        runtime = SimpleNamespace(world_size=8, rank=0, local_rank=0, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_guided_decoding_regex="",
            max_total_images=28,
            max_seq_length=4096,
            keep_recent_tool_image_messages=0,
            keep_recent_text_messages=20,
            max_image_side=640,
            max_image_pixels=0,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
        )
        fake_runtime = SimpleNamespace(
            enabled=True,
            settings={},
            ensure_weights_synced=lambda *a, **k: None,
            reset_prefix_cache=lambda: None,
            close=lambda: None,
            llm=mock.Mock(),
        )
        processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: []), batch_decode=mock.Mock())
        remote_lora_request = {
            "lora_name": "adapter-checkpoint-6",
            "lora_int_id": 1,
            "lora_path": "/tmp/checkpoint-6",
            "base_model_name": "/tmp/base-model",
        }
        with mock.patch.object(
            vllm_generation,
            "_should_use_static_external_launcher_runtime",
            return_value=True,
        ), mock.patch.object(
            vllm_generation,
            "build_remote_vllm_lora_request",
            return_value=remote_lora_request,
        ), mock.patch.object(
            vllm_generation,
            "_load_processor_or_placeholder",
            return_value=processor,
        ), mock.patch.object(
            vllm_generation,
            "create_vllm_runtime",
            return_value=fake_runtime,
        ) as create_runtime_mock, mock.patch.object(
            vllm_generation,
            "_resolve_vllm_base_model_path",
            return_value="/tmp/base-model",
        ):
            policy = vllm_generation.build_vllm_policy_from_model_path(
                args=args,
                runtime=runtime,
                model_path=Path("/tmp/checkpoint-6"),
                max_new_tokens=256,
                max_total_images=28,
                max_seq_length=4096,
                keep_recent_tool_image_messages=0,
                keep_recent_text_messages=20,
                max_image_side=640,
                max_image_pixels=0,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                prefer_direct_local_rank_runtime=True,
            )
        self.assertIsNone(policy.source_model)
        self.assertIs(policy.processor, processor)
        self.assertEqual(policy.remote_lora_request, remote_lora_request)
        self.assertEqual(create_runtime_mock.call_args.kwargs["model_path"], "/tmp/base-model")

    def test_recovery_vllm_policy_factory_prefers_direct_local_rank_runtime_for_static_eval(self) -> None:
        captured = {}

        def _fake_runtime_builder(*, args, runtime, model_path, prefer_direct_local_rank_runtime=False):
            captured["prefer_direct_local_rank_runtime"] = bool(prefer_direct_local_rank_runtime)
            captured["vllm_mode"] = getattr(args, "vllm_mode", None)
            captured["model_path"] = str(model_path)
            return SimpleNamespace(
                enabled=True,
                settings={},
                ensure_weights_synced=lambda *a, **k: None,
                reset_prefix_cache=lambda: None,
                close=lambda: None,
                llm=mock.Mock(),
            )

        runtime = SimpleNamespace(world_size=8, rank=0, local_rank=0, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_guided_decoding_regex="",
            max_total_images=28,
            max_seq_length=4096,
            keep_recent_tool_image_messages=0,
            keep_recent_text_messages=20,
            max_image_side=640,
            max_image_pixels=0,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
        )
        processor = object()
        with mock.patch.object(vllm_generation, "_maybe_load_adapter_source_model", return_value=(None, processor)), mock.patch.object(
            vllm_generation, "_resolve_sft_vllm_runtime_args", side_effect=AssertionError("server reroute should not be used for static recovery eval")
        ):
            factory = vllm_generation.build_recovery_vllm_policy_factory(
                args=args,
                runtime_builder=_fake_runtime_builder,
            )
            policy, cleanup = factory(
                checkpoint_path=Path("/tmp/checkpoint-6"),
                model_path="/unused",
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_3",
                rollout_eval_config=SimpleNamespace(
                    policy_max_new_tokens=256,
                    max_total_images=28,
                    max_seq_length=4096,
                    keep_recent_tool_image_messages=0,
                    keep_recent_text_messages=20,
                    max_image_side=640,
                    max_image_pixels=0,
                    use_generation_cache=True,
                ),
                runtime=runtime,
            )
        self.assertTrue(captured["prefer_direct_local_rank_runtime"])
        self.assertEqual(captured["vllm_mode"], "colocate")
        self.assertEqual(captured["model_path"], "/tmp/checkpoint-6")
        cleanup()

    def test_recovery_vllm_policy_factory_uses_remote_lora_for_static_external_launcher_adapter(self) -> None:
        runtime = SimpleNamespace(world_size=8, rank=0, local_rank=0, is_distributed=True)
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_guided_decoding_regex="",
            max_total_images=28,
            max_seq_length=4096,
            keep_recent_tool_image_messages=0,
            keep_recent_text_messages=20,
            max_image_side=640,
            max_image_pixels=0,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
        )
        fake_runtime = SimpleNamespace(
            enabled=True,
            settings={},
            ensure_weights_synced=lambda *a, **k: None,
            reset_prefix_cache=lambda: None,
            close=lambda: None,
            llm=mock.Mock(),
        )
        remote_lora_request = {
            "lora_name": "adapter-checkpoint-6",
            "lora_int_id": 1,
            "lora_path": "/tmp/checkpoint-6",
            "base_model_name": "/tmp/base-model",
        }
        processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: []), batch_decode=mock.Mock())
        with mock.patch.object(
            vllm_generation,
            "_should_use_static_external_launcher_runtime",
            return_value=True,
        ), mock.patch.object(
            vllm_generation,
            "build_remote_vllm_lora_request",
            return_value=remote_lora_request,
        ), mock.patch.object(
            vllm_generation,
            "_load_processor_or_placeholder",
            return_value=processor,
        ), mock.patch.object(
            vllm_generation,
            "create_vllm_runtime",
            return_value=fake_runtime,
        ) as create_runtime_mock, mock.patch.object(
            vllm_generation,
            "_resolve_vllm_base_model_path",
            return_value="/tmp/base-model",
        ):
            factory = vllm_generation.build_recovery_vllm_policy_factory(args=args)
            policy, cleanup = factory(
                checkpoint_path=Path("/tmp/checkpoint-6"),
                model_path="/unused",
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_3",
                rollout_eval_config=SimpleNamespace(
                    policy_max_new_tokens=256,
                    max_total_images=28,
                    max_seq_length=4096,
                    keep_recent_tool_image_messages=0,
                    keep_recent_text_messages=20,
                    max_image_side=640,
                    max_image_pixels=0,
                    use_generation_cache=True,
                ),
                runtime=runtime,
            )
        self.assertIsNone(policy.source_model)
        self.assertIs(policy.processor, processor)
        self.assertEqual(policy.remote_lora_request, remote_lora_request)
        self.assertEqual(create_runtime_mock.call_args.kwargs["model_path"], "/tmp/base-model")
        cleanup()


if __name__ == "__main__":
    unittest.main()


class RLVllmRuntimeSelectionTests(unittest.TestCase):
    def test_rl_create_vllm_runtime_prefers_direct_local_rank_runtime(self) -> None:
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.4,
            vllm_max_num_seqs=4,
            vllm_fallback_max_num_seqs=2,
        )
        runtime = SimpleNamespace(local_rank=3, rank=3, world_size=8, is_distributed=True, is_main_process=False)
        fake_runtime = object()
        with mock.patch.object(
            trl_grpo_trainer.shared_vllm_generation,
            "create_vllm_runtime",
            return_value=fake_runtime,
        ) as create_mock:
            result = trl_grpo_trainer.create_vllm_runtime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )
        self.assertIs(result, fake_runtime)
        _, kwargs = create_mock.call_args
        self.assertTrue(kwargs["prefer_direct_local_rank_runtime"])
        self.assertEqual(kwargs["args"].vllm_max_num_seqs, 4)

    def test_rl_create_vllm_runtime_falls_back_to_two_on_oom(self) -> None:
        args = SimpleNamespace(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.4,
            vllm_max_num_seqs=4,
            vllm_fallback_max_num_seqs=2,
        )
        runtime = SimpleNamespace(local_rank=1, rank=1, world_size=8, is_distributed=True, is_main_process=False)
        calls = []

        def _fake_create_vllm_runtime(*, args, runtime, model_path, prefer_direct_local_rank_runtime):
            calls.append((args.vllm_max_num_seqs, prefer_direct_local_rank_runtime))
            if args.vllm_max_num_seqs == 4:
                raise RuntimeError("CUDA out of memory while loading vLLM model")
            return "ok"

        with mock.patch.object(
            trl_grpo_trainer.shared_vllm_generation,
            "create_vllm_runtime",
            side_effect=_fake_create_vllm_runtime,
        ):
            result = trl_grpo_trainer.create_vllm_runtime(
                args=args,
                runtime=runtime,
                model_path="/models/qwen3-vl-8b-Instruct",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, [(4, True), (2, True)])
