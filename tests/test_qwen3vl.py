import sys
import types
import unittest
from unittest import mock

from saver_v3.model.qwen3vl import (
    build_qwen3vl_inputs,
    extract_vision_inputs,
    load_qwen3vl_model,
    load_qwen3vl_processor,
)
from saver_v3.model.qwen_policy import load_auto_processor_with_compat


class DummyProcessor:
    def __init__(self) -> None:
        self.padding_side = "right"
        self.tokenizer = types.SimpleNamespace(padding_side="right")
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(("template", kwargs))
        if kwargs.get("tokenize"):
            raise TypeError("tokenized chat template not supported in the test stub")
        return "PROMPT"

    def __call__(self, **kwargs):
        self.calls.append(("processor", kwargs))
        return kwargs


class Qwen3VLTests(unittest.TestCase):
    def test_load_qwen3vl_processor_sets_left_padding(self) -> None:
        processor = DummyProcessor()
        captured = {}

        class AutoProcessor:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                self.assertEqual(model_name, "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct")
                captured["kwargs"] = dict(kwargs)
                return processor

        with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoProcessor=AutoProcessor)}):
            loaded = load_qwen3vl_processor("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct")

        self.assertIs(loaded, processor)
        self.assertEqual(loaded.padding_side, "left")
        self.assertEqual(loaded.tokenizer.padding_side, "left")
        self.assertTrue(captured["kwargs"]["fix_mistral_regex"])

    def test_load_auto_processor_with_compat_falls_back_when_fix_mistral_regex_is_unsupported(self) -> None:
        processor = DummyProcessor()
        calls = []

        class AutoProcessor:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls.append(dict(kwargs))
                if "fix_mistral_regex" in kwargs:
                    raise TypeError("from_pretrained() got an unexpected keyword argument 'fix_mistral_regex'")
                return processor

        with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoProcessor=AutoProcessor)}):
            loaded = load_auto_processor_with_compat(
                "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct",
                trust_remote_code=True,
            )

        self.assertIs(loaded, processor)
        self.assertEqual(len(calls), 2)
        self.assertIn("fix_mistral_regex", calls[0])
        self.assertNotIn("fix_mistral_regex", calls[1])
        self.assertTrue(calls[1]["trust_remote_code"])

    def test_load_qwen3vl_model_locks_fa3(self) -> None:
        captured = {}

        class Qwen3VLForConditionalGeneration:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                captured["model_name"] = model_name
                captured["kwargs"] = kwargs
                return types.SimpleNamespace(model_name=model_name, kwargs=kwargs)

        with mock.patch.dict(
            sys.modules,
            {"transformers": types.SimpleNamespace(Qwen3VLForConditionalGeneration=Qwen3VLForConditionalGeneration)},
        ):
            model = load_qwen3vl_model(
                "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct",
                torch_dtype="auto",
                env={"SAVER_V3_ATTN_BACKEND": "fa3"},
                cuda_device_capabilities=[(9, 0)],
                module_available=lambda name: name == "flash_attn_interface",
            )

        self.assertEqual(model.model_name, "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct")
        self.assertEqual(captured["kwargs"]["attn_implementation"], "flash_attention_3")
        self.assertTrue(captured["kwargs"]["trust_remote_code"])

    def test_extract_vision_inputs_and_fallback_processing(self) -> None:
        processor = DummyProcessor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe the clip"},
                    {"type": "image", "image": "image-1"},
                    {"type": "video", "video": ["frame-1", "frame-2"]},
                ],
            }
        ]

        images, videos = extract_vision_inputs(messages)
        batch = build_qwen3vl_inputs(processor, messages)

        self.assertEqual(images, ["image-1"])
        self.assertEqual(videos, [["frame-1", "frame-2"]])
        self.assertEqual(batch["text"], "PROMPT")
        self.assertEqual(batch["images"], ["image-1"])
        self.assertEqual(batch["videos"], [["frame-1", "frame-2"]])

    def test_missing_transformers_raises_clear_error(self) -> None:
        with mock.patch("importlib.import_module", side_effect=ImportError("missing transformers")):
            with self.assertRaises(ImportError):
                load_qwen3vl_processor("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct")
