"""Tests for the parts of the runtime that do not need torch or Transformers.

The memory gate, option parsing and inspection all run before a single weight is
touched, which is exactly why they are testable here — and why they are worth
testing, since they decide whether a load is attempted at all.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from runtime import inspector, runtime as runtime_module
from runtime.errors import (
    GrammarCompileFailedError,
    GrammarUnsupportedError,
    InsufficientMemoryError,
    InvalidModelError,
    MissingWeightsError,
)
from runtime.models import GenerationOptions, RuntimePolicy

GIB = 1024 ** 3


def write_checkpoint(directory: Path, *, shards=((1, 4096),), config=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config or {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "max_position_embeddings": 8192,
    }))
    for index, size in shards:
        header = json.dumps({
            f"model.layers.{index}.weight": {
                "dtype": "BF16", "shape": [size // 2], "data_offsets": [0, size],
            },
        }).encode()
        (directory / f"model-0000{index}.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"\0" * size
        )
    return directory


class MemoryGateTest(unittest.TestCase):
    """The gate that stops a model loading into disk offload."""

    def setUp(self) -> None:
        self.llm = runtime_module.LocalLLM()
        self._device = runtime_module.available_device_bytes
        self._host = runtime_module.available_host_bytes

    def tearDown(self) -> None:
        runtime_module.available_device_bytes = self._device
        runtime_module.available_host_bytes = self._host

    def patch(self, device: int | None, host: int | None) -> None:
        runtime_module.available_device_bytes = lambda: device
        runtime_module.available_host_bytes = lambda: host

    def test_allows_a_model_that_fits_the_device(self) -> None:
        self.patch(device=48 * GIB, host=48 * GIB)
        self.llm._assert_fits(8 * GIB, RuntimePolicy())

    def test_rejects_exceeding_the_device_when_cpu_offload_is_refused(self) -> None:
        self.patch(device=16 * GIB, host=64 * GIB)
        with self.assertRaises(InsufficientMemoryError):
            self.llm._assert_fits(30 * GIB, RuntimePolicy(allowCpuOffload=False))

    def test_allows_cpu_offload_when_host_memory_can_hold_the_model(self) -> None:
        # A discrete GPU with plenty of system RAM behind it: offload is slow
        # but real, so the load is allowed — but only when explicitly opted in.
        self.patch(device=8 * GIB, host=64 * GIB)
        self.llm._assert_fits(30 * GIB, RuntimePolicy(allowCpuOffload=True))

    def test_default_policy_refuses_device_exceed_without_opt_in(self) -> None:
        self.patch(device=8 * GIB, host=64 * GIB)
        with self.assertRaises(InsufficientMemoryError) as caught:
            self.llm._assert_fits(30 * GIB, RuntimePolicy())
        self.assertIn("CPU offload", str(caught.exception))

    def test_rejects_a_model_larger_than_host_memory(self) -> None:
        # Unified memory: CPU offload buys nothing, so disk is the only place
        # left and the default policy forbids it. This is the case that used to
        # slip through and produce a model generating at unusable speed.
        self.patch(device=36 * GIB, host=36 * GIB)
        with self.assertRaises(InsufficientMemoryError) as caught:
            self.llm._assert_fits(60 * GIB, RuntimePolicy(allowCpuOffload=True))
        self.assertIn("disk", str(caught.exception))

    def test_load_within_tolerance_warns_but_proceeds(self) -> None:
        # Usable is 90 GiB of a 100 GiB device; 92 GiB is ~2% over the line.
        self.patch(device=100 * GIB, host=100 * GIB)
        with self.assertWarns(UserWarning):
            self.llm._assert_fits(92 * GIB, RuntimePolicy())

    def test_load_beyond_tolerance_refuses(self) -> None:
        # 100 GiB is ~11% over the 90 GiB usable line: past the 5% band.
        self.patch(device=100 * GIB, host=100 * GIB)
        with self.assertRaises(InsufficientMemoryError):
            self.llm._assert_fits(100 * GIB, RuntimePolicy())

    def test_disk_offload_opt_in_bypasses_the_gate_entirely(self) -> None:
        self.patch(device=4 * GIB, host=8 * GIB)
        self.llm._assert_fits(200 * GIB, RuntimePolicy(allowDiskOffload=True))

    def test_unknown_memory_never_blocks_a_load(self) -> None:
        # Guessing would be worse than allowing it: the load may well succeed.
        self.patch(device=None, host=None)
        self.llm._assert_fits(60 * GIB, RuntimePolicy())


class ContextLengthTest(unittest.TestCase):
    """The loaded worker sees the same window as inspection.

    Multimodal configs nest it under text_config; the loaded worker used to
    look only at the top level and reported unknown context, which collapsed
    every context-derived budget to a single token.
    """

    def length_of(self, config) -> int | None:
        llm = runtime_module.LocalLLM()
        llm.config = config
        return llm._context_length()

    def test_top_level_window(self) -> None:
        self.assertEqual(
            self.length_of(SimpleNamespace(max_position_embeddings=32768)), 32768,
        )

    def test_nested_text_config_window(self) -> None:
        self.assertEqual(
            self.length_of(SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=131072),
            )),
            131072,
        )

    def test_missing_window_is_unknown(self) -> None:
        self.assertIsNone(self.length_of(SimpleNamespace()))


class UnloadCachePurgeTest(unittest.TestCase):
    """Unload purges only available backends.

    Merely exposing empty_cache is not enough: a half-initialized backend
    aborts the interpreter when purged, and no except clause catches that.
    """

    def _torch(self, cuda=False, xpu=False, mps=False):
        calls = []

        def empty():
            calls.append(True)

        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: cuda, empty_cache=empty)
        torch.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps) if mps is not None else None
        )
        if xpu:
            torch.xpu = SimpleNamespace(is_available=lambda: True, empty_cache=empty)
        sys.modules["torch"] = torch
        return calls

    def setUp(self) -> None:
        self._saved = sys.modules.pop("torch", ...)

    def tearDown(self) -> None:
        sys.modules.pop("torch", None)
        if self._saved is not ...:
            sys.modules["torch"] = self._saved

    def unload_with(self, **backends):
        calls = self._torch(**backends)
        llm = runtime_module.LocalLLM()
        llm.model = object()
        llm.unload()
        return calls

    def test_unavailable_backends_are_not_purged(self) -> None:
        self.assertEqual(self.unload_with(), [])

    def test_available_backend_is_purged(self) -> None:
        self.assertEqual(len(self.unload_with(cuda=True)), 1)


class ThinkingChannelTest(unittest.TestCase):
    """Reasoning traces strip to the answered text, per vendor protocol."""

    def test_gemma_channel_keeps_final_text(self) -> None:
        self.assertEqual(
            runtime_module.strip_thinking_channels(
                "<|channel>thought\nI called it; result ok.\n<channel|>ok"
            ),
            "ok",
        )

    def test_qwen_think_block_is_dropped(self) -> None:
        self.assertEqual(
            runtime_module.strip_thinking_channels("<think>hmm</think>the value is ok"),
            "the value is ok",
        )

    def test_plain_text_passes_through(self) -> None:
        self.assertEqual(
            runtime_module.strip_thinking_channels("The validation value is ok."),
            "The validation value is ok.",
        )

    def test_unclosed_thought_drops_to_end(self) -> None:
        self.assertEqual(
            runtime_module.strip_thinking_channels("ok<think>never finished"),
            "ok",
        )

    def test_lone_closer_drops_orphaned_thought(self) -> None:
        # The prompt ends inside the open channel, so generation emits only
        # the closer after thinking.
        self.assertEqual(
            runtime_module.strip_thinking_channels(
                "The function returned ok.\n<channel|>ok"
            ),
            "ok",
        )

    def test_specials_clean_after_channel_strip(self) -> None:
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                assert text == "kept"
                assert add_special_tokens is False
                return [7]

            def decode(self, ids, skip_special_tokens=True):
                assert ids == [7]
                assert skip_special_tokens is True
                return "kept"

        llm = runtime_module.LocalLLM()
        llm.tokenizer = FakeTokenizer()
        self.assertEqual(llm._clean_special_tokens("kept"), "kept")

    def test_thinking_kwarg_reaches_the_template_only_when_set(self) -> None:
        seen = {}

        class FakeTokenizer:
            chat_template = "template"

            def apply_chat_template(self, _messages, **kwargs):
                seen.update(kwargs)
                return {"input_ids": [[1]]}

        llm = runtime_module.LocalLLM()
        llm.tokenizer = FakeTokenizer()
        llm._encode_chat([], GenerationOptions.from_params({}))
        self.assertNotIn("enable_thinking", seen)
        llm._encode_chat([], GenerationOptions.from_params({"enableThinking": True}))
        self.assertTrue(seen["enable_thinking"])


class OptionParsingTest(unittest.TestCase):
    def test_unknown_keys_are_dropped_rather_than_raising(self) -> None:
        # A newer extension paired with an older worker must degrade, not fail.
        options = GenerationOptions.from_params(
            {"maxNewTokens": 32, "somethingFromTheFuture": True}
        )
        self.assertEqual(options.maxNewTokens, 32)
        self.assertEqual(options.temperature, 0.7)

    def test_policy_defaults_refuse_disk_offload(self) -> None:
        policy = RuntimePolicy.from_params(None)
        self.assertFalse(policy.allowDiskOffload)
        self.assertFalse(policy.trustRemoteCode)

    def test_policy_defaults_refuse_cpu_offload(self) -> None:
        # Decided per scope: silent CPU offload is never the default; the
        # caller opts in explicitly. CPU offload that is allowed is slow but
        # real, so it stays available behind the flag.
        self.assertFalse(RuntimePolicy.from_params(None).allowCpuOffload)

    def test_json_schema_survives_option_parsing(self) -> None:
        schema = {"type": "object"}
        self.assertEqual(GenerationOptions.from_params({"jsonSchema": schema}).jsonSchema, schema)


class GrammarBackendTest(unittest.TestCase):
    """Schema-constrained kwargs without importing torch or xgrammar.

    The backend is stubbed in sys.modules: the point under test is the
    refusal codes and the wiring, not xgrammar's masking (proven separately
    against the provisioned interpreter).
    """

    STUBBED = ("xgrammar", "xgrammar.contrib", "xgrammar.contrib.hf", "transformers")

    def setUp(self) -> None:
        self.llm = runtime_module.LocalLLM()
        self.llm.config = SimpleNamespace(vocab_size=1000)
        self.llm.tokenizer = object()
        self._saved = {name: sys.modules.get(name, ...) for name in self.STUBBED}
        for name in self.STUBBED:
            sys.modules.pop(name, None)

    def tearDown(self) -> None:
        for name in self.STUBBED:
            sys.modules.pop(name, None)
        for name, module in self._saved.items():
            if module is not ...:
                sys.modules[name] = module

    def _stub_backend(self, compile_result=None, compile_error=None):
        calls = {"compiled": 0}

        class FakeCompiler:
            def compile_json_schema(self, _schema):
                calls["compiled"] += 1
                if compile_error is not None:
                    raise compile_error
                return compile_result

        package = ModuleType("xgrammar")
        package.GrammarCompiler = lambda _info: FakeCompiler()  # noqa: E731
        package.TokenizerInfo = SimpleNamespace(from_huggingface=lambda _tok, **_kw: "info")
        contrib = ModuleType("xgrammar.contrib")
        hf = ModuleType("xgrammar.contrib.hf")
        hf.LogitsProcessor = lambda compiled: ("processor", compiled)
        transformers = ModuleType("transformers")
        transformers.LogitsProcessorList = lambda items: ("list", list(items))
        sys.modules["xgrammar"] = package
        sys.modules["xgrammar.contrib"] = contrib
        sys.modules["xgrammar.contrib.hf"] = hf
        sys.modules["transformers"] = transformers
        return calls

    def options(self):
        return GenerationOptions.from_params(
            {"maxNewTokens": 64, "jsonSchema": {"type": "object"}}
        )

    def test_missing_backend_refuses_with_code(self) -> None:
        sys.modules["xgrammar"] = None  # type: ignore[assignment]
        with self.assertRaises(GrammarUnsupportedError) as caught:
            self.llm._generation_kwargs(self.options())
        self.assertEqual(caught.exception.code, "unsupported_grammar")

    def test_no_schema_wires_no_processor(self) -> None:
        kwargs = self.llm._generation_kwargs(GenerationOptions.from_params({}))
        self.assertNotIn("logits_processor", kwargs)
        self.assertEqual(kwargs["max_new_tokens"], 512)

    def test_compile_failure_maps_to_code(self) -> None:
        self._stub_backend(compile_error=RuntimeError("bad type"))
        with self.assertRaises(GrammarCompileFailedError) as caught:
            self.llm._generation_kwargs(self.options())
        self.assertEqual(caught.exception.code, "grammar_compile_failed")

    def test_compiled_grammar_is_cached_per_schema(self) -> None:
        calls = self._stub_backend(compile_result="grammar")
        first = self.llm._generation_kwargs(self.options())
        second = self.llm._generation_kwargs(self.options())
        self.assertEqual(calls["compiled"], 1)
        self.assertEqual(first["logits_processor"], ("list", [("processor", "grammar")]))
        self.assertEqual(second["logits_processor"], ("list", [("processor", "grammar")]))


class InspectorTest(unittest.TestCase):
    def test_reports_shards_and_weight_bytes_from_headers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = write_checkpoint(Path(raw) / "m", shards=((1, 4096), (2, 2048)))
            result = inspector.inspect(path)
        self.assertTrue(result.sharded)
        self.assertEqual(result.fileCount, 2)
        self.assertEqual(result.weightBytes, 6144)
        self.assertEqual(result.dtype, "BF16")
        self.assertEqual(result.contextLength, 8192)

    def test_detects_custom_code_and_quantization(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = write_checkpoint(Path(raw) / "m", config={
                "architectures": ["WeirdForCausalLM"],
                "model_type": "weird",
                "auto_map": {"AutoModelForCausalLM": "modeling_weird.WeirdModel"},
                "quantization_config": {
                    "config_groups": {"group_0": {"format": "nvfp4-pack-quantized"}},
                },
            })
            result = inspector.inspect(path)
        self.assertTrue(result.customCodeRequired)
        self.assertEqual(result.quantization, "nvfp4-pack-quantized")

    def test_falls_back_to_file_size_when_a_header_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "m"
            path.mkdir()
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"not a safetensors header")
            result = inspector.inspect(path)
        self.assertEqual(result.weightBytes, len(b"not a safetensors header"))

    def test_rejects_a_directory_with_no_weights(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "m"
            path.mkdir()
            (path / "config.json").write_text("{}")
            with self.assertRaises(MissingWeightsError):
                inspector.inspect(path)

    def test_header_cap_matches_typescript_side(self) -> None:
        # Parity contract with src/models/safetensorsDirectory.ts
        # MAX_SAFETENSORS_HEADER_BYTES. Decided: 128 MiB both sides.
        self.assertEqual(inspector.MAX_HEADER_BYTES, 128 * 1024 * 1024)

    def test_shared_oversize_fixture_is_refused_without_allocating(self) -> None:
        # Shared fixture with src/models/safetensorsDirectory.test.ts: a
        # 14-byte file claiming a 200 MiB header. Must return {} — the cap
        # check runs before any 200 MiB allocation.
        fixture = (
            Path(__file__).parent / "fixtures" / "oversize-header-claim.safetensors"
        )
        self.assertEqual(inspector.read_safetensors_header(fixture), {})

    def test_rejects_a_missing_directory(self) -> None:
        with self.assertRaises(InvalidModelError):
            inspector.inspect(Path("/definitely/not/here"))


if __name__ == "__main__":
    unittest.main()
