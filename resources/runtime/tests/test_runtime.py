"""Tests for the parts of the runtime that do not need torch or Transformers.

The memory gate, option parsing and inspection all run before a single weight is
touched, which is exactly why they are testable here — and why they are worth
testing, since they decide whether a load is attempted at all.
"""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from runtime import inspector, runtime as runtime_module
from runtime.errors import InsufficientMemoryError, InvalidModelError, MissingWeightsError
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
        # but real, so the load is allowed.
        self.patch(device=8 * GIB, host=64 * GIB)
        self.llm._assert_fits(30 * GIB, RuntimePolicy())

    def test_rejects_a_model_larger_than_host_memory(self) -> None:
        # Unified memory: CPU offload buys nothing, so disk is the only place
        # left and the default policy forbids it. This is the case that used to
        # slip through and produce a model generating at unusable speed.
        self.patch(device=36 * GIB, host=36 * GIB)
        with self.assertRaises(InsufficientMemoryError) as caught:
            self.llm._assert_fits(60 * GIB, RuntimePolicy())
        self.assertIn("disk", str(caught.exception))

    def test_disk_offload_opt_in_bypasses_the_gate_entirely(self) -> None:
        self.patch(device=4 * GIB, host=8 * GIB)
        self.llm._assert_fits(200 * GIB, RuntimePolicy(allowDiskOffload=True))

    def test_unknown_memory_never_blocks_a_load(self) -> None:
        # Guessing would be worse than allowing it: the load may well succeed.
        self.patch(device=None, host=None)
        self.llm._assert_fits(60 * GIB, RuntimePolicy())


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

    def test_rejects_a_missing_directory(self) -> None:
        with self.assertRaises(InvalidModelError):
            inspector.inspect(Path("/definitely/not/here"))


if __name__ == "__main__":
    unittest.main()
