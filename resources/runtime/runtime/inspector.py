"""Pre-load inspection of a local Safetensors checkpoint.

This module imports nothing beyond the standard library. Inspection has to work
before torch and Transformers are provisioned, and it has to be cheap enough to
run over a model list, so it reads the Safetensors headers directly rather than
opening the checkpoint through any framework.

Reading the header is not an exception to the delegate-upstream rule: the goal
is a size and capability estimate, not tensor access. Nothing here loads,
assembles, or interprets weights.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from .errors import InvalidModelError, MissingWeightsError
from .models import ModelInspection
from .protocol import log

# Safetensors stores an 8-byte little-endian header length followed by a JSON
# header. A header past this size means the file is not what it claims to be.
MAX_HEADER_BYTES = 128 * 1024 * 1024

# Bytes per element, used to report the checkpoint's dominant precision.
DTYPE_WIDTH = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def validate(path: Path) -> None:
    """Confirm the directory is shaped like a checkpoint before anything heavier.

    Kept deliberately shallow. Transformers stays authoritative for whether a
    checkpoint actually loads; this only rejects obvious mistakes fast.
    """

    if not path.exists():
        raise InvalidModelError(f"Model directory does not exist: {path}")
    if not path.is_dir():
        raise InvalidModelError(f"Model path is not a directory: {path}")
    if not (path / "config.json").exists():
        raise InvalidModelError(f"Missing config.json in {path}")
    if not list(path.glob("*.safetensors")):
        raise MissingWeightsError(f"No Safetensors checkpoint found in {path}")


def read_safetensors_header(file_path: Path) -> dict:
    """The JSON header of one Safetensors file, or an empty dict if unreadable.

    An unreadable header must not block inspection: the caller falls back to
    file sizes, which are always available.
    """

    try:
        with file_path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                log("warning", f"{file_path.name} is too short to hold a Safetensors header.")
                return {}
            (length,) = struct.unpack("<Q", prefix)
            if length <= 0 or length > MAX_HEADER_BYTES:
                log("warning", f"{file_path.name} declares an implausible header length ({length} bytes).")
                return {}
            return json.loads(handle.read(length).decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log("warning", f"Could not read the Safetensors header of {file_path.name}: {exc!r}")
        return {}


def _weight_summary(files: list[Path]) -> tuple[int, str | None]:
    """Total weight bytes and the dominant dtype across every shard."""

    total = 0
    widths: dict[str, int] = {}
    for file_path in files:
        header = read_safetensors_header(file_path)
        counted = 0
        for name, entry in header.items():
            if name == "__metadata__" or not isinstance(entry, dict):
                continue
            offsets = entry.get("data_offsets")
            if isinstance(offsets, list) and len(offsets) == 2:
                span = int(offsets[1]) - int(offsets[0])
                counted += max(0, span)
                dtype = entry.get("dtype")
                if isinstance(dtype, str):
                    widths[dtype] = widths.get(dtype, 0) + max(0, span)
        # A header that produced nothing still contributes its file size, so a
        # partially readable checkpoint is never reported as weightless.
        total += counted if counted else file_path.stat().st_size
    dominant = max(widths, key=lambda key: widths[key]) if widths else None
    return total, dominant


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Optional sidecars are routinely absent; the caller decides if it matters.
        return {}
    except (OSError, ValueError) as exc:
        log("warning", f"Could not read {path.name}: {exc!r}")
        return {}


def _context_length(config: dict) -> int | None:
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return _context_length(text_config)
    return None


def _quantization(config: dict) -> str | None:
    """The quantization scheme declared by the checkpoint, if any.

    Reported so the extension can warn before a load. Pre-quantized formats
    largely depend on CUDA-only kernels, so on Metal and Intel XPU they either
    fail or dequantize back to bf16 and use more memory than the unquantized
    checkpoint would have.
    """

    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return None
    for key in ("quant_method", "format"):
        value = quant.get(key)
        if isinstance(value, str):
            return value
    groups = quant.get("config_groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, dict) and isinstance(group.get("format"), str):
                return group["format"]
    return "unknown"


def _supports_chat(path: Path) -> bool:
    """Whether the tokenizer ships a chat template.

    Recent checkpoints keep the template in a standalone `chat_template.jinja`
    rather than inside `tokenizer_config.json`, so both locations count.
    """

    if (path / "chat_template.jinja").exists():
        return True
    return bool(_read_json(path / "tokenizer_config.json").get("chat_template"))


def inspect(path: Path) -> ModelInspection:
    """Describe a checkpoint without loading a single weight."""

    validate(path)
    config = _read_json(path / "config.json")
    files = sorted(path.glob("*.safetensors"))
    weight_bytes, dtype = _weight_summary(files)
    architectures = config.get("architectures")
    architecture = (
        architectures[0]
        if isinstance(architectures, list) and architectures
        else None
    )
    return ModelInspection(
        path=str(path),
        modelType=str(config.get("model_type", "unknown")),
        architecture=architecture,
        isEncoderDecoder=bool(config.get("is_encoder_decoder", False)),
        supportsChat=_supports_chat(path),
        contextLength=_context_length(config),
        weightFormat="safetensors",
        sharded=(path / "model.safetensors.index.json").exists() or len(files) > 1,
        fileCount=len(files),
        weightBytes=weight_bytes,
        dtype=dtype or (config.get("torch_dtype") or config.get("dtype")),
        quantization=_quantization(config),
        customCodeRequired="auto_map" in config,
        adapter=(path / "adapter_config.json").exists(),
    )
