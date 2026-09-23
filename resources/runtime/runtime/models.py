"""Data contracts exchanged with the extension.

Every dataclass here serialises straight to JSON through `dataclasses.asdict`.
Field names are the wire format, so they use the extension's camelCase rather
than Python's snake_case.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

# Bumped whenever a request or response shape changes incompatibly. The
# extension refuses to drive a worker whose version it does not recognise.
PROTOCOL_VERSION = 1


@dataclass(slots=True)
class RuntimeInfo:
    """Hardware and dependency detection, reported for capability and support.

    Detection is informational. PyTorch and Accelerate remain responsible for
    execution; nothing in this runtime branches on `backend` to choose a code
    path.
    """

    protocolVersion: int
    platform: str
    pythonVersion: str
    deviceType: str
    deviceName: str
    backend: str
    versions: dict[str, str]


@dataclass(slots=True)
class ModelCapabilities:
    """What a loaded model can actually do, in the extension's vocabulary."""

    completion: bool = True
    chat: bool = False
    streaming: bool = True
    encoderDecoder: bool = False
    customCodeRequired: bool = False
    quantized: bool = False
    adapter: bool = False
    contextLength: int | None = None


@dataclass(slots=True)
class ModelInspection:
    """Everything discoverable without loading weights.

    This is the gate in front of `model.load`. Accelerate will happily place a
    model far larger than device memory by offloading to CPU and then disk,
    producing a model that loads successfully and generates unusably slowly, so
    the decision has to be made before the load rather than after it.
    """

    path: str
    modelType: str
    architecture: str | None
    isEncoderDecoder: bool
    supportsChat: bool
    contextLength: int | None
    weightFormat: str
    sharded: bool
    fileCount: int
    weightBytes: int
    dtype: str | None
    quantization: str | None
    customCodeRequired: bool
    adapter: bool


@dataclass(slots=True)
class ModelInfo:
    """State of the model that is currently loaded."""

    path: str
    modelType: str
    architecture: str | None
    modelClass: str | None
    tokenizerClass: str | None
    dtype: str | None
    isEncoderDecoder: bool
    supportsChat: bool
    contextLength: int | None
    deviceMap: Any
    capabilities: ModelCapabilities
    runtime: RuntimeInfo


@dataclass(slots=True)
class GenerationOptions:
    """Caller overrides applied on top of the model's own GenerationConfig.

    Anything left unset stays at whatever the model shipped with; this runtime
    does not substitute its own opinion for a model's defaults.
    """

    maxNewTokens: int = 512
    temperature: float = 0.7
    topP: float = 0.95
    topK: int = 50
    repetitionPenalty: float = 1.0
    jsonSchema: dict[str, Any] | None = None
    # Reasoning-channel switch for the chat template. None leaves the template
    # default untouched; True/False is passed through as enable_thinking, the
    # documented kwarg thinking models (Qwen3, Gemma, Ministral, …) read.
    enableThinking: bool | None = None

    @classmethod
    def from_params(cls, raw: dict[str, Any] | None) -> "GenerationOptions":
        """Build options from a request, ignoring keys this version does not know.

        An older worker paired with a newer extension must degrade rather than
        fail, so unknown keys are dropped instead of raising.
        """

        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})


@dataclass(slots=True)
class RuntimePolicy:
    """The small set of knobs the extension is allowed to turn.

    Deliberately not a passthrough for every Transformers argument: vendor
    specifics belong to PyTorch and Accelerate, not to this API.
    """

    device: str = "auto"
    allowCpuOffload: bool = False
    allowDiskOffload: bool = False
    trustRemoteCode: bool = False

    @classmethod
    def from_params(cls, raw: dict[str, Any] | None) -> "RuntimePolicy":
        if not raw:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})


@dataclass(slots=True)
class WorkerStatus:
    """Worker lifecycle state, so the extension never infers it from UI timing."""

    state: str
    modelPath: str | None = None
    detail: str | None = None


WORKER_STATES = (
    "STARTING",
    "READY",
    "LOADING",
    "LOADED",
    "GENERATING",
    "UNLOADING",
    "FAILED",
    "STOPPED",
)
