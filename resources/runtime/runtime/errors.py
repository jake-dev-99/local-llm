"""Normalized error contract for the local Safetensors runtime.

Raw Transformers and PyTorch exceptions carry useful detail for logs but make a
poor public API: their types and messages change between upstream releases. The
worker maps everything onto these codes, and the extension branches on the code
rather than on exception text.
"""

from __future__ import annotations


class RuntimeErrorBase(Exception):
    """Base class for every error the worker reports over the protocol."""

    code = "internal_error"


class InvalidModelError(RuntimeErrorBase):
    code = "invalid_model"


class UnsupportedArchitectureError(RuntimeErrorBase):
    code = "unsupported_architecture"


class MissingTokenizerError(RuntimeErrorBase):
    code = "missing_tokenizer"


class MissingWeightsError(RuntimeErrorBase):
    code = "missing_weights"


class CustomCodeRequiredError(RuntimeErrorBase):
    code = "custom_code_required"


class UnsupportedQuantizationError(RuntimeErrorBase):
    code = "unsupported_quantization"


class DeviceUnavailableError(RuntimeErrorBase):
    code = "device_unavailable"


class InsufficientMemoryError(RuntimeErrorBase):
    code = "insufficient_memory"


class ContextOverflowError(RuntimeErrorBase):
    code = "context_overflow"


class ModelLoadFailedError(RuntimeErrorBase):
    code = "model_load_failed"


class ModelNotLoadedError(RuntimeErrorBase):
    code = "model_not_loaded"


class ChatNotSupportedError(RuntimeErrorBase):
    code = "chat_not_supported"


class GenerationFailedError(RuntimeErrorBase):
    code = "generation_failed"


class GenerationCancelledError(RuntimeErrorBase):
    code = "generation_cancelled"


class GenerationBusyError(RuntimeErrorBase):
    code = "generation_busy"


class UnknownMethodError(RuntimeErrorBase):
    code = "unknown_method"


class InvalidRequestError(RuntimeErrorBase):
    code = "invalid_request"


def error_code(exc: BaseException) -> str:
    """The protocol code for any exception, including ones from upstream."""

    return getattr(exc, "code", "internal_error")


class GrammarUnsupportedError(RuntimeErrorBase):
    code = "unsupported_grammar"


class GrammarCompileFailedError(RuntimeErrorBase):
    code = "grammar_compile_failed"
