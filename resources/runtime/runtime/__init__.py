"""Local Safetensors LLM runtime.

Orchestration only. Architecture support, tokenization, chat formatting, device
placement and kernels all belong to Transformers, Accelerate and PyTorch.
"""

from .models import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION"]
