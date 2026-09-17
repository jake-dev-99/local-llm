"""Model lifecycle and generation.

Everything model-specific is delegated: Transformers resolves the architecture,
tokenizer, chat template and generation defaults; Accelerate places the weights;
PyTorch selects the device backend. What remains here is orchestration —
lifecycle, a memory gate, streaming, and cancellation.

torch and Transformers are imported lazily so the worker can answer
`model.inspect` and `runtime.info` on a machine where the stack is still being
provisioned.
"""

from __future__ import annotations

import gc
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import (
    ChatNotSupportedError,
    ContextOverflowError,
    CustomCodeRequiredError,
    DeviceUnavailableError,
    GenerationCancelledError,
    GenerationFailedError,
    GrammarUnsupportedError,
    InsufficientMemoryError,
    ModelLoadFailedError,
    ModelNotLoadedError,
)
from .inspector import inspect as inspect_checkpoint
from .models import (
    PROTOCOL_VERSION,
    GenerationOptions,
    ModelCapabilities,
    ModelInfo,
    RuntimeInfo,
    RuntimePolicy,
)

TokenSink = Callable[[str], None]


@dataclass(slots=True)
class GenerationResult:
    """One completion, with the prompt size the model actually saw.

    The token count travels with the response because the runtime has it at
    hand; asking for it separately would cost a round trip per reply to report
    a number the worker already computed.
    """

    text: str
    inputTokens: int


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - provisioning failure path
        raise DeviceUnavailableError(
            "PyTorch is not available in this runtime environment."
        ) from exc
    return torch


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def detect_runtime() -> RuntimeInfo:
    """Report the active PyTorch backend.

    Informational only. No generation path branches on the result; it exists so
    the extension can show what hardware is in use and so support reports carry
    the dependency versions that decide architecture compatibility.
    """

    versions = {
        name: _package_version(name)
        for name in ("torch", "transformers", "accelerate", "safetensors", "peft")
    }
    base = {
        "protocolVersion": PROTOCOL_VERSION,
        "platform": platform.platform(),
        "pythonVersion": sys.version.split()[0],
        "versions": versions,
    }
    try:
        torch = _torch()
    except DeviceUnavailableError:
        return RuntimeInfo(
            deviceType="none", deviceName="PyTorch not installed",
            backend="none", **base,
        )

    if torch.cuda.is_available():
        # ROCm presents through the CUDA interface; torch.version.hip is what
        # separates an AMD build from an NVIDIA one.
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return RuntimeInfo(
            deviceType="cuda", deviceName=torch.cuda.get_device_name(0),
            backend=backend, **base,
        )
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = "Intel XPU"
        return RuntimeInfo(deviceType="xpu", deviceName=name, backend="xpu", **base)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return RuntimeInfo(
            deviceType="mps", deviceName="Apple Metal", backend="mps", **base,
        )
    return RuntimeInfo(
        deviceType="cpu", deviceName=platform.processor() or "CPU",
        backend="cpu", **base,
    )


def available_device_bytes() -> int | None:
    """Best-effort free memory on the active accelerator.

    Returns None when the backend exposes no usable figure, in which case the
    caller must not gate on memory rather than guess.
    """

    try:
        torch = _torch()
    except DeviceUnavailableError:
        return None
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return int(torch.xpu.get_device_properties(0).total_memory)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return int(torch.mps.recommended_max_memory())
    except Exception:
        return None
    return None


def available_host_bytes() -> int | None:
    """Total physical memory, or None where it cannot be determined.

    Deliberately standard-library only: adding a dependency to the baseline for
    one number would not earn its place in the dependency set.
    """

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except Exception:
        pass
    return None


def _cancel_criteria(cancelled: threading.Event):
    """A StoppingCriteria that ends generation once the cancel event is set.

    `generate` blocks, so cancellation cannot interrupt it from outside;
    checking a flag between steps is the supported way to stop early. The class
    is built here rather than at module scope so that importing this module
    never requires Transformers.
    """

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class CancelCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full(
                (input_ids.shape[0],),
                cancelled.is_set(),
                dtype=torch.bool,
                device=input_ids.device,
            )

    return StoppingCriteriaList([CancelCriteria()])


class LocalLLM:
    """One loaded model and the generation calls made against it."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.config: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.generation_config: Any = None
        self.policy = RuntimePolicy()
        self._runtime: RuntimeInfo | None = None
        self._cancelled = threading.Event()
        self._generating = threading.Lock()

    @property
    def runtime(self) -> RuntimeInfo:
        """Hardware detection, resolved on first use and cached.

        Deliberately not resolved in the constructor. Importing torch can abort
        the interpreter rather than raise — a duplicate OpenMP runtime is the
        common cause — and that must not prevent the worker from starting and
        answering the stdlib-only `model.inspect`.
        """

        if self._runtime is None:
            self._runtime = detect_runtime()
        return self._runtime

    # ---------------------------------------------------------------- lifecycle

    def load(self, path: str | Path, policy: RuntimePolicy | None = None) -> ModelInfo:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            GenerationConfig,
        )

        self.unload()
        policy = policy or RuntimePolicy()
        resolved = Path(path).expanduser().resolve()
        inspection = inspect_checkpoint(resolved)

        if inspection.customCodeRequired and not policy.trustRemoteCode:
            raise CustomCodeRequiredError(
                f"{resolved.name} ships custom model code. Loading it runs Python "
                "from the checkpoint, so it requires explicit opt-in."
            )
        self._assert_fits(inspection.weightBytes, policy)

        common = {
            "local_files_only": True,
            "trust_remote_code": policy.trustRemoteCode,
        }
        try:
            config = AutoConfig.from_pretrained(resolved, **common)
            tokenizer = AutoTokenizer.from_pretrained(resolved, **common)
            model_class = (
                AutoModelForSeq2SeqLM
                if getattr(config, "is_encoder_decoder", False)
                else AutoModelForCausalLM
            )
            model = model_class.from_pretrained(
                resolved,
                config=config,
                use_safetensors=True,
                # "auto" preserves the precision the checkpoint was saved in.
                dtype="auto",
                # Placement and offload belong to Accelerate.
                device_map=policy.device,
                **common,
            )
        except Exception as exc:
            message = str(exc)
            if "trust_remote_code" in message:
                raise CustomCodeRequiredError(message) from exc
            raise ModelLoadFailedError(message) from exc

        model.eval()
        try:
            generation_config = GenerationConfig.from_pretrained(
                resolved, local_files_only=True,
            )
        except Exception:
            generation_config = model.generation_config

        self.path = resolved
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.generation_config = generation_config
        self.policy = policy
        return self.info()

    def _assert_fits(self, weight_bytes: int, policy: RuntimePolicy) -> None:
        """Refuse a load that would land on disk rather than silently crawling.

        Accelerate never refuses: past device memory it offloads to CPU, and
        past host memory it offloads to disk, and the model still "loads" either
        way. Declining here is the difference between a clear error and an
        apparently working model generating at unusable speed.

        The check runs in two steps because the two policy flags gate different
        fallbacks: exceeding device memory is only acceptable when CPU offload is
        allowed, and exceeding host memory means disk offload is unavoidable no
        matter what CPU offload permits.
        """

        if policy.allowDiskOffload or weight_bytes <= 0:
            return

        def mib(value: int) -> int:
            return value // 1024 ** 2

        device = available_device_bytes()
        # Activations, the KV cache and framework overhead all sit on top of the
        # weights, so a checkpoint filling the whole device will not run.
        if device is not None and weight_bytes > int(device * 0.9):
            if not policy.allowCpuOffload:
                raise InsufficientMemoryError(
                    f"Model weights need about {mib(weight_bytes)} MiB and only "
                    f"{mib(int(device * 0.9))} MiB is usable on this device. "
                    "Enable CPU offload to load it anyway."
                )
            host = available_host_bytes()
            # On unified-memory hardware this is the same pool as the device, so
            # CPU offload buys nothing and disk is the only place left to go.
            if host is not None and weight_bytes > int(host * 0.9):
                raise InsufficientMemoryError(
                    f"Model weights need about {mib(weight_bytes)} MiB, more than "
                    f"the {mib(int(host * 0.9))} MiB usable on this machine. "
                    "Loading it would offload to disk and generate far too "
                    "slowly to use; a quantized build of this model will fit."
                )

    def unload(self) -> None:
        # Nothing was loaded, so there are no caches to drop. Returning early
        # also avoids importing torch on a machine where that import aborts.
        if self.model is None and self.path is None:
            return
        self.model = None
        self.tokenizer = None
        self.config = None
        self.generation_config = None
        self.path = None
        gc.collect()
        try:
            torch = _torch()
        except DeviceUnavailableError:
            return
        for empty in (
            getattr(getattr(torch, "cuda", None), "empty_cache", None),
            getattr(getattr(torch, "xpu", None), "empty_cache", None),
            getattr(getattr(torch, "mps", None), "empty_cache", None),
        ):
            try:
                if empty is not None:
                    empty()
            except Exception:
                pass

    # ------------------------------------------------------------ introspection

    def _require_loaded(self) -> None:
        if self.model is None:
            raise ModelNotLoadedError("No model is currently loaded.")

    def _context_length(self) -> int | None:
        for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            value = getattr(self.config, key, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _supports_chat(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def info(self) -> ModelInfo:
        self._require_loaded()
        architectures = getattr(self.config, "architectures", None)
        architecture = architectures[0] if architectures else None
        context_length = self._context_length()
        is_encoder_decoder = bool(getattr(self.config, "is_encoder_decoder", False))
        return ModelInfo(
            path=str(self.path),
            modelType=getattr(self.config, "model_type", "unknown"),
            architecture=architecture,
            modelClass=self.model.__class__.__name__,
            tokenizerClass=self.tokenizer.__class__.__name__,
            dtype=str(getattr(self.model, "dtype", None)),
            isEncoderDecoder=is_encoder_decoder,
            supportsChat=self._supports_chat(),
            contextLength=context_length,
            deviceMap=getattr(self.model, "hf_device_map", None),
            capabilities=ModelCapabilities(
                completion=True,
                chat=self._supports_chat(),
                streaming=True,
                encoderDecoder=is_encoder_decoder,
                customCodeRequired=self.policy.trustRemoteCode,
                quantized=hasattr(self.config, "quantization_config"),
                adapter=False,
                contextLength=context_length,
            ),
            runtime=self.runtime,
        )

    # ---------------------------------------------------------------- generation

    def _input_device(self):
        """The device holding the input embeddings.

        With `device_map="auto"` a model can be split across devices, so there is
        no single `model.device` to send inputs to.
        """

        return self.model.get_input_embeddings().weight.device

    def _move(self, inputs) -> dict:
        device = self._input_device()
        return {name: tensor.to(device) for name, tensor in inputs.items()}

    def _generation_kwargs(self, options: GenerationOptions) -> dict[str, Any]:
        if options.jsonSchema is not None:
            raise GrammarUnsupportedError(
                "Schema-constrained generation is not available in this runtime "
                "build. Install a grammar backend to enable tool calling."
            )
        kwargs: dict[str, Any] = {
            "max_new_tokens": options.maxNewTokens,
            "repetition_penalty": options.repetitionPenalty,
        }
        if options.temperature > 0:
            kwargs.update({
                "do_sample": True,
                "temperature": options.temperature,
                "top_p": options.topP,
                "top_k": options.topK,
            })
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _encode_chat(self, messages: list[dict[str, str]]):
        if not self._supports_chat():
            raise ChatNotSupportedError(
                "This model provides no chat template. Use completion mode."
            )
        # The tokenizer owns role formatting, special tokens and generation
        # markers; reproducing any of that here would drift per model family.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def _assert_context(self, prompt_tokens: int, options: GenerationOptions) -> None:
        limit = self._context_length()
        if limit is None:
            return
        needed = prompt_tokens + options.maxNewTokens
        if needed > limit:
            raise ContextOverflowError(
                f"Request needs {needed} tokens and the model context is {limit}. "
                "Shorten the input or lower maxNewTokens."
            )

    def count_tokens(self, text: str | None, messages: list | None) -> int:
        """Token count for a prompt, as the model's own tokenizer sees it.

        The extension enforces its input budget against a real count rather than
        an estimate, and only the loaded tokenizer can supply one. Chat messages
        are counted through the chat template so the count includes the role
        markers the model will actually receive.
        """

        self._require_loaded()
        if messages is not None:
            encoded = self._encode_chat(messages)
            return int(encoded["input_ids"].shape[-1])
        return len(self.tokenizer.encode(text or ""))

    def cancel(self) -> None:
        """Ask an in-flight generation to stop at the next token boundary."""

        self._cancelled.set()

    def generate(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        options: GenerationOptions | None = None,
        on_token: TokenSink | None = None,
    ) -> GenerationResult:
        """Run one generation, streaming through `on_token` when supplied.

        Streaming and non-streaming share this path so that cancellation, the
        context check and prompt trimming cannot drift apart between them.
        """

        self._require_loaded()
        if not self._generating.acquire(blocking=False):
            raise GenerationFailedError("A generation is already running.")
        try:
            options = options or GenerationOptions()
            self._cancelled.clear()
            inputs = (
                self._encode_chat(messages)
                if messages is not None
                else self.tokenizer(prompt or "", return_tensors="pt")
            )
            inputs = self._move(dict(inputs))
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            self._assert_context(prompt_tokens, options)
            kwargs = self._generation_kwargs(options)
            text = (
                self._generate_blocking(inputs, kwargs)
                if on_token is None
                else self._generate_streaming(inputs, kwargs, on_token)
            )
            return GenerationResult(text=text, inputTokens=prompt_tokens)
        finally:
            self._generating.release()

    def _trim(self, output, prompt_length: int) -> str:
        generated = (
            output if getattr(self.config, "is_encoder_decoder", False)
            else output[:, prompt_length:]
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _generate_blocking(self, inputs: dict, kwargs: dict) -> str:
        torch = _torch()
        prompt_length = int(inputs["input_ids"].shape[-1])
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    stopping_criteria=_cancel_criteria(self._cancelled),
                    **kwargs,
                )
        except Exception as exc:
            raise GenerationFailedError(str(exc)) from exc
        if self._cancelled.is_set():
            raise GenerationCancelledError("Generation was cancelled.")
        return self._trim(output, prompt_length)

    def _generate_streaming(self, inputs: dict, kwargs: dict, on_token: TokenSink) -> str:
        torch = _torch()
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        failure: list[BaseException] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        streamer=streamer,
                        stopping_criteria=_cancel_criteria(self._cancelled),
                        **kwargs,
                    )
            except BaseException as exc:  # surfaced after the stream drains
                failure.append(exc)
            finally:
                streamer.end()

        thread = threading.Thread(target=run, name="local-llm-generate", daemon=True)
        thread.start()
        collected: list[str] = []
        for chunk in streamer:
            if chunk:
                collected.append(chunk)
                on_token(chunk)
        thread.join()
        if failure:
            raise GenerationFailedError(str(failure[0])) from failure[0]
        if self._cancelled.is_set():
            raise GenerationCancelledError("Generation was cancelled.")
        return "".join(collected)
