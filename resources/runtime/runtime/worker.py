"""The stdin/stdout worker loop.

The reader stays on the main thread and generation runs on its own thread.
That separation is what makes cancellation possible: a generation that blocked
the reader could never receive the `generation.cancel` that stops it.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from .errors import (
    GenerationBusyError,
    InvalidRequestError,
    RuntimeErrorBase,
    UnknownMethodError,
)
from .inspector import inspect as inspect_checkpoint
from .models import GenerationOptions, RuntimePolicy, WorkerStatus
from .protocol import (
    emit,
    failure,
    log,
    parse_request,
    state_notification,
    success,
    token_notification,
)
from .runtime import LocalLLM, detect_runtime


class Worker:
    """Dispatches protocol requests against a single `LocalLLM`."""

    def __init__(self) -> None:
        self.llm = LocalLLM()
        self.status = WorkerStatus(state="STARTING")
        self._generation: threading.Thread | None = None
        self._active_request: object = None

    # -------------------------------------------------------------- lifecycle

    def set_state(self, state: str, detail: str | None = None) -> None:
        self.status = WorkerStatus(
            state=state,
            modelPath=str(self.llm.path) if self.llm.path else None,
            detail=detail,
        )
        emit(state_notification(state, detail))

    def run(self) -> None:
        self.set_state("READY")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = parse_request(line)
            except RuntimeErrorBase as exc:
                emit(failure(None, exc))
                continue
            self.dispatch(request)
        # stdin closing is the extension going away; stop rather than idle.
        self.shutdown()

    def shutdown(self) -> None:
        self.llm.cancel()
        if self._generation and self._generation.is_alive():
            self._generation.join(timeout=5)
        self.llm.unload()
        self.set_state("STOPPED")

    # --------------------------------------------------------------- dispatch

    def dispatch(self, request: dict) -> None:
        request_id = request.get("id")
        method = request["method"]
        params = request.get("params") or {}
        try:
            if method == "generate.chat" or method == "generate.complete":
                # `generate.complete` is a REQUEST (spec §6.2) and is one letter
                # from §18's `generation.complete` NOTIFICATION, which this
                # worker deliberately does not emit: the ordinary {id, ok,
                # result} response already signals completion, and two signals
                # for one request invite the client and worker to disagree about
                # which is authoritative.
                #
                # Generation is the one method that does not answer inline; it
                # replies from its own thread so the reader stays responsive.
                self.start_generation(request_id, method, params)
                return
            emit(success(request_id, self.handle(method, params)))
        except RuntimeErrorBase as exc:
            emit(failure(request_id, exc))
        except Exception as exc:  # upstream failure, normalised at the boundary
            log("error", f"{method} failed: {exc!r}")
            emit(failure(request_id, exc))

    def handle(self, method: str, params: dict):
        if method == "runtime.info":
            return detect_runtime()
        if method == "worker.status":
            return self.status
        if method == "model.inspect":
            return inspect_checkpoint(Path(self._require(params, "path")))
        if method == "model.load":
            self.set_state("LOADING")
            try:
                info = self.llm.load(
                    self._require(params, "path"),
                    RuntimePolicy.from_params(params.get("policy")),
                )
            except Exception:
                self.set_state("FAILED", "Model load failed.")
                raise
            self.set_state("LOADED")
            return info
        if method == "model.info":
            return self.llm.info()
        if method == "model.unload":
            self.set_state("UNLOADING")
            self.llm.unload()
            self.set_state("READY")
            return None
        if method == "model.tokenize":
            return {
                "tokens": self.llm.count_tokens(
                    params.get("text"), params.get("messages"),
                ),
            }
        if method == "generation.cancel":
            # §18 addresses a cancel to a specific request. One worker runs one
            # generation, so a cancel naming a stale request must be ignored
            # rather than stopping whatever started after it.
            target = params.get("requestId")
            if target is not None and target != self._active_request:
                raise InvalidRequestError(
                    f"Request {target} is not the generation in flight."
                )
            self.llm.cancel()
            return None
        if method == "worker.shutdown":
            self.shutdown()
            return None
        raise UnknownMethodError(f"Unknown method: {method}")

    @staticmethod
    def _require(params: dict, key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value:
            raise InvalidRequestError(f"Parameter '{key}' is required.")
        return value

    # ------------------------------------------------------------- generation

    def start_generation(self, request_id, method: str, params: dict) -> None:
        if self._generation and self._generation.is_alive():
            emit(failure(request_id, GenerationBusyError(
                "A generation is already running on this worker."
            )))
            return

        options = GenerationOptions.from_params(params.get("options"))
        stream = bool(params.get("stream", True))

        def sink(text: str) -> None:
            emit(token_notification(request_id, text))

        def run() -> None:
            self.set_state("GENERATING")
            try:
                if method == "generate.chat":
                    messages = params.get("messages")
                    if not isinstance(messages, list):
                        raise InvalidRequestError(
                            "Parameter 'messages' must be an array."
                        )
                    result = self.llm.generate(
                        messages=messages,
                        options=options,
                        on_token=sink if stream else None,
                    )
                else:
                    result = self.llm.generate(
                        prompt=self._require(params, "prompt"),
                        options=options,
                        on_token=sink if stream else None,
                    )
                emit(success(request_id, result))
            except RuntimeErrorBase as exc:
                emit(failure(request_id, exc))
            except Exception as exc:
                log("error", f"{method} failed: {exc!r}")
                emit(failure(request_id, exc))
            finally:
                self._active_request = None
                self.set_state("LOADED" if self.llm.model is not None else "READY")

        self._active_request = request_id
        self._generation = threading.Thread(
            target=run, name="local-llm-request", daemon=True,
        )
        self._generation.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Safetensors LLM worker")
    parser.add_argument("--worker", action="store_true",
                        help="Serve the JSON Lines protocol on stdin/stdout.")
    parser.add_argument("--model", help="Model directory, for one-shot use.")
    parser.add_argument("--prompt", help="Prompt, for one-shot use.")
    parser.add_argument("--inspect", help="Inspect a model directory and exit.")
    parser.add_argument("--probe", action="store_true",
                        help="Report runtime availability and exit.")
    args = parser.parse_args()

    if args.probe:
        # Run as its own process by the extension before the worker is trusted.
        # Importing torch can abort rather than raise, so the only safe place to
        # find that out is a process whose death costs nothing.
        import json as _json
        print(_json.dumps(asdict(detect_runtime()), default=str))
        return
    if args.worker:
        Worker().run()
        return
    if args.inspect:
        import json as _json
        print(_json.dumps(asdict(inspect_checkpoint(Path(args.inspect))), indent=2))
        return
    if args.model and args.prompt:
        llm = LocalLLM()
        llm.load(args.model)
        print(llm.generate(prompt=args.prompt).text)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
