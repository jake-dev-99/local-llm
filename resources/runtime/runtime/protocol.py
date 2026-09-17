"""Request and response framing for the JSON Lines worker protocol.

One JSON object per line. stdout carries protocol traffic and nothing else;
every log and traceback goes to stderr, because a stray print on stdout
corrupts the stream and desynchronises the client.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
from typing import Any

from .errors import InvalidRequestError, error_code

# Serialising to stdout from both the reader thread and a generation thread
# would interleave partial lines.
_write_lock = threading.Lock()


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def emit(message: dict) -> None:
    """Write one protocol message, atomically against other writers."""

    line = json.dumps(message, default=str)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(category: str, message: str) -> None:
    """Diagnostics go to stderr so they can never corrupt the protocol stream."""

    sys.stderr.write(f"[{category}] {message}\n")
    sys.stderr.flush()


def success(request_id: Any, result: Any = None) -> dict:
    return {"id": request_id, "ok": True, "result": _encode(result)}


def failure(request_id: Any, exc: BaseException) -> dict:
    return {
        "id": request_id,
        "ok": False,
        "error": {"code": error_code(exc), "message": str(exc)},
    }


def notify(method: str, params: dict) -> dict:
    """A server-initiated message, which carries no `id` and expects no reply."""

    return {"method": method, "params": params}


def token_notification(request_id: Any, text: str) -> dict:
    return notify("generation.token", {"requestId": request_id, "text": text})


def state_notification(state: str, detail: str | None = None) -> dict:
    return notify("worker.state", {"state": state, "detail": detail})


def parse_request(line: str) -> dict:
    """Decode one request line, rejecting anything that is not a usable object."""

    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"Request was not valid JSON: {exc}") from exc
    if not isinstance(request, dict):
        raise InvalidRequestError("Request must be a JSON object.")
    if not isinstance(request.get("method"), str):
        raise InvalidRequestError("Request must carry a string 'method'.")
    return request
