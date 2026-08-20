"""Stable Python exception hierarchy for Odon control errors."""

from __future__ import annotations

from typing import Any, Mapping


class OdonError(Exception):
    """Base class for all SDK errors."""


class ConnectionClosedError(OdonError):
    """The control connection closed before an operation completed."""


class ProtocolError(OdonError):
    """Odon or the client sent an invalid control-protocol message."""


class RequestTimeoutError(OdonError):
    """Python stopped waiting for a response; Odon may still be working."""


class TaskCancelledError(OdonError):
    """A long-running Odon task was cancelled."""


class TaskFailedError(OdonError):
    """A long-running Odon task completed with a structured failure."""

    def __init__(self, task_id: str, error: Any) -> None:
        super().__init__(f"task {task_id!r} failed: {error}")
        self.task_id = task_id
        self.error = error


class InstanceNotFoundError(OdonError):
    """No live Odon instance matched the requested selection."""


class MultipleInstancesError(OdonError):
    """More than one Odon instance is live and no selection was supplied."""


class RemoteError(OdonError):
    """A structured error returned by Odon."""

    def __init__(
        self,
        message: str,
        *,
        code: int,
        kind: str = "CONTROL_ERROR",
        data: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{kind}: {message}")
        self.message = message
        self.code = code
        self.kind = kind
        self.data = dict(data or {})


class AuthenticationError(RemoteError):
    pass


class ProtocolVersionError(RemoteError):
    pass


class InvalidParametersError(RemoteError):
    pass


class ResourceNotFoundError(RemoteError):
    pass


class NotReadyError(RemoteError):
    pass


class UnsupportedCapabilityError(RemoteError):
    pass


class ConflictError(RemoteError):
    pass


class PermissionDeniedError(RemoteError):
    pass


class WrongModeError(RemoteError):
    pass


class ResourceLimitError(RemoteError):
    pass


def remote_error_from_message(
    message: Mapping[str, Any],
    *,
    method: str | None = None,
    request_id: int | None = None,
) -> RemoteError:
    error = message.get("error")
    if not isinstance(error, Mapping):
        raise ProtocolError("JSON-RPC error response has no error object")
    data = error.get("data")
    if not isinstance(data, Mapping):
        data = {}
    kind = str(data.get("kind", "CONTROL_ERROR"))
    error_type = {
        "AUTHENTICATION_FAILED": AuthenticationError,
        "INCOMPATIBLE_PROTOCOL": ProtocolVersionError,
        "INVALID_PARAMS": InvalidParametersError,
        "RESOURCE_NOT_FOUND": ResourceNotFoundError,
        "NOT_READY": NotReadyError,
        "UNSUPPORTED": UnsupportedCapabilityError,
        "CONFLICT": ConflictError,
        "PERMISSION_DENIED": PermissionDeniedError,
        "WRONG_MODE": WrongModeError,
        "RESOURCE_LIMIT": ResourceLimitError,
    }.get(kind, RemoteError)
    exception = error_type(
        str(error.get("message", "unknown Odon error")),
        code=int(error.get("code", -32603)),
        kind=kind,
        data=data,
    )
    exception.method = method
    exception.request_id = request_id
    return exception
