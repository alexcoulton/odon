"""Small dependency-free models returned by the control handshake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_REQUESTED_CAPABILITIES = (
    "ui.shell.application_control",
    "ui.shell.chrome",
    "ui.shell.compose",
    "ui.shell.extension_place",
    "ui.shell.persistence",
    "ui.shell.read",
    "ui.shell.recovery",
)


@dataclass(frozen=True)
class Hello:
    protocol_version: int
    app_name: str
    app_version: str
    control_api_version: str
    instance_id: str
    session_id: str
    capabilities: frozenset[str]
    granted_capabilities: frozenset[str]
    max_inline_payload_bytes: int
    permission_policy: str

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "Hello":
        return cls(
            protocol_version=int(value["protocol_version"]),
            app_name=str(value["app_name"]),
            app_version=str(value["app_version"]),
            control_api_version=str(value["control_api_version"]),
            instance_id=str(value["instance_id"]),
            session_id=str(value["session_id"]),
            capabilities=frozenset(str(item) for item in value.get("capabilities", [])),
            granted_capabilities=frozenset(
                str(item) for item in value.get("granted_capabilities", [])
            ),
            max_inline_payload_bytes=int(value["max_inline_payload_bytes"]),
            permission_policy=str(value["permission_policy"]),
        )


@dataclass(frozen=True)
class Event:
    """One ordered notification emitted by an Odon control session."""

    name: str
    sequence: int
    revision: int
    source: str
    data: Any
    initiating_session_id: str | None = None
    initiating_request_id: Any = None

    @classmethod
    def from_params(cls, value: Mapping[str, Any]) -> "Event":
        return cls(
            name=str(value["event"]),
            sequence=int(value["sequence"]),
            revision=int(value["revision"]),
            source=str(value["source"]),
            data=value.get("data"),
            initiating_session_id=(
                str(value["initiating_session_id"])
                if value.get("initiating_session_id") is not None
                else None
            ),
            initiating_request_id=value.get("initiating_request_id"),
        )


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    label: str
    state: str
    progress: float | None
    phase: str
    phase_details: Any
    result: Any
    error: Any
    created_at_unix_ms: int
    completed_at_unix_ms: int | None
    cancellation_supported: bool
    owner_session_id: str

    @property
    def done(self) -> bool:
        return self.state in {"completed", "failed", "cancelled"}

    @classmethod
    def from_result(cls, value: Mapping[str, Any]) -> "TaskSnapshot":
        progress = value.get("progress")
        completed = value.get("completed_at_unix_ms")
        return cls(
            task_id=str(value["task_id"]),
            label=str(value.get("label", "")),
            state=str(value["state"]),
            progress=float(progress) if progress is not None else None,
            phase=str(value.get("phase", "")),
            phase_details=value.get("phase_details"),
            result=value.get("result"),
            error=value.get("error"),
            created_at_unix_ms=int(value.get("created_at_unix_ms", 0)),
            completed_at_unix_ms=int(completed) if completed is not None else None,
            cancellation_supported=bool(value.get("cancellation_supported", False)),
            owner_session_id=str(value.get("owner_session_id", "")),
        )
