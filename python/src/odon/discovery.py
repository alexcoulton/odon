"""Discovery of authenticated Odon instances without third-party dependencies."""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .errors import InstanceNotFoundError, MultipleInstancesError, ProtocolError


@dataclass(frozen=True)
class Instance:
    instance_id: str
    pid: int
    host: str
    port: int
    token: str
    app_version: str
    protocol_versions: tuple[int, ...]
    started_at_unix_ms: int
    manifest_path: Path
    project_path: Path | None = None

    @classmethod
    def from_manifest(cls, path: Path, value: Mapping[str, Any]) -> "Instance":
        endpoint = urlparse(str(value["endpoint"]))
        if endpoint.scheme != "tcp" or endpoint.hostname is None or endpoint.port is None:
            raise ProtocolError(f"unsupported Odon endpoint in {path}")
        project = value.get("project_path")
        return cls(
            instance_id=str(value["instance_id"]),
            pid=int(value["pid"]),
            host=endpoint.hostname,
            port=endpoint.port,
            token=str(value["token"]),
            app_version=str(value["app_version"]),
            protocol_versions=tuple(int(item) for item in value["protocol_versions"]),
            started_at_unix_ms=int(value["started_at_unix_ms"]),
            manifest_path=path,
            project_path=Path(project) if project is not None else None,
        )


def runtime_dir() -> Path:
    override = os.environ.get("ODON_RUNTIME_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "odon" / "runtime"
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return Path(runtime) / "odon"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "odon" / "runtime"
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "odon" / "runtime"


def list_instances(*, clean_stale: bool = True) -> list[Instance]:
    directory = runtime_dir()
    if not directory.exists():
        return []
    instances: list[Instance] = []
    for path in directory.glob("instance-*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ProtocolError(f"manifest is not an object: {path}")
            instance = Instance.from_manifest(path, value)
        except (OSError, ValueError, KeyError, TypeError, ProtocolError):
            if clean_stale:
                _unlink(path)
            continue
        if _endpoint_is_live(instance):
            instances.append(instance)
        elif clean_stale:
            _unlink(path)
    return sorted(
        instances,
        key=lambda instance: (-instance.started_at_unix_ms, instance.instance_id),
    )


def select_instance(instance: Instance | str | None = None) -> Instance:
    if isinstance(instance, Instance):
        return instance
    requested_id = instance or os.environ.get("ODON_INSTANCE_ID")
    instances = list_instances()
    if requested_id is not None:
        for candidate in instances:
            if candidate.instance_id == requested_id:
                return candidate
        raise InstanceNotFoundError(
            f"no running Odon instance has ID {requested_id!r}"
        )
    if not instances:
        raise InstanceNotFoundError("no running Odon instances were discovered")
    if len(instances) > 1:
        choices = ", ".join(instance.instance_id for instance in instances)
        raise MultipleInstancesError(
            "multiple Odon instances are running; pass instance=... or set "
            f"ODON_INSTANCE_ID. Available instances: {choices}"
        )
    return instances[0]


def _endpoint_is_live(instance: Instance) -> bool:
    try:
        with socket.create_connection((instance.host, instance.port), timeout=0.15):
            return True
    except OSError:
        return False


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
