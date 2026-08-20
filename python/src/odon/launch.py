"""Launch an installed Odon executable and connect through normal discovery."""

from __future__ import annotations

import os
import asyncio
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from .client import Client
from .async_client import AsyncClient
from .discovery import list_instances
from .errors import ConnectionClosedError, RequestTimeoutError


def launch(
    executable: str | Path,
    args: Sequence[str] = (),
    *,
    timeout: float = 20.0,
    env: Mapping[str, str] | None = None,
    terminate_on_failure: bool = True,
) -> Client:
    """Start an installed Odon process and return its authenticated client.

    The SDK does not distribute Odon or Python with the application. It watches
    the ordinary private runtime manifests and preferentially selects the
    manifest whose PID matches the child process.
    """

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    before = {instance.instance_id for instance in list_instances()}
    process_env = os.environ.copy()
    if env is not None:
        process_env.update({str(key): str(value) for key, value in env.items()})
    process = subprocess.Popen(
        [str(executable), *(str(argument) for argument in args)],
        env=process_env,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise ConnectionClosedError(
                    f"Odon exited with status {return_code} before publishing a control endpoint"
                )
            instances = list_instances()
            selected = next(
                (instance for instance in instances if instance.pid == process.pid),
                None,
            ) or next(
                (instance for instance in instances if instance.instance_id not in before),
                None,
            )
            if selected is not None:
                client = Client(instance=selected, timeout=timeout)
                client._launched_process = process
                return client
            time.sleep(0.05)
        raise RequestTimeoutError(
            f"timed out after {timeout:g}s waiting for Odon to publish a control endpoint"
        )
    except BaseException:
        if terminate_on_failure and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


async def launch_async(
    executable: str | Path,
    args: Sequence[str] = (),
    *,
    timeout: float = 20.0,
    env: Mapping[str, str] | None = None,
    terminate_on_failure: bool = True,
) -> AsyncClient:
    """Asynchronously launch an installed Odon process and connect to it."""

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    before = {
        instance.instance_id for instance in await asyncio.to_thread(list_instances)
    }
    process_env = os.environ.copy()
    if env is not None:
        process_env.update({str(key): str(value) for key, value in env.items()})
    process = await asyncio.create_subprocess_exec(
        str(executable), *(str(argument) for argument in args), env=process_env
    )
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise ConnectionClosedError(
                    f"Odon exited with status {process.returncode} before publishing a control endpoint"
                )
            instances = await asyncio.to_thread(list_instances)
            selected = next(
                (instance for instance in instances if instance.pid == process.pid), None
            ) or next(
                (instance for instance in instances if instance.instance_id not in before), None
            )
            if selected is not None:
                client = await AsyncClient.connect(instance=selected, timeout=timeout)
                client._launched_process = process
                return client
            await asyncio.sleep(0.05)
        raise RequestTimeoutError(
            f"timed out after {timeout:g}s waiting for Odon to publish a control endpoint"
        )
    except BaseException:
        if terminate_on_failure and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                process.kill()
                await process.wait()
        raise
