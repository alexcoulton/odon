"""Typed interaction subscriptions for Python-authored native Odon UI."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import inspect
import logging
import threading
import time
from typing import Any, Literal

from .errors import (
    ActionCancelledError,
    ActionExecutionError,
    ActionQueueFullError,
    ActionRejectedError,
    StaleActionError,
)
from .models import Event

logger = logging.getLogger("odon.ui.actions")

ExecutionPolicy = Literal["callback", "worker", "serial-worker"]
CoalescePolicy = Literal["all", "latest", "accumulate", "reject-while-busy"]


class UiInteractionDecodeError(ValueError):
    """An extension event was not a valid native UI interaction envelope."""


@dataclass(frozen=True)
class UiInteraction:
    """One normalized interaction emitted by an extension-owned component."""

    extension_id: str
    component_id: str
    action: str | None
    value: Any
    kind: Literal["action", "input"]
    event: Event

    @classmethod
    def from_event(cls, event: Event, *, extension_id: str) -> "UiInteraction":
        prefix = f"ui.extension:{extension_id}."
        if not event.name.startswith(prefix):
            raise UiInteractionDecodeError(
                f"event {event.name!r} does not belong to extension {extension_id!r}"
            )
        kind = event.name.removeprefix(prefix)
        if kind not in {"action", "input"}:
            raise UiInteractionDecodeError(
                f"event {event.name!r} is not an action or input interaction"
            )
        if not isinstance(event.data, dict):
            raise UiInteractionDecodeError("UI interaction data must be an object")
        component_id = event.data.get("component_id")
        if not isinstance(component_id, str) or not component_id.strip():
            raise UiInteractionDecodeError(
                "UI interaction requires a non-empty component_id"
            )
        raw_action = event.data.get("action")
        semantic: str | None = None
        if raw_action is not None:
            if not isinstance(raw_action, dict):
                raise UiInteractionDecodeError("UI interaction action must be an object")
            raw_semantic = raw_action.get("event")
            if raw_semantic is not None:
                if not isinstance(raw_semantic, str) or not raw_semantic.strip():
                    raise UiInteractionDecodeError(
                        "UI interaction action.event must be a non-empty string"
                    )
                semantic = raw_semantic
        return cls(
            extension_id=extension_id,
            component_id=component_id,
            action=semantic,
            value=event.data.get("value"),
            kind=kind,
            event=event,
        )


class InteractionSubscription:
    """A removable synchronous extension interaction subscription."""

    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove
        self._removed = False

    @property
    def removed(self) -> bool:
        return self._removed

    def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        self._remove()

    def _close_local(self) -> None:
        """Mark a subscription removed without issuing a disconnect-time RPC."""

        self._removed = True

    def __enter__(self) -> "InteractionSubscription":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.remove()


class AsyncInteractionSubscription:
    """A removable asynchronous extension interaction subscription."""

    def __init__(self, remove: Callable[[], Any]) -> None:
        self._remove = remove
        self._removed = False

    @property
    def removed(self) -> bool:
        return self._removed

    async def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        await self._remove()

    def _close_local(self) -> None:
        """Mark a subscription removed without issuing a disconnect-time RPC."""

        self._removed = True

    async def __aenter__(self) -> "AsyncInteractionSubscription":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.remove()


@dataclass(frozen=True)
class ActionWorkerSnapshot:
    """Detached diagnostics for one extension-owned action runner."""

    submitted: int
    executed: int
    completed: int
    failed: int
    cancelled: int
    rejected: int
    coalesced: int
    queue_depth: int
    running_actions: tuple[str, ...]
    closed: bool

    @property
    def running_action(self) -> str | None:
        return self.running_actions[0] if self.running_actions else None


def _validate_execution(value: str) -> ExecutionPolicy:
    if value not in {"callback", "worker", "serial-worker"}:
        raise ValueError(
            "action execution must be callback, worker, or serial-worker"
        )
    return value  # type: ignore[return-value]


def _validate_coalesce(value: str) -> CoalescePolicy:
    if value not in {"all", "latest", "accumulate", "reject-while-busy"}:
        raise ValueError(
            "action coalesce must be all, latest, accumulate, or reject-while-busy"
        )
    return value  # type: ignore[return-value]


class ActionContext:
    """Lifecycle and UI helpers for one synchronous extension action."""

    def __init__(
        self,
        runner: "ActionRunner",
        registration: "ActionRegistration",
        interaction: UiInteraction,
        *,
        generation: int,
        delta: float,
    ) -> None:
        self._runner = runner
        self.registration = registration
        self.interaction = interaction
        self.generation = generation
        self.delta = delta
        self._cancelled = threading.Event()
        self._task: Any = None

    @property
    def action(self) -> str:
        return self.registration.action

    @property
    def queue_key(self) -> str:
        return self.registration.queue_key

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def is_current(self) -> bool:
        return self._runner.is_current(
            self.queue_key, self.generation, self.registration.coalesce
        )

    @property
    def task(self) -> Any:
        return self._task

    def attach(self, task: Any) -> Any:
        """Attach the currently retained Odon task for progress and cancellation."""

        self._task = task
        if self.cancelled:
            self._cancel_task()
        return task

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise ActionCancelledError(
                f"action {self.action!r} generation {self.generation} was cancelled",
                action=self.action,
                queue_key=self.queue_key,
                generation=self.generation,
            )

    def ensure_current(self) -> None:
        self.check_cancelled()
        if not self.is_current:
            raise StaleActionError(
                f"action {self.action!r} generation {self.generation} was superseded",
                action=self.action,
                queue_key=self.queue_key,
                generation=self.generation,
            )

    def cancel(self) -> None:
        self._cancelled.set()
        self._cancel_task()

    def _cancel_task(self) -> None:
        task = self._task
        snapshot = getattr(task, "snapshot", None)
        if task is None or not bool(getattr(snapshot, "cancellation_supported", False)):
            return
        try:
            task.cancel()
        except Exception:
            logger.exception("Failed to cancel retained task for action %s", self.action)

    def _patch(self, values: Mapping[str, Any]) -> bool:
        contribution = self.registration.contribution
        if contribution is None:
            return False
        return self._runner.commit_if_current(
            self.queue_key,
            self.generation,
            self.registration.coalesce,
            lambda: contribution.patch_values(dict(values)),
        )

    def patch(self, values: Mapping[str, Any]) -> bool:
        """Patch arbitrary contribution values only if this generation still wins."""

        return self._patch(values)

    def commit(self, callback: Callable[[], Any]) -> bool:
        """Run a short local/SDK commit while submission generation is stable."""

        return self._runner.commit_if_current(
            self.queue_key,
            self.generation,
            self.registration.coalesce,
            callback,
        )

    def status(self, message: str) -> bool:
        component_id = self.registration.status_component_id
        if component_id is None:
            return False
        return self._patch({component_id: str(message)})

    def progress(self, value: float | None, message: str | None = None) -> bool:
        values: dict[str, Any] = {}
        component_id = self.registration.progress_component_id
        if component_id is not None and value is not None:
            values[component_id] = min(1.0, max(0.0, float(value)))
        if message is not None and self.registration.status_component_id is not None:
            values[self.registration.status_component_id] = str(message)
        return bool(values) and self._patch(values)

    def report_task(self, snapshot: Any) -> None:
        message = str(getattr(snapshot, "phase", "") or getattr(snapshot, "state", ""))
        self.progress(getattr(snapshot, "progress", None), message or None)

    def result(self, message: str = "Ready") -> bool:
        self.ensure_current()
        return self.status(message)

    @contextmanager
    def busy(self, message: str, *, ready: str = "Ready"):
        self.status(message)
        try:
            yield self
            self.ensure_current()
        except BaseException:
            raise
        else:
            self.status(ready)


class ActionRegistration:
    """A callable, removable synchronous extension action registration."""

    def __init__(
        self,
        runner: "ActionRunner",
        action: str,
        callback: Callable[[ActionContext, UiInteraction], Any],
        *,
        execution: ExecutionPolicy,
        coalesce: CoalescePolicy,
        queue_key: str,
        delta: float,
        contribution: Any = None,
        status_component_id: str | None = None,
        progress_component_id: str | None = None,
        on_error: Callable[[BaseException, ActionContext | None], Any] | None = None,
        on_remove: Callable[["ActionRegistration"], None] | None = None,
    ) -> None:
        self._runner = runner
        self.action = action
        self.callback = callback
        self.execution = execution
        self.coalesce = coalesce
        self.queue_key = queue_key
        self.delta = delta
        self.contribution = contribution
        self.status_component_id = status_component_id
        self.progress_component_id = progress_component_id
        self.on_error = on_error
        self._on_remove = on_remove
        self._subscription: InteractionSubscription | None = None
        self._removed = False

    @property
    def removed(self) -> bool:
        return self._removed

    def __call__(self, context: ActionContext, interaction: UiInteraction) -> Any:
        return self.callback(context, interaction)

    def submit(self, interaction: UiInteraction) -> bool:
        if self._removed:
            return False
        return self._runner.submit(self, interaction)

    def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        if self._subscription is not None:
            self._subscription.remove()
        self._runner.cancel_registration(self)
        if self._on_remove is not None:
            self._on_remove(self)

    def _close_local(self) -> None:
        """Make the handle inert during client-local cleanup."""

        self._removed = True
        if self._subscription is not None:
            self._subscription._close_local()


@dataclass
class _Invocation:
    registration: ActionRegistration
    interaction: UiInteraction
    generation: int
    delta: float


class ActionRunner:
    """Bounded extension-owned executor for callback, worker, and serial actions."""

    def __init__(self, *, max_queue: int = 128, concurrent_workers: int = 4) -> None:
        if max_queue < 1:
            raise ValueError("action max_queue must be at least 1")
        if concurrent_workers < 1:
            raise ValueError("action concurrent_workers must be at least 1")
        self.max_queue = max_queue
        self._condition = threading.Condition(threading.RLock())
        self._serial: deque[_Invocation] = deque()
        self._running: dict[int, ActionContext] = {}
        self._concurrent: set[Future[Any]] = set()
        self._concurrent_invocations: dict[Future[Any], _Invocation] = {}
        self._latest_generation: dict[str, int] = {}
        self._next_generation = 0
        self._closed = False
        self._submitted = 0
        self._executed = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._coalesced = 0
        self._concurrent_workers = concurrent_workers
        self._executor: ThreadPoolExecutor | None = None
        self._serial_worker: threading.Thread | None = None

    def snapshot(self) -> ActionWorkerSnapshot:
        with self._condition:
            pending_concurrent = sum(
                invocation.generation not in self._running
                for invocation in self._concurrent_invocations.values()
            )
            return ActionWorkerSnapshot(
                submitted=self._submitted,
                executed=self._executed,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled,
                rejected=self._rejected,
                coalesced=self._coalesced,
                queue_depth=len(self._serial) + pending_concurrent,
                running_actions=tuple(
                    context.action for context in self._running.values()
                ),
                closed=self._closed,
            )

    def is_current(
        self,
        queue_key: str,
        generation: int,
        coalesce: CoalescePolicy,
    ) -> bool:
        if coalesce in {"all", "reject-while-busy"}:
            return True
        with self._condition:
            return self._latest_generation.get(queue_key) == generation

    def commit_if_current(
        self,
        queue_key: str,
        generation: int,
        coalesce: CoalescePolicy,
        callback: Callable[[], Any],
    ) -> bool:
        with self._condition:
            if (
                coalesce not in {"all", "reject-while-busy"}
                and self._latest_generation.get(queue_key) != generation
            ):
                return False
            callback()
            return True

    def _ensure_serial_worker(self) -> None:
        if self._serial_worker is not None:
            return
        self._serial_worker = threading.Thread(
            target=self._run_serial,
            name="odon-extension-serial-actions",
            daemon=True,
        )
        self._serial_worker.start()

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._concurrent_workers,
                thread_name_prefix="odon-extension-action",
            )
        return self._executor

    def _new_generation(self) -> int:
        self._next_generation += 1
        return self._next_generation

    def _key_busy(self, queue_key: str) -> bool:
        return any(
            item.registration.queue_key == queue_key for item in self._serial
        ) or any(context.queue_key == queue_key for context in self._running.values())

    def submit(
        self, registration: ActionRegistration, interaction: UiInteraction
    ) -> bool:
        error: ActionExecutionError | None = None
        invocation: _Invocation | None = None
        execute_callback = False
        with self._condition:
            if self._closed:
                error = ActionRejectedError(
                    "extension action runner is closed",
                    action=registration.action,
                    queue_key=registration.queue_key,
                )
            elif (
                registration.coalesce == "reject-while-busy"
                and self._key_busy(registration.queue_key)
            ):
                error = ActionRejectedError(
                    f"action {registration.action!r} was rejected while its queue key is busy",
                    action=registration.action,
                    queue_key=registration.queue_key,
                )
            elif registration.execution == "worker" and registration.coalesce == "accumulate":
                error = ActionRejectedError(
                    "accumulate coalescing requires serial-worker execution",
                    action=registration.action,
                    queue_key=registration.queue_key,
                )
            else:
                generation = self._new_generation()
                invocation = _Invocation(
                    registration, interaction, generation, registration.delta
                )
                if registration.execution == "serial-worker":
                    invocation = self._coalesce_serial(invocation)
                    if invocation is not None and len(self._serial) >= self.max_queue:
                        error = ActionQueueFullError(
                            f"extension action queue is full at {self.max_queue} items",
                            action=registration.action,
                            queue_key=registration.queue_key,
                            generation=generation,
                        )
                        invocation = None
                    elif invocation is not None:
                        self._ensure_serial_worker()
                        self._serial.append(invocation)
                        self._condition.notify()
                elif registration.execution == "worker":
                    if registration.coalesce == "latest":
                        self._coalesce_concurrent_latest(registration.queue_key)
                    if len(self._concurrent) >= self.max_queue:
                        error = ActionQueueFullError(
                            f"extension worker action queue is full at {self.max_queue} items",
                            action=registration.action,
                            queue_key=registration.queue_key,
                            generation=generation,
                        )
                        invocation = None
                else:
                    execute_callback = True
                if error is None:
                    self._submitted += 1
                    if registration.coalesce != "accumulate":
                        self._latest_generation[registration.queue_key] = generation
            if error is not None:
                self._rejected += 1
        if error is not None:
            self._report_error(registration, error, None)
            return False
        if invocation is None:
            return True
        if execute_callback:
            self._execute(invocation)
        elif registration.execution == "worker":
            with self._condition:
                future = self._ensure_executor().submit(self._execute, invocation)
                self._concurrent.add(future)
                self._concurrent_invocations[future] = invocation
            future.add_done_callback(self._concurrent_done)
        return True

    def _coalesce_concurrent_latest(self, queue_key: str) -> None:
        for future, pending in tuple(self._concurrent_invocations.items()):
            if (
                pending.registration.queue_key == queue_key
                and future.cancel()
            ):
                self._concurrent.discard(future)
                self._concurrent_invocations.pop(future, None)
                self._coalesced += 1

    def _coalesce_serial(self, invocation: _Invocation) -> _Invocation | None:
        registration = invocation.registration
        if registration.coalesce == "latest":
            retained: deque[_Invocation] = deque()
            for pending in self._serial:
                if pending.registration.queue_key == registration.queue_key:
                    self._coalesced += 1
                else:
                    retained.append(pending)
            self._serial = retained
        elif registration.coalesce == "accumulate":
            for index in range(len(self._serial) - 1, -1, -1):
                pending = self._serial[index]
                if (
                    pending.registration.queue_key == registration.queue_key
                    and pending.registration.callback == registration.callback
                ):
                    combined = pending.delta + invocation.delta
                    self._coalesced += 1
                    del self._serial[index]
                    if combined == 0:
                        return None
                    invocation.delta = combined
                    return invocation
        return invocation

    def _run_serial(self) -> None:
        while True:
            with self._condition:
                while not self._serial and not self._closed:
                    self._condition.wait()
                if self._closed and not self._serial:
                    return
                invocation = self._serial.popleft()
            self._execute(invocation)

    def _execute(self, invocation: _Invocation) -> None:
        registration = invocation.registration
        context = ActionContext(
            self,
            registration,
            invocation.interaction,
            generation=invocation.generation,
            delta=invocation.delta,
        )
        with self._condition:
            if self._closed or registration.removed:
                self._cancelled += 1
                return
            if registration.coalesce == "accumulate":
                # Pending relative deltas are based on the state committed by the
                # active action. New clicks therefore do not stale that active step.
                self._latest_generation[registration.queue_key] = invocation.generation
            self._running[context.generation] = context
            self._executed += 1
        try:
            result = registration.callback(context, invocation.interaction)
            if inspect.isawaitable(result):
                raise TypeError(
                    "synchronous extension action callbacks must not return an awaitable"
                )
            context.check_cancelled()
        except (ActionCancelledError, StaleActionError):
            with self._condition:
                self._cancelled += 1
        except BaseException as error:
            with self._condition:
                self._failed += 1
            self._report_error(registration, error, context)
        else:
            with self._condition:
                self._completed += 1
        finally:
            with self._condition:
                self._running.pop(context.generation, None)

    def _report_error(
        self,
        registration: ActionRegistration,
        error: BaseException,
        context: ActionContext | None,
    ) -> None:
        if registration.on_error is not None:
            try:
                registration.on_error(error, context)
                return
            except Exception:
                logger.exception(
                    "Odon extension action error callback failed for %s",
                    registration.action,
                )
        if context is not None and registration.status_component_id is not None:
            try:
                context.status(f"Failed: {error}")
            except Exception:
                logger.exception(
                    "Failed to patch error status for Odon extension action %s",
                    registration.action,
                )
        logger.error(
            "Odon extension action %s failed: %s",
            registration.action,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

    def _concurrent_done(self, future: Future[Any]) -> None:
        with self._condition:
            self._concurrent.discard(future)
            self._concurrent_invocations.pop(future, None)

    def cancel_registration(self, registration: ActionRegistration) -> None:
        contexts: list[ActionContext] = []
        with self._condition:
            retained = deque(
                item for item in self._serial if item.registration is not registration
            )
            self._cancelled += len(self._serial) - len(retained)
            self._serial = retained
            contexts = [
                context
                for context in self._running.values()
                if context.registration is registration
            ]
            for future, invocation in tuple(self._concurrent_invocations.items()):
                if invocation.registration is registration and future.cancel():
                    self._concurrent.discard(future)
                    self._concurrent_invocations.pop(future, None)
                    self._cancelled += 1
        for context in contexts:
            context.cancel()

    def close(
        self, *, wait: bool = True, cancel: bool = True, timeout: float = 1.0
    ) -> None:
        """Stop accepting work and return within ``timeout`` seconds.

        Python cannot forcibly stop an uncooperative thread. Attached cancellable Odon
        tasks are cancelled, pending calls are discarded, and any remaining callback is
        detached after the bounded wait.
        """

        if timeout < 0:
            raise ValueError("action shutdown timeout must be non-negative")
        contexts: list[ActionContext]
        with self._condition:
            if self._closed:
                return
            self._closed = True
            if cancel:
                self._cancelled += len(self._serial)
                self._serial.clear()
            contexts = list(self._running.values()) if cancel else []
            self._condition.notify_all()
        for context in contexts:
            context.cancel()
        for future in tuple(self._concurrent):
            if cancel and future.cancel():
                with self._condition:
                    self._cancelled += 1
        deadline = time.monotonic() + timeout
        serial_worker = self._serial_worker
        if (
            wait
            and serial_worker is not None
            and threading.current_thread() is not serial_worker
        ):
            serial_worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if wait:
            for future in tuple(self._concurrent):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    future.result(timeout=remaining)
                except BaseException:
                    pass
        all_finished = all(future.done() for future in tuple(self._concurrent))
        if self._executor is not None:
            self._executor.shutdown(wait=wait and all_finished, cancel_futures=cancel)


class AsyncActionContext:
    """Lifecycle and UI helpers for one asynchronous extension action."""

    def __init__(
        self,
        runner: "AsyncActionRunner",
        registration: "AsyncActionRegistration",
        interaction: UiInteraction,
        *,
        generation: int,
        delta: float,
    ) -> None:
        self._runner = runner
        self.registration = registration
        self.interaction = interaction
        self.generation = generation
        self.delta = delta
        self._cancelled = False
        self._task: Any = None

    @property
    def action(self) -> str:
        return self.registration.action

    @property
    def queue_key(self) -> str:
        return self.registration.queue_key

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_current(self) -> bool:
        return self._runner.is_current(
            self.queue_key, self.generation, self.registration.coalesce
        )

    @property
    def task(self) -> Any:
        return self._task

    def attach(self, task: Any) -> Any:
        self._task = task
        if self.cancelled:
            asyncio.create_task(self._cancel_task())
        return task

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise ActionCancelledError(
                f"action {self.action!r} generation {self.generation} was cancelled",
                action=self.action,
                queue_key=self.queue_key,
                generation=self.generation,
            )

    def ensure_current(self) -> None:
        self.check_cancelled()
        if not self.is_current:
            raise StaleActionError(
                f"action {self.action!r} generation {self.generation} was superseded",
                action=self.action,
                queue_key=self.queue_key,
                generation=self.generation,
            )

    async def cancel(self) -> None:
        self._cancelled = True
        await self._cancel_task()

    async def _cancel_task(self) -> None:
        snapshot = getattr(self._task, "snapshot", None)
        if self._task is not None and bool(
            getattr(snapshot, "cancellation_supported", False)
        ):
            try:
                await self._task.cancel()
            except Exception:
                logger.exception("Failed to cancel retained task for action %s", self.action)

    async def _patch(self, values: Mapping[str, Any]) -> bool:
        contribution = self.registration.contribution
        if contribution is None or not self.is_current:
            return False
        await contribution.patch_values(dict(values))
        return self.is_current

    async def patch(self, values: Mapping[str, Any]) -> bool:
        """Patch contribution values when this async generation still wins."""

        return await self._patch(values)

    async def commit(self, callback: Callable[[], Any | Awaitable[Any]]) -> bool:
        """Run a short async commit after a generation check.

        Actor calls remain separately revisioned; callers should still perform a
        final ``ensure_current()`` after an awaited commit.
        """

        if not self.is_current:
            return False
        result = callback()
        if inspect.isawaitable(result):
            await result
        return self.is_current

    async def status(self, message: str) -> bool:
        component_id = self.registration.status_component_id
        if component_id is None:
            return False
        return await self._patch({component_id: str(message)})

    async def progress(
        self, value: float | None, message: str | None = None
    ) -> bool:
        values: dict[str, Any] = {}
        if self.registration.progress_component_id is not None and value is not None:
            values[self.registration.progress_component_id] = min(
                1.0, max(0.0, float(value))
            )
        if message is not None and self.registration.status_component_id is not None:
            values[self.registration.status_component_id] = str(message)
        return bool(values) and await self._patch(values)

    async def report_task(self, snapshot: Any) -> None:
        message = str(getattr(snapshot, "phase", "") or getattr(snapshot, "state", ""))
        await self.progress(getattr(snapshot, "progress", None), message or None)

    async def result(self, message: str = "Ready") -> bool:
        self.ensure_current()
        return await self.status(message)

    @asynccontextmanager
    async def busy(self, message: str, *, ready: str = "Ready"):
        await self.status(message)
        try:
            yield self
            self.ensure_current()
        except BaseException:
            raise
        else:
            await self.status(ready)


class AsyncActionRegistration:
    """A removable asynchronous extension action registration."""

    def __init__(
        self,
        runner: "AsyncActionRunner",
        action: str,
        callback: Callable[[AsyncActionContext, UiInteraction], Any | Awaitable[Any]],
        *,
        execution: ExecutionPolicy,
        coalesce: CoalescePolicy,
        queue_key: str,
        delta: float,
        contribution: Any = None,
        status_component_id: str | None = None,
        progress_component_id: str | None = None,
        on_error: Callable[
            [BaseException, AsyncActionContext | None], Any | Awaitable[Any]
        ]
        | None = None,
        on_remove: Callable[["AsyncActionRegistration"], Any] | None = None,
    ) -> None:
        self._runner = runner
        self.action = action
        self.callback = callback
        self.execution = execution
        self.coalesce = coalesce
        self.queue_key = queue_key
        self.delta = delta
        self.contribution = contribution
        self.status_component_id = status_component_id
        self.progress_component_id = progress_component_id
        self.on_error = on_error
        self._on_remove = on_remove
        self._subscription: AsyncInteractionSubscription | None = None
        self._removed = False

    @property
    def removed(self) -> bool:
        return self._removed

    def submit(self, interaction: UiInteraction) -> bool:
        return not self._removed and self._runner.submit(self, interaction)

    async def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        if self._subscription is not None:
            await self._subscription.remove()
        await self._runner.cancel_registration(self)
        if self._on_remove is not None:
            result = self._on_remove(self)
            if inspect.isawaitable(result):
                await result

    def _close_local(self) -> None:
        """Make the handle inert during async client-local cleanup."""

        self._removed = True
        if self._subscription is not None:
            self._subscription._close_local()


@dataclass
class _AsyncInvocation:
    registration: AsyncActionRegistration
    interaction: UiInteraction
    generation: int
    delta: float


class AsyncActionRunner:
    """Asyncio executor for callback, concurrent, and serialized extension actions."""

    def __init__(self, *, max_queue: int = 128) -> None:
        if max_queue < 1:
            raise ValueError("action max_queue must be at least 1")
        self.max_queue = max_queue
        self._serial: deque[_AsyncInvocation] = deque()
        self._serial_ready = asyncio.Event()
        self._running: dict[int, AsyncActionContext] = {}
        self._concurrent: set[asyncio.Task[Any]] = set()
        self._concurrent_invocations: dict[
            asyncio.Task[Any], _AsyncInvocation
        ] = {}
        self._latest_generation: dict[str, int] = {}
        self._next_generation = 0
        self._closed = False
        self._submitted = 0
        self._executed = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._rejected = 0
        self._coalesced = 0
        self._serial_worker: asyncio.Task[Any] | None = None

    def snapshot(self) -> ActionWorkerSnapshot:
        pending_concurrent = sum(
            invocation.generation not in self._running
            for invocation in self._concurrent_invocations.values()
        )
        return ActionWorkerSnapshot(
            submitted=self._submitted,
            executed=self._executed,
            completed=self._completed,
            failed=self._failed,
            cancelled=self._cancelled,
            rejected=self._rejected,
            coalesced=self._coalesced,
            queue_depth=len(self._serial) + pending_concurrent,
            running_actions=tuple(context.action for context in self._running.values()),
            closed=self._closed,
        )

    def is_current(
        self,
        queue_key: str,
        generation: int,
        coalesce: CoalescePolicy,
    ) -> bool:
        if coalesce in {"all", "reject-while-busy"}:
            return True
        return self._latest_generation.get(queue_key) == generation

    def _ensure_serial_worker(self) -> None:
        if self._serial_worker is None:
            self._serial_worker = asyncio.create_task(
                self._run_serial(), name="odon-extension-serial-actions"
            )

    def _key_busy(self, queue_key: str) -> bool:
        return any(
            item.registration.queue_key == queue_key for item in self._serial
        ) or any(context.queue_key == queue_key for context in self._running.values())

    def submit(
        self, registration: AsyncActionRegistration, interaction: UiInteraction
    ) -> bool:
        if self._closed:
            self._rejected += 1
            self._schedule_error(
                registration,
                ActionRejectedError(
                    "extension action runner is closed",
                    action=registration.action,
                    queue_key=registration.queue_key,
                ),
                None,
            )
            return False
        if (
            registration.coalesce == "reject-while-busy"
            and self._key_busy(registration.queue_key)
        ):
            self._rejected += 1
            self._schedule_error(
                registration,
                ActionRejectedError(
                    f"action {registration.action!r} was rejected while its queue key is busy",
                    action=registration.action,
                    queue_key=registration.queue_key,
                ),
                None,
            )
            return False
        if registration.execution == "worker" and registration.coalesce == "accumulate":
            self._rejected += 1
            self._schedule_error(
                registration,
                ActionRejectedError(
                    "accumulate coalescing requires serial-worker execution",
                    action=registration.action,
                    queue_key=registration.queue_key,
                ),
                None,
            )
            return False
        self._next_generation += 1
        generation = self._next_generation
        invocation = _AsyncInvocation(
            registration, interaction, generation, registration.delta
        )
        if registration.execution == "serial-worker":
            self._ensure_serial_worker()
            invocation = self._coalesce_serial(invocation)
            if invocation is None:
                self._submitted += 1
                return True
            if len(self._serial) >= self.max_queue:
                self._rejected += 1
                self._schedule_error(
                    registration,
                    ActionQueueFullError(
                        f"extension action queue is full at {self.max_queue} items",
                        action=registration.action,
                        queue_key=registration.queue_key,
                        generation=generation,
                    ),
                    None,
                )
                return False
            self._serial.append(invocation)
            self._serial_ready.set()
        elif registration.execution == "worker":
            if registration.coalesce == "latest":
                self._coalesce_concurrent_latest(registration.queue_key)
            if len(self._concurrent) >= self.max_queue:
                self._rejected += 1
                self._schedule_error(
                    registration,
                    ActionQueueFullError(
                        f"extension worker action queue is full at {self.max_queue} items",
                        action=registration.action,
                        queue_key=registration.queue_key,
                        generation=generation,
                    ),
                    None,
                )
                return False
            task = asyncio.create_task(self._execute(invocation))
            self._concurrent.add(task)
            self._concurrent_invocations[task] = invocation
            task.add_done_callback(self._concurrent_done)
        else:
            task = asyncio.create_task(self._execute(invocation))
            self._concurrent.add(task)
            self._concurrent_invocations[task] = invocation
            task.add_done_callback(self._concurrent_done)
        self._submitted += 1
        if registration.coalesce != "accumulate":
            self._latest_generation[registration.queue_key] = generation
        return True

    def _coalesce_concurrent_latest(self, queue_key: str) -> None:
        for task, pending in tuple(self._concurrent_invocations.items()):
            if (
                pending.registration.queue_key == queue_key
                and pending.generation not in self._running
                and task.cancel()
            ):
                self._concurrent.discard(task)
                self._concurrent_invocations.pop(task, None)
                self._coalesced += 1

    def _concurrent_done(self, task: asyncio.Task[Any]) -> None:
        self._concurrent.discard(task)
        self._concurrent_invocations.pop(task, None)

    def _coalesce_serial(
        self, invocation: _AsyncInvocation
    ) -> _AsyncInvocation | None:
        registration = invocation.registration
        if registration.coalesce == "latest":
            retained: deque[_AsyncInvocation] = deque()
            for pending in self._serial:
                if pending.registration.queue_key == registration.queue_key:
                    self._coalesced += 1
                else:
                    retained.append(pending)
            self._serial = retained
        elif registration.coalesce == "accumulate":
            for index in range(len(self._serial) - 1, -1, -1):
                pending = self._serial[index]
                if (
                    pending.registration.queue_key == registration.queue_key
                    and pending.registration.callback == registration.callback
                ):
                    combined = pending.delta + invocation.delta
                    self._coalesced += 1
                    del self._serial[index]
                    if combined == 0:
                        return None
                    invocation.delta = combined
                    return invocation
        return invocation

    async def _run_serial(self) -> None:
        while True:
            await self._serial_ready.wait()
            self._serial_ready.clear()
            if self._closed and not self._serial:
                return
            while self._serial:
                await self._execute(self._serial.popleft())

    async def _execute(self, invocation: _AsyncInvocation) -> None:
        registration = invocation.registration
        context = AsyncActionContext(
            self,
            registration,
            invocation.interaction,
            generation=invocation.generation,
            delta=invocation.delta,
        )
        if self._closed or registration.removed:
            self._cancelled += 1
            return
        if registration.coalesce == "accumulate":
            self._latest_generation[registration.queue_key] = invocation.generation
        self._running[context.generation] = context
        self._executed += 1
        try:
            result = registration.callback(context, invocation.interaction)
            if inspect.isawaitable(result):
                await result
            context.check_cancelled()
        except asyncio.CancelledError:
            self._cancelled += 1
            raise
        except (ActionCancelledError, StaleActionError):
            self._cancelled += 1
        except BaseException as error:
            self._failed += 1
            await self._report_error(registration, error, context)
        else:
            self._completed += 1
        finally:
            self._running.pop(context.generation, None)

    def _schedule_error(
        self,
        registration: AsyncActionRegistration,
        error: BaseException,
        context: AsyncActionContext | None,
    ) -> None:
        asyncio.create_task(self._report_error(registration, error, context))

    async def _report_error(
        self,
        registration: AsyncActionRegistration,
        error: BaseException,
        context: AsyncActionContext | None,
    ) -> None:
        if registration.on_error is not None:
            try:
                result = registration.on_error(error, context)
                if inspect.isawaitable(result):
                    await result
                return
            except Exception:
                logger.exception(
                    "Odon async extension action error callback failed for %s",
                    registration.action,
                )
        if context is not None and registration.status_component_id is not None:
            try:
                await context.status(f"Failed: {error}")
            except Exception:
                logger.exception(
                    "Failed to patch async action error status for %s",
                    registration.action,
                )
        logger.error("Odon async extension action %s failed: %s", registration.action, error)

    async def cancel_registration(
        self, registration: AsyncActionRegistration
    ) -> None:
        retained = deque(
            item for item in self._serial if item.registration is not registration
        )
        self._cancelled += len(self._serial) - len(retained)
        self._serial = retained
        for context in tuple(self._running.values()):
            if context.registration is registration:
                await context.cancel()
        for task, invocation in tuple(self._concurrent_invocations.items()):
            if (
                invocation.registration is registration
                and invocation.generation not in self._running
                and task.cancel()
            ):
                self._concurrent.discard(task)
                self._concurrent_invocations.pop(task, None)
                self._cancelled += 1

    async def close(self, *, cancel: bool = True, timeout: float = 1.0) -> None:
        """Stop the runner, with a bounded wait for cancellation cleanup."""

        if timeout < 0:
            raise ValueError("action shutdown timeout must be non-negative")
        if self._closed:
            return
        self._closed = True
        if cancel:
            self._cancelled += len(self._serial)
            self._serial.clear()
            for context in tuple(self._running.values()):
                await context.cancel()
            for task, invocation in tuple(self._concurrent_invocations.items()):
                if invocation.generation not in self._running and task.cancel():
                    self._cancelled += 1
                elif invocation.generation in self._running:
                    task.cancel()
        self._serial_ready.set()
        waiters = [*tuple(self._concurrent)]
        if self._serial_worker is not None:
            waiters.insert(0, self._serial_worker)
        if waiters:
            done, pending = await asyncio.wait(waiters, timeout=timeout)
            for task in pending:
                task.cancel()
