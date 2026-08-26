from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.errors import RequestTimeoutError, StaleActionError, TaskCancelledError
from odon.recipes import (
    ObjectPropertyUnavailableError,
    async_replace_object_source_and_style,
    replace_object_source_and_style,
    wait_for_viewer_readiness,
)


def ready(revision: int) -> Mapping[str, Any]:
    return {
        "loading": {
            "model_ready": True,
            "resources_ready": True,
            "geometry_ready": True,
            "canvas_ready": True,
            "presentation_ready": True,
            "projection_revision": revision,
            "presented_projection_revision": revision,
        }
    }


class FakeTask:
    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls
        self.snapshot = type("Snapshot", (), {"cancellation_supported": True})()

    def wait(self, timeout: float | None = None, *, progress: Any = None) -> Any:
        self.calls.append(("task.wait", timeout, progress is not None))
        return {"installed": True}


class FakeApplication:
    def __init__(self, calls: list[Any], states: list[Mapping[str, Any]] | None = None) -> None:
        self.calls = calls
        self.states = states or [ready(1)]
        self.index = 0

    def get_loading_state(self) -> Mapping[str, Any]:
        self.calls.append("readiness")
        value = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return value


class FakeObjects:
    def __init__(self, calls: list[Any], *, include_property: bool = True) -> None:
        self.calls = calls
        self.include_property = include_property
        self.task = FakeTask(calls)

    def set_overlay_visibility(self, visible: bool) -> None:
        self.calls.append(("visibility", visible))

    def set_style(self, **style: Any) -> None:
        self.calls.append(("style", style))

    def clear(self) -> None:
        self.calls.append("clear")

    def load(self, path: str, *, downsample_factor: float) -> FakeTask:
        self.calls.append(("load", path, downsample_factor))
        return self.task

    def get_state(self) -> Mapping[str, Any]:
        self.calls.append("object_state")
        return {"source": "new.parquet", "geometry_ready": True}

    def list_properties(self, *, offset: int, limit: int) -> Mapping[str, Any]:
        self.calls.append(("properties", offset, limit))
        columns = [
            {"name": "new_value", "numeric": True, "loaded": True}
            if self.include_property
            else {"name": "other", "numeric": True, "loaded": True}
        ]
        return {"columns": columns, "has_more": False}

    def color_by_continuous(self, property_name: str, **style: Any) -> None:
        self.calls.append(("continuous", property_name, style))


class FakeApp:
    def __init__(
        self,
        *,
        states: list[Mapping[str, Any]] | None = None,
        include_property: bool = True,
    ) -> None:
        self.calls: list[Any] = []
        self.application = FakeApplication(self.calls, states)
        self.objects = FakeObjects(self.calls, include_property=include_property)


class Context:
    def __init__(self, *, stale_at: int | None = None) -> None:
        self.calls: list[Any] = []
        self.stale_at = stale_at
        self.checks = 0

    def status(self, value: str) -> None:
        self.calls.append(("status", value))

    def ensure_current(self) -> None:
        self.checks += 1
        if self.stale_at == self.checks:
            raise StaleActionError(
                "superseded", action="select", queue_key="selection", generation=1
            )

    def attach(self, task: Any) -> Any:
        self.calls.append(("attach", task))
        return task

    def report_task(self, _snapshot: Any) -> None:
        pass

    def result(self, value: str) -> None:
        self.calls.append(("result", value))

    def commit(self, callback: Any) -> bool:
        self.ensure_current()
        callback()
        return True


class ObjectSourceStyleRecipeTests(unittest.TestCase):
    def test_neutralizes_and_presents_before_replacing_source(self) -> None:
        app = FakeApp()
        context = Context()
        result = replace_object_source_and_style(
            app,
            "new.parquet",
            "new_value",
            domain=(0, 10),
            context=context,
            poll_interval=0.001,
        )
        calls = app.calls
        neutral = calls.index(("style", {"color_mapping": {"mode": "single"}}))
        clear = calls.index("clear")
        load = calls.index(("load", "new.parquet", 1.0))
        properties = calls.index(("properties", 0, 200))
        continuous = next(
            index for index, call in enumerate(calls) if isinstance(call, tuple) and call[0] == "continuous"
        )
        reveal = calls.index(("visibility", True))
        readiness_indices = [index for index, call in enumerate(calls) if call == "readiness"]
        self.assertLess(neutral, readiness_indices[0])
        self.assertLess(readiness_indices[0], clear)
        self.assertLess(clear, load)
        self.assertLess(load, readiness_indices[1])
        self.assertLess(readiness_indices[1], properties)
        self.assertLess(properties, continuous)
        self.assertLess(continuous, reveal)
        self.assertLess(reveal, readiness_indices[-1])
        self.assertEqual(result.property_descriptor["name"], "new_value")
        self.assertEqual(context.calls[-1], ("result", "Ready"))

    def test_absent_property_never_commits_continuous_style_or_reveals(self) -> None:
        app = FakeApp(include_property=False)
        with self.assertRaises(ObjectPropertyUnavailableError):
            replace_object_source_and_style(
                app, "new.parquet", "missing", poll_interval=0.001
            )
        self.assertFalse(any(call[0] == "continuous" for call in app.calls if isinstance(call, tuple)))
        self.assertNotIn(("visibility", True), app.calls)
        self.assertEqual(app.calls[0], ("visibility", False))

    def test_stale_request_cannot_commit_final_style(self) -> None:
        app = FakeApp()
        with self.assertRaises(StaleActionError):
            replace_object_source_and_style(
                app,
                "new.parquet",
                "new_value",
                context=Context(stale_at=3),
                poll_interval=0.001,
            )
        self.assertFalse(any(call[0] == "continuous" for call in app.calls if isinstance(call, tuple)))

    def test_cancelled_load_keeps_the_overlay_neutral_and_hidden(self) -> None:
        app = FakeApp()

        def cancelled_wait(timeout: float | None = None, *, progress: Any = None) -> Any:
            app.calls.append(("task.wait.cancelled", timeout, progress is not None))
            raise TaskCancelledError("task was cancelled")

        app.objects.task.wait = cancelled_wait  # type: ignore[method-assign]
        with self.assertRaises(TaskCancelledError):
            replace_object_source_and_style(
                app, "new.parquet", "new_value", poll_interval=0.001
            )
        self.assertFalse(
            any(
                call[0] == "continuous"
                for call in app.calls
                if isinstance(call, tuple)
            )
        )
        self.assertNotIn(("visibility", True), app.calls)

    def test_readiness_timeout_states_work_may_still_be_running(self) -> None:
        app = FakeApp(
            states=[
                {
                    "loading": {
                        "model_ready": True,
                        "resources_ready": False,
                        "geometry_ready": True,
                        "canvas_ready": True,
                        "presentation_ready": False,
                    }
                }
            ]
        )
        with self.assertRaisesRegex(RequestTimeoutError, "may still be running"):
            wait_for_viewer_readiness(app, timeout=0.001, poll_interval=0.001)


class AsyncTask:
    snapshot = type("Snapshot", (), {"cancellation_supported": True})()

    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls

    async def wait(self, timeout: float | None = None, *, progress: Any = None) -> Any:
        self.calls.append(("task.wait", timeout, progress is not None))
        return {"installed": True}


class AsyncProxy:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.application = self
        self.objects = self
        self.task = AsyncTask(self.calls)

    async def get_loading_state(self) -> Mapping[str, Any]:
        self.calls.append("readiness")
        return ready(1)

    async def set_overlay_visibility(self, visible: bool) -> None:
        self.calls.append(("visibility", visible))

    async def set_style(self, **style: Any) -> None:
        self.calls.append(("style", style))

    async def clear(self) -> None:
        self.calls.append("clear")

    async def load(self, path: str, *, downsample_factor: float) -> AsyncTask:
        self.calls.append(("load", path, downsample_factor))
        return self.task

    async def get_state(self) -> Mapping[str, Any]:
        return {"source": "new.parquet"}

    async def list_properties(self, *, offset: int, limit: int) -> Mapping[str, Any]:
        return {"columns": [{"name": "new_value", "numeric": True}], "has_more": False}

    async def color_by_continuous(self, property_name: str, **style: Any) -> None:
        self.calls.append(("continuous", property_name, style))


class AsyncObjectSourceStyleRecipeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_recipe_has_the_same_ordered_barriers(self) -> None:
        app = AsyncProxy()
        result = await async_replace_object_source_and_style(
            app, "new.parquet", "new_value", poll_interval=0.001
        )
        self.assertEqual(result.property_descriptor["name"], "new_value")
        self.assertLess(
            app.calls.index(("style", {"color_mapping": {"mode": "single"}})),
            app.calls.index("clear"),
        )
        self.assertLess(
            next(index for index, call in enumerate(app.calls) if isinstance(call, tuple) and call[0] == "continuous"),
            app.calls.index(("visibility", True)),
        )


if __name__ == "__main__":
    unittest.main()
