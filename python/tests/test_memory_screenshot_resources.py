from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Mapping

from odon.async_resources import AsyncMemory, AsyncScreenshots
from odon.resources import Memory, Screenshots


class Tasks:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self.calls = calls

    def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        call = (method, {**dict(params or {}), "_label": label})
        self.calls.append(call)
        return call


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tasks = Tasks(self.calls)

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class AsyncTasks(Tasks):
    async def start(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        label: str | None = None,
    ) -> Any:
        return super().start(method, params, label=label)


class AsyncClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = AsyncTasks(self.calls)

    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class ResourceTests(unittest.TestCase):
    def test_memory_wrappers(self) -> None:
        client = Client()
        memory = Memory(client)  # type: ignore[arg-type]
        memory.get()
        memory.pin(2, channels=[0, "DAPI"], scope="all", force=True, if_revision=4)
        memory.unpin(2, scope="item", item="ROI-1")
        memory.unpin_all()
        memory.get_tile_loading()
        memory.set_tile_loading(
            workers=6,
            prefetch_mode="target_halo",
            prefetch_aggressiveness="aggressive",
            prefer_pinned_finer_levels=True,
        )
        self.assertEqual(client.calls[1][0], "memory.pin")
        self.assertEqual(client.calls[1][1]["channels"], [0, "DAPI"])
        self.assertTrue(client.calls[1][1]["force"])
        self.assertEqual(client.calls[1][1]["scope"], "all")
        self.assertEqual(client.calls[1][1]["if_revision"], 4)
        self.assertEqual(client.calls[2][1]["item"], "ROI-1")
        self.assertEqual(client.calls[5][1]["workers"], 6)

    def test_screenshot_settings_and_overwrite(self) -> None:
        client = Client()
        screenshots = Screenshots(client)  # type: ignore[arg-type]
        screenshots.get_settings()
        screenshots.set_settings(
            output_dir=Path("captures"),
            include_scale_bar=False,
            legend_scale=1.5,
        )
        screenshots.capture("image.png", overwrite=True)
        screenshots.capture_workspace("comparison.png", overwrite=True)
        screenshots.set_settings(clear_output_dir=True)
        self.assertEqual(client.calls[1][1]["output_dir"], "captures")
        self.assertFalse(client.calls[1][1]["include_scale_bar"])
        self.assertTrue(client.calls[2][1]["overwrite"])
        self.assertEqual(
            client.calls[3][0], "viewer.workspace.screenshot.capture"
        )
        self.assertIsNone(client.calls[4][1]["output_dir"])
        with self.assertRaises(ValueError):
            screenshots.set_settings(output_dir="captures", clear_output_dir=True)


class AsyncResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_wrappers(self) -> None:
        client = AsyncClient()
        memory = AsyncMemory(client)  # type: ignore[arg-type]
        screenshots = AsyncScreenshots(client)  # type: ignore[arg-type]
        await memory.pin(1)
        await memory.set_tile_loading(prefetch_mode="off")
        await screenshots.set_settings(include_legend=False)
        await screenshots.capture("image.png", overwrite=True)
        await screenshots.capture_workspace("comparison.png")
        self.assertEqual(client.calls[0][0], "memory.pin")
        self.assertEqual(client.calls[1][1]["prefetch_mode"], "off")
        self.assertFalse(client.calls[2][1]["include_legend"])
        self.assertTrue(client.calls[3][1]["overwrite"])
        self.assertEqual(
            client.calls[4][0], "viewer.workspace.screenshot.capture"
        )


if __name__ == "__main__":
    unittest.main()
