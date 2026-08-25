from __future__ import annotations

import unittest
from typing import Any, Mapping

from odon.async_resources import AsyncApplication, AsyncViewer
from odon.resources import Application, Viewer


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class AsyncClient(Client):
    async def call(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        call = (method, dict(params or {}))
        self.calls.append(call)
        return call


class ResourceTests(unittest.TestCase):
    def test_application_settings_and_recent_projects(self) -> None:
        client = Client()
        application = Application(client)  # type: ignore[arg-type]
        application.get_settings()
        application.update_settings(
            auto_contrast={"method": "p1_to_p99"},
            fast_object_rendering=False,
            show_extension_manager=True,
            shell_layout_startup_profiles={"single": "Review"},
            if_revision=3,
        )
        application.list_recent_projects()
        application.forget_recent_project("study.odon")
        application.clear_recent_projects()
        self.assertEqual(client.calls[1][1]["auto_contrast"]["method"], "p1_to_p99")
        self.assertFalse(client.calls[1][1]["fast_object_rendering"])
        self.assertTrue(client.calls[1][1]["show_extension_manager"])
        self.assertEqual(
            client.calls[1][1]["shell_layout_startup_profiles"],
            {"single": "Review"},
        )
        self.assertEqual(client.calls[1][1]["if_revision"], 3)
        self.assertEqual(client.calls[3][1]["path"], "study.odon")

    def test_application_lifecycle_requires_explicit_save_decision(self) -> None:
        client = Client()
        application = Application(client)  # type: ignore[arg-type]
        application.get_lifecycle()
        application.request_close()
        application.request_quit(save="discard", if_revision=8)
        self.assertEqual(client.calls[1][1]["save"], "prompt")
        self.assertEqual(client.calls[2][1]["save"], "discard")
        self.assertEqual(client.calls[2][1]["if_revision"], 8)

    def test_scale_bar_wrappers(self) -> None:
        client = Client()
        viewer = Viewer(client)  # type: ignore[arg-type]
        viewer.get_scale_bar()
        viewer.set_scale_bar(False, if_revision=5)
        self.assertEqual(
            client.calls[1],
            ("viewer.scale_bar.set", {"visible": False, "if_revision": 5}),
        )

    def test_panel_tab_wrappers(self) -> None:
        client = Client()
        viewer = Viewer(client)  # type: ignore[arg-type]
        viewer.set_left_tab("project", if_revision=4)
        viewer.set_right_tab("analysis")
        self.assertEqual(
            client.calls[0],
            ("viewer.ui.set_left_tab", {"tab": "project", "if_revision": 4}),
        )
        self.assertEqual(
            client.calls[1],
            ("viewer.ui.set_right_tab", {"tab": "analysis"}),
        )


class AsyncResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_application_and_scale_bar(self) -> None:
        client = AsyncClient()
        application = AsyncApplication(client)  # type: ignore[arg-type]
        viewer = AsyncViewer(client)  # type: ignore[arg-type]
        await application.update_settings(
            fast_object_rendering=True,
            show_extension_manager=False,
            shell_layout_startup_profiles={"project": "Home"},
        )
        await application.request_close(save="save")
        await viewer.set_scale_bar(True)
        self.assertEqual(client.calls[0][0], "app.settings.set")
        self.assertEqual(
            client.calls[0][1]["shell_layout_startup_profiles"],
            {"project": "Home"},
        )
        self.assertFalse(client.calls[0][1]["show_extension_manager"])
        self.assertEqual(client.calls[1][1]["save"], "save")
        self.assertTrue(client.calls[2][1]["visible"])

    async def test_async_panel_tab_wrappers(self) -> None:
        client = AsyncClient()
        viewer = AsyncViewer(client)  # type: ignore[arg-type]
        await viewer.set_left_tab("layers")
        await viewer.set_right_tab("measurements", if_revision=7)
        self.assertEqual(client.calls[0], ("viewer.ui.set_left_tab", {"tab": "layers"}))
        self.assertEqual(
            client.calls[1],
            ("viewer.ui.set_right_tab", {"tab": "measurements", "if_revision": 7}),
        )


if __name__ == "__main__":
    unittest.main()
