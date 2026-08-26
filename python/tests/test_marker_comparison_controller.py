from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Mapping

from odon.events import Events
from odon.recipes import (
    MarkerComparisonController,
    MarkerComparisonState,
)
from odon.ui import Ui


EXTENSION_ID = "org.example.marker-comparison"


def interaction(action: str, value: Any = None, sequence: int = 1) -> Mapping[str, Any]:
    return {
        "event": f"ui.extension:{EXTENSION_ID}.action",
        "sequence": sequence,
        "revision": sequence,
        "source": f"ui:{action}",
        "data": {
            "component_id": action,
            "value": value,
            "action": {"type": "emit", "event": action},
        },
    }


class Client:
    def __init__(self) -> None:
        self.closed = False
        self.events = Events(self)  # type: ignore[arg-type]

    def call(self, method: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        values = dict(params or {})
        if method == "ui.extensions.register":
            return {**values, "granted_capabilities": values["capabilities"]}
        return {}


class LoadTask:
    snapshot = type("Snapshot", (), {"cancellation_supported": False})()

    def __init__(
        self,
        objects: "Objects",
        source: str,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.objects = objects
        self.source = source
        self.started = started
        self.release = release

    def wait(self, timeout: float | None = None, *, progress: Any = None) -> Any:
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(timeout):
            raise TimeoutError("test load did not release")
        self.objects.source = self.source
        self.objects.loads.append(self.source)
        return {"source": self.source}


class Objects:
    def __init__(self) -> None:
        self.source: str | None = None
        self.loads: list[str] = []
        self.block_source: str | None = None
        self.started = threading.Event()
        self.release = threading.Event()

    def set_overlay_visibility(self, _visible: bool) -> None:
        pass

    def set_style(self, **_style: Any) -> None:
        pass

    def clear(self) -> None:
        self.source = None

    def load(self, path: str, *, downsample_factor: float) -> LoadTask:
        blocking = path == self.block_source and not self.started.is_set()
        return LoadTask(
            self,
            path,
            self.started if blocking else None,
            self.release if blocking else None,
        )

    def get_state(self) -> Mapping[str, Any]:
        return {"source": self.source}

    def list_properties(self, *, offset: int, limit: int) -> Mapping[str, Any]:
        marker = (self.source or "M1").rsplit("/", 1)[-1]
        return {
            "columns": [
                {"name": f"raw-{marker}", "numeric": True},
                {"name": f"flat-field-{marker}", "numeric": True},
                {"name": f"nimbus-{marker}", "numeric": True},
            ],
            "has_more": False,
        }


class PresentationObjects:
    def __init__(self) -> None:
        self.styles: list[Mapping[str, Any]] = []

    def set_style(self, **style: Any) -> None:
        self.styles.append(dict(style))

    def color_by_continuous(self, property_name: str, **style: Any) -> None:
        self.styles.append({"property": property_name, **style})


class Viewer:
    def __init__(self) -> None:
        self.objects = PresentationObjects()
        self.visible: list[str] = []
        self.active: str | None = None

    def set_visible_channels(self, channels: list[str], *, mode: str) -> None:
        self.visible = list(channels)

    def set_active_channel(self, channel: str) -> None:
        self.active = channel


class Application:
    def get_loading_state(self) -> Mapping[str, Any]:
        return {
            "loading": {
                "model_ready": True,
                "resources_ready": True,
                "geometry_ready": True,
                "canvas_ready": True,
                "presentation_ready": True,
                "projection_revision": 5,
                "presented_projection_revision": 5,
            }
        }


class App:
    def __init__(self) -> None:
        self.application = Application()
        self.objects = Objects()


class Contribution:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.history: list[Mapping[str, Any]] = []

    def patch_values(self, values: Mapping[str, Any]) -> None:
        self.values.update(values)
        self.history.append(dict(values))


class MarkerComparisonControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.extension = Ui(self.client).register_extension(
            id=EXTENSION_ID, name="Comparison", version="1"
        )
        self.app = App()
        self.viewer = Viewer()
        self.contribution = Contribution()
        markers = {"ROI1": ("M1", "M2", "M3")}
        self.controller = MarkerComparisonController(
            self.app,
            self.extension,
            self.contribution,
            self.viewer,
            rois=("ROI1",),
            markers=markers,
            fills=("raw", "flat-field", "nimbus"),
            channel_for=lambda _roi, marker: f"channel-{marker}",
            property_for=lambda _roi, marker, fill: f"{fill}-{marker}",
            source_for=lambda _roi, marker: f"/objects/{marker}",
            domain_for=lambda _roi, _marker, _fill: (0, 1),
            initial_state=MarkerComparisonState("ROI1", "M1", "raw"),
            timeout=1,
        )

    def tearDown(self) -> None:
        self.controller.close()
        self.extension._close_local()
        self.client.events._close()

    def wait_ready(
        self, marker: str, fill: str, *, minimum_generation: int = 0
    ) -> None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = self.controller.state
            if (
                state.phase == "ready"
                and state.marker == marker
                and state.fill == fill
                and state.generation >= minimum_generation
            ):
                return
            time.sleep(0.01)
        self.fail(f"controller did not reach {marker}/{fill}: {self.controller.state}")

    def test_fill_switch_reuses_source_and_keeps_channel_panel_and_style_aligned(self) -> None:
        self.controller.apply_initial()
        self.controller.install_actions()
        self.assertEqual(self.app.objects.loads, ["/objects/M1"])
        self.client.events._receive(interaction("fill-selected", "flat-field", 1))
        self.wait_ready("M1", "flat-field", minimum_generation=1)
        self.assertEqual(self.app.objects.loads, ["/objects/M1"])
        self.assertEqual(
            self.viewer.objects.styles[-2]["property"], "flat-field-M1"
        )
        self.client.events._receive(interaction("fill-selected", "nimbus", 2))
        self.wait_ready("M1", "nimbus", minimum_generation=2)
        self.assertEqual(self.app.objects.loads, ["/objects/M1"])
        self.assertEqual(self.viewer.visible, ["channel-M1"])
        self.assertEqual(self.viewer.active, "channel-M1")
        self.assertEqual(self.viewer.objects.styles[-2]["property"], "nimbus-M1")
        self.assertEqual(self.contribution.values["marker"], "M1")
        self.assertEqual(self.contribution.values["fill-mode"], "nimbus")

    def test_rapid_navigation_accumulates_from_the_active_step(self) -> None:
        self.controller.apply_initial()
        self.controller.install_actions()
        self.app.objects.block_source = "/objects/M2"
        self.client.events._receive(interaction("next-marker", sequence=1))
        self.assertTrue(self.app.objects.started.wait(1))
        self.client.events._receive(interaction("next-marker", sequence=2))
        self.client.events._receive(interaction("next-marker", sequence=3))
        deadline = time.monotonic() + 1
        while (
            self.extension.action_status().submitted < 3
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.app.objects.release.set()
        # Three Next clicks from M1 wrap to M1. The already-running M2 step may
        # complete, while the two pending clicks coalesce and skip M3 entirely.
        self.wait_ready("M1", "raw", minimum_generation=3)
        self.assertEqual(self.viewer.visible, ["channel-M1"])
        self.assertEqual(self.viewer.active, "channel-M1")
        self.assertEqual(self.viewer.objects.styles[-2]["property"], "raw-M1")
        self.assertEqual(
            self.app.objects.loads,
            ["/objects/M1", "/objects/M2", "/objects/M1"],
        )
        ready_markers = [
            values.get("marker")
            for values in self.contribution.history
            if str(values.get("status", "")).startswith("Ready")
        ]
        self.assertNotIn("M3", ready_markers)
        self.assertEqual(ready_markers[-1], "M1")

    def test_latest_marker_selection_rejects_stale_completion(self) -> None:
        self.controller.apply_initial()
        self.controller.install_actions()
        self.app.objects.block_source = "/objects/M2"
        self.client.events._receive(interaction("marker-selected", "M2", 1))
        self.assertTrue(self.app.objects.started.wait(1))
        self.client.events._receive(interaction("marker-selected", "M3", 2))
        deadline = time.monotonic() + 1
        while (
            self.extension.action_status().submitted < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.app.objects.release.set()
        self.wait_ready("M3", "raw", minimum_generation=2)
        ready_markers = [
            values.get("marker")
            for values in self.contribution.history
            if str(values.get("status", "")).startswith("Ready")
        ]
        self.assertNotIn("M2", ready_markers)
        self.assertEqual(ready_markers[-1], "M3")

    def test_roi_selection_during_pending_open_rejects_stale_roi(self) -> None:
        opened: list[str] = []
        roi_started = threading.Event()
        roi_release = threading.Event()

        class OpenOperation:
            snapshot = type("Snapshot", (), {"cancellation_supported": False})()

            def __init__(self, roi: str) -> None:
                self.roi = roi

            def wait(self, timeout: float | None = None, *, progress: Any = None) -> None:
                opened.append(self.roi)
                if self.roi == "ROI2":
                    roi_started.set()
                    if not roi_release.wait(timeout):
                        raise TimeoutError("test ROI open did not release")

        second = MarkerComparisonController(
            self.app,
            self.extension,
            self.contribution,
            self.viewer,
            rois=("ROI1", "ROI2", "ROI3"),
            markers={"ROI1": ("M1",), "ROI2": ("M1",), "ROI3": ("M1",)},
            fills=("raw", "flat-field", "nimbus"),
            channel_for=lambda roi, marker: f"channel-{roi}-{marker}",
            property_for=lambda _roi, marker, fill: f"{fill}-{marker}",
            source_for=lambda roi, marker: f"/objects/{roi}/{marker}",
            domain_for=lambda _roi, _marker, _fill: (0, 1),
            open_roi=lambda roi: OpenOperation(roi),
            initial_state=MarkerComparisonState("ROI1", "M1", "raw"),
            timeout=1,
        )
        try:
            second.apply_initial()
            second.install_actions()
            self.client.events._receive(interaction("roi-selected", "ROI2", 1))
            self.assertTrue(roi_started.wait(1))
            self.client.events._receive(interaction("roi-selected", "ROI3", 2))
            deadline = time.monotonic() + 1
            while (
                self.extension.action_status().submitted < 2
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            roi_release.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if (
                    second.state.phase == "ready"
                    and second.state.roi_id == "ROI3"
                    and second.state.generation >= 2
                ):
                    break
                time.sleep(0.01)
            else:
                self.fail(f"controller did not reach ROI3: {second.state}")
            self.assertEqual(opened, ["ROI2", "ROI3"])
            ready_rois = [
                values.get("roi")
                for values in self.contribution.history
                if str(values.get("status", "")).startswith("Ready")
            ]
            self.assertNotIn("ROI2", ready_rois)
            self.assertEqual(ready_rois[-1], "ROI3")
            self.assertEqual(self.viewer.visible, ["channel-ROI3-M1"])
        finally:
            second.close()


if __name__ == "__main__":
    unittest.main()
