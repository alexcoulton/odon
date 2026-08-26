from __future__ import annotations

import asyncio
import threading
import unittest
from typing import Any, Mapping

from odon.async_events import AsyncEvents
from odon.async_ui import AsyncUi
from odon.events import Events
from odon.models import Event
from odon.ui import Ui
from odon.ui_actions import UiInteraction, UiInteractionDecodeError


def event_params(
    *,
    kind: str = "action",
    component_id: str = "next-marker",
    action: str = "next-channel",
    value: Any = None,
    sequence: int = 1,
) -> Mapping[str, Any]:
    return {
        "event": f"ui.extension:org.example.interactions.{kind}",
        "sequence": sequence,
        "revision": sequence,
        "source": f"ui:{component_id}",
        "data": {
            "component_id": component_id,
            "value": value,
            "action": {"type": "emit", "event": action},
        },
    }


class SyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.events = Events(self)  # type: ignore[arg-type]

    def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        values = dict(params or {})
        self.calls.append((method, values))
        if method == "ui.extensions.register":
            return {**values, "granted_capabilities": values["capabilities"]}
        return {}


class AsyncClient:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Mapping[str, Any]]] = []
        self.events = AsyncEvents(self)  # type: ignore[arg-type]

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        values = dict(params or {})
        self.calls.append((method, values))
        if method == "ui.extensions.register":
            return {**values, "granted_capabilities": values["capabilities"]}
        return {}


class UiInteractionTests(unittest.TestCase):
    def test_action_and_input_envelopes_are_normalized(self) -> None:
        action_event = Event.from_params(event_params())
        action = UiInteraction.from_event(
            action_event, extension_id="org.example.interactions"
        )
        self.assertEqual(action.component_id, "next-marker")
        self.assertEqual(action.action, "next-channel")
        self.assertIsNone(action.value)
        self.assertEqual(action.kind, "action")
        self.assertIs(action.event, action_event)

        input_event = Event.from_params(
            event_params(
                kind="input",
                component_id="marker",
                action="marker-selected",
                value="CD183",
            )
        )
        selected = UiInteraction.from_event(
            input_event, extension_id="org.example.interactions"
        )
        self.assertEqual(selected.kind, "input")
        self.assertEqual(selected.value, "CD183")

    def test_value_component_families_share_one_shape(self) -> None:
        cases = (
            ("toggle", True),
            ("select", "CD183"),
            ("radio", "nimbus"),
            ("slider", 0.72),
            ("text-input", "marker query"),
        )
        for sequence, (component_id, value) in enumerate(cases, start=1):
            with self.subTest(component_id=component_id):
                event = Event.from_params(
                    event_params(
                        kind="input",
                        component_id=component_id,
                        action="value-changed",
                        value=value,
                        sequence=sequence,
                    )
                )
                item = UiInteraction.from_event(
                    event, extension_id="org.example.interactions"
                )
                self.assertEqual(item.component_id, component_id)
                self.assertEqual(item.action, "value-changed")
                self.assertEqual(item.value, value)
                self.assertEqual(item.event.revision, sequence)

    def test_malformed_or_wrong_extension_events_are_rejected(self) -> None:
        with self.assertRaises(UiInteractionDecodeError):
            UiInteraction.from_event(
                Event.from_params(event_params()), extension_id="org.example.other"
            )
        malformed = dict(event_params())
        malformed["data"] = {"action": {"event": "next-channel"}}
        with self.assertRaises(UiInteractionDecodeError):
            UiInteraction.from_event(
                Event.from_params(malformed),
                extension_id="org.example.interactions",
            )

    def test_extension_subscription_filters_and_is_removable(self) -> None:
        client = SyncClient()
        extension = Ui(client).register_extension(
            id="org.example.interactions", name="Interactions", version="1"
        )
        received: list[UiInteraction] = []
        delivered = threading.Event()

        def callback(interaction: UiInteraction) -> None:
            received.append(interaction)
            delivered.set()

        subscription = extension.on_interaction(
            callback, action="next-channel", component_id="next-marker"
        )
        try:
            client.events._receive(event_params(action="ignored", sequence=1))
            client.events._receive(event_params(sequence=2))
            self.assertTrue(delivered.wait(1))
            self.assertEqual([item.event.sequence for item in received], [2])
            subscription.remove()
            delivered.clear()
            client.events._receive(event_params(sequence=3))
            self.assertFalse(delivered.wait(0.05))
            self.assertTrue(subscription.removed)
            self.assertIn(
                ("events.unsubscribe", {"events": ["ui.extension:org.example.interactions.*"]}),
                client.calls,
            )
        finally:
            client.events._close()

    def test_raw_wildcard_subscription_remains_available(self) -> None:
        client = SyncClient()
        delivered = threading.Event()
        raw: list[Event] = []

        def receive(event: Event) -> None:
            raw.append(event)
            delivered.set()

        client.events.subscribe("ui.extension:*", receive)
        try:
            client.events._receive(event_params(sequence=17))
            self.assertTrue(delivered.wait(1))
            self.assertEqual(raw[0].sequence, 17)
            self.assertEqual(raw[0].data["component_id"], "next-marker")
        finally:
            client.events._close()


class AsyncUiInteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_subscription_supports_awaitable_callbacks_and_removal(self) -> None:
        client = AsyncClient()
        extension = await AsyncUi(client).register_extension(
            id="org.example.interactions", name="Interactions", version="1"
        )
        received: list[str] = []
        delivered = asyncio.Event()

        async def callback(interaction: UiInteraction) -> None:
            await asyncio.sleep(0)
            received.append(str(interaction.value))
            delivered.set()

        subscription = await extension.on_interaction(
            callback, action="marker-selected"
        )
        client.events._receive(
            event_params(
                kind="input",
                component_id="marker",
                action="marker-selected",
                value="CD183",
            )
        )
        await asyncio.wait_for(delivered.wait(), 1)
        self.assertEqual(received, ["CD183"])
        await subscription.remove()
        self.assertTrue(subscription.removed)
        client.events._close()


if __name__ == "__main__":
    unittest.main()
