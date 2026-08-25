from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_multiplex_review_cockpit.py"


def load_example():
    spec = importlib.util.spec_from_file_location(
        "python_multiplex_review_cockpit", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MultiplexReviewCockpitExampleTests(unittest.TestCase):
    def test_cockpit_plan_is_typed_and_self_consistent(self) -> None:
        module = load_example()
        layout = module.shell_layout()
        panel = module.review_panel()
        summary = module.plan_summary()

        layout.validate_for_mode("single")
        self.assertIsInstance(panel, ui.Panel)
        self.assertEqual(summary["dataset"], "synthetic_5ch.ome.zarr")
        self.assertEqual(summary["layout_nodes"], len(layout.nodes))
        self.assertEqual(summary["cockpit_mount"], module.EXPECTED_SHELL_MOUNT)
        self.assertEqual(summary["cockpit_title"], "Multiplex review")
        self.assertEqual(summary["selected_right_tab"], module.COCKPIT_NODE_ID)
        self.assertEqual(summary["profile"]["scope"], "session")
        self.assertEqual(
            set(summary["presets"]),
            {"overview", "nuclear-qc", "immune-context", "stromal-context"},
        )

    def test_panel_menu_and_toolbar_share_actor_command_ids(self) -> None:
        module = load_example()
        summary = module.plan_summary()
        command_ids = set(summary["command_ids"])

        self.assertEqual(set(summary["toolbar_commands"]), command_ids)
        self.assertTrue(command_ids.issubset(set(summary["menu_commands"])))
        self.assertEqual(
            summary["export_bindings"]["enabled"]["command_id"],
            module.command_id("export-review"),
        )
        self.assertEqual(
            summary["inspect_bindings"],
            {
                "visible": ui.command_state(
                    module.command_id("inspect-selection"), state="visible"
                ),
                "enabled": ui.command_state(
                    module.command_id("inspect-selection"), state="enabled"
                ),
            },
        )

        panel = module.review_panel().to_dict()
        actions: list[dict] = []

        def collect(component: dict) -> None:
            if isinstance(component.get("action"), dict):
                actions.append(component["action"])
            for child in component.get("children", []):
                collect(child)

        collect(panel)
        panel_commands = {
            action["params"]["command_id"]
            for action in actions
            if action.get("method") == "ui.commands.execute"
        }
        self.assertIn(module.command_id("nuclear-qc"), panel_commands)
        self.assertIn(module.command_id("immune-context"), panel_commands)
        self.assertIn(module.command_id("flag-view"), panel_commands)
        self.assertIn(module.command_id("export-review"), panel_commands)

    def test_predicates_cover_hidden_disabled_and_capability_states(self) -> None:
        module = load_example()
        objects = module.command_predicates("objects").to_dict()
        masks = module.command_predicates("masks").to_dict()
        dataset = module.command_predicates().to_dict()

        self.assertEqual(objects["visible"]["path"], "resources.objects")
        self.assertEqual(
            objects["enabled"],
            {
                "type": "state",
                "path": "selection.objects.count",
                "operator": "greater_than",
                "value": 0,
                "reason": "Select at least one object.",
            },
        )
        self.assertEqual(masks["enabled"]["path"], "resources.masks")
        self.assertEqual(dataset["visible"]["capability"], "viewer.read")
        self.assertEqual(dataset["enabled"]["path"], "resources.dataset")


if __name__ == "__main__":
    unittest.main()
