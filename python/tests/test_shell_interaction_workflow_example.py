from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_shell_interaction_control.py"


class PythonShellInteractionWorkflowExampleTests(unittest.TestCase):
    def test_interaction_and_recovery_plan_is_typed_and_self_consistent(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "python_shell_interaction_control", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        layout = module.interaction_layout()
        plan = module.plan_summary()

        layout.validate_for_mode("single")
        self.assertEqual(plan["dataset"], "synthetic_5ch.ome.zarr")
        self.assertEqual(plan["layout_nodes"], len(layout.nodes))
        self.assertEqual(plan["layout_nodes"], len(module.layouts.review().nodes) + 1)
        self.assertEqual(plan["baseline_ratio"], module.BASELINE_RATIO)
        self.assertEqual(plan["resized_ratio"], module.RESIZED_RATIO)
        self.assertEqual(plan["baseline_selected_id"], module.LAYERS_ID)
        self.assertEqual(plan["resized_selected_id"], module.PROJECT_ID)
        self.assertEqual(plan["focused_node_id"], module.PROJECT_ID)
        self.assertEqual(plan["recovery_command_id"], "app.shell.recover")
        self.assertEqual(plan["recovery_root_id"], module.RECOVERY_ROOT_ID)

        body = layout.node(module.BODY_ID)
        self.assertEqual(body.children[0], module.LEFT_COLLAPSIBLE_ID)
        self.assertEqual(
            body.split,
            ui.ShellSplit(
                ratio=module.BASELINE_RATIO,
                resizable=True,
                axis="horizontal",
            ),
        )
        collapsible = layout.node(module.LEFT_COLLAPSIBLE_ID)
        self.assertEqual(collapsible.type, ui.ShellLayoutType.COLLAPSIBLE)
        self.assertEqual(collapsible.children, (module.LEFT_PANEL_ID,))
        self.assertFalse(collapsible.collapsed)


if __name__ == "__main__":
    unittest.main()
