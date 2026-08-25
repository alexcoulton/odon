from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_shell_startup_mode_control.py"


class PythonShellStartupModeWorkflowExampleTests(unittest.TestCase):
    def test_startup_mode_plan_is_typed_isolated_and_self_consistent(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "python_shell_startup_mode_control", EXAMPLE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        layout = module.startup_layout()
        plan = module.plan_summary()

        layout.validate_for_mode("single")
        self.assertEqual(plan["dataset"], "synthetic_5ch.ome.zarr")
        self.assertEqual(plan["layout_nodes"], len(layout.nodes))
        self.assertEqual(plan["layout_nodes"], len(module.layouts.analysis().nodes) + 1)
        self.assertEqual(plan["profile_name"], module.PROFILE_NAME)
        self.assertEqual(plan["settings_environment_variable"], "ODON_SETTINGS_PATH")
        self.assertEqual(plan["startup_ratio"], module.STARTUP_RATIO)
        self.assertEqual(plan["startup_selected_id"], module.PROJECT_ID)
        self.assertEqual(plan["transition_focus_id"], module.PROJECT_ID)
        self.assertEqual(
            plan["transition_sequence"],
            ["project", "single", "project", "single"],
        )

        body = layout.node(module.BODY_ID)
        self.assertEqual(body.children[0], module.LEFT_COLLAPSIBLE_ID)
        self.assertEqual(
            body.split,
            ui.ShellSplit(
                ratio=module.STARTUP_RATIO,
                resizable=True,
                axis="horizontal",
            ),
        )
        collapsible = layout.node(module.LEFT_COLLAPSIBLE_ID)
        self.assertEqual(collapsible.children, (module.LEFT_PANEL_ID,))
        self.assertEqual(collapsible.title, "Startup-restored analysis")
        self.assertEqual(layout.node(module.LEFT_TABS_ID).selected_id, module.PROJECT_ID)


if __name__ == "__main__":
    unittest.main()
