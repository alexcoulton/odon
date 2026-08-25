from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_shell_control.py"


class PythonShellWorkflowExampleTests(unittest.TestCase):
    def test_dataset_workflow_plan_is_typed_and_self_consistent(self) -> None:
        spec = importlib.util.spec_from_file_location("python_shell_control", EXAMPLE)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        layout = module.shell_layout()
        toolbar = module.command_toolbar()
        summary = module.plan_summary()

        layout.validate_for_mode("single")
        self.assertEqual(summary["dataset"], "synthetic_5ch.ome.zarr")
        self.assertEqual(summary["layout_nodes"], len(layout.nodes))
        self.assertEqual(summary["toolbar_groups"], len(toolbar.groups))
        self.assertEqual(
            summary["binding"],
            {
                "type": "command_state",
                "command_id": "viewer.scale_bar.toggle",
                "state": "checked",
                "equals": True,
            },
        )
        self.assertIn("app.shell.recover", summary["toolbar_commands"])
        self.assertIn("viewer.masks.export_geojson", summary["toolbar_commands"])


if __name__ == "__main__":
    unittest.main()
