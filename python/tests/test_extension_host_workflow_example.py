from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_extension_host_control.py"


class PythonExtensionHostWorkflowExampleTests(unittest.TestCase):
    def test_extension_host_plan_is_typed_and_self_consistent(self) -> None:
        spec = importlib.util.spec_from_file_location("python_extension_host_control", EXAMPLE)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plan = module.plan_summary()
        panel = module.extension_panel()
        layout = module.layouts.review()

        layout.validate_for_mode("single")
        self.assertIsInstance(panel, ui.Panel)
        self.assertEqual(plan["layout_nodes"], len(layout.nodes))
        self.assertEqual(
            layout.node(plan["host_node_id"]).mount,
            ui.ShellMountId.EXTENSION_LEFT_SECTIONS,
        )
        self.assertIn(plan["host_node_id"], layout.node(plan["host_parent_id"]).children)
        self.assertIsNotNone(layout.node(plan["host_parent_id"]).selected_id)
        self.assertEqual(plan["contribution_location"], "left.sections")
        self.assertEqual(
            plan["expected_shell_mount"],
            "extension:org.odon.python_shell_host_example/host-proof",
        )
        self.assertEqual(
            plan["component_ids"],
            ["host-proof-panel", "explanation", "connection", "fit"],
        )


if __name__ == "__main__":
    unittest.main()
