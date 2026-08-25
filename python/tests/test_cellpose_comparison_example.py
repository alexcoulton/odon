from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_cellpose_comparison.py"


def load_example():
    spec = importlib.util.spec_from_file_location("python_cellpose_comparison", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def example_results(module):
    return {
        spec.id: {
            "id": spec.id,
            "title": spec.title,
            "description": spec.description,
            "settings": {
                "cellprob_threshold": spec.cellprob_threshold,
                "flow_threshold": spec.flow_threshold,
                "diameter": spec.diameter,
            },
            "cell_count": index + 10,
            "segmented_pixels": 100 + index,
            "coverage_percent": 2.5 + index,
            "median_area_px": 20.0 + index,
            "mean_area_px": 22.0 + index,
            "outline_count": index + 10,
            "geojson_path": f"/{spec.id}.geojson",
        }
        for index, spec in enumerate(module.RUN_SPECS)
    }


class CellposeComparisonExampleTests(unittest.TestCase):
    def test_plan_describes_shared_fixture_and_three_threshold_runs(self) -> None:
        module = load_example()
        plan = module.plan_summary()
        layout = module.comparison_layout()

        layout.validate_for_mode("single")
        self.assertEqual(plan["dataset"], "synthetic_5ch.ome.zarr")
        self.assertEqual(plan["input"], {"cytoplasm": ["PanCK", "CD3"], "nuclear": "DAPI"})
        self.assertEqual(
            [run["id"] for run in plan["runs"]],
            ["permissive", "balanced", "conservative"],
        )
        self.assertEqual(plan["viewer"]["panel_mount"], module.EXPECTED_SHELL_MOUNT)
        self.assertEqual(plan["viewer"]["selected_right_tab"], module.COMPARISON_NODE_ID)

    def test_panel_buttons_use_the_registered_extension_commands(self) -> None:
        module = load_example()
        panel = module.comparison_panel(example_results(module))

        self.assertIsInstance(panel, ui.Panel)
        run_grid = next(child for child in panel.children if child.id == "run-buttons")
        self.assertEqual(
            [button.action["params"]["command_id"] for button in run_grid.children],
            [module.command_id(spec.id) for spec in module.RUN_SPECS],
        )
        selected = next(child for child in panel.children if child.id == "selected-run")
        self.assertEqual(selected.value, "balanced")

    def test_fixture_channels_and_fingerprint_are_stable(self) -> None:
        module = load_example()

        self.assertEqual(
            module._channel_names(module.DEFAULT_DATASET),
            ["DAPI", "CD3", "PanCK", "Ki67", "Collagen"],
        )
        first = module._dataset_fingerprint(module.DEFAULT_DATASET)
        self.assertEqual(first, module._dataset_fingerprint(module.DEFAULT_DATASET))
        self.assertEqual(len(first), 64)

    def test_view_only_manifest_validation_is_typed(self) -> None:
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaises(FileNotFoundError):
                module.load_manifest(output_dir)
            (output_dir / "manifest.json").write_text('{"format":"wrong"}')
            with self.assertRaises(ValueError):
                module.load_manifest(output_dir)


if __name__ == "__main__":
    unittest.main()
