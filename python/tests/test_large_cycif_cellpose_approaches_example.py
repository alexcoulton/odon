from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def load_example():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    path = EXAMPLES / "python_cellpose_large_cycif_approaches.py"
    spec = importlib.util.spec_from_file_location("python_cellpose_large_cycif_approaches", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LargeCycifCellposeApproachTests(unittest.TestCase):
    def test_plan_separates_nuclear_and_membrane_guided_inputs(self) -> None:
        module = load_example()
        plan = module.approach_plan()
        approaches = {item["id"]: item for item in plan["approaches"]}

        self.assertEqual(plan["normalization"], "per-tile percentile 1.0–99.8")
        self.assertEqual(approaches["dapi-nuclei"]["model_type"], "nuclei")
        self.assertEqual(approaches["ecad-membrane"]["markers"], ("ECAD",))
        self.assertEqual(approaches["cd45-membrane"]["markers"], ("CD45",))
        self.assertEqual(approaches["dual-membrane-max"]["aggregation"], "max")
        self.assertEqual(approaches["broad-local-mean"]["aggregation"], "mean")
        self.assertEqual(approaches["broad-local-max"]["aggregation"], "max")

    def test_comparison_panel_accepts_approach_order(self) -> None:
        module = load_example()
        results = {}
        for approach in module.APPROACHES:
            results[approach.id] = {
                "id": approach.id,
                "title": approach.title,
                "description": approach.description,
                "cell_count": 10,
                "mean_area_px": 20.0,
                "median_area_px": 18.0,
                "segmented_pixels": 200,
                "settings": {
                    "cellprob_threshold": approach.cellprob_threshold,
                    "flow_threshold": approach.flow_threshold,
                    "min_size": approach.min_size,
                },
            }
        manifest = {
            "results": results,
            "result_order": [approach.id for approach in module.APPROACHES],
            "completed_tiles": 4,
            "total_tiles": 4,
        }

        panel = module.large.comparison_panel(manifest)
        serialized = panel.to_dict()
        self.assertEqual(module.large._result_ids(manifest), manifest["result_order"])
        for approach in module.APPROACHES:
            self.assertIn(f"show-{approach.id}", str(serialized))


if __name__ == "__main__":
    unittest.main()
