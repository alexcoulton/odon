from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def load_example():
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    path = EXAMPLES / "python_instanseg_large_cycif_pilot.py"
    spec = importlib.util.spec_from_file_location("python_instanseg_large_cycif_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InstanSegLargeCycifPilotTests(unittest.TestCase):
    def test_plan_is_matched_and_records_pixel_size_assumption(self) -> None:
        module = load_example()
        output = module.plan(max_tiles=2, pixel_size=0.5)

        self.assertEqual(output["format"], module.FORMAT)
        self.assertEqual(output["model"], "fluorescence_nuclei_and_cells")
        self.assertEqual(output["max_tiles"], 2)
        self.assertEqual(output["pixel_size_um"], 0.5)
        self.assertIn("implausible", output["pixel_size_note"])
        self.assertEqual(
            output["multiplex_markers"],
            ["DNA1", "ECAD", "panCK", "CD45", "CD3D", "CD8A"],
        )
        self.assertEqual(len(output["comparisons"]), 4)

    def test_cached_manifest_is_forced_to_channel_one(self) -> None:
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            output_base = Path(directory)
            run_dir = output_base / "run-test"
            run_dir.mkdir()
            (output_base / "latest.json").write_text(
                json.dumps({"run_dir": str(run_dir)})
            )
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": module.FORMAT,
                        "display_channel_indices": [0, 9, 11],
                        "results": {},
                    }
                )
            )

            manifest = module.load_manifest(output_base)

        self.assertEqual(manifest["display_channel_indices"], [0])

    def test_workflow_uses_native_continuous_styles_without_quantile_bins(self) -> None:
        module = load_example()
        source = Path(module.__file__).read_text()

        self.assertNotIn("mean_channel_1_quantile", source)
        self.assertNotIn("np.quantile", source)
        self.assertIn('"mode": "continuous"', source)
        self.assertIn('updated["shared_intensity_domain"]', source)


if __name__ == "__main__":
    unittest.main()
