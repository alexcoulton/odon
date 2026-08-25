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
    path = EXAMPLES / "python_cellpose_dapi_nuclei_sweep.py"
    spec = importlib.util.spec_from_file_location("python_cellpose_dapi_nuclei_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DapiNucleiSweepTests(unittest.TestCase):
    def test_plan_shares_matching_inferences_and_retains_small_candidate(self) -> None:
        module = load_example()
        output = module.plan()
        variants = {item["id"]: item for item in output["variants"]}

        self.assertEqual(output["normalization"], "per-tile percentile 1.0–99.8")
        self.assertEqual(output["shared_inferences_per_tile"], 3)
        self.assertEqual(variants["dapi-d18-small"]["diameter"], 18.0)
        self.assertEqual(variants["dapi-d18-small"]["cellprob_threshold"], -1.0)
        self.assertEqual(variants["dapi-d18-small"]["min_size"], 40)
        self.assertEqual(variants["dapi-d18-control"]["min_size"], 80)


if __name__ == "__main__":
    unittest.main()
