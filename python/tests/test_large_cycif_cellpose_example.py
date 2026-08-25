from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from odon import ui


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "python_cellpose_large_cycif.py"


def load_example():
    spec = importlib.util.spec_from_file_location("python_cellpose_large_cycif", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def result(module, spec, cells):
    return {
        "id": spec.id,
        "title": spec.title,
        "description": spec.description,
        "settings": {
            "cellprob_threshold": spec.cellprob_threshold,
            "flow_threshold": spec.flow_threshold,
            "min_size": spec.min_size,
        },
        "geojson_path": f"/{spec.id}.geojson",
        "cell_count": cells,
        "mean_area_px": 300.0,
        "median_area_px": 280.0,
        "segmented_pixels": cells * 300,
    }


class LargeCycifCellposeExampleTests(unittest.TestCase):
    def test_default_plan_maps_markers_and_avoids_dense_masks(self) -> None:
        module = load_example()
        plan = module.plan_summary()
        if plan.get("missing"):
            self.skipTest("external CyCIF fixture is not mounted")

        self.assertEqual(plan["shape"], [36, 27299, 20045])
        self.assertEqual(plan["nuclear"], {"name": "DNA1", "index": 0})
        self.assertEqual(
            [channel["index"] for channel in plan["cytoplasm"]],
            [9, 10, 11, 14, 15],
        )
        self.assertEqual(plan["tile_count"], 192)
        self.assertGreater(plan["dense_label_bytes_per_run"], 2_000_000_000)
        self.assertTrue(plan["strategy"]["resume"])

    def test_tile_ownership_regions_cover_each_pixel_exactly_once(self) -> None:
        module = load_example()
        tiles = list(module.iter_tiles(9, 7, tile_size=4, overlap=2))
        ownership = [[0 for _x in range(7)] for _y in range(9)]
        for tile in tiles:
            for y in range(tile.inner_y0, tile.inner_y1):
                for x in range(tile.inner_x0, tile.inner_x1):
                    ownership[y][x] += 1

        self.assertTrue(all(count == 1 for row in ownership for count in row))
        self.assertEqual(len({tile.index for tile in tiles}), len(tiles))

    def test_marker_csv_overrides_generic_ome_channel_labels(self) -> None:
        module = load_example()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "image.ome.zarr"
            (dataset / "0").mkdir(parents=True)
            (dataset / ".zattrs").write_text(
                json.dumps(
                    {
                        "omero": {"channels": [{"label": "Channel 1"}, {"label": "Channel 2"}]},
                        "multiscales": [{"name": "test"}],
                    }
                )
            )
            (dataset / "0" / ".zarray").write_text(
                json.dumps({"shape": [2, 10, 10], "dtype": "<u2", "chunks": [1, 5, 5]})
            )
            markers = Path(directory) / "markers.csv"
            with markers.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["marker_name"])
                writer.writeheader()
                writer.writerows([{"marker_name": "DNA1"}, {"marker_name": "panCK"}])

            self.assertEqual(module.marker_names(dataset, markers), ["DNA1", "panCK"])
            self.assertEqual(module.viewer_channel_names(dataset), ["Channel 1", "Channel 2"])

    def test_comparison_panel_and_commands_share_run_identity(self) -> None:
        module = load_example()
        results = {
            spec.id: result(module, spec, 100 + index)
            for index, spec in enumerate(module.RUN_SPECS)
        }
        manifest = {
            "results": results,
            "completed_tiles": 4,
            "total_tiles": 192,
        }
        panel = module.comparison_panel(manifest)
        layout = module.comparison_layout()

        self.assertIsInstance(panel, ui.Panel)
        layout.validate_for_mode("single")
        progress = next(child for child in panel.children if child.id == "tile-progress")
        self.assertAlmostEqual(progress.value, 4 / 192)
        buttons = next(child for child in panel.children if child.id == "run-buttons")
        self.assertEqual(buttons.columns, 1)
        self.assertEqual(
            [button.action["params"]["command_id"] for button in buttons.children],
            [module.command_id(spec.id) for spec in module.RUN_SPECS],
        )

    def test_result_text_accepts_non_cellpose_model_settings(self) -> None:
        module = load_example()
        text = module._result_text(
            {
                "title": "InstanSeg nuclei",
                "description": "DAPI only",
                "cell_count": 12,
                "object_label": "nuclei",
                "mean_area_px": 91.0,
                "median_area_px": 88.0,
                "settings": {
                    "model": "fluorescence_nuclei_and_cells",
                    "input_channels": ["DNA1"],
                },
            }
        )

        self.assertIn("12 nuclei", text)
        self.assertIn("Model: fluorescence_nuclei_and_cells", text)
        self.assertIn("Input: DNA1", text)


if __name__ == "__main__":
    unittest.main()
