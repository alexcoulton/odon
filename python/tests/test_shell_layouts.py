from __future__ import annotations

import unittest

from odon import layouts
from odon.ui import ShellMountId


class ShellLayoutTemplateTests(unittest.TestCase):
    def test_single_view_workflow_layouts_are_mode_valid_and_keep_explicit_extensions(self) -> None:
        mount = "extension:org.example.review/panel"
        for builder in (layouts.review, layouts.analysis, layouts.comparison):
            layout = builder(panel_mounts=[mount])
            layout.validate_for_mode("single")
            self.assertIn(ShellMountId.VIEWER_CANVAS, [node.mount for node in layout.nodes])
            self.assertIn(mount, [node.mount for node in layout.nodes])
            self.assertIn(
                ShellMountId.EXTENSION_RIGHT_TABS,
                [node.mount for node in layout.nodes],
            )

    def test_mosaic_triage_is_mode_valid(self) -> None:
        layout = layouts.mosaic_triage()
        layout.validate_for_mode("mosaic")
        self.assertIn(ShellMountId.MOSAIC_CANVAS, [node.mount for node in layout.nodes])
        self.assertIn(ShellMountId.MOSAIC_LAYOUT, [node.mount for node in layout.nodes])

    def test_presentation_supports_single_and_mosaic_with_optional_toolbar(self) -> None:
        single = layouts.presentation()
        single.validate_for_mode("single")
        self.assertEqual(len(single.nodes), 2)
        mosaic = layouts.presentation(mode="mosaic", show_toolbar=True)
        mosaic.validate_for_mode("mosaic")
        self.assertIn(ShellMountId.MOSAIC_TOP_BAR, [node.mount for node in mosaic.nodes])

    def test_workflow_builders_reject_ambiguous_extension_mounts(self) -> None:
        with self.assertRaises(ValueError):
            layouts.review(panel_mounts=["not-an-extension"])
        with self.assertRaises(ValueError):
            layouts.analysis(
                panel_mounts=[
                    "extension:org.example/panel",
                    "extension:org.example/panel",
                ]
            )
        with self.assertRaises(ValueError):
            layouts.presentation(mode="project")


if __name__ == "__main__":
    unittest.main()
