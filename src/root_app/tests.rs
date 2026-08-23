mod control_boundary_tests {
    use super::super::*;

    #[test]
    fn workspace_screenshot_crop_scales_clips_and_rejects_empty_rectangles() {
        assert_eq!(
            screenshot_crop_bounds(
                [800, 600],
                Some(egui::Rect::from_min_max(
                    egui::pos2(10.25, 20.5),
                    egui::pos2(110.75, 70.25),
                )),
                2.0,
            ),
            Some((20, 41, 222, 141))
        );
        assert_eq!(
            screenshot_crop_bounds(
                [100, 80],
                Some(egui::Rect::from_min_max(
                    egui::pos2(-20.0, -10.0),
                    egui::pos2(200.0, 100.0),
                )),
                1.0,
            ),
            Some((0, 0, 100, 80))
        );
        assert_eq!(
            screenshot_crop_bounds(
                [100, 80],
                Some(egui::Rect::from_min_max(
                    egui::pos2(120.0, 5.0),
                    egui::pos2(130.0, 10.0),
                )),
                1.0,
            ),
            None
        );
    }
}

mod native_control_cutover_tests {
    #[test]
    fn root_has_no_native_snapshot_translator() {
        let source = include_str!("../root_app.rs");
        assert!(!source.contains("mod native_control"));
        assert!(!source.contains("native_mosaic_before"));
        assert!(!source.contains("mosaic_native_control_intents"));
    }
}
