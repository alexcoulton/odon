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
    use super::super::*;

    #[test]
    fn root_has_no_native_snapshot_translator() {
        let source = include_str!("../root_app.rs");
        assert!(!source.contains("mod native_control"));
        assert!(!source.contains("native_mosaic_before"));
        assert!(!source.contains("mosaic_native_control_intents"));
    }

    #[test]
    fn native_command_presentations_share_one_actor_execution_intent() {
        let source = include_str!("../root_app.rs");
        assert_eq!(
            source.matches("method: \"ui.commands.execute\"").count(),
            1,
            "menu, shortcut, toolbar, and palette entry points must not build divergent requests"
        );
        assert_eq!(source.matches("command_execution_intent(").count(), 5);

        for command_id in [
            "app.window.close",
            "app.lifecycle.quit",
            "app.shell.recover",
        ] {
            let intent = command_execution_intent(crate::ui::CommandPresentationInvocation {
                command_id: command_id.to_string(),
                checked: None,
            });
            assert_eq!(intent.method, "ui.commands.execute");
            assert_eq!(intent.params, serde_json::json!({"command_id":command_id}));
        }
        let toggled = command_execution_intent(crate::ui::CommandPresentationInvocation {
            command_id: "viewer.scale_bar.toggle".to_string(),
            checked: Some(false),
        });
        assert_eq!(
            toggled.params,
            serde_json::json!({
                "command_id":"viewer.scale_bar.toggle",
                "checked":false,
            })
        );
    }
}
