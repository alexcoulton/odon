use super::*;
#[test]
fn rapid_comparison_waits_for_layout_then_fits_a_restored_off_image_camera() {
    let mut app = fixture_app();
    let expected_center = app.image_world_rect_lvl0().center();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.control_create_viewport(&serde_json::json!({
        "viewport_id": left,
        "layout": "horizontal",
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    let viewport_ids = app
        .viewport_workspace
        .as_ref()
        .unwrap()
        .viewports()
        .iter()
        .map(|viewport| viewport.id.clone())
        .collect::<Vec<_>>();
    for viewport_id in viewport_ids {
        let viewport = app
            .viewport_workspace
            .as_mut()
            .unwrap()
            .get_mut(&viewport_id)
            .unwrap();
        viewport.state.camera.center_world_lvl0 = egui::pos2(40_000.0, 8_000.0);
        viewport.state.camera.zoom_screen_per_lvl0_px = 0.01;
        viewport.state.last_canvas_rect = None;
    }
    let active_state = app
        .viewport_workspace
        .as_ref()
        .unwrap()
        .active()
        .state
        .clone();
    active_state.apply(&mut app);
    assert!(!app.control_workspace_canvas_ready());
    assert_eq!(app.control_viewport_canvas_ready(&left), Some(false));

    let ctx = egui::Context::default();
    ctx.begin_pass(egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        ..Default::default()
    });
    egui::CentralPanel::default().show(&ctx, |ui| {
        app.ui_viewport_workspace(ui, &ctx);
    });
    let _ = ctx.end_pass();

    assert!(app.control_workspace_canvas_ready());
    let fitted = app.control_fit_viewport_camera(&serde_json::json!({
        "viewport_id": left,
    }));
    assert!(fitted["result"].get("error").is_none(), "{fitted:#}");
    assert_eq!(
        fitted["result"]["center_world_lvl0"],
        serde_json::json!([expected_center.x, expected_center.y])
    );
    let right_camera = app.control_get_viewport_camera(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(
        right_camera["result"]["center_world_lvl0"],
        fitted["result"]["center_world_lvl0"]
    );
}
