use super::*;
#[test]
fn rapid_comparison_fits_in_the_actor_before_renderer_layout_is_ready() {
    let mut app = fixture_actor_app();
    let expected_center = app.image_world_rect_lvl0().center();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.create",
        serde_json::json!({"viewport_id": left, "layout": "horizontal"}),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    app.actor_command(
        "viewer.viewports.camera.set",
        serde_json::json!({
            "viewport_id": left,
            "center_world_lvl0": [40_000.0, 8_000.0],
            "zoom": 0.01,
        }),
    );
    assert!(!app.control_workspace_canvas_ready());
    assert_eq!(app.control_viewport_canvas_ready(&left), Some(false));

    let fitted = app.actor_command(
        "viewer.viewports.camera.fit",
        serde_json::json!({"viewport_id": left}),
    );
    assert_eq!(
        fitted["result"]["center_world_lvl0"],
        serde_json::json!([expected_center.x, expected_center.y])
    );

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
    let right_camera = app.control_get_viewport_camera(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(
        right_camera["result"]["center_world_lvl0"],
        fitted["result"]["center_world_lvl0"]
    );
}
