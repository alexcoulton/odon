use super::*;
#[test]
fn workspace_canvas_rect_is_the_union_of_both_viewport_canvases() {
    let mut app = fixture_actor_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "layout": "horizontal",
        }),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let left_id = ViewportId::new(left).unwrap();
    let right_id = ViewportId::new(right).unwrap();
    app.last_canvas_rect = Some(egui::Rect::from_min_max(
        egui::pos2(210.0, 20.0),
        egui::pos2(410.0, 220.0),
    ));
    app.sync_runtime_to_active_viewport();
    app.viewport_workspace
        .as_mut()
        .unwrap()
        .get_mut(&left_id)
        .unwrap()
        .state
        .last_canvas_rect = Some(egui::Rect::from_min_max(
        egui::pos2(0.0, 10.0),
        egui::pos2(200.0, 210.0),
    ));
    assert_eq!(
        app.viewport_workspace.as_ref().unwrap().active_id(),
        &right_id
    );

    let rect = app.workspace_canvas_rect().unwrap();
    assert_eq!(rect.min, egui::pos2(0.0, 10.0));
    assert_eq!(rect.max, egui::pos2(410.0, 220.0));
}
