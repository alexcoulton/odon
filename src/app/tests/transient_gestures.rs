use super::*;
#[test]
fn activating_or_removing_a_viewport_cancels_transient_edit_gestures() {
    let mut app = fixture_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.control_create_viewport(&serde_json::json!({
        "layout": "horizontal",
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    app.selection_rect_start_world = Some(egui::pos2(1.0, 2.0));
    app.selection_rect_current_world = Some(egui::pos2(3.0, 4.0));
    app.selection_lasso_world.push(egui::pos2(5.0, 6.0));
    app.drawing_mask_polygon.push(egui::pos2(7.0, 8.0));

    let activated = app.control_set_active_viewport(&serde_json::json!({
        "viewport_id": left,
    }));
    assert_eq!(activated["changed"], true);
    assert!(app.selection_rect_start_world.is_none());
    assert!(app.selection_rect_current_world.is_none());
    assert!(app.selection_lasso_world.is_empty());
    assert!(app.drawing_mask_polygon.is_empty());

    app.selection_rect_start_world = Some(egui::pos2(1.0, 2.0));
    app.drawing_mask_polygon.push(egui::pos2(7.0, 8.0));
    let removed = app.control_remove_viewport(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(removed["removed"], true);
    assert!(app.selection_rect_start_world.is_none());
    assert!(app.drawing_mask_polygon.is_empty());
}
