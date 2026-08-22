use super::*;
#[test]
fn explicit_camera_fit_uses_target_canvas_and_propagates_one_link_transaction() {
    let mut app = fixture_app();
    let expected_center = app.image_world_rect_lvl0().center();
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
    let left_id = ViewportId::new(left.clone()).unwrap();
    app.viewport_workspace
        .as_mut()
        .unwrap()
        .get_mut(&left_id)
        .unwrap()
        .state
        .last_canvas_rect = Some(egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(320.0, 200.0),
    ));

    let fitted = app.control_fit_viewport_camera(&serde_json::json!({
        "viewport_id": left,
    }));
    assert!(fitted.get("error").is_none(), "{fitted:#}");
    assert!(fitted["result"].get("error").is_none(), "{fitted:#}");
    assert_eq!(
        fitted["result"]["center_world_lvl0"],
        serde_json::json!([expected_center.x, expected_center.y])
    );
    assert_eq!(
        fitted["affected_viewport_ids"],
        serde_json::json!([left, right])
    );
    assert!(fitted["link_transaction_id"].is_string());
    let left_camera = app.control_get_viewport_camera(&serde_json::json!({
        "viewport_id": left,
    }));
    let right_camera = app.control_get_viewport_camera(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(
        left_camera["result"]["center_world_lvl0"],
        right_camera["result"]["center_world_lvl0"]
    );
    assert_eq!(
        left_camera["result"]["zoom_screen_per_lvl0_px"],
        right_camera["result"]["zoom_screen_per_lvl0_px"]
    );
}
