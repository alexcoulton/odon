use super::*;
#[test]
fn actor_camera_fit_uses_logical_geometry_and_propagates_one_link_transaction() {
    let mut app = fixture_actor_app();
    let expected_center = app.image_world_rect_lvl0().center();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.create",
        serde_json::json!({"layout": "horizontal"}),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let fitted = app.actor_command(
        "viewer.viewports.camera.fit",
        serde_json::json!({"viewport_id": left}),
    );
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
