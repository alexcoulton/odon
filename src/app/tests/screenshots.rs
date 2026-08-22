use super::*;
#[test]
fn viewport_screenshot_queue_keeps_targets_independent_and_cleans_removed_view() {
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
    let first = app.control_capture_screenshot(&serde_json::json!({
        "viewport_id": left,
        "path": std::env::temp_dir().join("odon-left-viewport.png"),
    }));
    let second = app.control_capture_screenshot(&serde_json::json!({
        "viewport_id": right,
        "path": std::env::temp_dir().join("odon-right-viewport.png"),
    }));
    assert_eq!(first["queued"], true);
    assert_eq!(second["queued"], true);
    assert_eq!(app.screenshot_pending.len(), 2);
    assert_eq!(app.screenshot_pending[0].viewport_id.as_str(), left);
    assert_eq!(app.screenshot_pending[1].viewport_id.as_str(), right);

    let removed = app.control_remove_viewport(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(removed["removed"], true);
    assert_eq!(app.screenshot_pending.len(), 1);
    assert_eq!(app.screenshot_pending[0].viewport_id.as_str(), left);
}
