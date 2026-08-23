use super::*;
#[test]
fn viewport_screenshot_queue_keeps_targets_independent_and_cleans_removed_view() {
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
    let (completion_tx, _completion_rx) = crossbeam_channel::bounded(2);
    let preferences = odon::model::ScreenshotPreferences::default();
    app.request_actor_screenshot(61, Some(&left), &preferences, completion_tx.clone())
        .unwrap();
    app.request_actor_screenshot(62, Some(&right), &preferences, completion_tx)
        .unwrap();
    assert_eq!(app.screenshot_capture.pending.len(), 2);
    assert_eq!(app.screenshot_capture.pending[0].viewport_id.as_str(), left);
    assert_eq!(
        app.screenshot_capture.pending[1].viewport_id.as_str(),
        right
    );

    let removed = app.actor_command(
        "viewer.viewports.remove",
        serde_json::json!({
            "viewport_id": right,
        }),
    );
    assert_eq!(removed["removed"], true);
    assert_eq!(app.screenshot_capture.pending.len(), 1);
    assert_eq!(app.screenshot_capture.pending[0].viewport_id.as_str(), left);
}
