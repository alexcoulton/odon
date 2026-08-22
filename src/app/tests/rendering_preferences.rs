use super::*;
#[test]
fn rendering_preferences_are_independent_and_revision_guarded_per_viewport() {
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

    let changed = app.control_set_viewport_rendering(&serde_json::json!({
        "viewport_id": left,
        "smooth_pixels": false,
        "show_scale_bar": false,
        "show_hud": false,
        "show_tile_debug": true,
        "if_presentation_revision": 1,
    }));
    assert_eq!(changed["presentation_revision"], 2, "{changed:#}");
    assert_eq!(changed["result"]["rendering"]["smooth_pixels"], false);

    let left_state = app.control_get_viewport_rendering(&serde_json::json!({
        "viewport_id": left,
    }));
    let right_state = app.control_get_viewport_rendering(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(left_state["result"]["show_hud"], false);
    assert_eq!(left_state["result"]["show_tile_debug"], true);
    assert_eq!(right_state["result"]["smooth_pixels"], true);
    assert_eq!(right_state["result"]["show_scale_bar"], true);
    assert_eq!(right_state["result"]["show_hud"], true);
    assert_eq!(right_state["result"]["show_tile_debug"], false);

    let stale = app.control_set_viewport_rendering(&serde_json::json!({
        "viewport_id": left,
        "show_hud": true,
        "if_presentation_revision": 1,
    }));
    assert_eq!(stale["revision_domain"], "presentation");
    assert_eq!(stale["current_revision"], 2);
}
