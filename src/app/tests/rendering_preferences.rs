use super::*;
#[test]
fn rendering_preferences_are_independent_and_revision_guarded_per_viewport() {
    let mut app = fixture_actor_app();
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

    let changed = app.actor_command(
        "viewer.viewports.rendering.set",
        serde_json::json!({
            "viewport_id": left,
            "smooth_pixels": false,
            "show_scale_bar": false,
            "show_hud": false,
            "show_tile_debug": true,
            "if_presentation_revision": 1,
        }),
    );
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

    let stale = app
        .try_actor_command(
            "viewer.viewports.rendering.set",
            serde_json::json!({
                "viewport_id": left,
                "show_hud": true,
                "if_presentation_revision": 1,
            }),
        )
        .expect_err("stale presentation revision must conflict");
    assert_eq!(stale.kind, odon::control::ControlErrorKind::Conflict);
    let data = stale.data.expect("conflict data");
    assert_eq!(data["revision_domain"], "presentation");
    assert_eq!(data["current_revision"], 2);
}
