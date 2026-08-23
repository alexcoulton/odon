use super::*;
#[test]
fn explicit_layer_get_set_keeps_presentations_independent() {
    let mut app = fixture_actor_app();
    let mask_id = app.actor_command(
        "viewer.masks.layers.create",
        serde_json::json!({"name": "Shared mask data"}),
    )["id"]
        .as_u64()
        .expect("created mask ID");
    let mask_layer_id = format!("mask:{mask_id}");
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
    let default_mask_presentation = app.actor_query(
        "viewer.viewports.layers.get",
        serde_json::json!({
            "viewport_id": right,
            "layer_id": mask_layer_id.clone(),
        }),
    )["result"]["presentation"]
        .clone();
    let left_revision = app.control_get_viewport(&serde_json::json!({
        "viewport_id": left,
    }))["presentation_revision"]
        .as_u64()
        .unwrap();

    let changed = app.actor_command(
        "viewer.viewports.layers.set",
        serde_json::json!({
            "viewport_id": left,
            "layer_id": mask_layer_id.clone(),
            "presentation": {
                "opacity": 0.2,
                "width_screen_px": 4.0,
                "display_mode": "translucent_fill",
                "color_rgb": [10, 20, 30],
            },
            "if_presentation_revision": left_revision,
        }),
    );
    assert_eq!(
        changed["presentation_revision"],
        left_revision + 1,
        "{changed:#}"
    );
    assert!(
        (changed["result"]["layer"]["presentation"]["opacity"]
            .as_f64()
            .unwrap()
            - 0.2)
            .abs()
            < 1.0e-6
    );

    let left_mask = app.actor_query(
        "viewer.viewports.layers.get",
        serde_json::json!({"viewport_id": left, "layer_id": mask_layer_id.clone()}),
    );
    let right_mask = app.actor_query(
        "viewer.viewports.layers.get",
        serde_json::json!({"viewport_id": right, "layer_id": mask_layer_id}),
    );
    assert_eq!(
        left_mask["result"]["presentation"]["color_rgb"],
        serde_json::json!([10, 20, 30])
    );
    assert_eq!(
        left_mask["result"]["presentation"]["display_mode"],
        "translucent_fill"
    );
    assert_eq!(
        right_mask["result"]["presentation"]["opacity"],
        default_mask_presentation["opacity"]
    );
    assert_eq!(
        right_mask["result"]["presentation"]["display_mode"],
        default_mask_presentation["display_mode"]
    );

    let channel = app.actor_command(
        "viewer.viewports.layers.set",
        serde_json::json!({
            "viewport_id": right,
            "layer_id": "channel:0",
            "presentation": {
                "visible": true,
                "color_rgb": [1, 2, 3],
                "window": {"min": 5.0, "max": 50.0},
            },
        }),
    );
    assert_eq!(
        channel["result"]["layer"]["presentation"]["color_rgb"],
        serde_json::json!([1, 2, 3])
    );
    assert_eq!(
        channel["result"]["layer"]["presentation"]["window"]["min"],
        5.0
    );
}
