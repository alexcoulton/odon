use super::*;
#[test]
fn background_actor_preserves_migrated_viewport_control_results() {
    let mut app = fixture_actor_app();
    let viewport_id = "viewport-1";

    let operations = [
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["CD3", "PanCK"],
                "mode": "only",
            }),
        ),
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["Collagen"],
                "mode": "show",
            }),
        ),
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["CD3"],
                "mode": "hide",
            }),
        ),
        (
            "viewer.viewports.channels.set_active",
            serde_json::json!({"viewport_id": viewport_id, "channel": "PanCK"}),
        ),
        (
            "viewer.viewports.channels.set_color",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channel": "PanCK",
                "color_rgb": [12, 34, 56],
            }),
        ),
        (
            "viewer.viewports.channels.set_contrast",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channel": "PanCK",
                "min": 100.0,
                "max": 1000.0,
            }),
        ),
        (
            "viewer.viewports.channels.set_order",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": [4, 3, 2, 1, 0],
                "mode": "exact",
            }),
        ),
        (
            "viewer.viewports.channels.set_group",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["PanCK", "Collagen"],
                "name": "Stroma",
                "color_rgb": [90, 80, 70],
            }),
        ),
        (
            "viewer.viewports.rendering.set",
            serde_json::json!({
                "viewport_id": viewport_id,
                "smooth_pixels": true,
                "show_scale_bar": false,
                "show_hud": false,
                "show_tile_debug": true,
            }),
        ),
        (
            "viewer.viewports.planes.set",
            serde_json::json!({"viewport_id": viewport_id, "mode": "xy", "slice": 99}),
        ),
    ];

    for (method, params) in operations {
        let actor = app.actor_command(method, params);
        assert_eq!(actor["viewport_id"], viewport_id, "{method}");
        assert!(actor["result"].is_object(), "{method}: {actor:#}");
    }

    let actor_channels = app.actor_query(
        "viewer.viewports.channels.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_channels =
        app.control_get_viewport_channels(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_channels["result"], renderer_channels["result"]);

    let actor_groups = app.actor_query(
        "viewer.viewports.channels.list_groups",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_groups =
        app.control_get_viewport_channel_groups(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_groups["result"], renderer_groups["result"]);

    let actor_plane = app.actor_query(
        "viewer.viewports.planes.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_plane =
        app.control_get_viewport_plane(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_plane["result"], renderer_plane["result"]);

    let actor_rendering = app.actor_query(
        "viewer.viewports.rendering.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_rendering =
        app.control_get_viewport_rendering(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_rendering["result"], renderer_rendering["result"]);
}
