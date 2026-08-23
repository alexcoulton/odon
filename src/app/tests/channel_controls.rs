use super::*;
#[test]
fn channel_and_view_controls_preserve_external_semantics() {
    let mut app = fixture_actor_app();

    let active = app.actor_command(
        "viewer.channels.set_active",
        serde_json::json!({"name": "Ki67"}),
    );
    assert_eq!(active["result"]["active_channel"]["name"], "Ki67");

    let only = app.actor_command(
        "viewer.channels.set_visible",
        serde_json::json!({"channels": ["CD3", "PanCK"], "mode": "only"}),
    );
    assert_eq!(only["result"]["changed"], true);
    assert_eq!(visible_channel_names(&app), vec!["CD3", "PanCK"]);
    assert_eq!(
        app.actor_query("viewer.channels.get_active", serde_json::json!({}))["active_channel"]["name"],
        "CD3"
    );

    let before_invalid = app.actor_query("viewer.channels.list_visible", serde_json::json!({}));
    let invalid = app.try_actor_command(
        "viewer.channels.set_visible",
        serde_json::json!({"channels": ["CD3", "missing"], "mode": "only"}),
    );
    assert!(invalid.is_err());
    assert_eq!(
        app.actor_query("viewer.channels.list_visible", serde_json::json!({})),
        before_invalid
    );

    app.actor_command(
        "viewer.channels.set_visible",
        serde_json::json!({"channels": ["Collagen"], "mode": "show"}),
    );
    app.actor_command(
        "viewer.channels.set_visible",
        serde_json::json!({"channels": ["CD3"], "mode": "hide"}),
    );
    assert_eq!(visible_channel_names(&app), vec!["PanCK", "Collagen"]);

    let contrast = app.actor_command(
        "viewer.channels.set_contrast",
        serde_json::json!({"channel": "PanCK", "min": 100.0, "max": 1000.0}),
    );
    assert_eq!(contrast["contrast"]["name"], "PanCK");
    assert_eq!(contrast["contrast"]["min"], 100.0);
    assert_eq!(contrast["contrast"]["max"], 1000.0);
    let invalid_contrast = app.try_actor_command(
        "viewer.channels.set_contrast",
        serde_json::json!({"channel": "PanCK", "min": 1000.0, "max": 100.0}),
    );
    assert!(invalid_contrast.is_err());
    assert_eq!(
        app.actor_query(
            "viewer.channels.get_contrast",
            serde_json::json!({"channel": "PanCK"})
        )["contrast"]["max"],
        1000.0
    );

    let color = app.actor_command(
        "viewer.channels.set_color",
        serde_json::json!({"channel": "PanCK", "color_rgb": [12, 34, 56]}),
    );
    assert_eq!(
        color["result"]["channel"]["color_rgb"],
        serde_json::json!([12, 34, 56])
    );
    let note = app.actor_command(
        "viewer.channels.set_note",
        serde_json::json!({"channel": "PanCK", "note": "epithelial marker"}),
    );
    assert_eq!(note["channel"]["note"], "epithelial marker");

    let transform = app.actor_command(
        "viewer.channels.set_transform",
        serde_json::json!({
            "channel": "PanCK",
            "offset_world": [4.0, -3.0],
            "scale": [1.25, 0.75],
            "rotation_rad": 0.5
        }),
    );
    assert_eq!(transform["changed"], true);
    assert_eq!(
        transform["transform"]["offset_world"],
        serde_json::json!([4.0, -3.0])
    );
    assert_eq!(
        transform["transform"]["scale"],
        serde_json::json!([1.25, 0.75])
    );
    let before_invalid_transform = app.actor_query(
        "viewer.channels.get_transform",
        serde_json::json!({"channel": "PanCK"}),
    );
    assert!(
        app.try_actor_command(
            "viewer.channels.set_transform",
            serde_json::json!({"channel": "PanCK", "scale": [0.0, 1.0]})
        )
        .is_err()
    );
    assert_eq!(
        app.actor_query(
            "viewer.channels.get_transform",
            serde_json::json!({"channel": "PanCK"})
        ),
        before_invalid_transform
    );
    let reset = app.actor_command(
        "viewer.channels.reset_transform",
        serde_json::json!({"channel": "PanCK"}),
    );
    assert_eq!(
        reset["transform"]["offset_world"],
        serde_json::json!([0.0, 0.0])
    );
    assert_eq!(reset["transform"]["scale"], serde_json::json!([1.0, 1.0]));
    assert_eq!(reset["transform"]["rotation_rad"], 0.0);

    let native_layers =
        app.actor_query("viewer.native_layers.list", serde_json::json!({}))["layers"].clone();
    assert_eq!(native_layers[0]["layer_id"], "channel:0");
    assert_eq!(native_layers[0]["stack"], "channels");
    let hidden = app.actor_command(
        "viewer.native_layers.set_visibility",
        serde_json::json!({"layer_id": "channel:2", "visible": false}),
    );
    assert_eq!(hidden["result"]["layer"]["visible"], false);
    let active = app.actor_command(
        "viewer.native_layers.set_active",
        serde_json::json!({"layer_id": "channel:2"}),
    );
    assert_eq!(active["result"]["layer"]["active"], true);
    let reordered = app.actor_command(
        "viewer.native_layers.set_order",
        serde_json::json!({
            "stack": "channels",
            "layers": ["channel:4", "channel:3", "channel:2", "channel:1", "channel:0"]
        }),
    );
    assert_eq!(reordered["result"]["changed"], true);
    assert_eq!(app.channel_layer_order, vec![4, 3, 2, 1, 0]);
    let moved = app.actor_command(
        "viewer.native_layers.set_offset",
        serde_json::json!({"layer_id": "channel:2", "offset_world": [9.0, -2.0]}),
    );
    assert_eq!(
        moved["result"]["layer"]["offset_world"],
        serde_json::json!([9.0, -2.0])
    );
    let reset_offset = app.actor_command(
        "viewer.native_layers.reset_offset",
        serde_json::json!({"layer_id": "channel:2"}),
    );
    assert_eq!(
        reset_offset["result"]["layer"]["offset_world"],
        serde_json::json!([0.0, 0.0])
    );
    app.actor_command(
        "viewer.native_layers.set_order",
        serde_json::json!({
            "stack": "channels",
            "layers": ["channel:0", "channel:1", "channel:2", "channel:3", "channel:4"]
        }),
    );
    app.actor_command(
        "viewer.native_layers.set_visibility",
        serde_json::json!({"layer_id": "channel:2", "visible": true}),
    );
    app.actor_command(
        "viewer.native_layers.set_active",
        serde_json::json!({"layer_id": "channel:1"}),
    );

    let panels = app.actor_command(
        "viewer.panels.set",
        serde_json::json!({"left": false, "right": true}),
    );
    assert_eq!(
        panels["result"]["panels"],
        serde_json::json!({"left": false, "right": true})
    );
    assert!(
        app.try_actor_command("viewer.panels.set", serde_json::json!({}))
            .is_err()
    );

    let smooth = app.actor_command(
        "viewer.rendering.set_smooth_pixels",
        serde_json::json!({"smooth": false}),
    );
    assert_eq!(smooth["result"]["changed"], true);
    assert_eq!(smooth["result"]["smooth_pixels"]["smooth"], false);

    let listed_first = app.actor_command(
        "viewer.channels.set_order",
        serde_json::json!({
            "channels": ["Collagen", "DAPI"],
            "mode": "listed_first"
        }),
    );
    assert_eq!(listed_first["changed"], true);
    assert_eq!(
        &app.channel_layer_order,
        &[4usize, 0usize, 1usize, 2usize, 3usize]
    );
    let before_invalid_order = app.channel_layer_order.clone();
    let invalid_order = app.try_actor_command(
        "viewer.channels.set_order",
        serde_json::json!({
            "channels": ["DAPI", "missing"],
            "mode": "listed_first"
        }),
    );
    assert!(invalid_order.is_err());
    assert_eq!(app.channel_layer_order, before_invalid_order);
    let exact = app.actor_command(
        "viewer.channels.set_order",
        serde_json::json!({
            "channels": ["Ki67", "PanCK", "CD3", "DAPI", "Collagen"],
            "mode": "exact"
        }),
    );
    assert_eq!(exact["mode"], "exact");
    assert_eq!(app.channel_layer_order, vec![3, 2, 1, 0, 4]);

    let grouped = app.actor_command(
        "viewer.channels.set_group",
        serde_json::json!({
            "channels": ["CD3", "PanCK"],
            "group": "Markers",
            "color_rgb": [18, 52, 86],
            "inherit_color": false,
            "replace_group_members": true
        }),
    );
    let group_id = grouped["result"]["group_id"]
        .as_u64()
        .expect("channel group id");
    let groups = app.current_layer_groups();
    let group = groups
        .channel_groups
        .iter()
        .find(|group| group.id == group_id)
        .expect("created channel group");
    assert_eq!(group.name, "Markers");
    assert_eq!(group.color_rgb, [0x12, 0x34, 0x56]);
    assert!(!groups.channel_members["CD3"].inherit_color);
    assert!(!groups.channel_members["PanCK"].inherit_color);

    let camera = app.actor_command(
        "viewer.camera.set",
        serde_json::json!({"center_world_lvl0": [12.0, 34.0], "zoom": 2.0}),
    );
    assert_eq!(
        camera["camera"]["center_world_lvl0"],
        serde_json::json!([12.0, 34.0])
    );
    assert_eq!(camera["camera"]["zoom_screen_per_lvl0_px"], 2.0);
    assert_eq!(
        app.actor_command("viewer.camera.zoom_out", serde_json::json!({"factor": 2.0}))["camera"]["zoom_screen_per_lvl0_px"],
        serde_json::json!(1.0)
    );
    assert!(
        app.try_actor_command("viewer.camera.zoom_in", serde_json::json!({"factor": 0.0}))
            .is_err()
    );
    let fitted = app.actor_command("viewer.camera.fit", serde_json::json!({}));
    assert!(
        fitted["camera"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .is_some_and(|zoom| zoom > 0.0)
    );

    let screenshot_path = std::env::temp_dir().join(format!(
        "odon-control-screenshot-{}.png",
        std::process::id()
    ));
    app.request_screenshot_png(screenshot_path);
    assert_eq!(
        app.screenshot_pending
            .front()
            .map(|pending| pending.request.id),
        Some(1)
    );
    assert!(app.screenshot_in_flight.is_empty());
}
