use super::*;
#[test]
fn channel_and_view_controls_preserve_external_semantics() {
    let mut app = fixture_app();

    let active = app.control_set_active_channel(&serde_json::json!({"name": "Ki67"}));
    assert_eq!(active["active_channel"]["name"], "Ki67");

    let only = app.control_set_visible_channels(
        &serde_json::json!({"channels": ["CD3", "PanCK"], "mode": "only"}),
    );
    assert_eq!(only["changed"], true);
    assert_eq!(visible_channel_names(&app), vec!["CD3", "PanCK"]);
    assert_eq!(app.control_active_channel_snapshot()["name"], "CD3");

    let before_invalid = app.control_visible_channel_snapshot();
    let invalid = app.control_set_visible_channels(
        &serde_json::json!({"channels": ["CD3", "missing"], "mode": "only"}),
    );
    assert!(invalid.get("error").is_some());
    assert_eq!(app.control_visible_channel_snapshot(), before_invalid);

    app.control_set_visible_channels(
        &serde_json::json!({"channels": ["Collagen"], "mode": "show"}),
    );
    app.control_set_visible_channels(&serde_json::json!({"channels": ["CD3"], "mode": "hide"}));
    assert_eq!(visible_channel_names(&app), vec!["PanCK", "Collagen"]);

    let contrast = app.control_set_channel_contrast(
        &serde_json::json!({"channel": "PanCK", "min": 100.0, "max": 1000.0}),
    );
    assert_eq!(contrast["name"], "PanCK");
    assert_eq!(contrast["min"], 100.0);
    assert_eq!(contrast["max"], 1000.0);
    let invalid_contrast = app.control_set_channel_contrast(
        &serde_json::json!({"channel": "PanCK", "min": 1000.0, "max": 100.0}),
    );
    assert!(invalid_contrast.get("error").is_some());
    assert_eq!(
        app.control_get_channel_contrast(&serde_json::json!({"channel": "PanCK"}))["max"],
        1000.0
    );

    let color = app.control_set_channel_color(
        &serde_json::json!({"channel": "PanCK", "color_rgb": [12, 34, 56]}),
    );
    assert_eq!(
        color["channel"]["color_rgb"],
        serde_json::json!([12, 34, 56])
    );
    let note = app.control_set_channel_note(
        &serde_json::json!({"channel": "PanCK", "note": "epithelial marker"}),
    );
    assert_eq!(note["channel"]["note"], "epithelial marker");

    let transform = app.control_set_channel_transform(&serde_json::json!({
        "channel": "PanCK",
        "offset_world": [4.0, -3.0],
        "scale": [1.25, 0.75],
        "rotation_rad": 0.5
    }));
    assert_eq!(transform["changed"], true);
    assert_eq!(
        transform["transform"]["offset_world"],
        serde_json::json!([4.0, -3.0])
    );
    assert_eq!(
        transform["transform"]["scale"],
        serde_json::json!([1.25, 0.75])
    );
    let before_invalid_transform =
        app.control_get_channel_transform(&serde_json::json!({"channel": "PanCK"}));
    assert!(
        app.control_set_channel_transform(
            &serde_json::json!({"channel": "PanCK", "scale": [0.0, 1.0]})
        )
        .get("error")
        .is_some()
    );
    assert_eq!(
        app.control_get_channel_transform(&serde_json::json!({"channel": "PanCK"})),
        before_invalid_transform
    );
    let reset = app.control_reset_channel_transform(&serde_json::json!({"channel": "PanCK"}));
    assert_eq!(
        reset["transform"]["offset_world"],
        serde_json::json!([0.0, 0.0])
    );
    assert_eq!(reset["transform"]["scale"], serde_json::json!([1.0, 1.0]));
    assert_eq!(reset["transform"]["rotation_rad"], 0.0);

    let native_layers = app.control_native_layer_snapshot_list();
    assert_eq!(native_layers[0]["layer_id"], "channel:0");
    assert_eq!(native_layers[0]["stack"], "channels");
    let hidden = app.control_set_native_layer_visibility(
        &serde_json::json!({"layer_id": "channel:2", "visible": false}),
    );
    assert_eq!(hidden["layer"]["visible"], false);
    let active = app.control_set_active_native_layer(&serde_json::json!({"layer_id": "channel:2"}));
    assert_eq!(active["layer"]["active"], true);
    let reordered = app.control_set_native_layer_order(&serde_json::json!({
        "stack": "channels",
        "layers": ["channel:4", "channel:3", "channel:2", "channel:1", "channel:0"]
    }));
    assert_eq!(reordered["changed"], true);
    assert_eq!(app.channel_layer_order, vec![4, 3, 2, 1, 0]);
    let moved = app.control_set_native_layer_offset(
        &serde_json::json!({"layer_id": "channel:2", "offset_world": [9.0, -2.0]}),
    );
    assert_eq!(
        moved["layer"]["offset_world"],
        serde_json::json!([9.0, -2.0])
    );
    let reset_offset =
        app.control_reset_native_layer_offset(&serde_json::json!({"layer_id": "channel:2"}));
    assert_eq!(
        reset_offset["layer"]["offset_world"],
        serde_json::json!([0.0, 0.0])
    );
    app.control_set_native_layer_order(&serde_json::json!({
        "stack": "channels",
        "layers": ["channel:0", "channel:1", "channel:2", "channel:3", "channel:4"]
    }));
    app.control_set_native_layer_visibility(
        &serde_json::json!({"layer_id": "channel:2", "visible": true}),
    );
    app.control_set_active_native_layer(&serde_json::json!({"layer_id": "channel:1"}));

    let panels = app.control_set_side_panels(&serde_json::json!({
        "left": false,
        "right": true
    }));
    assert_eq!(
        panels["panels"],
        serde_json::json!({"left": false, "right": true})
    );
    assert!(
        app.control_set_side_panels(&serde_json::json!({}))
            .get("error")
            .is_some()
    );

    let smooth = app.control_set_smooth_pixels(&serde_json::json!({"smooth": false}));
    assert_eq!(smooth["changed"], true);
    assert_eq!(smooth["smooth_pixels"]["smooth"], false);

    let listed_first = app.control_set_channel_order(&serde_json::json!({
        "channels": ["Collagen", "DAPI"],
        "mode": "listed_first"
    }));
    assert_eq!(listed_first["changed"], true);
    assert_eq!(
        &app.channel_layer_order,
        &[4usize, 0usize, 1usize, 2usize, 3usize]
    );
    let before_invalid_order = app.channel_layer_order.clone();
    let invalid_order = app.control_set_channel_order(&serde_json::json!({
        "channels": ["DAPI", "missing"],
        "mode": "listed_first"
    }));
    assert!(invalid_order.get("error").is_some());
    assert_eq!(app.channel_layer_order, before_invalid_order);
    let exact = app.control_set_channel_order(&serde_json::json!({
        "channels": ["Ki67", "PanCK", "CD3", "DAPI", "Collagen"],
        "mode": "exact"
    }));
    assert_eq!(exact["mode"], "exact");
    assert_eq!(app.channel_layer_order, vec![3, 2, 1, 0, 4]);

    let grouped = app.control_set_channel_group(&serde_json::json!({
        "channels": ["CD3", "PanCK"],
        "group": "Markers",
        "color": "#123456",
        "inherit_color": false,
        "replace_group_members": true
    }));
    let group_id = grouped["group_id"].as_u64().expect("channel group id");
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

    let camera = app.control_set_camera(&serde_json::json!({
        "center_world_lvl0": [12.0, 34.0],
        "zoom": 2.0
    }));
    assert_eq!(camera["center_world_lvl0"], serde_json::json!([12.0, 34.0]));
    assert_eq!(camera["zoom_screen_per_lvl0_px"], 2.0);
    assert_eq!(
        app.control_zoom(0.5)["zoom_screen_per_lvl0_px"],
        serde_json::json!(1.0)
    );
    assert!(app.control_zoom(0.0).get("error").is_some());
    assert!(app.control_fit_to_view().get("error").is_some());

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
