use super::*;
#[test]
fn explicit_layer_get_set_keeps_presentations_independent() {
    let mut app = fixture_app();
    app.mask_layers.push(MaskLayer {
        id: 42,
        name: "Shared mask data".to_string(),
        visible: true,
        opacity: 0.5,
        width_screen_px: 2.0,
        display_mode: MaskDisplayMode::OutlineOnly,
        color_rgb: [255, 255, 255],
        offset_world: egui::Vec2::ZERO,
        editable: false,
        polygons_world: Vec::new(),
        raster_display: None,
        source_geojson: None,
    });
    app.rebuild_layer_orders();
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
    let left_revision = app.control_get_viewport(&serde_json::json!({
        "viewport_id": left,
    }))["presentation_revision"]
        .as_u64()
        .unwrap();

    let changed = app.control_set_viewport_layer(&serde_json::json!({
        "viewport_id": left,
        "layer_id": "mask:42",
        "presentation": {
            "opacity": 0.2,
            "width_screen_px": 4.0,
            "display_mode": "translucent_fill",
            "color_rgb": [10, 20, 30],
        },
        "if_presentation_revision": left_revision,
    }));
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

    let left_mask = app.control_get_viewport_layer(&serde_json::json!({
        "viewport_id": left,
        "layer_id": "mask:42",
    }));
    let right_mask = app.control_get_viewport_layer(&serde_json::json!({
        "viewport_id": right,
        "layer_id": "mask:42",
    }));
    assert_eq!(
        left_mask["result"]["presentation"]["color_rgb"],
        serde_json::json!([10, 20, 30])
    );
    assert_eq!(
        left_mask["result"]["presentation"]["display_mode"],
        "translucent_fill"
    );
    assert_eq!(right_mask["result"]["presentation"]["opacity"], 0.5);
    assert_eq!(
        right_mask["result"]["presentation"]["display_mode"],
        "outline_only"
    );

    let channel = app.control_set_viewport_layer(&serde_json::json!({
        "viewport_id": right,
        "layer_id": "channel:0",
        "presentation": {
            "visible": true,
            "color_rgb": [1, 2, 3],
            "window": {"min": 5.0, "max": 50.0},
        },
    }));
    assert_eq!(
        channel["result"]["layer"]["presentation"]["color_rgb"],
        serde_json::json!([1, 2, 3])
    );
    assert_eq!(
        channel["result"]["layer"]["presentation"]["window"]["min"],
        5.0
    );
}
