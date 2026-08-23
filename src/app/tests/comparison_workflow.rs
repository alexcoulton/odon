use super::*;
#[test]
fn motivating_two_property_comparison_runs_end_to_end_on_one_document() {
    let mut app = fixture_app();
    let object_path = std::env::temp_dir().join(format!(
        "odon-multiview-acceptance-{}-{}.geojson",
        std::process::id(),
        app.active_render_id
    ));
    let objects = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "cell-a", "properties": {"marker_a": 1.0, "marker_b": 9.0}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[5,0],[5,5],[0,5],[0,0]]]}},
            {"type": "Feature", "id": "cell-b", "properties": {"marker_a": 4.0, "marker_b": 2.0}, "geometry": {"type": "Polygon", "coordinates": [[[10,0],[15,0],[15,5],[10,5],[10,0]]]}},
            {"type": "Feature", "id": "cell-c", "properties": {"marker_a": 7.0, "marker_b": 5.0}, "geometry": {"type": "Polygon", "coordinates": [[[20,0],[25,0],[25,5],[20,5],[20,0]]]}}
        ]
    });
    std::fs::write(&object_path, serde_json::to_vec(&objects).unwrap()).unwrap();
    app.seg_objects.load_path(object_path.clone(), 1.0);
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.seg_objects.object_count() != 3 && Instant::now() < deadline {
        app.seg_objects.tick();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(app.seg_objects.object_count(), 3);
    app.rebuild_layer_orders();

    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.control_create_viewport(&serde_json::json!({
        "viewport_id": left,
        "title": "Marker B",
        "layout": "horizontal",
        "ratio": 0.55,
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let left_style = app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": left,
        "fill_cells": true,
        "fill_opacity": 0.65,
        "color_property": "marker_a",
    }));
    let right_style = app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": right,
        "fill_cells": true,
        "fill_opacity": 0.65,
        "color_property": "marker_b",
    }));
    assert!(left_style.get("error").is_none(), "{left_style:#}");
    assert!(right_style.get("error").is_none(), "{right_style:#}");
    let left_value = left_style["result"]["style"]["legend"][0]["value"]
        .as_str()
        .unwrap()
        .to_string();
    let right_value = right_style["result"]["style"]["legend"][0]["value"]
        .as_str()
        .unwrap()
        .to_string();
    let left_palette = app.control_set_viewport_object_legend(&serde_json::json!({
        "viewport_id": left,
        "entries": [{"value": left_value, "color_rgb": [255, 0, 0]}],
    }));
    let right_palette = app.control_set_viewport_object_legend(&serde_json::json!({
        "viewport_id": right,
        "entries": [{"value": right_value, "color_rgb": [0, 255, 0]}],
    }));
    assert!(left_palette.get("error").is_none(), "{left_palette:#}");
    assert!(right_palette.get("error").is_none(), "{right_palette:#}");

    app.set_viewport_links(ViewportLinks {
        camera: true,
        plane: true,
        selection: true,
    });
    let navigation = app.control_set_viewport_camera(&serde_json::json!({
        "viewport_id": left,
        "center_world_lvl0": [12.0, 18.0],
        "zoom": 2.25,
    }));
    assert_eq!(
        navigation["affected_viewport_ids"],
        serde_json::json!([left, right])
    );
    let plane = app.control_set_viewport_plane(&serde_json::json!({
        "viewport_id": right,
        "mode": "xy",
        "slice": 0,
    }));
    assert!(plane.get("error").is_none(), "{plane:#}");
    let selected = app.control_select_object_ids(&serde_json::json!({
        "target": "objects",
        "ids": ["cell-b"],
        "mode": "replace",
    }));
    assert_eq!(selected["matched_count"], 1);
    assert_eq!(
        app.control_object_selection_signature()["selection"]["selection_count"],
        1
    );

    let workspace = app.control_viewport_workspace_snapshot();
    assert_eq!(workspace["layout"], "horizontal");
    assert!((workspace["ratio"].as_f64().unwrap() - 0.55).abs() < 1.0e-6);
    assert_eq!(workspace["shared_resources"]["document_instances"], 1);
    assert_eq!(workspace["shared_resources"]["dataset_instances"], 1);
    assert_eq!(
        workspace["shared_resources"]["primary_object_geometry_instances"],
        1
    );
    assert_eq!(workspace["shared_resources"]["primary_object_count"], 3);
    let viewport = |id: &str| {
        workspace["viewports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|viewport| viewport["viewport_id"] == id)
            .unwrap()
    };
    assert_eq!(viewport(&left)["objects"]["color_property"], "marker_a");
    assert_eq!(viewport(&right)["objects"]["color_property"], "marker_b");
    assert_eq!(
        viewport(&left)["camera"]["center_world_lvl0"],
        viewport(&right)["camera"]["center_world_lvl0"]
    );

    app.request_screenshot_png_for_viewport(
        std::env::temp_dir().join("odon-acceptance-left.png"),
        ViewportId::new(&left).unwrap(),
    );
    app.request_screenshot_png_for_viewport(
        std::env::temp_dir().join("odon-acceptance-right.png"),
        ViewportId::new(&right).unwrap(),
    );

    let level = app.dataset.levels.last().unwrap();
    let view = ViewPlaneSelection {
        mode: ViewPlaneMode::Xy,
        slice_level0: 0,
    };
    let left_key = TileKey {
        render_id: 77_001,
        view,
        level: level.index,
        tile_y: 0,
        tile_x: 0,
    };
    let right_key = TileKey {
        render_id: 77_002,
        ..left_key
    };
    app.loader
        .set_active_render_ids(HashSet::from([left_key.render_id, right_key.render_id]));
    app.loader
        .set_active_keys(HashSet::from([left_key, right_key]));
    for (key, color_rgb) in [(left_key, [1.0, 0.0, 0.0]), (right_key, [0.0, 1.0, 0.0])] {
        app.loader
            .tx
            .send(TileRequest {
                key,
                channels: vec![RenderChannel {
                    index: 0,
                    color_rgb,
                    window: (0.0, app.dataset.abs_max),
                }],
            })
            .unwrap();
    }
    let mut rendered = HashMap::new();
    for _ in 0..2 {
        let response = app
            .loader
            .rx
            .recv_timeout(Duration::from_secs(5))
            .expect("viewport tile completion");
        let TileWorkerResponse::Tile(tile) = response else {
            panic!("expected successful tile");
        };
        rendered.insert(tile.key.render_id, tile.rgba);
    }
    assert_ne!(rendered[&77_001], rendered[&77_002]);
    let stats = app.loader.stats();
    assert_eq!(stats.decode_requests, 2);
    assert_eq!(stats.source_reads, 1);
    assert_eq!(stats.cache_hits, 1);
    assert!(stats.decoded_cache_bytes > 0);

    let _ = std::fs::remove_file(object_path);
}
