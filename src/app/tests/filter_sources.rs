use super::*;
#[test]
fn filter_sensitive_operations_require_and_honor_an_explicit_source() {
    let mut app = fixture_app();
    let temp = std::env::temp_dir().join(format!(
        "odon-multiview-filter-{}-{}.geojson",
        std::process::id(),
        app.active_render_id
    ));
    let fixture = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "a", "properties": {"class": "tumor", "score": 1.0}, "geometry": {"type": "Polygon", "coordinates": [[[0,0],[5,0],[5,5],[0,5],[0,0]]]}},
            {"type": "Feature", "id": "b", "properties": {"class": "immune", "score": 2.5}, "geometry": {"type": "Polygon", "coordinates": [[[10,0],[15,0],[15,5],[10,5],[10,0]]]}},
            {"type": "Feature", "id": "c", "properties": {"class": "tumor", "score": 3.0}, "geometry": {"type": "Polygon", "coordinates": [[[20,0],[25,0],[25,5],[20,5],[20,0]]]}}
        ]
    });
    std::fs::write(&temp, serde_json::to_vec(&fixture).unwrap()).unwrap();
    app.seg_objects.load_path(temp.clone(), 1.0);
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
    app.control_set_viewport_object_filter(&serde_json::json!({
        "viewport_id": left,
        "query": "class == 'tumor'",
    }));
    let right = app.control_create_viewport(&serde_json::json!({
        "layout": "horizontal",
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    app.control_set_viewport_object_filter(&serde_json::json!({
        "viewport_id": right,
        "query": "class == 'immune'",
    }));

    let ambiguous = app.control_select_filtered_objects(&serde_json::json!({
        "target": "objects",
    }));
    assert!(
        ambiguous["error"]
            .as_str()
            .unwrap()
            .contains("require viewport_id")
    );

    let left_selection = app.control_select_filtered_objects(&serde_json::json!({
        "target": "objects",
        "viewport_id": left,
        "mode": "replace",
    }));
    assert_eq!(left_selection["result"]["matched_count"], 2);
    assert_eq!(
        app.control_viewport_workspace_snapshot()["active_viewport_id"],
        right
    );

    let standalone = app.control_select_filtered_objects(&serde_json::json!({
        "target": "objects",
        "filter_query": "score >= 2.5",
        "mode": "replace",
    }));
    assert_eq!(standalone["matched_count"], 2);
    let right_filter = app.control_get_viewport_object_filter(&serde_json::json!({
        "viewport_id": right,
    }));
    assert_eq!(right_filter["result"]["query"]["text"], "class == 'immune'");

    let all_histogram = app.control_object_histogram(&serde_json::json!({
        "target": "objects",
        "property": "score",
        "use_all_objects": true,
    }));
    assert_eq!(all_histogram["count"], 3);
    assert_eq!(all_histogram["filtered"], false);
    let left_histogram = app.control_object_histogram(&serde_json::json!({
        "target": "objects",
        "property": "score",
        "viewport_id": left,
    }));
    assert_eq!(left_histogram["result"]["count"], 2);

    let _ = std::fs::remove_file(temp);
}
