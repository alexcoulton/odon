use super::*;
#[test]
fn object_filter_and_overlay_controls_have_stable_state_and_errors() {
    let mut app = fixture_app();
    let request =
        DeepLinkRequest::parse_arg("odon://open?filter=id:cell-1%7Cid:cell-2&filter_logic=or")
            .expect("parse object-filter deep link")
            .expect("Odon deep link");

    app.apply_deep_link_request(&request);

    let filter = app.seg_objects.filter_snapshot_json();
    assert_eq!(filter["mode"], "simple");
    assert_eq!(filter["logic"], "any");
    assert_eq!(
        filter["simple"]["clauses"],
        serde_json::json!([
            {"enabled": true, "property": "id", "query": "cell-1"},
            {"enabled": true, "property": "id", "query": "cell-2"}
        ])
    );

    let query = app.control_set_object_filter_query(&serde_json::json!({
        "target": "objects",
        "query": "unknown_property == 3"
    }));
    assert!(
        query["error"]
            .as_str()
            .is_some_and(|error| error.contains("object layer is empty"))
    );
    assert_eq!(app.seg_objects.filter_snapshot_json()["mode"], "simple");

    app.seg_objects.clear_filter();
    let cleared = app.seg_objects.filter_snapshot_json();
    assert_eq!(cleared["active"], false);
    assert_eq!(cleared["mode"], "simple");

    let visibility = app.control_set_object_overlay_visibility(
        &serde_json::json!({"target": "all", "visible": false}),
    );
    assert_eq!(visibility["segmentation_labels"], false);
    assert_eq!(visibility["segmentation_geojson"], false);
    assert_eq!(visibility["segmentation_objects"], false);
    assert!(
        app.control_set_object_overlay_visibility(
            &serde_json::json!({"target": "unknown", "visible": true})
        )
        .get("error")
        .is_some()
    );
}
