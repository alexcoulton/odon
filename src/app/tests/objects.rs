use super::*;
#[test]
fn object_filter_and_overlay_controls_have_stable_state_and_errors() {
    let mut app = fixture_actor_app();
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

    app.seg_objects.clear_filter();
    let cleared = app.seg_objects.filter_snapshot_json();
    assert_eq!(cleared["active"], false);
    assert_eq!(cleared["mode"], "simple");

    let visibility = app.actor_command(
        "viewer.objects.set_visibility",
        serde_json::json!({"target": "all", "visible": false}),
    );
    assert_eq!(visibility["overlay"]["segmentation_labels"], false);
    assert_eq!(visibility["overlay"]["segmentation_geojson"], false);
    assert_eq!(visibility["overlay"]["segmentation_objects"], false);
    assert!(
        app.try_actor_command(
            "viewer.objects.set_visibility",
            serde_json::json!({"target": "unknown", "visible": true})
        )
        .is_err()
    );
}
