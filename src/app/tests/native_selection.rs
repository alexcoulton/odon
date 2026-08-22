use super::*;
#[test]
fn native_object_selection_replays_as_one_generation_checked_actor_transaction() {
    let mut app = fixture_app();
    let path = std::env::temp_dir().join(format!(
        "odon-native-object-selection-{}-{}.geojson",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "type":"FeatureCollection",
            "features":[
                {"type":"Feature","id":"cell-a","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]] }},
                {"type":"Feature","id":"cell-b","properties":{},"geometry":{"type":"Polygon","coordinates":[[[20,20],[30,20],[30,30],[20,30],[20,20]]]}}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let resource = crate::objects::load_control_object_resource(path.clone(), 1.0).unwrap();
    assert!(app.install_control_actor_object_resource(1, &resource));
    app.control_actor_object_selection_generation = 2;
    let before = app.control_object_selection_projection_snapshot();
    app.seg_objects
        .install_control_selection(&[1], Some(1))
        .unwrap();
    app.record_native_object_selection_intent(&before);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.selection.state.replace");
    assert_eq!(intents[0].params["expected_generation"], 2);

    let mut model = odon::model::AppModel::project();
    model.install_dataset(&app.dataset);
    let (document_generation, resource_generation) =
        model.begin_object_resource_load(path.to_string_lossy());
    assert!(model.install_object_resource_for_generation(
        document_generation,
        resource_generation,
        std::sync::Arc::new(resource),
    ));
    let committed = model
        .dispatch(intents[0].method, &intents[0].params)
        .expect("native object selection transaction is actor-owned")
        .expect("native object selection transaction commits")
        .response;
    assert_eq!(committed["selection"]["selection_count"], 1);
    assert_eq!(committed["selection"]["primary"]["id"], "cell-b");
    std::fs::remove_file(path).unwrap();
}
