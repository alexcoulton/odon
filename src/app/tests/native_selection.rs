use super::*;
#[test]
fn native_object_click_submits_generation_checked_actor_transaction_without_local_mutation() {
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
    app.active_layer = LayerId::SegmentationObjects;
    assert!(app.commit_point_selection_to_layer(
        LayerId::SegmentationObjects,
        egui::pos2(25.0, 25.0),
        false,
        false,
    ));
    assert_eq!(app.seg_objects.selection_count(), 0);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.selection.state.replace");
    assert_eq!(intents[0].params["expected_generation"], 2);
    assert_eq!(
        intents[0].params["state"]["selected_indices"],
        serde_json::json!([1])
    );
    assert_eq!(intents[0].params["state"]["primary_index"], 1);
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();

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

#[test]
fn native_id_selection_submits_targeted_actor_command_without_local_mutation() {
    let mut app = fixture_app();
    let path = std::env::temp_dir().join(format!(
        "odon-native-id-selection-{}-{}.geojson",
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

    assert_eq!(
        app.select_objects_by_ids_target(
            &["cell-b".to_string()],
            crate::spatialdata::PositiveCellSelectionTarget::SegmentationObjects,
        ),
        Some((1, 1))
    );
    assert_eq!(app.seg_objects.selection_count(), 0);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.selection.select_ids");
    assert_eq!(intents[0].params["target"], "segmentation_objects");
    assert_eq!(intents[0].params["ids"], serde_json::json!(["cell-b"]));
    assert_eq!(intents[0].params["mode"], "replace");
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();
    std::fs::remove_file(path).unwrap();
}

#[test]
fn native_rect_and_lasso_gestures_submit_actor_commands_before_mutating_selection() {
    let mut app = fixture_app();
    let path = std::env::temp_dir().join(format!(
        "odon-native-gesture-selection-{}-{}.geojson",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "type":"FeatureCollection",
            "features":[
                {"type":"Feature","id":"cell-a","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[10,0],[10,10],[0,10],[0,0]]]}}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let resource = crate::objects::load_control_object_resource(path.clone(), 1.0).unwrap();
    assert!(app.install_control_actor_object_resource(3, &resource));
    app.active_layer = LayerId::SegmentationObjects;
    app.seg_objects_offset_world = egui::vec2(3.0, 4.0);

    assert_eq!(
        app.commit_rect_selection_to_active_layer(
            egui::Rect::from_min_max(egui::pos2(5.0, 7.0), egui::pos2(15.0, 17.0)),
            true,
        ),
        0
    );
    assert_eq!(app.seg_objects.selection_count(), 0);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.select_rect");
    assert_eq!(intents[0].params["target"], "segmentation_objects");
    assert_eq!(
        intents[0].params["world_rect"],
        serde_json::json!([2.0, 3.0, 12.0, 13.0])
    );
    assert_eq!(intents[0].params["mode"], "add");
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();

    assert_eq!(
        app.commit_lasso_selection_to_active_layer(
            &[
                egui::pos2(3.0, 4.0),
                egui::pos2(13.0, 4.0),
                egui::pos2(3.0, 14.0),
            ],
            false,
        ),
        0
    );
    assert_eq!(app.seg_objects.selection_count(), 0);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.select_lasso");
    assert_eq!(
        intents[0].params["points"],
        serde_json::json!([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    );
    assert_eq!(intents[0].params["mode"], "replace");
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();

    app.commit_clear_object_selection(LayerId::SegmentationObjects);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.objects.clear_selection");
    assert_eq!(intents[0].params["target"], "segmentation_objects");
    odon::control::ControlCommand::decode(intents[0].method, intents[0].params.clone()).unwrap();
    std::fs::remove_file(path).unwrap();
}
