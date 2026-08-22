use super::*;
#[test]
fn native_mask_commit_replays_as_one_generation_checked_actor_transaction() {
    let mut app = fixture_app();
    let before = app.control_mask_projection_snapshot();
    let created = app.control_create_mask_layer(&serde_json::json!({"name":"Native mask"}));
    let layer_id = created["id"].as_u64().unwrap();
    app.control_add_mask_polygon(&serde_json::json!({
        "id":layer_id,
        "vertices":[[1.0,2.0],[5.0,2.0],[5.0,7.0]],
    }));
    app.record_native_mask_intent(&before);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.masks.state.replace");
    assert_eq!(intents[0].params["expected_generation"], 1);

    let mut model = odon::model::AppModel::project();
    model.install_dataset(&app.dataset);
    let committed = model
        .dispatch(intents[0].method, &intents[0].params)
        .expect("native mask transaction is actor-owned")
        .expect("native mask transaction commits")
        .response;
    assert_eq!(committed["generation"], 2);
    assert_eq!(committed["layers"][0]["name"], "Native mask");
    assert_eq!(
        committed["layers"][0]["polygons_world"][0]
            .as_array()
            .unwrap()
            .len(),
        4
    );
}
