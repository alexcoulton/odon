use super::*;
#[test]
fn native_layer_commit_replays_as_one_viewport_actor_transaction() {
    let mut app = fixture_app();
    let renderer_workspace = app.control_viewport_workspace_snapshot();
    let before = app.control_native_layers_projection_snapshot();
    app.channels[0].visible = false;
    app.channels[0].color_rgb = [7, 8, 9];
    app.channel_offsets_world[0] = egui::vec2(11.0, -3.0);
    app.channel_layer_order.reverse();
    app.active_layer = LayerId::Channel(2);
    app.record_native_layers_intent(&before);
    let intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    assert_eq!(intents[0].method, "viewer.viewports.layers.state.replace");
    assert_eq!(intents[0].params["if_presentation_revision"], 1);

    let mut model = odon::model::AppModel::project();
    model
        .bootstrap_dataset_from_renderer(&app.dataset, &renderer_workspace)
        .unwrap();
    model
        .dispatch(intents[0].method, &intents[0].params)
        .expect("native layer transaction is actor-owned")
        .expect("native layer transaction commits");
    let workspace = model.render_workspace_snapshot().unwrap();
    let viewport = &workspace["viewports"][0];
    assert_eq!(viewport["channels"][0]["visible"], false);
    assert_eq!(
        viewport["channels"][0]["color_rgb"],
        serde_json::json!([7, 8, 9])
    );
    assert_eq!(
        viewport["native_layers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|layer| layer["layer_id"] == "channel:0")
            .unwrap()["offset_world"],
        serde_json::json!([11.0, -3.0])
    );
    assert_eq!(
        viewport["channel_order"],
        serde_json::json!([4, 3, 2, 1, 0])
    );
    let active = viewport["native_layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|layer| layer["active"] == true)
        .unwrap();
    assert_eq!(active["layer_id"], "channel:2");
}
