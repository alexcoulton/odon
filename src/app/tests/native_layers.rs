use super::*;
use odon::control::ControlCommand;

fn actor_model_from_renderer(app: &mut OmeZarrViewerApp) -> odon::model::AppModel {
    let renderer_workspace = app.control_viewport_workspace_snapshot();
    let mut model = odon::model::AppModel::project();
    model
        .bootstrap_dataset_from_renderer(&app.dataset, &renderer_workspace)
        .expect("fixture renderer state bootstraps the actor model");
    app.control_actor_workspace_revision = 1;
    model
}

fn take_one(app: &mut OmeZarrViewerApp, expected_method: &str) -> NativeControlIntent {
    let mut intents = app.take_native_control_intents();
    assert_eq!(intents.len(), 1);
    let intent = intents.remove(0);
    assert_eq!(intent.method, expected_method);
    assert_eq!(intent.params["viewport_id"], "viewport-1");
    assert_eq!(intent.params["if_presentation_revision"], 1);
    ControlCommand::decode(intent.method, intent.params.clone())
        .expect("native interaction emits a typed actor command");
    intent
}

fn replay(model: &mut odon::model::AppModel, intent: NativeControlIntent) {
    model
        .dispatch(intent.method, &intent.params)
        .expect("native layer command is actor-owned")
        .expect("native layer command commits");
}

#[test]
fn native_layer_visibility_commits_directly_without_mutating_the_renderer() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    assert!(app.channels[0].visible);

    assert!(app.submit_native_layer_visibility(LayerId::Channel(0), false));
    assert!(
        app.channels[0].visible,
        "actor-owned native interaction must wait for projection reconciliation"
    );
    let intent = take_one(&mut app, "viewer.viewports.layers.set_visibility");
    replay(&mut model, intent);

    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(workspace["viewports"][0]["channels"][0]["visible"], false);
}

#[test]
fn native_layer_active_and_order_interactions_emit_direct_actor_commands() {
    let mut active_app = fixture_app();
    let mut active_model = actor_model_from_renderer(&mut active_app);
    let active_before = active_app.active_layer;
    active_app.commit_active_layer(LayerId::Channel(2));
    assert_eq!(active_app.active_layer, active_before);
    let active_intent = take_one(&mut active_app, "viewer.viewports.layers.set_active");
    replay(&mut active_model, active_intent);
    let workspace = active_model.render_workspace_snapshot().unwrap();
    let active = workspace["viewports"][0]["native_layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|layer| layer["active"] == true)
        .unwrap();
    assert_eq!(active["layer_id"], "channel:2");

    let mut order_app = fixture_app();
    let mut order_model = actor_model_from_renderer(&mut order_app);
    let before = order_app.channel_layer_order.clone();
    let channel_count = order_app.channels.len();
    assert!(
        order_app
            .submit_native_layer_order("channels", (0..channel_count).rev().map(LayerId::Channel),)
    );
    assert_eq!(order_app.channel_layer_order, before);
    let order_intent = take_one(&mut order_app, "viewer.viewports.layers.set_order");
    replay(&mut order_model, order_intent);
    assert_eq!(
        order_model.render_workspace_snapshot().unwrap()["viewports"][0]["channel_order"],
        serde_json::json!([4, 3, 2, 1, 0])
    );
}

#[test]
fn native_layer_bulk_visibility_and_offsets_are_atomic_actor_transactions() {
    let mut visibility_app = fixture_app();
    let mut visibility_model = actor_model_from_renderer(&mut visibility_app);
    assert!(
        visibility_app
            .submit_native_layer_visibilities([LayerId::Channel(0), LayerId::Channel(1)], false,)
    );
    assert!(visibility_app.channels[0].visible);
    assert!(!visibility_app.channels[1].visible);
    let visibility_intent = take_one(&mut visibility_app, "viewer.viewports.layers.state.replace");
    replay(&mut visibility_model, visibility_intent);
    let workspace = visibility_model.render_workspace_snapshot().unwrap();
    assert_eq!(workspace["viewports"][0]["channels"][0]["visible"], false);
    assert_eq!(workspace["viewports"][0]["channels"][1]["visible"], false);

    let mut offset_app = fixture_app();
    let mut offset_model = actor_model_from_renderer(&mut offset_app);
    assert!(offset_app.commit_layer_offsets(&[LayerOffsetEntry {
        layer: LayerId::Channel(0),
        offset_world: egui::vec2(11.0, -3.0),
    }]));
    assert_eq!(offset_app.channel_offsets_world[0], egui::Vec2::ZERO);
    let offset_intent = take_one(&mut offset_app, "viewer.viewports.layers.state.replace");
    replay(&mut offset_model, offset_intent);
    let workspace = offset_model.render_workspace_snapshot().unwrap();
    let channel = workspace["viewports"][0]["native_layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|layer| layer["layer_id"] == "channel:0")
        .unwrap();
    assert_eq!(channel["offset_world"], serde_json::json!([11.0, -3.0]));
}

#[test]
fn actor_native_layer_projection_is_the_only_committed_renderer_write() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    assert!(app.submit_native_layer_visibility(LayerId::Channel(0), false));
    let intent = take_one(&mut app, "viewer.viewports.layers.set_visibility");
    replay(&mut model, intent);
    assert!(app.channels[0].visible);

    let projection = model.render_workspace_snapshot().unwrap();
    app.apply_control_actor_workspace_projection(&projection)
        .expect("actor projection reconciles renderer state");
    assert!(!app.channels[0].visible);
    assert!(app.take_native_control_intents().is_empty());
}

#[test]
fn native_layer_commits_queue_before_first_projection_without_mutating_renderer_state() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 0;
    let active_before = app.active_layer;
    let offset_before = app.layer_offset_world(LayerId::Channel(0));

    app.commit_active_layer(LayerId::Channel(1));
    let active = take_one(&mut app, "viewer.viewports.layers.set_active");
    assert_eq!(active.params["if_presentation_revision"], 1);
    assert_eq!(app.active_layer, active_before);

    assert!(app.commit_layer_offsets(&[LayerOffsetEntry {
        layer: LayerId::Channel(0),
        offset_world: egui::vec2(12.0, -4.0),
    }]));
    let offset = take_one(&mut app, "viewer.viewports.layers.state.replace");
    assert_eq!(offset.params["if_presentation_revision"], 1);
    assert_eq!(app.layer_offset_world(LayerId::Channel(0)), offset_before);
}

#[test]
fn first_native_layer_projection_cannot_feed_an_active_mask_command_back_to_the_actor() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 0;
    app.control_actor_mask_generation = 1;
    app.active_layer = LayerId::Mask(99);
    let mut projection = app.control_native_layer_snapshot_list();
    projection
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|layer| layer["layer_id"] == "channel:0")
        .unwrap()["active"] = serde_json::json!(true);

    app.apply_control_actor_native_layers_projection(&projection)
        .expect("native layer projection applies without command emulation");

    assert!(
        app.take_native_control_intents().is_empty(),
        "projection reconciliation must never emit a semantic command"
    );
}

#[test]
fn native_layer_projection_clears_an_explicit_null_channel_window() {
    let mut app = fixture_app();
    app.channels[0].window = Some((10.0, 20.0));
    let mut projection = app.control_native_layer_snapshot_list();
    let channel = projection
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|layer| layer["layer_id"] == "channel:0")
        .unwrap();
    channel["presentation"]["window"] = serde_json::Value::Null;

    app.apply_control_actor_native_layers_projection(&projection)
        .expect("native layer projection accepts the actor's automatic-window state");

    assert_eq!(app.channels[0].window, None);
    assert!(app.take_native_control_intents().is_empty());
}

#[test]
fn native_channel_transform_is_revision_checked_and_projection_committed() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    assert!(app.submit_native_channel_transform(
        0,
        Some(egui::vec2(3.0, -2.0)),
        Some(egui::vec2(1.5, 0.75)),
        Some(0.25),
    ));
    assert_eq!(app.channel_offsets_world[0], egui::Vec2::ZERO);
    assert_eq!(app.channel_scales[0], egui::Vec2::splat(1.0));
    let intent = take_one(&mut app, "viewer.channels.set_transform");
    replay(&mut model, intent);

    let projection = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        projection["channel_transforms"][0]["offset_world"],
        serde_json::json!([3.0, -2.0])
    );
    assert_eq!(projection["viewports"][0]["presentation_revision"], 2);
    app.apply_control_actor_workspace_projection(&projection)
        .expect("transform projection commits renderer state");
    assert_eq!(app.channel_offsets_world[0], egui::vec2(3.0, -2.0));
    assert_eq!(app.channel_scales[0], egui::vec2(1.5, 0.75));
}
