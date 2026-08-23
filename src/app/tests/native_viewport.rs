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
    ControlCommand::decode(intent.method, intent.params.clone())
        .expect("native viewport interaction emits a typed actor command");
    intent
}

fn replay(model: &mut odon::model::AppModel, intent: NativeControlIntent) {
    model
        .dispatch(intent.method, &intent.params)
        .expect("native viewport command is actor-owned")
        .expect("native viewport command commits");
}

#[test]
fn native_workspace_topology_commands_do_not_fall_back_to_renderer_mutation() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 0;
    let before = app.viewport_workspace.as_ref().unwrap().revision();

    assert!(app.submit_native_viewport_intent(
        "viewer.workspace.layout.set",
        serde_json::json!({"layout":"horizontal","ratio":0.4}),
    ));

    let intent = take_one(&mut app, "viewer.workspace.layout.set");
    assert_eq!(intent.params["layout"], "horizontal");
    assert_eq!(app.viewport_workspace.as_ref().unwrap().revision(), before);
}

#[test]
fn native_camera_fit_queues_before_first_projection_without_mutating_renderer_state() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 0;
    let before = app.camera.clone();
    let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    let world = egui::Rect::from_min_size(egui::pos2(100.0, 200.0), egui::vec2(400.0, 300.0));

    assert!(app.fit_camera_to_world_rect(viewport, world));
    let intent = take_one(&mut app, "viewer.viewports.camera.set");
    assert_eq!(intent.params["if_navigation_revision"], 1);
    assert_eq!(app.camera.center_world_lvl0, before.center_world_lvl0);
    assert_eq!(
        app.camera.zoom_screen_per_lvl0_px,
        before.zoom_screen_per_lvl0_px
    );
}

#[test]
fn detached_workspace_retains_the_explicit_native_command_scope() {
    let mut app = fixture_app();
    let _workspace = app.viewport_workspace.take().expect("fixture workspace");
    app.control_actor_workspace_revision = 1;
    app.native_viewport_command_scope = Some(NativeViewportCommandScope {
        viewport_id: "viewport-1".to_string(),
        navigation_revision: 7,
        presentation_revision: 11,
    });

    assert!(app.submit_native_active_viewport_rendering(true, true, false, false));
    let rendering = take_one(&mut app, "viewer.viewports.rendering.set");
    assert_eq!(rendering.params["viewport_id"], "viewport-1");
    assert_eq!(rendering.params["if_presentation_revision"], 11);

    assert!(app.submit_native_active_viewport_plane(ViewPlaneMode::Xy, Some(3)));
    let plane = take_one(&mut app, "viewer.viewports.planes.set");
    assert_eq!(plane.params["viewport_id"], "viewport-1");
    assert_eq!(plane.params["if_navigation_revision"], 7);
}

#[test]
fn native_camera_commit_is_revision_checked_and_replays_through_the_actor() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    let before = app.camera.clone();
    let desired = Camera {
        center_world_lvl0: egui::pos2(123.0, 456.0),
        zoom_screen_per_lvl0_px: 2.5,
    };

    assert!(app.submit_native_camera(&desired));
    assert_eq!(app.camera.center_world_lvl0, before.center_world_lvl0);
    assert_eq!(
        app.camera.zoom_screen_per_lvl0_px,
        before.zoom_screen_per_lvl0_px
    );
    let intent = take_one(&mut app, "viewer.viewports.camera.set");
    assert_eq!(intent.params["if_navigation_revision"], 1);
    replay(&mut model, intent);

    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        workspace["viewports"][0]["camera"]["center_world_lvl0"],
        serde_json::json!([123.0, 456.0])
    );
    assert_eq!(
        workspace["viewports"][0]["camera"]["zoom_screen_per_lvl0_px"],
        2.5
    );
}

#[test]
fn native_channel_group_commit_waits_for_actor_projection() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    let before = app.current_layer_groups();
    let mut desired = before.clone();
    desired.channel_groups.push(ProjectChannelGroup {
        id: 9,
        name: "Native group".to_string(),
        expanded: true,
        color_rgb: [20, 30, 40],
    });
    desired.channel_members.insert(
        app.channels[1].name.clone(),
        ProjectChannelGroupMember {
            group_id: 9,
            inherit_color: true,
        },
    );

    assert!(app.commit_current_channel_groups(desired));
    assert_eq!(app.current_layer_groups(), before);
    let intent = take_one(&mut app, "viewer.viewports.channels.set_group");
    assert_eq!(intent.params["replace_all"], true);
    replay(&mut model, intent);

    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(workspace["viewports"][0]["channel_groups"][0]["id"], 9);
    assert_eq!(
        workspace["viewports"][0]["channel_groups"][0]["name"],
        "Native group"
    );
}

#[test]
fn native_quick_contrast_is_one_atomic_layer_transaction() {
    let mut app = fixture_app();
    let mut model = actor_model_from_renderer(&mut app);
    let before = app.channels[0].window;

    app.apply_channel_window_to_indices(&[0, 1], 10.0, 80.0);
    assert_eq!(app.channels[0].window, before);
    let intent = take_one(&mut app, "viewer.viewports.layers.state.replace");
    replay(&mut model, intent);

    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        workspace["viewports"][0]["channels"][0]["window"],
        serde_json::json!({"min":10.0,"max":80.0})
    );
    assert_eq!(
        workspace["viewports"][0]["channels"][1]["window"],
        serde_json::json!({"min":10.0,"max":80.0})
    );
}

#[test]
fn native_object_filter_command_keeps_worker_evaluation_and_revision_guard() {
    let mut app = fixture_app();
    actor_model_from_renderer(&mut app);
    let filter = app.seg_objects.viewport_filter_state();

    assert!(app.submit_native_object_filter_at("viewport-1", 4, &filter));
    let intent = take_one(&mut app, "viewer.viewports.objects.filter.set");
    assert_eq!(intent.params["if_presentation_revision"], 4);
    assert!(intent.params.get("mode").is_some());
}
