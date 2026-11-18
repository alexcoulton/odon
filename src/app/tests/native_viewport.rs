use super::*;
use odon::control::ControlCommand;

fn actor_model_fixture(app: &mut OmeZarrViewerApp) -> odon::model::AppModel {
    let mut model = odon::model::AppModel::project();
    model
        .bootstrap_dataset(&app.dataset)
        .expect("fixture installs the actor dataset");
    let projection = model
        .render_workspace_snapshot()
        .expect("installed dataset has an actor projection");
    app.apply_control_actor_workspace_projection(&projection)
        .expect("fixture consumes the canonical actor projection");
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
fn native_channel_controls_queue_before_first_projection_without_mutating_renderer_state() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 0;
    let visible_before = app.channels[0].visible;
    let window_before = app.channels[0].window;

    <OmeZarrViewerApp as channels_panel::ChannelListHost>::set_channel_visible(
        &mut app,
        0,
        !visible_before,
    );
    let visibility = take_one(&mut app, "viewer.viewports.channels.set_visible");
    assert_eq!(visibility.params["if_presentation_revision"], 1);
    assert_eq!(app.channels[0].visible, visible_before);

    app.apply_channel_window_to_indices(&[0], 10.0, 80.0);
    let contrast = take_one(&mut app, "viewer.viewports.layers.state.replace");
    assert_eq!(contrast.params["if_presentation_revision"], 1);
    assert_eq!(app.channels[0].window, window_before);

    let scale_bar_before = app.show_scale_bar;
    assert!(app.submit_native_scale_bar_visibility(!scale_bar_before));
    let rendering = take_one(&mut app, "viewer.viewports.rendering.set");
    assert_eq!(rendering.params["if_presentation_revision"], 1);
    assert_eq!(app.show_scale_bar, scale_bar_before);
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
    let mut model = actor_model_fixture(&mut app);
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
    let mut model = actor_model_fixture(&mut app);
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
    let mut model = actor_model_fixture(&mut app);
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
    actor_model_fixture(&mut app);
    let filter = app.seg_objects.viewport_filter_state();

    assert!(app.submit_native_object_filter_at("viewport-1", 4, &filter));
    let intent = take_one(&mut app, "viewer.viewports.objects.filter.set");
    assert_eq!(intent.params["if_presentation_revision"], 4);
    assert!(intent.params.get("mode").is_some());
}

#[test]
fn late_projection_of_an_earlier_pan_step_does_not_rewind_the_gesture() {
    let mut app = fixture_app();
    let mut model = actor_model_fixture(&mut app);

    // First pan step: applied locally and reported straight away.
    app.camera.center_world_lvl0 = egui::pos2(10.0, 20.0);
    app.record_native_camera_gesture();
    // Two more steps happen while that command is still in flight; nothing else is queued.
    app.camera.center_world_lvl0 = egui::pos2(30.0, 40.0);
    app.record_native_camera_gesture();
    app.camera.center_world_lvl0 = egui::pos2(50.0, 60.0);
    app.record_native_camera_gesture();

    let first = take_one(&mut app, "viewer.viewports.camera.set");
    assert_eq!(first.params["center_world_lvl0"][0], 10.0);
    assert!(first.params.get("if_navigation_revision").is_none());
    replay(&mut model, first);
    let first_projection = model.render_workspace_snapshot().unwrap();

    // The projection of the first step arrives late and must not rewind the camera.
    app.apply_control_actor_workspace_projection(&first_projection)
        .expect("stale projection still applies");
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(50.0, 60.0));

    // Only the newest position is sent next, and the actor accepts it even though the renderer
    // never reported the intermediate navigation revision.
    app.flush_native_camera_hold();
    let latest = take_one(&mut app, "viewer.viewports.camera.set");
    assert_eq!(latest.params["center_world_lvl0"][0], 50.0);
    replay(&mut model, latest);
    let latest_projection = model.render_workspace_snapshot().unwrap();
    app.apply_control_actor_workspace_projection(&latest_projection)
        .expect("current projection applies");
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(50.0, 60.0));
}

#[test]
fn navigation_only_projection_moves_cameras_without_reapplying_the_workspace() {
    let mut app = fixture_app();
    let _model = actor_model_fixture(&mut app);
    let viewport_id = app
        .viewport_workspace
        .as_ref()
        .unwrap()
        .active_id()
        .as_str()
        .to_string();
    let navigation = |x: f32, revision: u64| {
        vec![odon::model::ProjectedViewportNavigation {
            viewport_id: viewport_id.clone(),
            navigation_revision: revision,
            camera: odon::model::ProjectedCamera {
                center_world_lvl0: [x, 7.0],
                zoom_screen_per_lvl0_px: 3.0,
            },
        }]
    };

    assert!(app.apply_control_actor_navigation(&navigation(100.0, 5)));
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(100.0, 7.0));
    assert_eq!(app.camera.zoom_screen_per_lvl0_px, 3.0);
    let workspace = app.viewport_workspace.as_ref().unwrap();
    assert_eq!(workspace.active().navigation_revision, 5);
    assert_eq!(
        workspace.active().state.camera.center_world_lvl0,
        egui::pos2(100.0, 7.0)
    );

    // A pan gesture in progress keeps owning the camera on this path too.
    app.camera.center_world_lvl0 = egui::pos2(500.0, 7.0);
    app.record_native_camera_gesture();
    assert!(app.apply_control_actor_navigation(&navigation(100.0, 6)));
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(500.0, 7.0));
    assert_eq!(
        app.viewport_workspace
            .as_ref()
            .unwrap()
            .active()
            .navigation_revision,
        6
    );

    // A different viewport set cannot be realized from navigation alone.
    let mut other = navigation(1.0, 7);
    other[0].viewport_id = "viewport-unknown".to_string();
    assert!(!app.apply_control_actor_navigation(&other));
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(500.0, 7.0));
}
