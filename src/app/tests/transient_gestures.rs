use super::*;
#[test]
fn activating_or_removing_a_viewport_cancels_transient_edit_gestures() {
    let mut app = fixture_actor_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "layout": "horizontal",
        }),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    app.selection_rect_start_world = Some(egui::pos2(1.0, 2.0));
    app.selection_rect_current_world = Some(egui::pos2(3.0, 4.0));
    app.selection_lasso_world.push(egui::pos2(5.0, 6.0));
    app.drawing_mask_polygon.push(egui::pos2(7.0, 8.0));

    let activated = app.actor_command(
        "viewer.viewports.set_active",
        serde_json::json!({
            "viewport_id": left,
        }),
    );
    assert_eq!(activated["changed"], true);
    assert!(app.selection_rect_start_world.is_none());
    assert!(app.selection_rect_current_world.is_none());
    assert!(app.selection_lasso_world.is_empty());
    assert!(app.drawing_mask_polygon.is_empty());

    app.selection_rect_start_world = Some(egui::pos2(1.0, 2.0));
    app.drawing_mask_polygon.push(egui::pos2(7.0, 8.0));
    let removed = app.actor_command(
        "viewer.viewports.remove",
        serde_json::json!({
            "viewport_id": right,
        }),
    );
    assert_eq!(removed["removed"], true);
    assert!(app.selection_rect_start_world.is_none());
    assert!(app.drawing_mask_polygon.is_empty());
}

#[test]
fn cancelling_actor_owned_layer_gestures_restores_preview_state() {
    let mut app = fixture_app();
    app.control_actor_workspace_revision = 1;
    app.layer_move = Some(LayerMoveState {
        targets: vec![LayerOffsetEntry {
            layer: LayerId::Channel(0),
            offset_world: egui::Vec2::ZERO,
        }],
        actor_scope: Some(("viewport-1".to_string(), 1)),
    });
    app.channel_offsets_world[0] = egui::vec2(10.0, -4.0);
    assert!(app.control_projection_gesture_active());
    app.cancel_viewport_transient_gestures();
    assert_eq!(app.channel_offsets_world[0], egui::Vec2::ZERO);
    assert!(!app.control_projection_gesture_active());

    app.layer_transform = Some(LayerTransformState {
        layer: LayerId::Channel(0),
        kind: LayerTransformKind::Scale,
        start_offset_world: egui::Vec2::ZERO,
        start_scale: egui::Vec2::splat(1.0),
        start_rotation_rad: 0.0,
        pivot_screen: egui::Pos2::ZERO,
        start_pointer_screen: egui::Pos2::ZERO,
        start_angle_rad: 0.0,
        start_len_screen: 1.0,
        actor_scope: Some(("viewport-1".to_string(), 1)),
    });
    app.channel_scales[0] = egui::vec2(2.0, 3.0);
    app.channel_rotations_rad[0] = 0.5;
    app.cancel_viewport_transient_gestures();
    assert_eq!(app.channel_scales[0], egui::Vec2::splat(1.0));
    assert_eq!(app.channel_rotations_rad[0], 0.0);
}
