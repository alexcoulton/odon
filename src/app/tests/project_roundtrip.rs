use super::*;
#[test]
fn layer_transforms_order_masks_and_ui_roundtrip_through_project_state() {
    let mut app = fixture_app();
    app.project_space.add_roi_source(app.dataset.source.clone());
    let mut mask = MaskLayer {
        id: 17,
        name: "Review exclusion".to_string(),
        visible: false,
        opacity: 0.45,
        width_screen_px: 3.0,
        display_mode: MaskDisplayMode::TranslucentFill,
        color_rgb: [12, 34, 56],
        offset_world: egui::vec2(8.0, -4.0),
        editable: true,
        polygons_world: Vec::new(),
        raster_display: None,
        source_geojson: None,
    };
    mask.add_closed_polygon(vec![
        egui::pos2(1.0, 2.0),
        egui::pos2(11.0, 2.0),
        egui::pos2(11.0, 12.0),
    ]);
    app.mask_layers.push(mask);
    app.next_mask_layer_id = 18;
    app.mask_layers_project_dirty = true;
    app.rebuild_layer_orders();

    app.channel_offsets_world[1] = egui::vec2(3.0, 5.0);
    app.channel_scales[1] = egui::vec2(1.25, 0.75);
    app.channel_rotations_rad[1] = 0.25;
    app.channels[1].note = "registration reference".to_string();
    app.channels[1].visible = true;
    app.active_layer = LayerId::Mask(17);
    app.overlay_layer_order
        .retain(|id| *id != LayerId::Mask(17));
    app.overlay_layer_order.insert(0, LayerId::Mask(17));
    app.show_left_panel = false;
    app.show_right_panel = true;
    app.smooth_pixels = false;

    app.push_layer_offsets_undo_snapshot(&[LayerId::Channel(1), LayerId::Mask(17)]);
    app.channel_offsets_world[1] = egui::vec2(30.0, 50.0);
    app.mask_layers[0].offset_world = egui::vec2(80.0, -40.0);
    assert!(app.undo_last_edit());
    assert_eq!(app.channel_offsets_world[1], egui::vec2(3.0, 5.0));
    assert_eq!(app.mask_layers[0].offset_world, egui::vec2(8.0, -4.0));
    assert!(!app.undo_last_edit(), "undo stack is exhausted");

    sync_complete_state_to_active_viewport_for_test(&mut app);
    let source = app.dataset.source.clone();
    let project = app.take_project_space();
    let view = project.roi_view_state(&source).expect("saved ROI view");
    assert_eq!(view.channel_order, app.channel_layer_order);
    assert_eq!(view.channels[1].offset_world, Some([3.0, 5.0]));
    assert_eq!(view.channels[1].scale, Some([1.25, 0.75]));
    assert_eq!(view.channels[1].rotation_rad, Some(0.25));
    assert_eq!(
        view.channels[1].note.as_deref(),
        Some("registration reference")
    );
    assert_eq!(
        view.overlay_order.first().map(String::as_str),
        Some("mask:17")
    );
    assert_eq!(view.overlay_visibility["mask:17"], false);
    assert_eq!(view.overlay_offsets_world["mask:17"], [8.0, -4.0]);

    let mut restored = fixture_app();
    restored.set_project_space(project);
    assert_eq!(restored.channel_offsets_world[1], egui::vec2(3.0, 5.0));
    assert_eq!(restored.channel_scales[1], egui::vec2(1.25, 0.75));
    assert_eq!(restored.channel_rotations_rad[1], 0.25);
    assert_eq!(restored.channels[1].note, "registration reference");
    assert_eq!(restored.active_layer, LayerId::Mask(17));
    assert_eq!(
        restored.overlay_layer_order.first(),
        Some(&LayerId::Mask(17))
    );
    assert_eq!(restored.mask_layers.len(), 1);
    assert!(!restored.mask_layers[0].visible);
    assert_eq!(restored.mask_layers[0].offset_world, egui::vec2(8.0, -4.0));
    assert_eq!(restored.mask_layers[0].polygons_world[0].len(), 4);
    assert!(!restored.show_left_panel);
    assert!(restored.show_right_panel);
    assert!(!restored.smooth_pixels);
}
