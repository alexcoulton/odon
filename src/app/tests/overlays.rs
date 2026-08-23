use super::*;
#[test]
fn extended_overlay_presentation_roundtrips_per_viewport() {
    fn add_mask(app: &mut OmeZarrViewerApp) {
        app.mask_layers.push(MaskLayer {
            id: 41,
            name: "Comparison mask".to_string(),
            visible: true,
            opacity: 0.5,
            width_screen_px: 2.0,
            display_mode: MaskDisplayMode::OutlineOnly,
            color_rgb: [255, 255, 255],
            offset_world: egui::Vec2::ZERO,
            editable: false,
            polygons_world: Vec::new(),
            raster_display: None,
            source_geojson: None,
        });
        app.rebuild_layer_orders();
    }

    let mut app = fixture_app();
    add_mask(&mut app);
    app.mask_layers[0].opacity = 0.2;
    app.mask_layers[0].display_mode = MaskDisplayMode::TranslucentFill;
    app.mask_layers[0].color_rgb = [10, 20, 30];
    app.cell_points.visible = true;
    app.cell_points.style.radius_screen_px = 3.0;
    let mut app = ActorAppFixture::new(app);
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.create",
        serde_json::json!({"layout": "horizontal"}),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    app.mask_layers[0].opacity = 0.8;
    app.mask_layers[0].display_mode = MaskDisplayMode::FilledPreview;
    app.mask_layers[0].color_rgb = [200, 100, 50];
    app.cell_points.style.radius_screen_px = 9.0;

    app.sync_current_view_state_into_project_space();
    let saved = app
        .project_space
        .roi_view_state(&app.dataset.source)
        .cloned()
        .unwrap();
    let encoded = serde_json::to_value(saved).unwrap();
    let decoded: ProjectRoiViewState = serde_json::from_value(encoded).unwrap();

    let mut restored = fixture_app();
    add_mask(&mut restored);
    restored
        .project_space
        .set_roi_view_state(&restored.dataset.source, decoded);
    restored.apply_view_state_from_project_space();
    let workspace = restored.viewport_workspace.as_ref().unwrap();
    let left = &workspace
        .get(&ViewportId::new(left).unwrap())
        .unwrap()
        .state;
    let right = &workspace
        .get(&ViewportId::new(right).unwrap())
        .unwrap()
        .state;
    assert!((left.masks[0].opacity - 0.2).abs() < 1e-6);
    assert_eq!(left.masks[0].display_mode, MaskDisplayMode::TranslucentFill);
    assert_eq!(left.masks[0].color_rgb, [10, 20, 30]);
    assert!((left.cell_points_style.radius_screen_px - 3.0).abs() < 1e-6);
    assert!((right.masks[0].opacity - 0.8).abs() < 1e-6);
    assert_eq!(right.masks[0].display_mode, MaskDisplayMode::FilledPreview);
    assert_eq!(right.masks[0].color_rgb, [200, 100, 50]);
    assert!((right.cell_points_style.radius_screen_px - 9.0).abs() < 1e-6);
}
