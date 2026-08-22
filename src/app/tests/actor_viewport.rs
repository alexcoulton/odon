use super::*;
#[test]
fn background_actor_preserves_migrated_viewport_control_results() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open OME-Zarr fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let settings = AutoContrastSettings {
        enabled_on_open: false,
        ..AutoContrastSettings::default()
    };
    let mut app =
        OmeZarrViewerApp::new_runtime(&egui::Context::default(), false, dataset, store, settings);
    let viewport_id = "viewport-1";

    let operations = [
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["CD3", "PanCK"],
                "mode": "only",
            }),
        ),
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["Collagen"],
                "mode": "show",
            }),
        ),
        (
            "viewer.viewports.channels.set_visible",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["CD3"],
                "mode": "hide",
            }),
        ),
        (
            "viewer.viewports.channels.set_active",
            serde_json::json!({"viewport_id": viewport_id, "channel": "PanCK"}),
        ),
        (
            "viewer.viewports.channels.set_color",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channel": "PanCK",
                "color_rgb": [12, 34, 56],
            }),
        ),
        (
            "viewer.viewports.channels.set_contrast",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channel": "PanCK",
                "min": 100.0,
                "max": 1000.0,
            }),
        ),
        (
            "viewer.viewports.channels.set_order",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": [4, 3, 2, 1, 0],
                "mode": "exact",
            }),
        ),
        (
            "viewer.viewports.channels.set_group",
            serde_json::json!({
                "viewport_id": viewport_id,
                "channels": ["PanCK", "Collagen"],
                "name": "Stroma",
                "color_rgb": [90, 80, 70],
            }),
        ),
        (
            "viewer.viewports.rendering.set",
            serde_json::json!({
                "viewport_id": viewport_id,
                "smooth_pixels": true,
                "show_scale_bar": false,
                "show_hud": false,
                "show_tile_debug": true,
            }),
        ),
        (
            "viewer.viewports.planes.set",
            serde_json::json!({"viewport_id": viewport_id, "mode": "xy", "slice": 99}),
        ),
    ];

    for (method, params) in operations {
        let renderer = match method {
            "viewer.viewports.channels.set_visible" => app.control_set_viewport_channels(&params),
            "viewer.viewports.channels.set_active" => {
                app.control_set_viewport_active_channel(&params)
            }
            "viewer.viewports.channels.set_color" => {
                app.control_set_viewport_channel_color(&params)
            }
            "viewer.viewports.channels.set_contrast" => {
                app.control_set_viewport_channel_contrast(&params)
            }
            "viewer.viewports.channels.set_order" => {
                app.control_set_viewport_channel_order(&params)
            }
            "viewer.viewports.channels.set_group" => {
                app.control_set_viewport_channel_group(&params)
            }
            "viewer.viewports.rendering.set" => app.control_set_viewport_rendering(&params),
            "viewer.viewports.planes.set" => app.control_set_viewport_plane(&params),
            _ => unreachable!(),
        };
        let actor = actor_call(&mut model, method, params);
        assert_eq!(
            actor["result"], renderer["result"],
            "{method} changed its public result after actor migration"
        );
        assert_eq!(actor["viewport_id"], renderer["viewport_id"], "{method}");
        assert_eq!(
            actor["navigation_revision"], renderer["navigation_revision"],
            "{method}"
        );
        assert_eq!(
            actor["presentation_revision"], renderer["presentation_revision"],
            "{method}"
        );
    }

    let actor_channels = actor_call(
        &mut model,
        "viewer.viewports.channels.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_channels =
        app.control_get_viewport_channels(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_channels["result"], renderer_channels["result"]);

    let actor_groups = actor_call(
        &mut model,
        "viewer.viewports.channels.list_groups",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_groups =
        app.control_get_viewport_channel_groups(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_groups["result"], renderer_groups["result"]);

    let actor_plane = actor_call(
        &mut model,
        "viewer.viewports.planes.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_plane =
        app.control_get_viewport_plane(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_plane["result"], renderer_plane["result"]);

    let actor_rendering = actor_call(
        &mut model,
        "viewer.viewports.rendering.get",
        serde_json::json!({"viewport_id": viewport_id}),
    );
    let renderer_rendering =
        app.control_get_viewport_rendering(&serde_json::json!({"viewport_id": viewport_id}));
    assert_eq!(actor_rendering["result"], renderer_rendering["result"]);
}
