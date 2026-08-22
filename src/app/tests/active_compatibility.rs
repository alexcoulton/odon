use super::*;
#[test]
fn background_actor_preserves_active_view_compatibility_methods() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open OME-Zarr fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let mut app = OmeZarrViewerApp::new_runtime(
        &egui::Context::default(),
        false,
        dataset,
        store,
        AutoContrastSettings {
            enabled_on_open: false,
            ..AutoContrastSettings::default()
        },
    );

    assert_eq!(
        actor_call(&mut model, "viewer.channels.list", serde_json::json!({})),
        serde_json::json!({"mode":"single","channels":app.control_channel_snapshot()})
    );
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.channels.list_visible",
            serde_json::json!({}),
        ),
        serde_json::json!({"mode":"single","channels":app.control_visible_channel_snapshot()})
    );
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.channels.get_active",
            serde_json::json!({}),
        ),
        serde_json::json!({"mode":"single","active_channel":app.control_active_channel_snapshot()})
    );

    let visible = serde_json::json!({"channels":["CD3","PanCK"],"mode":"only"});
    let renderer = app.control_set_visible_channels(&visible);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_visible", visible),
        serde_json::json!({"mode":"single","result":renderer})
    );
    let active = serde_json::json!({"channel":"PanCK"});
    let renderer = app.control_set_active_channel(&active);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_active", active),
        serde_json::json!({"mode":"single","result":renderer})
    );
    let contrast = serde_json::json!({"channel":"PanCK","min":100.0,"max":1000.0});
    let renderer = app.control_set_channel_contrast(&contrast);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_contrast", contrast),
        serde_json::json!({"mode":"single","contrast":renderer})
    );

    let note = serde_json::json!({"channel":"PanCK","note":"epithelial marker"});
    let renderer = app.control_set_channel_note(&note);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_note", note),
        renderer
    );
    let transform = serde_json::json!({
        "channel":"PanCK",
        "offset_world":[4.0,-2.0],
        "scale":[1.2,0.8],
        "rotation_rad":0.25,
    });
    let renderer = app.control_set_channel_transform(&transform);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_transform", transform),
        renderer
    );
    let selector = serde_json::json!({"channel":"PanCK"});
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.channels.get_transform",
            selector.clone()
        ),
        app.control_get_channel_transform(&selector)
    );

    let order = serde_json::json!({"channels":[4,3,2,1,0],"mode":"exact"});
    let renderer = app.control_set_channel_order(&order);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_order", order),
        renderer
    );
    let presentation = serde_json::json!({"search":"CD","sort":"visible_first"});
    let renderer = app.control_set_channel_presentation(&presentation);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.presentation.set", presentation),
        renderer
    );
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.channels.presentation.get",
            serde_json::json!({})
        ),
        app.control_channel_presentation_json()
    );
    let group = serde_json::json!({
        "channels":["CD3","PanCK"],
        "name":"Comparison markers",
        "color_rgb":[20,40,60],
    });
    let renderer = app.control_set_channel_group(&group);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.set_group", group),
        serde_json::json!({"mode":"single","result":renderer})
    );
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.channels.list_groups",
            serde_json::json!({})
        ),
        serde_json::json!({
            "mode":"single",
            "groups":app.control_channel_groups_snapshot(),
        })
    );

    let renderer = app.control_reset_channel_transform(&selector);
    assert_eq!(
        actor_call(&mut model, "viewer.channels.reset_transform", selector),
        renderer
    );

    let camera = serde_json::json!({"center_x":123.0,"center_y":234.0,"zoom":2.5});
    let mut renderer =
        serde_json::json!({"mode":"single","camera":app.control_set_camera(&camera)});
    let mut actor = actor_call(&mut model, "viewer.camera.set", camera);
    renderer["camera"]
        .as_object_mut()
        .unwrap()
        .remove("viewport");
    actor["camera"].as_object_mut().unwrap().remove("viewport");
    assert_eq!(actor, renderer);

    let plane = serde_json::json!({"mode":"xy","slice":99});
    let renderer = app.control_set_plane(&plane);
    assert_eq!(
        actor_call(&mut model, "viewer.planes.set", plane),
        serde_json::json!({"mode":"single","result":renderer})
    );
    let renderer = app.control_step_plane(&serde_json::json!({"step": 1}), true);
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.planes.next",
            serde_json::json!({"step":1}),
        ),
        serde_json::json!({"mode":"single","result":renderer})
    );

    let renderer = app.control_set_smooth_pixels(&serde_json::json!({"smooth":false}));
    assert_eq!(
        actor_call(
            &mut model,
            "viewer.rendering.set_smooth_pixels",
            serde_json::json!({"smooth":false}),
        ),
        serde_json::json!({"mode":"single","result":renderer})
    );

    let panels = serde_json::json!({"left":false,"right":true});
    let renderer = app.control_set_side_panels(&panels);
    assert_eq!(
        actor_call(&mut model, "viewer.panels.set", panels),
        serde_json::json!({"mode":"single","result":renderer})
    );
    assert_eq!(
        actor_call(&mut model, "viewer.panels.get", serde_json::json!({})),
        serde_json::json!({"mode":"single","panels":app.control_side_panels_snapshot()})
    );
}
