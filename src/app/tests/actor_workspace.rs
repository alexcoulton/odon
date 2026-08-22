use super::*;
#[test]
fn background_actor_preserves_workspace_topology_and_link_transactions() {
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

    let create_params = serde_json::json!({
        "source_viewport_id": "viewport-1",
        "title": "Comparison",
        "layout": "horizontal",
        "ratio": 0.65,
    });
    let renderer_create = app.control_create_viewport(&create_params);
    let actor_create = actor_call(&mut model, "viewer.viewports.clone", create_params);
    assert_eq!(actor_create["viewport_id"], renderer_create["viewport_id"]);
    assert_eq!(
        workspace_topology(&actor_create["workspace"]),
        workspace_topology(&renderer_create["workspace"])
    );
    let right = actor_create["viewport_id"].as_str().unwrap().to_string();

    let rename_params = serde_json::json!({"viewport_id": right, "title": "Renamed comparison"});
    let renderer_rename = app.control_rename_viewport(&rename_params);
    let actor_rename = actor_call(&mut model, "viewer.viewports.rename", rename_params.clone());
    assert_eq!(actor_rename, renderer_rename);

    let ids = vec!["viewport-1".to_string(), right.clone()];
    let layout_params = serde_json::json!({
        "layout": "vertical",
        "ratio": 0.6,
        "viewports": ids,
    });
    let renderer_layout = app.control_set_viewport_layout(&layout_params);
    let actor_layout = actor_call(&mut model, "viewer.workspace.layout.set", layout_params);
    assert_eq!(actor_layout, renderer_layout);

    let unlink_params = serde_json::json!({"camera": false, "plane": false, "selection": true});
    let renderer_unlink = app.control_set_viewport_links(&unlink_params);
    let actor_unlink = actor_call(&mut model, "viewer.viewport_links.set", unlink_params);
    assert_eq!(actor_unlink["changed"], renderer_unlink["changed"]);
    assert_eq!(actor_unlink["links"], renderer_unlink["links"]);
    assert_eq!(
        actor_unlink["affected_viewport_ids"],
        renderer_unlink["affected_viewport_ids"]
    );
    assert_eq!(
        workspace_topology(&actor_unlink["workspace"]),
        workspace_topology(&renderer_unlink["workspace"])
    );

    let camera_params = serde_json::json!({
        "viewport_id": "viewport-1",
        "center_world_lvl0": [123.0, 234.0],
        "zoom": 2.5,
    });
    let mut renderer_camera = app.control_set_viewport_camera(&camera_params);
    let mut actor_camera = actor_call(&mut model, "viewer.viewports.camera.set", camera_params);
    renderer_camera["result"]
        .as_object_mut()
        .unwrap()
        .remove("viewport");
    actor_camera["result"]
        .as_object_mut()
        .unwrap()
        .remove("viewport");
    assert_eq!(actor_camera, renderer_camera);

    let relink_params = serde_json::json!({"camera": true, "plane": true, "selection": true});
    let renderer_relink = app.control_set_viewport_links(&relink_params);
    let actor_relink = actor_call(&mut model, "viewer.viewport_links.set", relink_params);
    assert_eq!(actor_relink["changed"], renderer_relink["changed"]);
    assert_eq!(actor_relink["links"], renderer_relink["links"]);
    assert_eq!(
        actor_relink["affected_viewport_ids"],
        renderer_relink["affected_viewport_ids"]
    );
    assert_eq!(
        workspace_topology(&actor_relink["workspace"]),
        workspace_topology(&renderer_relink["workspace"])
    );

    let renderer_swap = app.control_swap_viewports();
    let actor_swap = actor_call(&mut model, "viewer.workspace.swap", serde_json::json!({}));
    assert_eq!(
        workspace_topology(&actor_swap["workspace"]),
        workspace_topology(&renderer_swap["workspace"])
    );

    let remove_params = serde_json::json!({"viewport_id": "viewport-1"});
    let renderer_remove = app.control_remove_viewport(&remove_params);
    let actor_remove = actor_call(&mut model, "viewer.viewports.remove", remove_params);
    assert_eq!(actor_remove["removed"], renderer_remove["removed"]);
    assert_eq!(actor_remove["viewport_id"], renderer_remove["viewport_id"]);
    assert_eq!(
        workspace_topology(&actor_remove["workspace"]),
        workspace_topology(&renderer_remove["workspace"])
    );
}
