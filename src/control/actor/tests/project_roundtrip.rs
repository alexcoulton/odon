use super::*;
#[test]
fn project_open_edit_and_save_roundtrip_without_a_ui_frame() {
    let unique = format!(
        "odon-actor-project-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let input = std::env::temp_dir().join(format!("{unique}-input.json"));
    let output = std::env::temp_dir().join(format!("{unique}-output.json"));
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let project_file = json!({
        "version": 6,
        "config": {
            "rois": [{
                "id": "fixture",
                "path": fixture,
                "display_name": "Fixture",
                "meta": {"cohort": "A"}
            }],
            "default_dataset": "images"
        },
        "state": {
            "browser": {"selected": [], "focused": null},
            "view_presets": [{"name": "Overview", "spec": {"visible_channels": ["DAPI"]}}]
        }
    });
    fs::write(&input, serde_json::to_string_pretty(&project_file).unwrap()).unwrap();

    let channels = spawn_test_actor();
    let (open, open_rx) = request("project.open", json!({"path":input}));
    channels.request_tx.send(open).unwrap();
    let opened = open_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("project open completes without UI drain")
        .unwrap();
    assert_eq!(opened["project"]["roi_count"], 1);
    assert_eq!(opened["project"]["view_count"], 1);
    assert_eq!(channels.legacy_rx.len(), 0);

    let (update, update_rx) = request(
        "project.rois.update",
        json!({"target_id":"fixture","changes":{"display_name":"Edited in actor"}}),
    );
    channels.request_tx.send(update).unwrap();
    update_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (resource, resource_rx) = request(
        "data.resources.register",
        json!({
            "resource_id": "project-resource",
            "uri": "file:///tmp/project-resource.zarr",
            "format": "ome-zarr",
            "ownership": "project",
            "coordinate_space": {"axes": ["y", "x"], "scale": [1.0, 1.0]},
        }),
    );
    channels.request_tx.send(resource).unwrap();
    resource_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("resource registration completes without UI drain")
        .unwrap();
    let (layer, layer_rx) = request(
        "viewer.layers.add",
        json!({
            "layer_id": "project-layer",
            "name": "Project labels",
            "kind": "labels",
            "data_resource_id": "project-resource",
            "ownership": "project",
        }),
    );
    channels.request_tx.send(layer).unwrap();
    layer_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("layer installation completes without UI drain")
        .unwrap();

    // Saving immediately after the layer reply must observe the layer transaction. Both
    // commands now share the actor mailbox; there is no cross-mailbox notification race.
    let (save, save_rx) = request("project.save_as", json!({"path":output}));
    channels.request_tx.send(save).unwrap();
    let saved = save_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("project save completes without UI drain")
        .unwrap();
    assert_eq!(saved["saved"], true);
    assert_eq!(channels.legacy_rx.len(), 0);
    let (recent, recent_rx) = request("app.recent_projects.list", json!({}));
    channels.request_tx.send(recent).unwrap();
    let recent = recent_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        recent["projects"][0]["path"].as_str(),
        Some(output.to_string_lossy().as_ref())
    );
    assert_eq!(
        recent["projects"][1]["path"].as_str(),
        Some(input.to_string_lossy().as_ref())
    );

    let written: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(written["version"], 6);
    assert_eq!(
        written["config"]["rois"][0]["display_name"],
        "Edited in actor"
    );
    assert_eq!(written["state"]["view_presets"][0]["name"], "Overview");
    assert_eq!(
        written["config"]["control_resources"][0]["resource_id"],
        "project-resource"
    );
    assert_eq!(
        written["config"]["control_layers"][0]["layer_id"],
        "project-layer"
    );
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.project.load_generation, 1);
    assert_eq!(
        projection.project.saved_path.as_deref(),
        Some(output.as_path())
    );
    assert!(!projection.project.dirty);

    let _ = fs::remove_file(input);
    let _ = fs::remove_file(output);
}
