use super::*;
#[test]
fn saved_view_capture_and_apply_complete_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let left = workspace_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    for (method, params) in [
        (
            "viewer.viewports.channels.set_visible",
            json!({"viewport_id":left,"channels":[2],"mode":"only"}),
        ),
        (
            "viewer.viewports.channels.set_active",
            json!({"viewport_id":left,"channel":2}),
        ),
        (
            "viewer.viewports.camera.set",
            json!({"viewport_id":left,"center_world_lvl0":[111.0,222.0],"zoom_screen_per_lvl0_px":2.25}),
        ),
    ] {
        let (command, reply) = request(method, params);
        channels.request_tx.send(command).unwrap();
        reply.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
    }
    let (capture, capture_rx) = request(
        "project.views.capture",
        json!({"name":"Actor capture","viewport_id":left}),
    );
    channels.request_tx.send(capture).unwrap();
    let captured = capture_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("saved view capture completes without a UI frame")
        .unwrap();
    assert_eq!(captured["captured"], true);
    assert_eq!(captured["view"]["spec"]["channel_ref"]["label"], "PanCK");

    let (change, change_rx) = request(
        "viewer.viewports.channels.set_visible",
        json!({"viewport_id":left,"channels":[0],"mode":"only"}),
    );
    channels.request_tx.send(change).unwrap();
    change_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (move_camera, move_camera_rx) = request(
        "viewer.viewports.camera.set",
        json!({"viewport_id":left,"center_world_lvl0":[5.0,6.0],"zoom_screen_per_lvl0_px":0.5}),
    );
    channels.request_tx.send(move_camera).unwrap();
    move_camera_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (apply, apply_rx) = request("project.views.apply", json!({"name":"Actor capture"}));
    channels.request_tx.send(apply).unwrap();
    assert_eq!(
        apply_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("saved view apply completes without a UI frame")
            .unwrap()["applied"],
        true
    );
    let (viewport, viewport_rx) = request("viewer.viewports.get", json!({"viewport_id":left}));
    channels.request_tx.send(viewport).unwrap();
    let viewport = viewport_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        viewport["channels"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|channel| channel["visible"] == true)
            .map(|channel| channel["index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(
        viewport["camera"]["center_world_lvl0"],
        json!([111.0, 222.0])
    );

    let (create_resource_view, create_resource_view_rx) = request(
        "project.views.create",
        json!({
            "name":"Needs object load",
            "spec":{"segmentation_source":"geoparquet","fill_cells":true},
        }),
    );
    channels.request_tx.send(create_resource_view).unwrap();
    create_resource_view_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (apply_resource_view, apply_resource_view_rx) =
        request("project.views.apply", json!({"name":"Needs object load"}));
    channels.request_tx.send(apply_resource_view).unwrap();
    assert_eq!(
        apply_resource_view_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap_err()
            .kind,
        ControlErrorKind::ResourceNotFound
    );
}

#[test]
fn saved_view_resource_load_is_one_actor_worker_transaction() {
    let channels = spawn_test_actor_with_objects();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let object_path =
        std::env::temp_dir().join(format!("odon-saved-view-objects-{}", std::process::id()));
    let mut roi = ProjectRoi {
        id: "roi-a".to_string(),
        display_name: Some("ROI A".to_string()),
        segpath: Some(object_path),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        dataset_path,
    ));
    let source_key = roi.source_key().unwrap();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(ProjectModelSnapshot {
            config: ProjectConfig {
                rois: vec![roi.clone()],
                ..ProjectConfig::default()
            },
            rois: vec![roi],
            focused_source_key: Some(source_key.clone()),
            selected_source_keys: vec![source_key],
            saved_path: Some(std::env::temp_dir().join("saved-view-project.odon.json")),
            config_generation: 1,
            ..ProjectModelSnapshot::default()
        }))
        .unwrap();
    let (open, open_response) = request("project.rois.open", json!({"roi":"roi-a"}));
    channels.request_tx.send(open).unwrap();
    open_response
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (clear, clear_response) = request("viewer.objects.source.clear", json!({}));
    channels.request_tx.send(clear).unwrap();
    clear_response
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (create, create_response) = request(
        "project.views.create",
        json!({
            "name":"Reload objects",
            "spec":{"segmentation_source":"geoparquet","fill_cells":true},
        }),
    );
    channels.request_tx.send(create).unwrap();
    create_response
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (apply, apply_response) = request("project.views.apply", json!({"name":"Reload objects"}));
    channels.request_tx.send(apply).unwrap();
    assert_eq!(
        apply_response
            .recv_timeout(Duration::from_secs(3))
            .expect("saved-view worker transaction")
            .unwrap()["applied"],
        true
    );
    let (objects, objects_response) = request("viewer.objects.get_state", json!({}));
    channels.request_tx.send(objects).unwrap();
    assert_eq!(
        objects_response
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["state"]["object_count"],
        2
    );
}
