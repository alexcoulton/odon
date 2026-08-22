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
    assert_eq!(channels.legacy_rx.len(), 0);

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
    let legacy = channels
        .legacy_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("resource-loading view apply remains an explicit hybrid route");
    assert_eq!(legacy.command.method(), "project.views.apply");
    legacy.reply.send(Ok(json!({"applied":true}))).unwrap();
    apply_resource_view_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
}
