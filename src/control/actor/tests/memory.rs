use super::*;

fn open_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
}

#[test]
fn tile_loading_policy_commits_without_a_frame_and_accepts_renderer_observations() {
    let channels = spawn_test_actor();
    open_fixture(&channels);
    let (set, set_rx) = request(
        "memory.tiles.set",
        json!({
            "workers":1,
            "prefetch_mode":"off",
            "prefetch_aggressiveness":"aggressive",
            "prefer_pinned_finer_levels":false,
        }),
    );
    channels.request_tx.send(set).unwrap();
    let policy = set_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(policy["workers"], 1);
    assert_eq!(policy["prefetch_mode"], "off");
    assert_eq!(policy["prefetch_aggressiveness"], "aggressive");
    assert_eq!(policy["prefer_pinned_finer_levels"], false);
    assert_eq!(policy["presentation_pending"], true);
    let generation = policy["generation"].as_u64().unwrap();

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.tile_loading_policy.workers(), 1);
    assert_eq!(
        projection.tile_loading_policy.prefetch_mode(),
        crate::model::TilePrefetchMode::Off
    );

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let mut observed = workspace_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    observed.as_object_mut().unwrap().insert(
        "tile_loading_observation".to_string(),
        json!({
            "cache":{"loaded":7,"capacity":256,"in_flight":2},
            "target_level":3,
            "realized_generation":generation,
            "status":"Tile loading policy realized by renderer.",
        }),
    );
    channels
        .model_tx
        .send(ActorModelUpdate::RendererWorkspaceObserved {
            workspace: observed,
            based_on_projection_revision: projection.revision,
        })
        .unwrap();

    let (get, get_rx) = request("memory.tiles.get", json!({}));
    channels.request_tx.send(get).unwrap();
    let current = get_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(current["generation"], generation);
    assert_eq!(current["realized_generation"], generation);
    assert_eq!(current["presentation_pending"], false);
    assert_eq!(current["cache"]["loaded"], 7);
    assert_eq!(current["cache"]["in_flight"], 2);
    assert_eq!(current["target_level"], 3);
    assert_eq!(channels.legacy_rx.len(), 0);
}

#[test]
fn memory_pin_loads_shared_level_data_and_unpins_without_a_frame() {
    let channels = spawn_test_actor();
    open_fixture(&channels);
    let (state, state_rx) = request("memory.get", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let level = state["levels"].as_array().unwrap().len() - 1;

    let (pin, pin_rx) = request(
        "memory.pin",
        json!({"level":level,"channels":[0,1],"force":true}),
    );
    channels.request_tx.send(pin).unwrap();
    let pinned = pin_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(pinned["started"], true);
    assert_eq!(pinned["completed"], true);
    assert_eq!(pinned["level"], level);
    assert_eq!(pinned["memory"]["running"], false);
    assert_eq!(pinned["memory"]["levels"][level]["status"], "loaded");
    assert_eq!(pinned["memory"]["levels"][level]["channels_loaded"], 2);
    assert!(pinned["memory"]["pinned_bytes"].as_u64().unwrap() > 0);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.pinned_levels.len(), 1);
    assert_eq!(projection.pinned_levels[0].level(), level);
    assert_eq!(projection.pinned_levels[0].channels_loaded(), 2);

    let (unpin, unpin_rx) = request("memory.unpin", json!({"level":level}));
    channels.request_tx.send(unpin).unwrap();
    let unpinned = unpin_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(unpinned["unloaded"], true);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert!(projection.pinned_levels.is_empty());
    assert_eq!(channels.legacy_rx.len(), 0);
}

#[test]
fn unpinning_supersedes_a_late_memory_pin_install() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, _) = OmeZarrDataset::open_local(&fixture).unwrap();
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let spec = model
        .prepare_memory_pin(&json!({"level":0,"channels":[0],"force":true}))
        .unwrap();
    model.dispatch("memory.unpin", &json!({"level":0}));
    let stale =
        ControlPinnedLevelResource::new(0, 1, 1, std::iter::once((0, 0)).collect(), vec![1]);
    assert!(
        model
            .install_memory_pin(&spec, Arc::new(stale), None)
            .is_none()
    );
    assert!(model.pinned_level_resources().is_empty());
}
