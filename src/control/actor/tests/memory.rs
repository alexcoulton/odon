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
            "cache_mode":"custom",
            "cache_budget_bytes":268435456,
            "channel_history":"current_only",
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
    assert_eq!(policy["cache_mode"], "custom");
    assert_eq!(policy["cache_budget_bytes"], 268435456);
    assert_eq!(policy["channel_history"], "current_only");
    assert_eq!(policy["presentation_pending"], true);
    let generation = policy["generation"].as_u64().unwrap();

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.tile_loading_policy.workers(), 1);
    assert_eq!(
        projection.tile_loading_policy.prefetch_mode(),
        crate::model::TilePrefetchMode::Off
    );

    let observed = json!({
        "tile_loading_observation": {
            "cache":{
                "loaded":7,
                "capacity":32768,
                "in_flight":2,
                "pending_cpu_bytes":1048576,
                "uploaded_texture_bytes":2097152,
                "effective_budget_bytes":268435456
            },
            "target_level":3,
            "realized_generation":generation,
            "status":"Tile loading policy realized by renderer.",
        },
    });
    channels
        .model_tx
        .send(ActorModelUpdate::RendererObservation {
            observation: observed,
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
    assert_eq!(current["cache"]["pending_cpu_bytes"], 1048576);
    assert_eq!(current["cache"]["uploaded_texture_bytes"], 2097152);
    assert_eq!(current["cache"]["effective_budget_bytes"], 268435456);
    assert_eq!(current["target_level"], 3);
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
    assert_eq!(projection.memory_state["running"], false);
    assert_eq!(projection.memory_state["levels"][level]["status"], "loaded");
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
    assert_eq!(
        projection.memory_state["levels"][level]["status"],
        "unloaded"
    );
    assert!(projection.pinned_levels.is_empty());
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
    assert_eq!(model.memory_projection_state()["running"], true);
    model.dispatch("memory.unpin", &json!({"level":0}));
    assert_eq!(model.memory_projection_state()["running"], false);
    let stale =
        ControlPinnedLevelResource::new(0, 1, 1, std::iter::once((0, 0)).collect(), vec![1]);
    assert!(
        model
            .install_memory_pin(&spec, Arc::new(stale), None)
            .is_none()
    );
    assert!(model.pinned_level_resources().is_empty());
}

#[test]
fn cancelling_memory_pin_rejects_its_late_result() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, _) = OmeZarrDataset::open_local(&fixture).unwrap();
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let spec = model
        .prepare_memory_pin(&json!({"level":0,"channels":[0],"force":true}))
        .unwrap();
    assert!(model.cancel_memory_pin(&spec, "cancelled by test"));
    let stale =
        ControlPinnedLevelResource::new(0, 1, 1, std::iter::once((0, 0)).collect(), vec![1]);
    assert!(
        model
            .install_memory_pin(&spec, Arc::new(stale), None)
            .is_none()
    );
    assert_eq!(model.memory_projection_state()["running"], false);
    assert!(model.pinned_level_resources().is_empty());
}

#[test]
fn unchanged_memory_projection_reuses_its_immutable_snapshot() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, _) = OmeZarrDataset::open_local(&fixture).unwrap();
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let first = model.memory_projection_state();
    let unchanged = model.memory_projection_state();
    assert!(Arc::ptr_eq(&first, &unchanged));

    model
        .prepare_memory_pin(&json!({"level":0,"channels":[0],"force":true}))
        .unwrap();
    let pending = model.memory_projection_state();
    assert!(!Arc::ptr_eq(&first, &pending));
    assert_eq!(pending["running"], true);
}

#[test]
fn mosaic_unpin_supersedes_a_late_memory_pin_install() {
    let opened = crate::data::document::open_local_ome_zarr(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr"),
    )
    .unwrap();
    let mut model = AppModel::project();
    model
        .bootstrap_mosaic(ControlMosaicResource {
            generation: 1,
            source: "memory-test".to_string(),
            base_dir: None,
            initial_columns: Some(1),
            metadata_columns: Arc::new(Vec::new()),
            items: Arc::new(vec![ControlMosaicItemResource {
                id: 0,
                roi_id: "ROI-A".to_string(),
                metadata: std::collections::HashMap::new(),
                document: opened.into_control(),
                segmentation_path: None,
            }]),
        })
        .unwrap();
    let spec = model
        .prepare_mosaic_memory_pin(
            &json!({"level":0,"channels":[0],"scope":"focused","force":true}),
        )
        .unwrap();
    let _ = model
        .dispatch("memory.unpin", &json!({"level":0,"scope":"focused"}))
        .unwrap()
        .unwrap();
    let stale =
        ControlPinnedLevelResource::new(0, 1, 1, std::iter::once((0, 0)).collect(), vec![1]);
    assert!(
        model
            .install_mosaic_memory_pin(
                &spec,
                MosaicMemoryPinResult {
                    loaded: vec![(0, stale)],
                    failures: Vec::new(),
                },
                None,
            )
            .is_none()
    );
    assert!(model.mosaic_pinned_level_resources().is_empty());
    assert_eq!(model.memory_projection_state()["running"], false);
}
