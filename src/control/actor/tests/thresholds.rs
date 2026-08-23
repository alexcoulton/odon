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
fn threshold_preview_configure_refresh_apply_and_cancel_without_a_frame() {
    let channels = spawn_test_actor();
    open_fixture(&channels);

    let (levels, levels_rx) = request("viewer.thresholds.levels.list", json!({}));
    channels.request_tx.send(levels).unwrap();
    let levels = levels_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let level = levels["default_full_level"].as_u64().unwrap();

    let (start, start_rx) = request(
        "viewer.thresholds.preview.start",
        json!({
            "scope":"entire_image",
            "level":level,
            "channel":0,
            "threshold":0,
            "min_component_pixels":1,
        }),
    );
    channels.request_tx.send(start).unwrap();
    let started = start_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(started["active"], true);
    assert_eq!(started["preview"]["scope"], "entire_image");
    assert_eq!(started["preview"]["channel_index"], 0);
    assert!(started["preview"]["included_pixels"].as_u64().unwrap() > 0);
    let source_max = started["preview"]["source_max"].as_u64().unwrap();

    let (configure, configure_rx) = request(
        "viewer.thresholds.preview.configure",
        json!({"threshold":source_max,"min_component_pixels":1}),
    );
    channels.request_tx.send(configure).unwrap();
    let configured = configure_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(configured["preview"]["threshold"], source_max);

    let (refresh, refresh_rx) = request("viewer.thresholds.preview.refresh", json!({}));
    channels.request_tx.send(refresh).unwrap();
    let refreshed = refresh_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(refreshed["active"], true);
    assert_eq!(refreshed["preview"]["threshold"], source_max);

    let (apply, apply_rx) = request("viewer.thresholds.preview.apply", json!({}));
    channels.request_tx.send(apply).unwrap();
    let applied = apply_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(applied["applied"], true);
    assert!(applied["polygon_count"].as_u64().unwrap() > 0);

    let (masks, masks_rx) = request("viewer.masks.layers.list", json!({}));
    channels.request_tx.send(masks).unwrap();
    let masks = masks_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(masks["total"], 1);
    assert_eq!(masks["layers"][0]["display_mode"], "filled_preview");

    let (restart, restart_rx) = request(
        "viewer.thresholds.preview.start",
        json!({"scope":"entire_image","level":level,"threshold":0}),
    );
    channels.request_tx.send(restart).unwrap();
    restart_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (cancel, cancel_rx) = request("viewer.thresholds.preview.cancel", json!({}));
    channels.request_tx.send(cancel).unwrap();
    assert_eq!(
        cancel_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["cancelled"],
        true
    );

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert!(projection.threshold_preview.is_none());
    assert_eq!(
        projection.workspace.as_ref().unwrap()["masks"]["layers"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn cancelling_threshold_preview_rejects_a_late_worker_install() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, _) = OmeZarrDataset::open_local(&fixture).unwrap();
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let level = model
        .dispatch("viewer.thresholds.levels.list", &json!({}))
        .unwrap()
        .unwrap()
        .response["default_full_level"]
        .as_u64()
        .unwrap();
    let spec = model
        .prepare_threshold_preview_load(
            &json!({"scope":"entire_image","level":level,"threshold":0}),
            false,
        )
        .unwrap();
    model.cancel_threshold_preview().unwrap();
    let stale = ControlThresholdPreviewResource {
        generation: spec.operation_generation,
        channel_index: spec.channel_index,
        channel_name: spec.channel_name,
        scope: spec.scope,
        level: spec.level,
        downsample: spec.downsample,
        x0: spec.x0,
        y0: spec.y0,
        width: 1,
        height: 1,
        values: Arc::new(vec![1]),
        included: Arc::new(vec![true]),
        threshold: 0,
        min_component_pixels: 1,
    };
    assert!(
        model
            .install_threshold_preview(
                spec.document_generation,
                spec.operation_generation,
                Arc::new(stale),
            )
            .is_none()
    );
}
