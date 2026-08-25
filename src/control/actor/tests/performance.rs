use super::*;

use std::time::Instant;

use crate::model::ControlObjectFeature;

#[test]
fn local_model_command_median_round_trip_stays_below_five_milliseconds() {
    const WARMUP_SAMPLES: usize = 32;
    const MEASURED_SAMPLES: usize = 257;

    let channels = spawn_test_actor();
    for _ in 0..WARMUP_SAMPLES {
        let (command, reply) = request("app.get_state", json!({}));
        channels.request_tx.send(command).unwrap();
        reply
            .recv_timeout(Duration::from_secs(1))
            .expect("warmup request completes without a frame")
            .unwrap();
    }

    let mut elapsed_micros = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        let (command, reply) = request("app.get_state", json!({}));
        let started = Instant::now();
        channels.request_tx.send(command).unwrap();
        reply
            .recv_timeout(Duration::from_secs(1))
            .expect("measured request completes without a frame")
            .unwrap();
        elapsed_micros.push(started.elapsed().as_micros());
    }
    elapsed_micros.sort_unstable();
    let median_micros = elapsed_micros[elapsed_micros.len() / 2];
    let p95_micros = elapsed_micros[elapsed_micros.len() * 95 / 100];
    println!(
        "actor local model command: median={median_micros}us p95={p95_micros}us samples={MEASURED_SAMPLES}"
    );
    assert!(
        median_micros < 5_000,
        "median local model-command round trip was {median_micros}us; budget is <5000us"
    );
}

#[test]
fn camera_projection_reuses_large_object_resource_without_copying_geometry() {
    let (dataset, _) = OmeZarrDataset::open_local(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr"),
    )
    .expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let polygon = Arc::new(vec![vec![
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ]]);
    let features = Arc::new(
        (0..45_000)
            .map(|index| ControlObjectFeature {
                id: format!("feature-{index}"),
                bbox_world: [0.0, 0.0, 10.0, 10.0],
                centroid_world: [5.0, 5.0],
                polygons_world: Arc::clone(&polygon),
                point_position_world: None,
                area_px: 100.0,
                perimeter_px: 40.0,
                properties: serde_json::Map::new(),
            })
            .collect::<Vec<_>>(),
    );
    let resource = Arc::new(ControlObjectResource {
        source: PathBuf::from("large-objects.geojson"),
        downsample_factor: 1.0,
        features: Arc::clone(&features),
        property_names: Arc::new(vec!["id".to_string()]),
        numeric_summaries: Arc::new(Default::default()),
        renderer_payload: None,
    });
    let (document_generation, resource_generation) =
        model.begin_object_resource_load(resource.source.to_string_lossy().to_string());
    assert!(model.install_object_resource_for_generation(
        document_generation,
        resource_generation,
        Arc::clone(&resource),
    ));

    let (projection_tx, projection_rx) = crossbeam_channel::bounded(1);
    let diagnostics = ActorDiagnostics::shared();
    let wake: UiWake = Arc::new(|| {});
    publish_projection(
        &mut model,
        None,
        &projection_tx,
        &projection_rx,
        &wake,
        &diagnostics,
    );
    let before = projection_rx.recv().unwrap();
    let before_resource = before.object_resource.as_ref().unwrap();
    assert!(Arc::ptr_eq(before_resource, &resource));
    assert!(Arc::ptr_eq(&before_resource.features, &features));
    let workspace_text = serde_json::to_string(before.workspace.as_ref().unwrap()).unwrap();
    assert!(!workspace_text.contains("feature-44999"));

    let viewport_id = before.workspace.as_ref().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.camera.set",
            &json!({"viewport_id":viewport_id,"center_world_lvl0":[32.0,48.0],"zoom":2.0}),
        )
        .unwrap()
        .unwrap();
    publish_projection(
        &mut model,
        None,
        &projection_tx,
        &projection_rx,
        &wake,
        &diagnostics,
    );
    let after = projection_rx.recv().unwrap();
    let after_resource = after.object_resource.as_ref().unwrap();
    assert!(Arc::ptr_eq(before_resource, after_resource));
    assert!(Arc::ptr_eq(&after_resource.features, &features));
}
