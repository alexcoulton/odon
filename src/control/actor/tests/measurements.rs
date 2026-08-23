use super::*;

fn open_measurement_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (load, load_rx) = request(
        "viewer.objects.source.load",
        json!({"path":"measurement-polygons.geojson"}),
    );
    channels.request_tx.send(load).unwrap();
    load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
}

#[test]
fn measurements_configure_compute_and_publish_properties_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    open_measurement_fixture(&channels);

    let (configure, configure_rx) = request(
        "viewer.measurements.configure",
        json!({
            "metric":"mean",
            "level":3,
            "concurrency":2,
            "filtered_only":false,
            "prefix":"actor_mean_"
        }),
    );
    channels.request_tx.send(configure).unwrap();
    let configured = configure_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(configured["metric"], "mean");
    assert_eq!(configured["level"], 3);
    assert_eq!(configured["target_count"], 2);

    let (start, start_rx) = request("viewer.measurements.start", json!({}));
    channels.request_tx.send(start).unwrap();
    let completed = start_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(completed["started"], true);
    assert_eq!(completed["completed"], true);
    assert_eq!(completed["measurement"]["running"], false);
    assert_eq!(completed["measurement"]["progress"]["completed"], 2);
    let generated = completed["measurement"]["generated_properties"]
        .as_array()
        .unwrap();
    assert_eq!(generated.len(), 5);
    assert!(
        generated
            .iter()
            .all(|property| property.as_str().unwrap().starts_with("actor_mean_"))
    );

    let property = generated[0].as_str().unwrap();
    let (values, values_rx) = request(
        "viewer.objects.properties.values",
        json!({"property":property,"offset":0,"limit":10}),
    );
    channels.request_tx.send(values).unwrap();
    let values = values_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(values["values"].as_array().unwrap().len(), 2);
    assert!(values["values"][0]["value"].as_f64().is_some());

    let (properties, properties_rx) = request("viewer.measurements.properties.list", json!({}));
    channels.request_tx.send(properties).unwrap();
    let state = properties_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(state["generated_properties"].as_array().unwrap().len(), 5);

    let (cancel, cancel_rx) = request("viewer.measurements.cancel", json!({}));
    channels.request_tx.send(cancel).unwrap();
    assert_eq!(
        cancel_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["cancelled"],
        false
    );
}

#[test]
fn cancelled_measurement_rejects_a_late_result() {
    let mut model = AppModel::project();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, _) = OmeZarrDataset::open_local(&fixture).unwrap();
    model.install_dataset(&dataset);
    let (document_generation, resource_generation) =
        model.begin_object_resource_load("measurement-polygons.geojson");
    assert!(model.install_object_resource_for_generation(
        document_generation,
        resource_generation,
        Arc::new(ControlObjectResource {
            source: PathBuf::from("measurement-polygons.geojson"),
            downsample_factor: 1.0,
            features: Arc::new(vec![crate::model::ControlObjectFeature {
                id: "cell".to_string(),
                bbox_world: [0.0, 0.0, 10.0, 10.0],
                centroid_world: [5.0, 5.0],
                polygons_world: Arc::new(vec![vec![
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 10.0],
                    [0.0, 10.0],
                    [0.0, 0.0],
                ]]),
                point_position_world: None,
                area_px: 100.0,
                perimeter_px: 40.0,
                properties: serde_json::Map::new(),
            }]),
            property_names: Arc::new(vec!["id".to_string()]),
            renderer_payload: None,
        }),
    ));
    let spec = model.prepare_measurement(&json!({"level":3})).unwrap();
    assert_eq!(
        model.cancel_measurement(&json!({})).unwrap()["cancelled"],
        true
    );
    let original = model.object_resource().unwrap();
    assert!(
        model
            .install_measurement(&spec, original.as_ref().clone(), 1)
            .is_none()
    );
}

#[test]
fn spatial_shape_measurement_target_is_resolved_by_the_actor() {
    let channels = spawn_test_actor_with_objects();
    open_measurement_fixture(&channels);
    let (request, reply) = request(
        "viewer.measurements.get",
        json!({"target":"spatial_shape","layer_id":7}),
    );
    channels.request_tx.send(request).unwrap();
    let error = reply
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::ResourceNotFound);
}
