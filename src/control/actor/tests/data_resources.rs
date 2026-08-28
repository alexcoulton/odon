use super::*;

#[test]
fn registered_nested_zarr_group_opens_as_primary_document() {
    let channels = spawn_test_actor();
    let fixtures_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let source_uri = url::Url::from_directory_path(&fixtures_root)
        .expect("fixture directory URI")
        .to_string();
    let (register, register_rx) = request(
        "data.resources.register",
        json!({
            "resource_id": "resource:primary-image",
            "uri": source_uri,
            "format": "ome-zarr",
            "group_path": "synthetic_5ch.ome.zarr",
            "coordinate_space": {
                "axes": ["c", "y", "x"],
                "scale": [1.0, 1.0, 1.0],
                "translation": [0.0, 0.0, 0.0]
            }
        }),
    );
    channels.request_tx.send(register).unwrap();
    let registered = register_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("registration reply")
        .expect("registration succeeds");
    assert_eq!(registered["group_path"], "synthetic_5ch.ome.zarr");

    let (register_label, register_label_rx) = request(
        "data.resources.register",
        json!({
            "resource_id": "resource:labels",
            "uri": source_uri,
            "format": "ome-zarr",
            "group_path": "synthetic_5ch.ome.zarr",
            "coordinate_space": {"axes": ["c", "y", "x"]}
        }),
    );
    channels.request_tx.send(register_label).unwrap();
    register_label_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("label registration reply")
        .expect("label registration succeeds");

    let (open, open_rx) = request(
        "datasets.open_resource",
        json!({
            "resource_id": "resource:primary-image",
            "layers": [{
                "layer_id": "layer:nuclei",
                "data_resource_id": "resource:labels",
                "name": "Nuclei",
                "kind": "labels"
            }]
        }),
    );
    channels.request_tx.send(open).unwrap();
    let opened = open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("registered resource open reply")
        .expect("registered resource opens");
    assert_eq!(opened["resource_id"], "resource:primary-image");
    assert_eq!(opened["source_uri"], source_uri);
    assert_eq!(opened["group_path"], "synthetic_5ch.ome.zarr");
    assert_eq!(
        opened["path"],
        fixtures_root
            .join("synthetic_5ch.ome.zarr")
            .to_string_lossy()
            .as_ref()
    );
    assert_eq!(opened["model_ready"], true);
    assert_eq!(opened["resources_ready"], true);
    assert_eq!(opened["presentation_ready"], false);
    assert_eq!(opened["layers"][0]["layer_id"], "layer:nuclei");

    let (list, list_rx) = request("viewer.layers.list", json!({}));
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("layer list reply")
        .expect("layer list succeeds");
    assert_eq!(listed["layers"][0]["data_resource_id"], "resource:labels");
}

#[test]
fn registered_primary_resource_rejects_unsupported_sources_before_loading() {
    let channels = spawn_test_actor();
    let (register, register_rx) = request(
        "data.resources.register",
        json!({
            "resource_id": "resource:table",
            "uri": "file:///tmp/table.parquet",
            "format": "parquet",
            "coordinate_space": {"axes": ["y", "x"]}
        }),
    );
    channels.request_tx.send(register).unwrap();
    register_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("registration reply")
        .expect("registration succeeds");

    let (open, open_rx) = request(
        "datasets.open_resource",
        json!({"resource_id": "resource:table"}),
    );
    channels.request_tx.send(open).unwrap();
    let error = open_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("open rejection")
        .expect_err("non-Zarr primary resource is rejected");
    assert_eq!(error.kind, ControlErrorKind::InvalidParams);
    let data = error.data.expect("structured error data");
    assert_eq!(data["phase"], "validate_resource");
    assert_eq!(data["resource_id"], "resource:table");
}

#[test]
fn registered_geoparquet_layers_are_actor_owned_selection_targets() {
    let channels = spawn_test_actor_with_objects();
    let fixtures_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let source_uri = url::Url::from_directory_path(&fixtures_root)
        .expect("fixture directory URI")
        .to_string();
    for (resource_id, uri, format, group_path) in [
        (
            "resource:primary-image",
            source_uri.clone(),
            "ome-zarr",
            Some("synthetic_5ch.ome.zarr"),
        ),
        (
            "resource:cells",
            url::Url::from_file_path("/tmp/actor-owned-cells.parquet")
                .expect("object file URI")
                .to_string(),
            "geoparquet",
            None,
        ),
    ] {
        let (register, register_rx) = request(
            "data.resources.register",
            json!({
                "resource_id": resource_id,
                "uri": uri,
                "format": format,
                "group_path": group_path,
                "coordinate_space": {
                    "axes": ["y", "x"],
                    "affine": {
                        "input_axes": ["x", "y"],
                        "output_axes": ["x", "y"],
                        "matrix": [[1.0, 0.25, 3.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]],
                        "direction": "input_to_reference",
                        "pixel_center": "center"
                    }
                }
            }),
        );
        channels.request_tx.send(register).unwrap();
        register_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("registration reply")
            .expect("registration succeeds");
    }

    let (open, open_rx) = request(
        "datasets.open_resource",
        json!({
            "resource_id": "resource:primary-image",
            "layers": [{
                "layer_id": "spatial_shape:42",
                "data_resource_id": "resource:cells",
                "name": "Cells",
                "kind": "shapes"
            }]
        }),
    );
    channels.request_tx.send(open).unwrap();
    let opened = open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("registered resource open reply")
        .expect("registered composite opens");
    assert_eq!(opened["layers"][0]["layer_id"], "spatial_shape:42");

    let (select, select_rx) = request(
        "viewer.objects.selection.select_ids",
        json!({
            "target": "spatial_shape",
            "layer_id": 42,
            "ids": ["cell-a"],
            "mode": "replace"
        }),
    );
    channels.request_tx.send(select).unwrap();
    let selected = select_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("selection reply")
        .expect("selection succeeds");
    assert_eq!(selected["selection"]["selection_count"], 1);
    assert_eq!(selected["selection"]["selected"][0]["id"], "cell-a");

    let (properties, properties_rx) = request(
        "viewer.objects.properties.list",
        json!({"target": "spatial_shape", "layer_id": 42}),
    );
    channels.request_tx.send(properties).unwrap();
    let properties = properties_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("property reply")
        .expect("properties are actor-owned");
    assert!(
        properties["columns"]
            .as_array()
            .unwrap()
            .iter()
            .any(|column| column["name"] == "score")
    );

    let (remove, remove_rx) = request(
        "viewer.layers.remove",
        json!({"layer_id": "spatial_shape:42"}),
    );
    channels.request_tx.send(remove).unwrap();
    remove_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("remove reply")
        .expect("actor-owned layer removal succeeds");

    let (selection, selection_rx) = request(
        "viewer.objects.get_selection",
        json!({"target": "spatial_shape", "layer_id": 42}),
    );
    channels.request_tx.send(selection).unwrap();
    let removed = selection_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("selection reply")
        .expect_err("removed actor target is unavailable");
    assert_eq!(removed.kind, ControlErrorKind::ResourceNotFound);
}
