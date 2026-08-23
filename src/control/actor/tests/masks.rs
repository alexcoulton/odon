use super::*;
#[test]
fn mask_edits_commit_and_project_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (create, create_rx) = request("viewer.masks.layers.create", json!({"name":"Python cells"}));
    channels.request_tx.send(create).unwrap();
    let created = create_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let layer_id = created["id"].as_u64().unwrap();
    let (native_mask, native_mask_rx) = request(
        "viewer.native_layers.get",
        json!({"layer_id":format!("mask:{layer_id}")}),
    );
    channels.request_tx.send(native_mask).unwrap();
    let native_mask = native_mask_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(native_mask["layer"]["kind"], "mask");
    assert_eq!(native_mask["layer"]["active"], true);

    let (add, add_rx) = request(
        "viewer.masks.polygons.add",
        json!({
            "id":layer_id,
            "coordinate_space":"world",
            "vertices":[[10.0,20.0],[30.0,20.0],[30.0,40.0]],
        }),
    );
    channels.request_tx.send(add).unwrap();
    assert_eq!(
        add_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["added"],
        true
    );

    let (select, select_rx) = request(
        "viewer.masks.selection.set",
        json!({"id":layer_id,"index":0,"vertex_index":1}),
    );
    channels.request_tx.send(select).unwrap();
    assert_eq!(
        select_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["selection"]["vertex_index"],
        1
    );

    let (list, list_rx) = request("viewer.masks.layers.list", json!({}));
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(listed["total"], 1);
    assert_eq!(listed["layers"][0]["polygon_count"], 1);

    let geojson =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_objects.geojson");
    let (import, import_rx) = request(
        "viewer.masks.import_geojson",
        json!({"path":geojson,"name":"Imported cells","editable":false}),
    );
    channels.request_tx.send(import).unwrap();
    let imported = import_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(imported["imported"], true);
    assert_eq!(imported["layer"]["editable"], false);
    assert_eq!(imported["polygon_count"], 2);
    let imported_layer_id = imported["layer_id"].as_u64().unwrap();
    let (reload, reload_rx) = request(
        "viewer.masks.import_geojson",
        json!({
            "path":geojson,
            "name":"Ignored on stable reload",
            "editable":false,
            "replace_layer_id":imported_layer_id,
        }),
    );
    channels.request_tx.send(reload).unwrap();
    let reloaded = reload_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(reloaded["reloaded"], true);
    assert_eq!(reloaded["id"], imported_layer_id);
    let (reloaded_list, reloaded_list_rx) = request("viewer.masks.layers.list", json!({}));
    channels.request_tx.send(reloaded_list).unwrap();
    assert_eq!(
        reloaded_list_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["total"],
        2
    );

    let (persistence, persistence_rx) = request("viewer.masks.persistence.get", json!({}));
    channels.request_tx.send(persistence).unwrap();
    let persistence = persistence_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(persistence["dirty"], true);
    assert_eq!(persistence["persisted_layer_count"], Value::Null);
    let (sync, sync_rx) = request("viewer.masks.persistence.sync", json!({}));
    channels.request_tx.send(sync).unwrap();
    let synced = sync_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(synced["persistence"]["dirty"], false);
    assert_eq!(synced["persistence"]["persisted_layer_count"], 2);

    let output = std::env::temp_dir().join(format!(
        "odon-actor-mask-export-{}-{}.geojson",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (export, export_rx) = request(
        "viewer.masks.export_geojson",
        json!({"path":output,"overwrite":true}),
    );
    channels.request_tx.send(export).unwrap();
    let exported = export_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(exported["output_ready"], true);
    assert_eq!(exported["layer_count"], 2);
    assert_eq!(exported["polygon_count"], 3);
    let written: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(written["features"].as_array().unwrap().len(), 3);
    let _ = fs::remove_file(output);

    let projection = channels.presentation_rx.try_recv().unwrap();
    let masks = &projection.workspace.as_ref().unwrap()["masks"];
    assert_eq!(masks["layers"][0]["name"], "Python cells");
    assert_eq!(
        masks["layers"][0]["polygons_world"][0]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(masks["selection"]["vertex_index"], 1);
    assert_eq!(masks["dirty"], false);
    assert_eq!(projection.project.rois[0].mask_layers.len(), 2);
}

#[test]
fn drawn_masks_append_clear_and_reload_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (create, create_rx) = request("viewer.masks.layers.create", json!({"name":"Drawn"}));
    channels.request_tx.send(create).unwrap();
    let layer_id = create_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap()["id"]
        .as_u64()
        .unwrap();
    let (add, add_rx) = request(
        "viewer.masks.polygons.add",
        json!({
            "id":layer_id,
            "coordinate_space":"world",
            "sync_project":true,
            "vertices":[[2.0,4.0],[8.0,4.0],[8.0,10.0]],
        }),
    );
    channels.request_tx.send(add).unwrap();
    let added = add_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(added["persistence"]["dirty"], false);
    assert_eq!(added["persistence"]["persisted_layer_count"], 1);

    let output = std::env::temp_dir().join(format!(
        "odon-actor-mask-append-{}-{}.geojson",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (save, save_rx) = request(
        "viewer.masks.persistence.append_geojson",
        json!({
            "path":output,
            "downsample_factor":2.0,
            "roi_root":"fixture.ome.zarr",
        }),
    );
    channels.request_tx.send(save).unwrap();
    let saved = save_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(saved["saved"], true);
    assert_eq!(saved["applied_to_current_document"], true);
    assert_eq!(saved["appended_polygon_count"], 1);
    assert_eq!(saved["mask"]["cleared_polygon_count"], 1);

    let written: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(written["features"].as_array().unwrap().len(), 1);
    assert_eq!(
        written["features"][0]["geometry"]["coordinates"][0][0],
        json!([1.0, 2.0])
    );

    let (list, list_rx) = request("viewer.masks.layers.list", json!({}));
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(listed["total"], 2);
    assert_eq!(listed["layers"][0]["polygon_count"], 0);
    assert_eq!(listed["layers"][1]["editable"], false);
    assert_eq!(listed["layers"][1]["polygon_count"], 1);
    let (polygons, polygons_rx) = request(
        "viewer.masks.polygons.list",
        json!({"id":listed["layers"][1]["id"]}),
    );
    channels.request_tx.send(polygons).unwrap();
    let polygons = polygons_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        polygons["polygons"][0]["vertices_world"][0],
        json!([2.0, 4.0])
    );

    let _ = fs::remove_file(output);
}
