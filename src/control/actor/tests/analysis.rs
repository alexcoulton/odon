use super::*;

fn open_objects(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (load, load_rx) = request(
        "viewer.objects.source.load",
        json!({"path":"test-analysis.geojson"}),
    );
    channels.request_tx.send(load).unwrap();
    load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
}

#[test]
fn analysis_state_compute_warmup_and_presets_complete_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    open_objects(&channels);
    let state = json!({
        "threshold_set_name":"Demo",
        "threshold_elements":[{
            "name":"Score positive",
            "scope":{"kind":"composite"},
            "rules":[{
                "column_key":"score",
                "op":"greater_equal",
                "value":0.5,
                "value_transform":"none"
            }]
        }],
        "follow_active_channel":true,
        "selection_elements":[],
        "show_selection_overlay":true
    });
    let (set, set_rx) = request("viewer.analysis.set", json!({"state":state}));
    channels.request_tx.send(set).unwrap();
    let set = set_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(set["state"]["threshold_set_name"], "Demo");
    assert_eq!(set["numeric_properties"], json!(["score"]));

    let (histogram, histogram_rx) = request(
        "viewer.analysis.histogram",
        json!({"property":"score","bins":8}),
    );
    channels.request_tx.send(histogram).unwrap();
    let histogram = histogram_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(histogram["count"], 2);
    assert_eq!(histogram["bins"].as_array().unwrap().len(), 8);

    let (suggest, suggest_rx) = request(
        "viewer.analysis.suggest_thresholds",
        json!({"property":"score","method":"quantiles","count":2}),
    );
    channels.request_tx.send(suggest).unwrap();
    let suggested = suggest_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(suggested["sample_count"], 2);
    assert_eq!(suggested["levels"].as_array().unwrap().len(), 1);

    let (warmup, warmup_rx) = request("viewer.analysis.warmup.start", json!({}));
    channels.request_tx.send(warmup).unwrap();
    let warmed = warmup_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(warmed["ready"], true);
    assert_eq!(warmed["completed"], 1);

    let output = std::env::temp_dir().join(format!(
        "odon-analysis-preset-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (export, export_rx) = request(
        "viewer.analysis.presets.export",
        json!({"path":output,"overwrite":true}),
    );
    channels.request_tx.send(export).unwrap();
    assert_eq!(
        export_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()["call_count"],
        1
    );
    let (clear, clear_rx) = request(
        "viewer.analysis.set",
        json!({"state":{"threshold_elements":[],"selection_elements":[]}}),
    );
    channels.request_tx.send(clear).unwrap();
    clear_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (import, import_rx) = request("viewer.analysis.presets.import", json!({"path":output}));
    channels.request_tx.send(import).unwrap();
    assert_eq!(
        import_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()["call_count"],
        1
    );
    let _ = std::fs::remove_file(output);

    let (get, get_rx) = request("viewer.analysis.get", json!({}));
    channels.request_tx.send(get).unwrap();
    let current = get_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(current["state"]["threshold_set_name"], "Demo");
    assert_eq!(
        current["state"]["threshold_elements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(channels.legacy_rx.len(), 0);
}

#[test]
fn spatial_shape_analysis_target_retains_its_explicit_legacy_route() {
    let channels = spawn_test_actor_with_objects();
    open_objects(&channels);
    let (request, _reply) = request(
        "viewer.analysis.get",
        json!({"target":"spatial_shape","shape_id":7}),
    );
    channels.request_tx.send(request).unwrap();
    let forwarded = channels
        .legacy_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap();
    assert_eq!(forwarded.command.method(), "viewer.analysis.get");
}
