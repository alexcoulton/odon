use super::*;
#[test]
fn dataset_inspection_completes_on_a_worker_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let scope = fixture.to_string_lossy().into_owned();
    let (inspect, inspect_rx) = request("datasets.inspect", json!({"path":fixture}));
    channels.request_tx.send(inspect).unwrap();
    let inspected = inspect_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(inspected["kind"], "ome_zarr");
    assert_eq!(inspected["can_open"], true);
    assert_eq!(inspected["metadata"]["level_count"], 4);
    assert_eq!(
        inspected["metadata"]["channels"].as_array().unwrap().len(),
        5
    );

    let (loading, loading_rx) = request("app.get_loading_state", json!({}));
    channels.request_tx.send(loading).unwrap();
    let loading = loading_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let key = format!("dataset_inspection:{scope}");
    assert_eq!(
        loading["loading"]["operations"][key.as_str()]["phase"],
        "ready"
    );

    let missing = std::env::temp_dir().join(format!(
        "odon-missing-dataset-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (inspect, inspect_rx) = request("datasets.inspect", json!({"path":missing}));
    channels.request_tx.send(inspect).unwrap();
    let inspected = inspect_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(inspected["error"], "dataset path does not exist");
    assert!(inspected.get("kind").is_none());
    assert!(inspected.get("can_open").is_none());
}
