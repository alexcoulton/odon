use super::*;
#[test]
fn deep_link_resolution_uses_current_or_external_actor_project_without_a_frame() {
    let channels = spawn_test_actor();
    let (add, add_rx) = request(
        "project.rois.add",
        json!({"id":"current-roi","path":"/tmp/current.ome.zarr"}),
    );
    channels.request_tx.send(add).unwrap();
    add_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let workers_before = channels.diagnostics.snapshot()["workers"]["started"]
        .as_u64()
        .unwrap();
    let (resolve, resolve_rx) = request(
        "deep_links.resolve",
        json!({"request":{"roi":"current-roi"}}),
    );
    channels.request_tx.send(resolve).unwrap();
    let current = resolve_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(current["resolved"], true);
    assert_eq!(current["resolution"]["project_source"], "current");
    assert_eq!(current["resolution"]["roi"]["id"], "current-roi");
    assert_eq!(
        channels.diagnostics.snapshot()["workers"]["started"],
        workers_before
    );

    let project =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.project.json");
    let (resolve, resolve_rx) = request(
        "deep_links.resolve",
        json!({
            "request":{
                "project_path":project,
                "roi":"synthetic_5ch.ome.zarr",
            }
        }),
    );
    channels.request_tx.send(resolve).unwrap();
    let external = resolve_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(external["resolved"], true);
    assert_eq!(external["resolution"]["project_source"], "project_file");
    assert_eq!(
        external["resolution"]["roi"]["id"],
        "synthetic_5ch.ome.zarr"
    );
    assert!(
        external["resolution"]["roi"]["source"]["Local"]
            .as_str()
            .is_some_and(|path| path.ends_with("fixtures/synthetic_5ch.ome.zarr"))
    );

    let (example, example_rx) = request(
        "deep_links.resolve",
        json!({"request":{"example":"synthetic"}}),
    );
    channels.request_tx.send(example).unwrap();
    let example = example_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(example["resolved"], true);
    assert_eq!(example["request"]["channel"], "DAPI");
    assert_eq!(example["resolution"]["project_source"], "project_file");
}
