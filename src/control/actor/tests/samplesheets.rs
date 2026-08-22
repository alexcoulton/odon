use super::*;
#[test]
fn samplesheet_and_discovery_transactions_complete_without_a_ui_frame() {
    let unique = format!(
        "odon-actor-samplesheet-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    fs::create_dir_all(&root).unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let input = root.join("input.csv");
    let output = root.join("output.csv");
    fs::write(
        &input,
        format!("id,path,cohort\nfixture,{},A\n", fixture.display()),
    )
    .unwrap();
    let discovered = root.join("discovered.ome.zarr");
    fs::create_dir_all(&discovered).unwrap();
    fs::write(discovered.join(".zattrs"), "{}").unwrap();

    let channels = spawn_test_actor();
    let (inspect, inspect_rx) = request(
        "project.samplesheets.inspect",
        json!({"path":input,"offset":0,"limit":20}),
    );
    channels.request_tx.send(inspect).unwrap();
    let inspected = inspect_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("inspection completes without UI drain")
        .unwrap();
    assert_eq!(inspected["valid"], true);
    assert_eq!(inspected["total"], 1);

    let (import, import_rx) = request("project.samplesheets.import", json!({"path":input}));
    channels.request_tx.send(import).unwrap();
    let imported = import_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("import completes without UI drain")
        .unwrap();
    assert_eq!(imported["project"]["roi_count"], 1);
    assert_eq!(imported["project"]["rois"][0]["id"], "fixture");

    let (export, export_rx) = request(
        "project.samplesheets.export",
        json!({"path":output,"overwrite":true}),
    );
    channels.request_tx.send(export).unwrap();
    let exported = export_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("export completes without UI drain")
        .unwrap();
    assert_eq!(exported["output_ready"], true);
    assert!(exported["bytes"].as_u64().is_some_and(|bytes| bytes > 0));
    assert!(fs::read_to_string(&output).unwrap().contains("fixture"));

    let (discover, discover_rx) = request("project.discovery.add_root", json!({"path":root}));
    channels.request_tx.send(discover).unwrap();
    let discovered = discover_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("discovery completes without UI drain")
        .unwrap();
    assert_eq!(discovered["added"], 1);
    assert_eq!(discovered["project"]["roi_count"], 2);
    assert_eq!(channels.legacy_rx.len(), 0);
    assert_eq!(channels.presentation_rx.len(), 1);
    assert_eq!(
        channels
            .presentation_rx
            .try_recv()
            .unwrap()
            .project
            .rois
            .len(),
        2
    );

    fs::remove_dir_all(root).unwrap();
}
