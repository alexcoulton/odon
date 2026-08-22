use super::*;
#[test]
fn saturated_legacy_queue_returns_backpressure_without_stalling_the_actor() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();

    let mut replies = Vec::new();
    for _ in 0..=ACTOR_QUEUE_CAPACITY {
        let (legacy, reply) = request("viewer.screenshot.capture", json!({}));
        channels.request_tx.send(legacy).unwrap();
        replies.push(reply);
    }
    let error = replies
        .last()
        .unwrap()
        .recv_timeout(Duration::from_secs(1))
        .expect("overflowing request receives an explicit reply")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
    assert!(error.message.contains("legacy UI command queue is full"));
    assert_eq!(channels.legacy_rx.len(), ACTOR_QUEUE_CAPACITY);

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    assert!(
        workspace_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("actor remains responsive")
            .is_ok()
    );
}
