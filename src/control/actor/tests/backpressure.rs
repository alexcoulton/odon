use super::*;
#[test]
fn saturated_mutation_queue_during_presentation_does_not_stall_the_actor() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("dataset projection");
    channels
        .model_tx
        .send(ActorModelUpdate::PresentationApplied(projection.revision))
        .unwrap();

    let output = std::env::temp_dir().join(format!(
        "odon-presentation-backpressure-{}",
        std::process::id()
    ));
    let (first, _first_reply) = request(
        "viewer.screenshot.capture",
        json!({"path":output.with_extension("first.png")}),
    );
    channels.request_tx.send(first).unwrap();
    let mut deferred_replies = Vec::new();
    for index in 0..=ACTOR_QUEUE_CAPACITY {
        let (mutation, reply) = request("viewer.panels.set", json!({"left":index % 2 == 0}));
        channels.request_tx.send(mutation).unwrap();
        deferred_replies.push(reply);
    }
    let error = deferred_replies
        .last()
        .unwrap()
        .recv_timeout(Duration::from_secs(1))
        .expect("overflowing request receives an explicit reply")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
    assert!(error.message.contains("actor mutation queue is full"));

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    assert!(
        workspace_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("actor remains responsive")
            .is_ok()
    );
}
