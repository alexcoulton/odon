use super::*;
#[test]
fn actor_rejects_ready_only_worker_commands_during_transition() {
    let channels = spawn_test_actor();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapMode(ModelMode::Transition))
        .unwrap();
    let (import, reply) = request(
        "project.samplesheets.import",
        json!({"path":"does-not-need-to-exist.csv"}),
    );
    channels.request_tx.send(import).unwrap();
    let error = reply
        .recv_timeout(Duration::from_secs(1))
        .expect("actor replies without starting a worker")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
    assert_eq!(error.data.as_ref().unwrap()["mode"], "transition");
    assert_eq!(channels.diagnostics.snapshot()["workers"]["started"], 0);
}
