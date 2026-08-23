use super::*;
#[test]
fn actor_owned_commands_never_fall_back_to_the_ui_queue_while_loading() {
    let channels = spawn_test_actor();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapMode(ModelMode::Transition))
        .unwrap();
    let (workspace, reply) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let error = reply
        .recv_timeout(Duration::from_secs(1))
        .expect("actor replies without a frame")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
}
