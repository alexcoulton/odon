use super::*;
#[test]
fn actor_publishes_scoped_and_active_compatibility_events_before_replying() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();

    let event_hub = EventHub::shared();
    let (event_tx, event_rx) = crossbeam_channel::bounded(8);
    event_hub.register("observer".to_string(), event_tx);
    event_hub
        .subscribe("observer", vec!["*".to_string()])
        .unwrap();
    let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "viewer.viewports.camera.set",
                json!({
                    "viewport_id": "viewport-1",
                    "center_world_lvl0": [123.0, 234.0],
                    "zoom": 2.0,
                }),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "initiator".to_string(),
            request_id: Some(json!(42)),
            event_hub,
            task_registry,
            task_id: None,
        })
        .unwrap();

    let reply = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("actor reply")
        .unwrap();
    assert_eq!(reply["_control"]["revision"], 1);

    // The actor publishes both events before sending the reply. Therefore neither receive
    // should have to wait once the reply is observable.
    let scoped = event_rx
        .try_recv()
        .expect("scoped viewport event was published before the reply");
    let compatibility = event_rx
        .try_recv()
        .expect("active-view compatibility event was published before the reply");
    assert_eq!(
        scoped["params"]["event"],
        "viewer.viewports.navigation.changed"
    );
    assert_eq!(scoped["params"]["source"], "viewport:viewport-1");
    assert_eq!(scoped["params"]["sequence"], 1);
    assert_eq!(scoped["params"]["revision"], 1);
    assert_eq!(compatibility["params"]["event"], "viewer.camera.changed");
    assert_eq!(compatibility["params"]["source"], "viewer:active");
    assert_eq!(compatibility["params"]["sequence"], 2);
    assert_eq!(compatibility["params"]["revision"], 1);
}
