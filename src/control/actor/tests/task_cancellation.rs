use super::*;
#[test]
fn actor_owned_task_cancellation_remains_responsive_during_worker_io() {
    struct BlockingInspector {
        started: Sender<()>,
        release: Mutex<Receiver<()>>,
    }

    impl DatasetInspector for BlockingInspector {
        fn inspect(&self, path: &std::path::Path) -> DatasetInspection {
            let _ = self.started.send(());
            let _ = self.release.lock().unwrap().recv();
            DatasetInspection::failed(
                crate::data::document::DatasetInspectionKind::Unsupported,
                path.to_path_buf(),
                "test inspection completed",
            )
        }
    }

    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(Arc::clone(&events));
    let channels = spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        None,
        Some(Arc::new(BlockingInspector {
            started: started_tx,
            release: Mutex::new(release_rx),
        })),
        None,
        None,
    )
    .unwrap();
    let task = channels
        .task_service
        .create("blocked inspection", "test", true)
        .unwrap();
    let (reply, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode("datasets.inspect", json!({"path":"blocked-source"}))
                .unwrap(),
            reply,
            session_id: "test".to_string(),
            request_id: None,
            event_hub: events,
            task_registry: channels.task_service.registry(),
            task_id: Some(task.task_id.clone()),
        })
        .unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("worker began inspection");

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    assert_eq!(
        state_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("actor remains responsive")
            .unwrap()["mode"],
        "project"
    );
    let cancelled = channels.task_service.cancel(&task.task_id).unwrap();
    assert_eq!(cancelled.state, TaskState::Cancelled);
    release_tx.send(()).unwrap();
    let error = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("cancelled worker result settles")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Cancelled);
    assert_eq!(
        channels.task_service.get(&task.task_id).unwrap().state,
        TaskState::Cancelled
    );
    assert_eq!(channels.legacy_rx.len(), 0);
}
