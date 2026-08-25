use super::*;
use std::sync::Barrier;

fn shell_call(
    tx: &Sender<OdonControlRequest>,
    method: &str,
    params: Value,
    session_id: impl Into<String>,
) -> Result<Value, ControlError> {
    let event_hub = EventHub::shared();
    let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
    let (reply, result) = crossbeam_channel::bounded(1);
    tx.send(OdonControlRequest {
        command: ControlCommand::decode(method, params).unwrap(),
        reply,
        session_id: session_id.into(),
        request_id: None,
        event_hub,
        task_registry,
        task_id: None,
    })
    .unwrap();
    result.recv_timeout(Duration::from_secs(2)).unwrap()
}

#[test]
fn rapid_native_shell_gestures_and_concurrent_python_patches_preserve_revision_order() {
    let channels = spawn_test_actor();
    let initial = shell_call(&channels.request_tx, "ui.shell.get", json!({}), "observer").unwrap();
    let workspace = initial["layout"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["mount"] == "builtin:project-workspace")
        .and_then(|node| node["id"].as_str())
        .expect("project workspace node")
        .to_string();

    // Native drag frames are renderer-local, but many completed gestures may still arrive faster
    // than a Python client can refetch. Exercise that actor queue in one burst without weakening
    // the actor's revision serialization.
    let mut rapid_replies = Vec::new();
    for index in 0..48_u64 {
        let event_hub = EventHub::shared();
        let (reply, result) = crossbeam_channel::bounded(1);
        channels
            .request_tx
            .send(OdonControlRequest {
                command: ControlCommand::decode(
                    "ui.shell.patch_layout",
                    json!({
                        "sizes":{workspace.clone():{"width":700.0 + index as f64}},
                        "transaction_id":format!("native-gesture-{index}"),
                    }),
                )
                .unwrap(),
                reply,
                session_id: "native-ui".to_string(),
                request_id: None,
                task_registry: TaskRegistry::shared(Arc::clone(&event_hub)),
                event_hub,
                task_id: None,
            })
            .unwrap();
        rapid_replies.push(result);
    }
    let mut previous_revision = initial["revision"].as_u64().unwrap();
    for reply in rapid_replies {
        let result = reply
            .recv_timeout(Duration::from_secs(2))
            .expect("rapid native shell reply")
            .unwrap();
        let revision = result["revision"].as_u64().unwrap();
        assert_eq!(revision, previous_revision + 1);
        previous_revision = revision;
    }

    let after_gestures =
        shell_call(&channels.request_tx, "ui.shell.get", json!({}), "observer").unwrap();
    assert_eq!(after_gestures["revision"], previous_revision);
    let shared_revision = previous_revision;

    // All Python clients race from one legitimate snapshot. Exactly one revision-guarded patch
    // may commit; every loser must receive the refetch/merge/retry contract.
    let client_count = 12;
    let barrier = Arc::new(Barrier::new(client_count + 1));
    let mut clients = Vec::new();
    for index in 0..client_count {
        let tx = channels.request_tx.clone();
        let barrier = Arc::clone(&barrier);
        let workspace = workspace.clone();
        clients.push(std::thread::spawn(move || {
            barrier.wait();
            shell_call(
                &tx,
                "ui.shell.patch_layout",
                json!({
                    "if_shell_revision":shared_revision,
                    "sizes":{workspace:{"width":1000.0 + index as f64}},
                    "transaction_id":format!("python-race-{index}"),
                }),
                format!("python-client-{index}"),
            )
        }));
    }
    barrier.wait();

    let mut committed = Vec::new();
    let mut conflicted = 0;
    for client in clients {
        match client.join().expect("Python race client") {
            Ok(result) => committed.push(result),
            Err(error) => {
                assert_eq!(error.kind, ControlErrorKind::Conflict);
                let data = error.data.expect("conflict retry metadata");
                assert_eq!(data["expected_revision"], shared_revision);
                assert_eq!(data["retry_strategy"], "refetch_merge_retry");
                assert_eq!(data["snapshot_method"], "ui.shell.get");
                conflicted += 1;
            }
        }
    }
    assert_eq!(committed.len(), 1);
    assert_eq!(conflicted, client_count - 1);
    assert_eq!(committed[0]["revision"], shared_revision + 1);

    let reconciled =
        shell_call(&channels.request_tx, "ui.shell.get", json!({}), "observer").unwrap();
    assert_eq!(reconciled["revision"], shared_revision + 1);
    let winning_width = reconciled["layout"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == workspace)
        .and_then(|node| node.pointer("/size/width"))
        .and_then(Value::as_f64)
        .expect("winning workspace width");
    assert!((1000.0..1012.0).contains(&winning_width));

    let retried = shell_call(
        &channels.request_tx,
        "ui.shell.patch_layout",
        json!({
            "if_shell_revision":reconciled["revision"],
            "sizes":{workspace:{"width":2000.0}},
            "transaction_id":"python-refetch-retry",
        }),
        "python-retry-client",
    )
    .expect("a conflicted client can commit after refetching");
    assert_eq!(retried["revision"], shared_revision + 2);
    assert_eq!(retried["change"]["transaction_id"], "python-refetch-retry");
}
