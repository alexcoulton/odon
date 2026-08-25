use super::*;

pub(in crate::mcp) fn spawn_control_server(
    addr: &str,
    publish: bool,
    ctx: egui::Context,
    tx: Sender<OdonControlRequest>,
    services: ControlServerServices,
) -> anyhow::Result<ControlServerPublication> {
    let listener = TcpListener::bind(addr)?;
    let local_addr = listener.local_addr()?;
    listener.set_nonblocking(false)?;
    let manifest = if publish {
        Some(crate::control::discovery::InstanceManifestGuard::publish(
            crate::control::discovery::InstanceManifest::new(local_addr)?,
        )?)
    } else {
        None
    };
    let identity = Arc::new(ControlServerIdentity {
        instance_id: manifest
            .as_ref()
            .map(|guard| guard.manifest().instance_id.clone())
            .unwrap_or(crate::control::discovery::random_uuid_like()?),
        expected_token: manifest
            .as_ref()
            .map(|guard| guard.manifest().token.clone()),
        allow_legacy: !publish,
        event_hub: services.event_hub,
        task_registry: services.task_registry,
        task_service: services.task_service,
        resource_registry: services.resource_registry,
        ui_registry: services.ui_registry,
        actor_diagnostics: services.actor_diagnostics,
    });
    thread::Builder::new()
        .name("odon-control-bridge".to_string())
        .spawn(move || serve_control_bridge(listener, tx, local_addr, identity, ctx))?;
    Ok(ControlServerPublication {
        local_addr,
        manifest,
    })
}

fn serve_control_bridge(
    listener: TcpListener,
    tx: Sender<OdonControlRequest>,
    address: SocketAddr,
    identity: Arc<ControlServerIdentity>,
    ctx: egui::Context,
) {
    eprintln!("odon control server listening on {address}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let tx = tx.clone();
                let ctx = ctx.clone();
                let identity = Arc::clone(&identity);
                let _ = thread::Builder::new()
                    .name("odon-control-client".to_string())
                    .spawn(move || handle_control_client(stream, tx, identity, ctx));
            }
            Err(err) => eprintln!("odon control bridge accept failed: {err}"),
        }
    }
}

fn handle_control_client(
    stream: TcpStream,
    tx: Sender<OdonControlRequest>,
    identity: Arc<ControlServerIdentity>,
    ctx: egui::Context,
) {
    let Ok(mut write_stream) = stream.try_clone() else {
        return;
    };
    let (outbound_tx, outbound_rx) = crossbeam_channel::bounded::<Value>(256);
    let writer_thread = thread::Builder::new()
        .name("odon-control-writer".to_string())
        .spawn(move || {
            while let Ok(message) = outbound_rx.recv() {
                if writeln!(write_stream, "{}", message).is_err() || write_stream.flush().is_err() {
                    break;
                }
            }
        });
    if writer_thread.is_err() {
        return;
    }
    let reader = BufReader::new(stream);
    let mut state = match ConnectionState::new(&identity, outbound_tx.clone()) {
        Ok(state) => state,
        Err(_) => return,
    };
    identity
        .event_hub
        .register(state.hello_server.session_id.clone(), outbound_tx.clone());
    let (work_tx, work_rx) = crossbeam_channel::bounded::<ControlWork>(64);
    let mut workers = Vec::new();
    for index in 0..4 {
        let work_rx = work_rx.clone();
        let app_tx = tx.clone();
        let app_ctx = ctx.clone();
        let outbound = outbound_tx.clone();
        if let Ok(worker) = thread::Builder::new()
            .name(format!("odon-control-worker-{index}"))
            .spawn(move || {
                while let Ok(mut work) = work_rx.recv() {
                    if let Some(response) =
                        handle_control_line(&work.line, &app_tx, &app_ctx, &mut work.state)
                    {
                        let _ = outbound.send_timeout(response, Duration::from_secs(5));
                    }
                }
            })
        {
            workers.push(worker);
        }
    }
    for line in reader.lines() {
        let line = match line {
            Ok(line) if line.len() as u64 > MAX_INLINE_PAYLOAD_BYTES => {
                state.close_after_response = true;
                let _ = outbound_tx.send_timeout(
                    json_rpc_error(
                        Value::Null,
                        &ControlError::new(
                            ControlErrorKind::ResourceLimit,
                            "control message exceeds the negotiated inline payload limit",
                        ),
                    ),
                    Duration::from_secs(5),
                );
                break;
            }
            Ok(line) => line,
            Err(err) => {
                let _ = outbound_tx.send_timeout(
                    json_rpc_error(
                        Value::Null,
                        &ControlError::new(
                            ControlErrorKind::Internal,
                            format!("read failed: {err}"),
                        ),
                    ),
                    Duration::from_secs(5),
                );
                break;
            }
        };
        if state.hello_complete {
            match work_tx.try_send(ControlWork {
                line,
                state: state.clone(),
            }) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    let _ = outbound_tx.send_timeout(
                        json_rpc_error(
                            Value::Null,
                            &ControlError::new(
                                ControlErrorKind::NotReady,
                                "this control connection's request queue is full",
                            ),
                        ),
                        Duration::from_secs(5),
                    );
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => break,
            }
        } else {
            if let Some(response) = handle_control_line(&line, &tx, &ctx, &mut state)
                && outbound_tx
                    .send_timeout(response, Duration::from_secs(5))
                    .is_err()
            {
                break;
            }
            if state.close_after_response {
                break;
            }
        }
    }
    drop(work_tx);
    for worker in workers {
        let _ = worker.join();
    }
    identity
        .resource_registry
        .cleanup_session(&state.hello_server.session_id);
    let ui_cleanup = identity
        .ui_registry
        .cleanup_session(&state.hello_server.session_id);
    if !ui_cleanup.extensions.is_empty()
        && super::dispatch::call_actor_for_cleanup(
            &tx,
            &identity,
            "ui.commands.cleanup_extensions",
            json!({"extensions":ui_cleanup.extensions}),
        )
        .is_ok()
    {
        ctx.request_repaint();
    }
    reconcile_shell_focus_after_ui_cleanup(&tx, &identity, &ctx, &ui_cleanup.unavailable_mounts);
    identity.event_hub.remove(&state.hello_server.session_id);
}

fn reconcile_shell_focus_after_ui_cleanup(
    tx: &Sender<OdonControlRequest>,
    identity: &ControlServerIdentity,
    ctx: &egui::Context,
    unavailable_mounts: &[String],
) {
    if unavailable_mounts.is_empty() {
        return;
    }
    let Ok(snapshot) =
        super::dispatch::call_actor_for_cleanup(tx, identity, "ui.shell.get", json!({}))
    else {
        return;
    };
    let Some(params) =
        focus_reconciliation_patch(&snapshot, unavailable_mounts, identity.ui_registry.as_ref())
    else {
        return;
    };
    if super::dispatch::call_actor_for_cleanup(tx, identity, "ui.shell.patch_layout", params)
        .is_ok()
    {
        ctx.request_repaint();
    }
}

pub(super) fn focus_reconciliation_patch(
    snapshot: &Value,
    unavailable_mounts: &[String],
    ui_registry: &UiRegistry,
) -> Option<Value> {
    let nodes = snapshot.pointer("/layout/nodes")?.as_array()?;
    let node = |id: &str| {
        nodes
            .iter()
            .find(|node| node.get("id").and_then(Value::as_str) == Some(id))
    };
    let unavailable = |id: &str| {
        let mount = node(id)?.get("mount").and_then(Value::as_str)?;
        Some(
            unavailable_mounts
                .iter()
                .any(|candidate| candidate == mount)
                || (mount.starts_with("builtin:extension-host.")
                    && !ui_registry.shell_mount_available(mount, snapshot)),
        )
    };
    let active = snapshot.get("active_region_id").and_then(Value::as_str);
    let focused = snapshot.get("focused_node_id").and_then(Value::as_str);
    if !active.and_then(&unavailable).unwrap_or(false)
        && !focused.and_then(&unavailable).unwrap_or(false)
    {
        return None;
    }
    let required_mount = match snapshot.get("mode").and_then(Value::as_str)? {
        "project" => "builtin:project-workspace",
        "single" => "builtin:viewer-canvas",
        "mosaic" => "builtin:mosaic-canvas",
        _ => return None,
    };
    let fallback = nodes.iter().find_map(|node| {
        (node.get("mount").and_then(Value::as_str) == Some(required_mount))
            .then(|| node.get("id").and_then(Value::as_str))
            .flatten()
    })?;
    Some(json!({
        "mode":snapshot.get("mode")?.clone(),
        "if_shell_revision":snapshot.get("revision")?.clone(),
        "active_region_id":fallback,
        "focused_node_id":fallback,
        "transaction_id":"extension-disconnect-focus-reconciliation",
    }))
}

struct ControlWork {
    line: String,
    state: ConnectionState,
}
