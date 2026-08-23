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
    identity
        .ui_registry
        .cleanup_session(&state.hello_server.session_id);
    identity.event_hub.remove(&state.hello_server.session_id);
}

struct ControlWork {
    line: String,
    state: ConnectionState,
}
