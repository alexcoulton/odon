use super::*;

pub(super) fn wait_for_project_object_preload(
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    thread::sleep(Duration::from_millis(50));
    loop {
        if let Err(error) = ensure_task_not_cancelled(tasks, task_id) {
            if let Ok(command) = ControlCommand::decode("project.objects.preload.clear", json!({}))
            {
                let (reply, _) = crossbeam_channel::bounded(1);
                let _ = tx.send_timeout(
                    OdonControlRequest {
                        command,
                        reply,
                        session_id: session_id.to_string(),
                        request_id: None,
                        event_hub: Arc::clone(event_hub),
                        task_registry: tasks.registry(),
                        task_id: None,
                    },
                    Duration::from_secs(1),
                );
                ctx.request_repaint();
            }
            return Err(error);
        }
        let command = ControlCommand::decode("project.objects.preload.get", json!({}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting project preload checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during a project preload check",
            )
        })??;
        let loading = state
            .get("loading")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let done = state.get("done").and_then(Value::as_u64).unwrap_or(0);
        let total = state.get("total").and_then(Value::as_u64).unwrap_or(0);
        let failed = state.get("failed").and_then(Value::as_u64).unwrap_or(0);
        let progress = (total > 0).then(|| done.min(total) as f64 / total as f64);
        let _ = tasks.progress(
            task_id,
            progress,
            format!("preloaded {done}/{total} object sources ({failed} failed)"),
        );
        if !loading {
            if failed > 0 && failed == total {
                return Err(ControlError::new(
                    ControlErrorKind::Internal,
                    format!("all {failed} project object sources failed to preload"),
                ));
            }
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn wait_for_mosaic_object_load(
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    thread::sleep(Duration::from_millis(50));
    loop {
        if let Err(error) = ensure_task_not_cancelled(tasks, task_id) {
            if let Ok(command) = ControlCommand::decode("mosaic.objects.cancel_load", json!({})) {
                let (reply, _) = crossbeam_channel::bounded(1);
                let _ = tx.send_timeout(
                    OdonControlRequest {
                        command,
                        reply,
                        session_id: session_id.to_string(),
                        request_id: None,
                        event_hub: Arc::clone(event_hub),
                        task_registry: tasks.registry(),
                        task_id: None,
                    },
                    Duration::from_secs(1),
                );
                ctx.request_repaint();
            }
            return Err(error);
        }
        let command = ControlCommand::decode("mosaic.objects.get_state", json!({}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting mosaic object checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during a mosaic object check",
            )
        })??;
        let objects = state.get("objects").unwrap_or(&state);
        let pending = objects
            .get("requested_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let loading = objects
            .get("requested_loading")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let _ = tasks.progress(
            task_id,
            None,
            format!("waiting for {pending} selected ROI object load(s); {loading} reading"),
        );
        if pending == 0 {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn wait_for_deep_link_application(
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    thread::sleep(Duration::from_millis(50));
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        let command = ControlCommand::decode("get_loading_state", json!({}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting deep-link readiness checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during a deep-link readiness check",
            )
        })??;
        let mode = state
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("transition");
        let pending = state
            .get("pending_deep_link")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let busy = state
            .get("loading")
            .and_then(|loading| loading.get("busy"))
            .and_then(Value::as_bool)
            .or_else(|| state.get("busy").and_then(Value::as_bool))
            .unwrap_or(true);
        let _ = tasks.progress(
            task_id,
            None,
            format!("deep link: mode={mode}, queued={pending}, viewer_busy={busy}"),
        );
        if !pending && mode != "transition" && !busy {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn wait_for_object_property_load(
    initial: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    let property = find_string_field(initial, "property")
        .ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::Internal,
                "property load response omitted property",
            )
        })?
        .to_string();
    let target = find_string_field(initial, "target").unwrap_or("segmentation_objects");
    let mut params = json!({"property": property, "offset": 0, "limit": 1, "target": target});
    if target == "spatial_shape"
        && let Some(layer_id) = find_u64_field(initial, "layer_id")
    {
        params["layer_id"] = json!(layer_id);
    }
    thread::sleep(Duration::from_millis(50));
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        let command = ControlCommand::decode("viewer.objects.properties.values", params.clone())?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting object property checks",
            )
        })?;
        ctx.request_repaint();
        match response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during an object property check",
            )
        })? {
            Ok(state) => {
                let load_required = find_bool_field(&state, "load_required").unwrap_or(false);
                if !load_required {
                    return Ok(());
                }
                let loading = find_bool_field(&state, "loading").unwrap_or(false);
                if !loading {
                    return Err(ControlError::new(
                        ControlErrorKind::Application,
                        format!("object property '{property}' did not load"),
                    ));
                }
            }
            Err(error) if error.kind == ControlErrorKind::InvalidParams => return Err(error),
            Err(_) => {}
        }
        let _ = tasks.progress(
            task_id,
            None,
            format!("loading object property '{property}'"),
        );
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn wait_for_object_source_load(
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    thread::sleep(Duration::from_millis(50));
    loop {
        if let Err(error) = ensure_task_not_cancelled(tasks, task_id) {
            if let Ok(command) =
                ControlCommand::decode("viewer.objects.source.cancel_load", json!({}))
            {
                let (reply, _) = crossbeam_channel::bounded(1);
                let _ = tx.send_timeout(
                    OdonControlRequest {
                        command,
                        reply,
                        session_id: session_id.to_string(),
                        request_id: None,
                        event_hub: Arc::clone(event_hub),
                        task_registry: tasks.registry(),
                        task_id: None,
                    },
                    Duration::from_secs(1),
                );
                ctx.request_repaint();
            }
            return Err(error);
        }
        let command =
            ControlCommand::decode("viewer.objects.get_state", json!({"target": "objects"}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting object-source checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during an object-source check",
            )
        })??;
        let loading = find_bool_field(&state, "loading_data").unwrap_or(false);
        let status = find_string_field(&state, "status").unwrap_or("loading objects");
        let _ = tasks.progress(task_id, None, status);
        if !loading {
            if status.starts_with("Object load failed:") {
                return Err(ControlError::new(ControlErrorKind::Application, status));
            }
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn wait_for_control_operation(
    state_method: &str,
    cancel_method: Option<&str>,
    params: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    thread::sleep(Duration::from_millis(50));
    loop {
        if let Err(error) = ensure_task_not_cancelled(tasks, task_id) {
            if let Some(cancel_method) = cancel_method
                && let Ok(command) = ControlCommand::decode(cancel_method, params.clone())
            {
                let (reply, _) = crossbeam_channel::bounded(1);
                let _ = tx.send_timeout(
                    OdonControlRequest {
                        command,
                        reply,
                        session_id: session_id.to_string(),
                        request_id: None,
                        event_hub: Arc::clone(event_hub),
                        task_registry: tasks.registry(),
                        task_id: None,
                    },
                    Duration::from_secs(1),
                );
                ctx.request_repaint();
            }
            return Err(error);
        }
        let command = ControlCommand::decode(state_method, params.clone())?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                format!("Odon stopped accepting {state_method} checks"),
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                format!("Odon closed during a {state_method} check"),
            )
        })??;
        let running = find_bool_field(&state, "running").unwrap_or(false);
        let status = find_string_field(&state, "status").unwrap_or(state_method);
        let completed = find_u64_field(&state, "completed");
        let total = find_u64_field(&state, "total");
        let progress = completed.zip(total).and_then(|(completed, total)| {
            (total > 0).then(|| completed.min(total) as f64 / total as f64)
        });
        let _ = tasks.progress(task_id, progress, status);
        if !running {
            if status.starts_with("Measurements failed:") || status.starts_with("Export failed:") {
                return Err(ControlError::new(ControlErrorKind::Application, status));
            }
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn wait_for_application_ready(
    operation: &str,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    session_id: &str,
    event_hub: &Arc<EventHub>,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    let expected_mode = match operation {
        "project.open" => "project",
        "datasets.open_mosaic_samplesheet" => "mosaic",
        _ => "single",
    };
    thread::sleep(Duration::from_millis(50));
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        let command = ControlCommand::decode("get_loading_state", json!({}))?;
        let (reply, response) = crossbeam_channel::bounded(1);
        tx.send_timeout(
            OdonControlRequest {
                command,
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(event_hub),
                task_registry: tasks.registry(),
                task_id: None,
            },
            Duration::from_secs(5),
        )
        .map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon stopped accepting readiness checks",
            )
        })?;
        ctx.request_repaint();
        let state = response.recv().map_err(|_| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "Odon closed during a readiness check",
            )
        })??;
        if application_state_is_ready(&state, expected_mode) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(super) fn application_state_is_ready(state: &Value, expected_mode: &str) -> bool {
    let mode_matches = state.get("mode").and_then(Value::as_str) == Some(expected_mode);
    let busy = state
        .get("loading")
        .and_then(|loading| loading.get("busy"))
        .and_then(Value::as_bool)
        .or_else(|| state.get("busy").and_then(Value::as_bool))
        .unwrap_or(true);
    let loading = state.get("loading").unwrap_or(state);
    let actor_readiness = loading.get("model_ready").is_some()
        || loading.get("resources_ready").is_some()
        || loading.get("geometry_ready").is_some();
    let work_ready = if actor_readiness {
        loading
            .get("model_ready")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            && loading
                .get("resources_ready")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && loading
                .get("geometry_ready")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    } else {
        expected_mode == "project"
            || state
                .get("loading")
                .and_then(|loading| loading.get("canvas_ready"))
                .and_then(Value::as_bool)
                .or_else(|| state.get("canvas_ready").and_then(Value::as_bool))
                .unwrap_or(false)
    };
    mode_matches && !busy && work_ready
}

pub(super) fn wait_for_output_path(
    value: &Value,
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    let Some(path) = find_string_field(value, "path") else {
        return Ok(());
    };
    let path = std::path::PathBuf::from(path);
    loop {
        ensure_task_not_cancelled(tasks, task_id)?;
        if std::fs::metadata(&path).is_ok_and(|metadata| metadata.len() > 0) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn ensure_task_not_cancelled(
    tasks: &crate::control::TaskServiceHandle,
    task_id: &str,
) -> Result<(), ControlError> {
    if tasks.get(task_id)?.state == TaskState::Cancelled {
        Err(
            ControlError::new(ControlErrorKind::Cancelled, "task was cancelled")
                .with_data(json!({"task_id": task_id})),
        )
    } else {
        Ok(())
    }
}

fn find_string_field<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    match value {
        Value::Object(object) => object.get(field).and_then(Value::as_str).or_else(|| {
            object
                .values()
                .find_map(|value| find_string_field(value, field))
        }),
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_string_field(value, field)),
        _ => None,
    }
}

fn find_u64_field(value: &Value, field: &str) -> Option<u64> {
    match value {
        Value::Object(object) => object.get(field).and_then(Value::as_u64).or_else(|| {
            object
                .values()
                .find_map(|value| find_u64_field(value, field))
        }),
        Value::Array(values) => values.iter().find_map(|value| find_u64_field(value, field)),
        _ => None,
    }
}

fn find_bool_field(value: &Value, field: &str) -> Option<bool> {
    match value {
        Value::Object(object) => object.get(field).and_then(Value::as_bool).or_else(|| {
            object
                .values()
                .find_map(|value| find_bool_field(value, field))
        }),
        Value::Array(values) => values
            .iter()
            .find_map(|value| find_bool_field(value, field)),
        _ => None,
    }
}
