use super::*;

pub(super) fn run_batch(
    params: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &mut ConnectionState,
) -> Result<Value, ControlError> {
    if params
        .get("atomic")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Err(ControlError::new(
            ControlErrorKind::Unsupported,
            "atomic batches are not supported in protocol v1",
        ));
    }
    let operations = params
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ControlError::invalid_params("system.batch", "operations must be an array")
        })?;
    if operations.is_empty() || operations.len() > 128 {
        return Err(ControlError::invalid_params(
            "system.batch",
            "operations must contain between 1 and 128 commands",
        ));
    }
    let mut results = Vec::with_capacity(operations.len());
    for (index, operation) in operations.iter().enumerate() {
        let method = operation
            .get("method")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "system.batch",
                    format!("operation {index} has no method"),
                )
            })?;
        if method.starts_with("system.")
            || method.starts_with("events.")
            || method.starts_with("tasks.")
            || method.starts_with("ui.")
            || method.starts_with("data.")
            || method.starts_with("viewer.layers.")
        {
            return Err(ControlError::invalid_params(
                "system.batch",
                format!("operation {index} is not an application command"),
            ));
        }
        let result = dispatch_to_app(
            method,
            operation
                .get("params")
                .cloned()
                .unwrap_or_else(|| json!({})),
            tx,
            ctx,
            state,
            None,
        )?;
        results.push(json!({"method": method, "result": result}));
    }
    Ok(json!({
        "atomic": false,
        "results": results,
        "revision": state.event_hub.revision(),
    }))
}

pub(super) fn start_task(
    params: &Value,
    tx: &Sender<OdonControlRequest>,
    ctx: &egui::Context,
    state: &ConnectionState,
) -> Result<Value, ControlError> {
    let method = params
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params("tasks.start", "method is required"))?;
    if method.starts_with("system.")
        || method.starts_with("events.")
        || method.starts_with("tasks.")
    {
        return Err(ControlError::invalid_params(
            "tasks.start",
            "only application control methods can run as tasks",
        ));
    }
    let command = ControlCommand::decode(
        method,
        params.get("params").cloned().unwrap_or_else(|| json!({})),
    )?;
    if !command.starts_task() {
        return Err(ControlError::invalid_params(
            "tasks.start",
            format!(
                "{} is an immediate method; call it directly instead of starting a task",
                command.method()
            ),
        ));
    }
    let operation_method = command.method().to_string();
    let operation_params = command.params().clone();
    let snapshot = state.task_service.create(
        params
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or(method),
        state.hello_server.session_id.clone(),
        true,
    )?;
    let task_id = snapshot.task_id.clone();
    let (reply_tx, reply_rx) = crossbeam_channel::bounded::<Result<Value, ControlError>>(1);
    tx.try_send(OdonControlRequest {
        command,
        reply: reply_tx,
        session_id: state.hello_server.session_id.clone(),
        request_id: None,
        event_hub: Arc::clone(&state.event_hub),
        task_registry: Arc::clone(&state.task_registry),
        task_id: Some(task_id.clone()),
    })
    .map_err(|error| match error {
        crossbeam_channel::TrySendError::Full(_) => ControlError::new(
            ControlErrorKind::NotReady,
            "Odon control queue is full; retry later",
        ),
        crossbeam_channel::TrySendError::Disconnected(_) => ControlError::new(
            ControlErrorKind::NotReady,
            "Odon app is not accepting control requests",
        ),
    })?;
    ctx.request_repaint();
    let tasks = state.task_service.clone();
    let app_tx = tx.clone();
    let app_ctx = ctx.clone();
    let session_id = state.hello_server.session_id.clone();
    let event_hub = Arc::clone(&state.event_hub);
    thread::Builder::new()
        .name("odon-control-task".to_string())
        .spawn(move || match reply_rx.recv() {
            Ok(Ok(value)) => {
                let settled = if matches!(
                    operation_method.as_str(),
                    "project.open"
                        | "datasets.open_ome_zarr"
                        | "datasets.open_tiff"
                        | "datasets.open_spatialdata"
                        | "datasets.open_xenium"
                        | "datasets.open_http"
                        | "datasets.open_s3"
                        | "datasets.open_mosaic_samplesheet"
                        | "project.rois.open"
                ) {
                    let _ = tasks.progress(&task_id, None, "waiting for viewer readiness");
                    wait_for_application_ready(
                        &operation_method,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "project.objects.preload.start" {
                    let _ = tasks.progress(&task_id, Some(0.0), "preloading project objects");
                    wait_for_project_object_preload(
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if matches!(
                    operation_method.as_str(),
                    "mosaic.objects.load" | "mosaic.objects.load_selected"
                ) {
                    let _ = tasks.progress(&task_id, None, "loading selected mosaic objects");
                    wait_for_mosaic_object_load(
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "deep_links.apply" {
                    let _ = tasks.progress(&task_id, None, "applying deep link");
                    wait_for_deep_link_application(
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "viewer.objects.properties.load" {
                    let _ = tasks.progress(&task_id, None, "loading object property");
                    wait_for_object_property_load(
                        &value,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if matches!(
                    operation_method.as_str(),
                    "viewer.objects.source.load" | "viewer.objects.source.reload"
                ) {
                    let _ = tasks.progress(&task_id, None, "loading object source");
                    wait_for_object_source_load(
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "viewer.analysis.warmup.start" {
                    let _ = tasks.progress(&task_id, None, "warming analysis properties");
                    wait_for_control_operation(
                        "viewer.analysis.warmup.get",
                        None,
                        &operation_params,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "viewer.measurements.start" {
                    let _ = tasks.progress(&task_id, Some(0.0), "measuring polygon intensities");
                    wait_for_control_operation(
                        "viewer.measurements.get",
                        Some("viewer.measurements.cancel"),
                        &operation_params,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if matches!(
                    operation_method.as_str(),
                    "exports.objects.start"
                        | "exports.objects.export_csv"
                        | "exports.objects.export_geoparquet"
                ) {
                    let _ = tasks.progress(&task_id, None, "exporting enriched objects");
                    wait_for_control_operation(
                        "exports.objects.get_state",
                        None,
                        &operation_params,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method == "memory.pin" {
                    let _ = tasks.progress(&task_id, None, "pinning image level in RAM");
                    wait_for_control_operation(
                        "memory.get",
                        None,
                        &operation_params,
                        &app_tx,
                        &app_ctx,
                        &session_id,
                        &event_hub,
                        &tasks,
                        &task_id,
                    )
                } else if operation_method.ends_with(".screenshot.capture") {
                    let _ = tasks.progress(&task_id, None, "waiting for screenshot output");
                    wait_for_output_path(&value, &tasks, &task_id)
                } else {
                    Ok(())
                };
                match settled {
                    Ok(()) => {
                        let _ = tasks.complete(&task_id, value);
                    }
                    Err(error) => {
                        if error.kind != ControlErrorKind::Cancelled {
                            let _ = tasks.fail(&task_id, &error);
                        }
                    }
                }
            }
            Ok(Err(error)) => {
                if error.kind != ControlErrorKind::Cancelled {
                    let _ = tasks.fail(&task_id, &error);
                }
            }
            Err(_) => {
                let _ = tasks.fail(
                    &task_id,
                    &ControlError::new(
                        ControlErrorKind::NotReady,
                        "Odon closed before the task completed",
                    ),
                );
            }
        })
        .map_err(|error| {
            ControlError::new(
                ControlErrorKind::Internal,
                format!("failed to start task monitor: {error}"),
            )
        })?;
    serde_json::to_value(snapshot).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize task: {error}"),
        )
    })
}
