use super::completion::{CompletionContext, reject_cancelled_request, request_is_cancelled};
use super::*;

pub(super) fn finish(completion: LoadCompletion, context: CompletionContext<'_>) {
    let CompletionContext {
        model,
        render_document,
        presentation_tx,
        presentation_coalesce_rx,
        wake_ui,
        diagnostics,
        ..
    } = context;
    match completion {
        LoadCompletion::ObjectResource {
            document_generation,
            resource_generation,
            request,
            path,
            downsample_factor,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_object_resource_for_generation(
                    document_generation,
                    resource_generation,
                    "object load was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "object load");
                return;
            }
            match result {
                Ok(resource) => {
                    let object_count = resource.features.len();
                    let property_count = resource.property_names.len();
                    if model.install_object_resource_for_generation(
                        document_generation,
                        resource_generation,
                        Arc::new(resource),
                    ) {
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        finish_request(
                            request,
                            json!({
                                "queued": true,
                                "loaded": true,
                                "path": path.to_string_lossy(),
                                "downsample_factor": downsample_factor,
                                "object_count": object_count,
                                "property_count": property_count,
                                "model_ready": true,
                                "resources_ready": true,
                                "presentation_ready": false,
                            }),
                            diagnostics,
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "object load was superseded by a newer document or object request",
                        )));
                    }
                }
                Err(error) => {
                    let message = format!("failed to load object resource: {error}");
                    if model.fail_object_resource_for_generation(
                        document_generation,
                        resource_generation,
                        &message,
                    ) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message)
                                .with_data(json!({"path": path.to_string_lossy()})),
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "failed object load was superseded by a newer request",
                        )));
                    }
                }
            }
        }
        LoadCompletion::Labels {
            document_generation,
            label_generation,
            request,
            name,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_label_load_for_generation(
                    document_generation,
                    label_generation,
                    "label load was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "label load");
                return;
            }
            match result {
                Ok(resource) => {
                    if model.install_label_resource_for_generation(
                        document_generation,
                        label_generation,
                        Arc::new(resource),
                    ) {
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        match model.dispatch("viewer.labels.get", &json!({})) {
                            Some(Ok(outcome)) => {
                                finish_request(request, outcome.response, diagnostics)
                            }
                            _ => reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(
                                    ControlErrorKind::Application,
                                    "label resource installed but its state could not be read",
                                ),
                            ),
                        }
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "label load was superseded by a newer document or label request",
                            ),
                        );
                    }
                }
                Err(error) => {
                    let message = format!("load labels/{name} failed: {error}");
                    if model.fail_label_load_for_generation(
                        document_generation,
                        label_generation,
                        &message,
                    ) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed label load was superseded by a newer request",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::MemoryPin {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                if model.cancel_memory_pin(&spec, "Memory pinning was cancelled") {
                    publish_projection(
                        model,
                        render_document.clone(),
                        presentation_tx,
                        presentation_coalesce_rx,
                        wake_ui,
                        diagnostics,
                    );
                }
                reject_cancelled_request(request, diagnostics, "memory pinning");
                return;
            }
            match result {
                Ok(MemoryPinWorkerResult { system, outcome }) => match outcome {
                    MemoryPinWorkerOutcome::Confirmation {
                        risk,
                        projected_bytes,
                        available_bytes,
                    } => {
                        if let Some(response) = model.finish_memory_pin_confirmation(
                            &spec,
                            system,
                            risk,
                            projected_bytes,
                            available_bytes,
                        ) {
                            publish_projection(
                                model,
                                render_document.clone(),
                                presentation_tx,
                                presentation_coalesce_rx,
                                wake_ui,
                                diagnostics,
                            );
                            finish_request(request, response, diagnostics);
                        } else {
                            diagnostics
                                .stale_worker_completions
                                .fetch_add(1, Ordering::Relaxed);
                            reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(
                                    ControlErrorKind::Conflict,
                                    "memory pin confirmation was superseded by newer state",
                                ),
                            );
                        }
                    }
                    MemoryPinWorkerOutcome::Loaded(resource) => {
                        if let Some(response) =
                            model.install_memory_pin(&spec, Arc::new(resource), system)
                        {
                            publish_projection(
                                model,
                                render_document.clone(),
                                presentation_tx,
                                presentation_coalesce_rx,
                                wake_ui,
                                diagnostics,
                            );
                            finish_request(request, response, diagnostics);
                        } else {
                            diagnostics
                                .stale_worker_completions
                                .fetch_add(1, Ordering::Relaxed);
                            reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(
                                    ControlErrorKind::Conflict,
                                    "memory pin result was superseded by newer state",
                                ),
                            );
                        }
                    }
                },
                Err(error) => {
                    let message = format!("memory pin failed: {error}");
                    if model.fail_memory_pin(&spec, &message) {
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed memory pin was superseded by newer state",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::ThresholdLoad {
            request,
            spec,
            result,
        } => {
            finish_threshold_preview_completion(
                model,
                render_document,
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
                request,
                spec.document_generation,
                spec.operation_generation,
                result,
            );
        }
        LoadCompletion::ThresholdRecompute {
            request,
            spec,
            result,
        } => {
            finish_threshold_preview_completion(
                model,
                render_document,
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
                request,
                spec.document_generation,
                spec.operation_generation,
                result,
            );
        }
        LoadCompletion::ThresholdApply {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_threshold_operation(
                    spec.document_generation,
                    spec.operation_generation,
                    "Threshold apply was cancelled",
                );
                publish_projection(
                    model,
                    render_document.clone(),
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                );
                reject_cancelled_request(request, diagnostics, "threshold apply");
                return;
            }
            match result {
                Ok(polygons) => {
                    if let Some(mut response) = model.install_threshold_mask(&spec, polygons) {
                        if request
                            .command
                            .params()
                            .get("sync_project")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            match model.sync_masks_to_project() {
                                Ok(synced) => {
                                    response["persistence"] = synced["persistence"].clone()
                                }
                                Err(error) => {
                                    reject_actor_request(request, diagnostics, error);
                                    return;
                                }
                            }
                        }
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        finish_request(request, response, diagnostics);
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "threshold apply was superseded by newer state",
                            ),
                        );
                    }
                }
                Err(error) => {
                    let message = format!("failed to apply threshold preview: {error}");
                    if model.fail_threshold_operation(
                        spec.document_generation,
                        spec.operation_generation,
                        &message,
                    ) {
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed threshold apply was superseded by newer state",
                            ),
                        );
                    }
                }
            }
        }
        _ => unreachable!("completion domain mismatch"),
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_threshold_preview_completion(
    model: &mut AppModel,
    render_document: &Option<Arc<RenderDocument>>,
    presentation_tx: &Sender<RenderProjection>,
    presentation_coalesce_rx: &Receiver<RenderProjection>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
    request: OdonControlRequest,
    document_generation: u64,
    operation_generation: u64,
    result: anyhow::Result<ControlThresholdPreviewResource>,
) {
    if request_is_cancelled(&request) {
        model.fail_threshold_operation(
            document_generation,
            operation_generation,
            "Threshold preview was cancelled",
        );
        publish_projection(
            model,
            render_document.clone(),
            presentation_tx,
            presentation_coalesce_rx,
            wake_ui,
            diagnostics,
        );
        reject_cancelled_request(request, diagnostics, "threshold preview");
        return;
    }
    match result {
        Ok(preview) => {
            if let Some(response) = model.install_threshold_preview(
                document_generation,
                operation_generation,
                Arc::new(preview),
            ) {
                publish_projection(
                    model,
                    render_document.clone(),
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                );
                finish_request(request, response, diagnostics);
            } else {
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Conflict,
                        "threshold preview was superseded by newer state",
                    ),
                );
            }
        }
        Err(error) => {
            let message = format!("failed to load threshold preview: {error}");
            if model.fail_threshold_operation(document_generation, operation_generation, &message) {
                publish_projection(
                    model,
                    render_document.clone(),
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                );
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(ControlErrorKind::Application, message),
                );
            } else {
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Conflict,
                        "failed threshold preview was superseded by newer state",
                    ),
                );
            }
        }
    }
}
