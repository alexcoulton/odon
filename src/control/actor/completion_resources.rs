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
        _ => unreachable!("completion domain mismatch"),
    }
}
