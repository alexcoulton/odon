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
        LoadCompletion::ObjectFilter {
            document_generation,
            resource_generation,
            operation_generation,
            viewport_id,
            expected_presentation_revision,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_object_filter_for_generation(
                    &viewport_id,
                    operation_generation,
                    "Object filter evaluation cancelled",
                );
                reject_cancelled_request(request, diagnostics, "object filter evaluation");
                return;
            }
            match result {
                Ok(result) => {
                    if let Some(response) = model.install_object_filter_for_generation(
                        document_generation,
                        resource_generation,
                        operation_generation,
                        &viewport_id,
                        expected_presentation_revision,
                        result,
                    ) {
                        let response = if matches!(
                            request.command.method(),
                            "viewer.objects.set_filter" | "viewer.objects.filters.set_model"
                        ) {
                            json!({
                                "target":"segmentation_objects",
                                "filter":response["result"].clone(),
                            })
                        } else {
                            response
                        };
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
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "object filter evaluation was superseded by newer object or viewport state",
                        )));
                    }
                }
                Err(error) => {
                    let current = model.fail_object_filter_for_generation(
                        &viewport_id,
                        operation_generation,
                        format!("Invalid object filter: {error}"),
                    );
                    if current {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::InvalidParams,
                                format!("invalid object filter: {error}"),
                            ),
                        );
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "failed object filter evaluation was superseded by newer state",
                        )));
                    }
                }
            }
        }
        LoadCompletion::ObjectSelectionFilter {
            document_generation,
            resource_generation,
            selection_generation,
            operation_generation,
            request,
            mode,
            limit,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_object_selection_filter_for_generation(
                    operation_generation,
                    "Object selection filter cancelled",
                );
                reject_cancelled_request(request, diagnostics, "object selection filter");
                return;
            }
            match result {
                Ok(result) => {
                    if let Some(response) = model.install_object_selection_filter_for_generation(
                        document_generation,
                        resource_generation,
                        selection_generation,
                        operation_generation,
                        result,
                        &mode,
                        limit,
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
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "object selection filter was superseded by newer object or selection state",
                        )));
                    }
                }
                Err(error) => {
                    model.fail_object_selection_filter_for_generation(
                        operation_generation,
                        format!("Invalid standalone object filter: {error}"),
                    );
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::InvalidParams,
                            format!("invalid standalone object filter: {error}"),
                        ),
                    );
                }
            }
        }
        LoadCompletion::AnalysisCompute {
            request,
            spec,
            kind,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_analysis_operation(&spec, "Analysis was cancelled");
                reject_cancelled_request(request, diagnostics, "analysis");
                return;
            }
            match result {
                Ok(response) => {
                    let response = if matches!(kind, AnalysisComputeKind::Warmup) {
                        response
                            .get("completed")
                            .and_then(Value::as_u64)
                            .and_then(|value| usize::try_from(value).ok())
                            .and_then(|completed| model.finish_analysis_warmup(&spec, completed))
                    } else {
                        model.finish_analysis_operation(&spec).then_some(response)
                    };
                    if let Some(response) = response {
                        if matches!(kind, AnalysisComputeKind::Warmup) {
                            publish_projection(
                                model,
                                render_document.clone(),
                                presentation_tx,
                                presentation_coalesce_rx,
                                wake_ui,
                                diagnostics,
                            );
                        }
                        finish_request(request, response, diagnostics);
                    } else {
                        reject_stale_analysis(request, diagnostics);
                    }
                }
                Err(error) => {
                    let message = format!("analysis failed: {error}");
                    if model.fail_analysis_operation(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_analysis(request, diagnostics);
                    }
                }
            }
        }
        LoadCompletion::AnalysisPresetImport {
            request,
            spec,
            path,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_analysis_operation(&spec, "Analysis preset import was cancelled");
                reject_cancelled_request(request, diagnostics, "analysis preset import");
                return;
            }
            match result {
                Ok(state) => match model.install_analysis_preset(&spec, state, &path) {
                    Some(Ok(response)) => {
                        publish_projection(
                            model,
                            render_document.clone(),
                            presentation_tx,
                            presentation_coalesce_rx,
                            wake_ui,
                            diagnostics,
                        );
                        finish_request(request, response, diagnostics);
                    }
                    Some(Err(error)) => reject_actor_request(request, diagnostics, error),
                    None => reject_stale_analysis(request, diagnostics),
                },
                Err(error) => {
                    let message = format!("failed to import analysis preset: {error}");
                    if model.fail_analysis_operation(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_analysis(request, diagnostics);
                    }
                }
            }
        }
        LoadCompletion::AnalysisPresetExport {
            request,
            spec,
            path,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_analysis_operation(&spec, "Analysis preset export was cancelled");
                reject_cancelled_request(request, diagnostics, "analysis preset export");
                return;
            }
            match result {
                Ok(call_count) if model.finish_analysis_operation(&spec) => finish_request(
                    request,
                    json!({"exported":true,"path":path.to_string_lossy(),"call_count":call_count}),
                    diagnostics,
                ),
                Ok(_) => reject_stale_analysis(request, diagnostics),
                Err(error) => {
                    let message = format!("failed to export analysis preset: {error}");
                    if model.fail_analysis_operation(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_analysis(request, diagnostics);
                    }
                }
            }
        }
        LoadCompletion::Measurement {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_measurement(&spec, "Measurement was cancelled");
                reject_cancelled_request(request, diagnostics, "measurement");
                return;
            }
            match result {
                Ok((resource, measured)) => {
                    if let Some(response) = model.install_measurement(&spec, resource, measured) {
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
                        reject_stale_analysis(request, diagnostics);
                    }
                }
                Err(error) => {
                    let message = format!("measurement failed: {error}");
                    if model.fail_measurement(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_analysis(request, diagnostics);
                    }
                }
            }
        }
        LoadCompletion::ObjectExport {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_object_export(&spec, "Object export was cancelled");
                reject_cancelled_request(request, diagnostics, "object export");
                return;
            }
            match result {
                Ok(result) => {
                    if let Some(response) = model.finish_object_export(&spec, &result) {
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
                        reject_stale_object_export(request, diagnostics);
                    }
                }
                Err(error) => {
                    let message = format!("object export failed: {error}");
                    if model.fail_object_export(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_object_export(request, diagnostics);
                    }
                }
            }
        }
        _ => unreachable!("completion domain mismatch"),
    }
}

fn reject_stale_object_export(request: OdonControlRequest, diagnostics: &ActorDiagnostics) {
    diagnostics
        .stale_worker_completions
        .fetch_add(1, Ordering::Relaxed);
    reject_actor_request(
        request,
        diagnostics,
        ControlError::new(
            ControlErrorKind::Conflict,
            "object export was superseded by newer document or object state",
        ),
    );
}

fn reject_stale_analysis(request: OdonControlRequest, diagnostics: &ActorDiagnostics) {
    diagnostics
        .stale_worker_completions
        .fetch_add(1, Ordering::Relaxed);
    reject_actor_request(
        request,
        diagnostics,
        ControlError::new(
            ControlErrorKind::Conflict,
            "analysis result was superseded by newer object or analysis state",
        ),
    );
}
