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
        _ => unreachable!("completion domain mismatch"),
    }
}
