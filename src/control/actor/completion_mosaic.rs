use super::completion::{CompletionContext, reject_cancelled_request, request_is_cancelled};
use super::*;

pub(super) fn finish(completion: LoadCompletion, context: CompletionContext<'_>) {
    let CompletionContext {
        model,
        render_document,
        remote_session,
        presentation_tx,
        presentation_coalesce_rx,
        wake_ui,
        diagnostics,
        ..
    } = context;
    match completion {
        LoadCompletion::MosaicOpen {
            generation,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_mosaic_open(generation, "Mosaic opening was cancelled");
                reject_cancelled_request(request, diagnostics, "mosaic opening");
                return;
            }
            match result {
                Ok(result) => {
                    if result
                        .s3_session_generation
                        .is_some_and(|generation| !remote_session.is_current(generation))
                    {
                        model.fail_mosaic_open(
                            generation,
                            "S3 session changed during mosaic opening",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "S3 session changed during mosaic opening",
                            ),
                        );
                        return;
                    }
                    let source = result.resource.source.clone();
                    let roi_count = result.resource.items.len();
                    match model.install_mosaic_for_generation(generation, result.resource) {
                        Ok(true) => {
                            *render_document = None;
                            publish_projection(
                                model,
                                None,
                                presentation_tx,
                                presentation_coalesce_rx,
                                wake_ui,
                                diagnostics,
                            );
                            finish_request(
                                request,
                                json!({
                                    "opened":true,
                                    "mode":"mosaic",
                                    "source":source,
                                    "roi_count":roi_count,
                                    "model_ready":true,
                                    "resources_ready":true,
                                    "presentation_ready":false,
                                }),
                                diagnostics,
                            );
                        }
                        Ok(false) => {
                            reject_stale_mosaic(request, diagnostics, "mosaic opening");
                        }
                        Err(error) => {
                            let message =
                                format!("could not restore mosaic project state: {error}");
                            model.fail_mosaic_open(generation, &message);
                            reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(ControlErrorKind::Application, message),
                            );
                        }
                    }
                }
                Err(error) => {
                    let message = format!("failed to open mosaic: {error}");
                    if model.fail_mosaic_open(generation, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_mosaic(request, diagnostics, "mosaic opening");
                    }
                }
            }
        }
        LoadCompletion::MosaicObjects {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_mosaic_object_load(&spec, "Mosaic object loading was cancelled");
                reject_cancelled_request(request, diagnostics, "mosaic object loading");
                return;
            }
            match result {
                Ok(result) => {
                    if let Some(response) = model.finish_mosaic_object_load(&spec, result) {
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
                        reject_stale_mosaic(request, diagnostics, "mosaic object loading");
                    }
                }
                Err(error) => {
                    let message = format!("mosaic object loading failed: {error}");
                    if model.fail_mosaic_object_load(&spec, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_mosaic(request, diagnostics, "mosaic object loading");
                    }
                }
            }
        }
        LoadCompletion::MosaicMemoryPin {
            request,
            spec,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_mosaic_memory_pin(&spec, "Mosaic memory pinning was cancelled");
                reject_cancelled_request(request, diagnostics, "mosaic memory pinning");
                return;
            }
            match result {
                Ok(MosaicMemoryPinWorkerResult { system, outcome }) => {
                    let response = match outcome {
                        MosaicMemoryPinWorkerOutcome::Confirmation {
                            risk,
                            projected_bytes,
                            available_bytes,
                        } => model.finish_mosaic_memory_confirmation(
                            &spec,
                            system,
                            risk,
                            projected_bytes,
                            available_bytes,
                        ),
                        MosaicMemoryPinWorkerOutcome::Loaded(result) => {
                            model.install_mosaic_memory_pin(&spec, result, system)
                        }
                    };
                    if let Some(response) = response {
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
                        reject_stale_mosaic(request, diagnostics, "mosaic memory pinning");
                    }
                }
                Err(error) => {
                    let message = format!("mosaic memory pinning failed: {error}");
                    if model.fail_mosaic_memory_pin(&spec, &message) {
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
                        reject_stale_mosaic(request, diagnostics, "mosaic memory pinning");
                    }
                }
            }
        }
        _ => unreachable!("non-mosaic completion reached mosaic completion dispatcher"),
    }
}

fn reject_stale_mosaic(
    request: OdonControlRequest,
    diagnostics: &ActorDiagnostics,
    operation: &str,
) {
    diagnostics
        .stale_worker_completions
        .fetch_add(1, Ordering::Relaxed);
    reject_actor_request(
        request,
        diagnostics,
        ControlError::new(
            ControlErrorKind::Conflict,
            format!("{operation} was superseded by newer mosaic state"),
        ),
    );
}
