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
        LoadCompletion::DatasetInspect {
            operation_generation,
            operation_scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_dataset_inspection(
                    &operation_scope,
                    operation_generation,
                    "Dataset inspection was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "dataset inspection");
                return;
            }
            if !model.finish_dataset_inspection(
                &operation_scope,
                operation_generation,
                "Dataset inspection complete",
            ) {
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
            }
            match serde_json::to_value(result) {
                Ok(value) => finish_request(request, value, diagnostics),
                Err(error) => reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Internal,
                        format!("failed to serialize dataset inspection: {error}"),
                    ),
                ),
            }
        }
        LoadCompletion::DeepLinkResolve {
            operation_generation,
            operation_scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_deep_link_resolution(
                    &operation_scope,
                    operation_generation,
                    "Deep-link resolution was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "deep-link resolution");
                return;
            }
            if !model.finish_deep_link_resolution(&operation_scope, operation_generation) {
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
            }
            finish_request(
                request,
                deep_link_resolution_response(result.request, result.resolution),
                diagnostics,
            );
        }
        LoadCompletion::OmeZarr {
            generation,
            request,
            path,
            result,
        } => {
            let cancelled = request
                .task_id
                .as_deref()
                .and_then(|task_id| request.task_registry.get(task_id).ok())
                .is_some_and(|task| task.state == TaskState::Cancelled);
            if cancelled {
                model.fail_dataset_open_for_generation(generation, "dataset open was cancelled");
                let _ = request.reply.send(Err(ControlError::new(
                    ControlErrorKind::Cancelled,
                    "dataset open was cancelled",
                )));
                return;
            }
            match result {
                Ok((opened, label_available, root_label_resource)) => {
                    let root_label_resource = root_label_resource.map(Arc::new);
                    if !model.install_document_for_generation(
                        generation,
                        opened.descriptor.clone(),
                        label_available,
                        root_label_resource.clone(),
                    ) {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "dataset open result was superseded by a newer request",
                        )
                        .with_data(
                            json!({"path": path.to_string_lossy(), "generation": generation}),
                        )));
                        return;
                    }
                    *render_document = Some(Arc::new(RenderDocument { generation, opened }));
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
                            "opened": true,
                            "mode": "single",
                            "kind": "ome_zarr",
                            "path": path.to_string_lossy(),
                            "model_ready": true,
                            "resources_ready": true,
                            "presentation_ready": false,
                        }),
                        diagnostics,
                    );
                }
                Err(error) => {
                    let message = format!("failed to open OME-Zarr dataset: {error}");
                    if model.fail_dataset_open_for_generation(generation, &message) {
                        diagnostics
                            .rejected_requests
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Application,
                            message,
                        )
                        .with_data(json!({"path": path.to_string_lossy()}))));
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        diagnostics.record_reply_time(request.command.queue_age());
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "failed dataset open was superseded by a newer request",
                        )
                        .with_data(
                            json!({"path": path.to_string_lossy(), "generation": generation}),
                        )));
                    }
                }
            }
        }
        LoadCompletion::RemoteList {
            session_generation,
            operation_generation,
            operation_scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_remote_listing(
                    &operation_scope,
                    operation_generation,
                    "Remote S3 listing was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "remote S3 listing");
                return;
            }
            if !remote_session.is_current(session_generation) {
                model.cancel_remote_listing(
                    &operation_scope,
                    operation_generation,
                    "S3 session changed before listing completed",
                );
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Conflict,
                        "S3 listing was superseded by a session change",
                    ),
                );
                return;
            }
            match result {
                Ok(listing) => {
                    if !model.finish_remote_listing(&operation_scope, operation_generation) {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "S3 listing was superseded by a newer request",
                            ),
                        );
                        return;
                    }
                    let session = remote_session.snapshot();
                    finish_request(
                        request,
                        json!({
                            "endpoint": session["endpoint"],
                            "region": session["region"],
                            "bucket": session["bucket"],
                            "prefix": listing.prefix,
                            "parent_prefix": listing.parent_prefix,
                            "current_is_dataset": listing.current_is_dataset,
                            "entries": listing.entries,
                            "session_generation": session_generation,
                        }),
                        diagnostics,
                    );
                }
                Err(error) => {
                    let message = format!("failed to list S3 prefix: {error}");
                    model.fail_remote_listing(&operation_scope, operation_generation, &message);
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(ControlErrorKind::Application, message),
                    );
                }
            }
        }
        LoadCompletion::RemoteOpen {
            generation,
            session_generation,
            request,
            identity,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_dataset_open_for_generation(generation, "remote open was cancelled");
                remote_session.finish_s3_open(generation);
                reject_cancelled_request(request, diagnostics, "remote dataset open");
                return;
            }
            if let Some(session_generation) = session_generation
                && !remote_session.is_current(session_generation)
            {
                model.fail_dataset_open_for_generation(
                    generation,
                    "S3 session changed before dataset open completed",
                );
                remote_session.finish_s3_open(generation);
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Conflict,
                        "S3 dataset open was superseded by a session change",
                    ),
                );
                return;
            }
            match result {
                Ok((opened, label_available, root_label_resource)) => {
                    let root_label_resource = root_label_resource.map(Arc::new);
                    if !model.install_document_for_generation(
                        generation,
                        opened.descriptor.clone(),
                        label_available,
                        root_label_resource,
                    ) {
                        remote_session.finish_s3_open(generation);
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "remote dataset open was superseded by a newer request",
                            ),
                        );
                        return;
                    }
                    *render_document = Some(Arc::new(RenderDocument { generation, opened }));
                    remote_session.finish_s3_open(generation);
                    publish_projection(
                        model,
                        render_document.clone(),
                        presentation_tx,
                        presentation_coalesce_rx,
                        wake_ui,
                        diagnostics,
                    );
                    let mut response = remote_open_response(&identity);
                    response.as_object_mut().unwrap().extend([
                        ("opened".to_string(), Value::Bool(true)),
                        ("mode".to_string(), Value::String("single".to_string())),
                        ("model_ready".to_string(), Value::Bool(true)),
                        ("resources_ready".to_string(), Value::Bool(true)),
                        ("presentation_ready".to_string(), Value::Bool(false)),
                    ]);
                    finish_request(request, response, diagnostics);
                }
                Err(error) => {
                    remote_session.finish_s3_open(generation);
                    let message = match identity {
                        RemoteOpenIdentity::Http { .. } => {
                            format!("failed to open remote OME-Zarr: {error}")
                        }
                        RemoteOpenIdentity::S3 { .. } => {
                            format!("failed to open S3 OME-Zarr: {error}")
                        }
                    };
                    if model.fail_dataset_open_for_generation(generation, &message) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message)
                                .with_data(remote_open_response(&identity)),
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
                                "failed remote dataset open was superseded by a newer request",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::ChannelIntensity {
            generation,
            request,
            result,
        } => {
            let cancelled = request
                .task_id
                .as_deref()
                .and_then(|task_id| request.task_registry.get(task_id).ok())
                .is_some_and(|task| task.state == TaskState::Cancelled);
            if cancelled {
                diagnostics.record_reply_time(request.command.queue_age());
                let _ = request.reply.send(Err(ControlError::new(
                    ControlErrorKind::Cancelled,
                    "channel intensity statistics were cancelled",
                )));
                return;
            }
            if generation != model.document_generation() {
                diagnostics
                    .stale_worker_completions
                    .fetch_add(1, Ordering::Relaxed);
                diagnostics.record_reply_time(request.command.queue_age());
                let _ = request.reply.send(Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    "channel intensity statistics were superseded by a newer document",
                )));
                return;
            }
            match result {
                Ok(value) => finish_request(request, value, diagnostics),
                Err(error) => {
                    diagnostics
                        .rejected_requests
                        .fetch_add(1, Ordering::Relaxed);
                    diagnostics.record_reply_time(request.command.queue_age());
                    let _ = request.reply.send(Err(ControlError::new(
                        ControlErrorKind::Application,
                        format!("failed to read channel intensity statistics: {error}"),
                    )));
                }
            }
        }
        _ => unreachable!("completion domain mismatch"),
    }
}

fn remote_open_response(identity: &RemoteOpenIdentity) -> Value {
    match identity {
        RemoteOpenIdentity::Http { url } => json!({
            "kind":"http_ome_zarr",
            "url":url,
        }),
        RemoteOpenIdentity::S3 {
            endpoint,
            region,
            bucket,
            prefix,
        } => json!({
            "kind":"s3_ome_zarr",
            "endpoint":endpoint,
            "region":region,
            "bucket":bucket,
            "prefix":prefix,
        }),
    }
}
