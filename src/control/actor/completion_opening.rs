use super::completion::{CompletionContext, reject_cancelled_request, request_is_cancelled};
use super::*;

pub(super) fn finish(completion: LoadCompletion, context: CompletionContext<'_>) {
    let CompletionContext {
        model,
        render_document,
        remote_session,
        resource_registry,
        presentation_tx,
        presentation_coalesce_rx,
        load_job_tx,
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
        LoadCompletion::DeepLinkApply {
            operation_generation,
            guard,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_deep_link_apply(
                    operation_generation,
                    guard,
                    "Deep-link application was cancelled",
                );
                reject_cancelled_request(request, diagnostics, "deep-link application");
                return;
            }
            match result {
                Ok(result) => {
                    if result
                        .opened
                        .s3_session_generation
                        .is_some_and(|generation| !remote_session.is_current(generation))
                    {
                        model.supersede_deep_link_apply(
                            operation_generation,
                            "S3 session changed during deep-link application",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "S3 session changed during deep-link application",
                            ),
                        );
                        return;
                    }
                    let DeepLinkApplyWorkerResult {
                        deep_link,
                        project,
                        project_source,
                        opened,
                        object_filter,
                    } = result;
                    let ProjectRoiOpenWorkerResult {
                        opened,
                        roi,
                        saved_view,
                        label_available,
                        label_resource,
                        object_resource,
                        reuse_current,
                        ..
                    } = opened;
                    let descriptor = opened.descriptor.clone();
                    let kind = match descriptor.kind {
                        crate::data::document::DocumentKind::OmeZarr => "ome_zarr",
                        crate::data::document::DocumentKind::Tiff => "tiff",
                        crate::data::document::DocumentKind::SpatialData => "spatialdata",
                        crate::data::document::DocumentKind::Xenium => "xenium",
                    };
                    let mut candidate = model.clone();
                    let installed = candidate.install_deep_link_apply_for_generation(
                        operation_generation,
                        guard,
                        project,
                        &roi,
                        reuse_current,
                        descriptor,
                        label_available,
                        label_resource.map(Arc::new),
                        object_resource,
                        saved_view.as_ref(),
                        &deep_link,
                        object_filter,
                    );
                    match installed {
                        Ok(Some((document_generation, notes))) => {
                            let project = candidate.project_snapshot();
                            if let Err(error) = resource_registry.replace_project_manifest(
                                &project.config.control_resources,
                                &project.config.control_layers,
                            ) {
                                model.fail_deep_link_apply(
                                    operation_generation,
                                    guard,
                                    error.message.clone(),
                                );
                                reject_actor_request(request, diagnostics, error);
                                return;
                            }
                            let roi_id = roi.id.clone();
                            let source = roi.source_display();
                            let project_path = project.saved_path.clone();
                            *model = candidate;
                            *render_document = Some(Arc::new(RenderDocument {
                                generation: document_generation,
                                opened,
                            }));
                            if project_source == "project_file"
                                && let Some(path) = project_path
                            {
                                enqueue_recent_project_persistence(
                                    model,
                                    path,
                                    load_job_tx,
                                    diagnostics,
                                );
                            }
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
                                    "applied":true,
                                    "settled":true,
                                    "url":deep_link.to_url(),
                                    "request":deep_link,
                                    "resolution":{
                                        "project_source":project_source,
                                        "project_path":project.saved_path,
                                        "roi":roi,
                                    },
                                    "opened":{
                                        "mode":"single",
                                        "kind":kind,
                                        "roi":roi_id,
                                        "source":source,
                                        "reused_document":reuse_current,
                                    },
                                    "notes":notes,
                                    "model_ready":true,
                                    "resources_ready":true,
                                    "presentation_ready":false,
                                }),
                                diagnostics,
                            );
                        }
                        Ok(None) => {
                            model.supersede_deep_link_apply(
                                operation_generation,
                                "Deep-link application was superseded",
                            );
                            reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(
                                    ControlErrorKind::Conflict,
                                    "deep-link application was superseded by newer state",
                                ),
                            );
                        }
                        Err(error) => {
                            if model.fail_deep_link_apply(
                                operation_generation,
                                guard,
                                error.message.clone(),
                            ) {
                                publish_projection(
                                    model,
                                    render_document.clone(),
                                    presentation_tx,
                                    presentation_coalesce_rx,
                                    wake_ui,
                                    diagnostics,
                                );
                                reject_actor_request(request, diagnostics, error);
                            } else {
                                model.supersede_deep_link_apply(
                                    operation_generation,
                                    "Deep-link application was superseded",
                                );
                                reject_actor_request(
                                    request,
                                    diagnostics,
                                    ControlError::new(
                                        ControlErrorKind::Conflict,
                                        "deep-link application was superseded by newer state",
                                    ),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    let message = format!("failed to apply deep link: {error}");
                    if model.fail_deep_link_apply(operation_generation, guard, message.clone()) {
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
                        model.supersede_deep_link_apply(
                            operation_generation,
                            "Deep-link application was superseded",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed deep-link application was superseded by newer state",
                            ),
                        );
                    }
                }
            }
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
                    *render_document = Some(Arc::new(RenderDocument {
                        generation,
                        opened: opened.into_control(),
                    }));
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
        LoadCompletion::Tiff {
            generation,
            request,
            path,
            z,
            t,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_dataset_open_for_generation(generation, "TIFF open was cancelled");
                reject_cancelled_request(request, diagnostics, "TIFF dataset open");
                return;
            }
            match result {
                Ok(opened) => {
                    if !model.install_document_for_generation(
                        generation,
                        opened.descriptor.clone(),
                        Vec::new(),
                        None,
                    ) {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "TIFF dataset open was superseded by a newer request",
                            )
                            .with_data(json!({
                                "path": path.to_string_lossy(),
                                "generation": generation,
                            })),
                        );
                        return;
                    }
                    *render_document = Some(Arc::new(RenderDocument {
                        generation,
                        opened: opened.into_control(),
                    }));
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
                            "kind": "tiff",
                            "path": path.to_string_lossy(),
                            "plane": {"z": z, "t": t},
                            "model_ready": true,
                            "resources_ready": true,
                            "presentation_ready": false,
                        }),
                        diagnostics,
                    );
                }
                Err(error) => {
                    let message = format!("failed to open TIFF plane Z={z}, T={t}: {error}");
                    if model.fail_dataset_open_for_generation(generation, &message) {
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
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed TIFF open was superseded by a newer request",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::ProjectRoiOpen {
            operation_generation,
            scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_project_roi_open(
                    &scope,
                    operation_generation,
                    "Project ROI open cancelled",
                );
                reject_cancelled_request(request, diagnostics, "project ROI open");
                return;
            }
            match result {
                Ok(result) => {
                    if result
                        .s3_session_generation
                        .is_some_and(|generation| !remote_session.is_current(generation))
                    {
                        model.supersede_project_roi_open(
                            operation_generation,
                            "S3 session changed during project ROI open",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "S3 session changed during project ROI open",
                            ),
                        );
                        return;
                    }
                    let descriptor = result.opened.descriptor.clone();
                    let kind = match descriptor.kind {
                        crate::data::document::DocumentKind::OmeZarr => "ome_zarr",
                        crate::data::document::DocumentKind::Tiff => "tiff",
                        crate::data::document::DocumentKind::SpatialData => "spatialdata",
                        crate::data::document::DocumentKind::Xenium => "xenium",
                    };
                    let mut candidate = model.clone();
                    let installed = candidate.install_project_roi_for_generation(
                        &scope,
                        operation_generation,
                        &result.roi,
                        descriptor,
                        result.label_available,
                        result.label_resource.map(Arc::new),
                        result.object_resource,
                        result.saved_view.as_ref(),
                    );
                    match installed {
                        Ok(Some(document_generation)) => {
                            let roi_id = result.roi.id.clone();
                            let source = result.roi.source_display();
                            *model = candidate;
                            *render_document = Some(Arc::new(RenderDocument {
                                generation: document_generation,
                                opened: result.opened,
                            }));
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
                                    "kind": kind,
                                    "roi": roi_id,
                                    "source": source,
                                    "model_ready": true,
                                    "resources_ready": true,
                                    "presentation_ready": false,
                                }),
                                diagnostics,
                            );
                        }
                        Ok(None) => {
                            model.supersede_project_roi_open(
                                operation_generation,
                                "Project ROI open was superseded",
                            );
                            reject_actor_request(
                                request,
                                diagnostics,
                                ControlError::new(
                                    ControlErrorKind::Conflict,
                                    "project ROI open was superseded by a newer transaction",
                                ),
                            );
                        }
                        Err(error) => {
                            if model.fail_project_roi_open(
                                &scope,
                                operation_generation,
                                error.message.clone(),
                            ) {
                                reject_actor_request(request, diagnostics, error);
                            } else {
                                reject_actor_request(
                                    request,
                                    diagnostics,
                                    ControlError::new(
                                        ControlErrorKind::Conflict,
                                        "project ROI open was superseded by a newer transaction",
                                    ),
                                );
                            }
                        }
                    }
                }
                Err(error) => {
                    let message = format!("failed to open project ROI: {error}");
                    if model.fail_project_roi_open(&scope, operation_generation, message.clone()) {
                        // Publish only readiness/project state. The prior render document and
                        // semantic dataset remain intact because the transaction never committed.
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
                        model.supersede_project_roi_open(
                            operation_generation,
                            "Project ROI open was superseded",
                        );
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed project ROI open was superseded by a newer transaction",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::SpatialData {
            generation,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model
                    .fail_dataset_open_for_generation(generation, "SpatialData open was cancelled");
                reject_cancelled_request(request, diagnostics, "SpatialData open");
                return;
            }
            match result {
                Ok((opened, identity)) => commit_alternate_document(
                    model,
                    render_document,
                    generation,
                    request,
                    opened,
                    json!({
                        "kind":"spatialdata",
                        "path":identity.root.to_string_lossy(),
                        "image":identity.image,
                        "extra_images":identity.extra_images,
                        "labels":identity.labels,
                        "shapes":identity.shapes,
                        "points":identity.points,
                        "points_max":identity.points_max,
                    }),
                    "SpatialData open was superseded by a newer request",
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                ),
                Err(error) => fail_alternate_document(
                    model,
                    generation,
                    request,
                    format!("failed to open SpatialData dataset: {error}"),
                    "failed SpatialData open was superseded by a newer request",
                    diagnostics,
                ),
            }
        }
        LoadCompletion::Xenium {
            generation,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_dataset_open_for_generation(generation, "Xenium open was cancelled");
                reject_cancelled_request(request, diagnostics, "Xenium open");
                return;
            }
            match result {
                Ok((opened, identity)) => commit_alternate_document(
                    model,
                    render_document,
                    generation,
                    request,
                    opened,
                    json!({
                        "kind":"xenium",
                        "path":identity.root.to_string_lossy(),
                        "imagery":identity.imagery,
                        "imagery_path":identity.imagery_path.to_string_lossy(),
                        "cells_loaded":identity.cells_loaded,
                        "transcripts_loaded":identity.transcripts_loaded,
                        "pixel_size_um":identity.pixel_size_um,
                    }),
                    "Xenium open was superseded by a newer request",
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                ),
                Err(error) => fail_alternate_document(
                    model,
                    generation,
                    request,
                    format!("failed to open Xenium dataset: {error}"),
                    "failed Xenium open was superseded by a newer request",
                    diagnostics,
                ),
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
                    *render_document = Some(Arc::new(RenderDocument {
                        generation,
                        opened: opened.into_control(),
                    }));
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

#[allow(clippy::too_many_arguments)]
fn commit_alternate_document(
    model: &mut AppModel,
    render_document: &mut Option<Arc<RenderDocument>>,
    generation: u64,
    request: OdonControlRequest,
    opened: OpenedDocument<AlternateDocumentResource>,
    mut response: Value,
    superseded_message: &str,
    presentation_tx: &Sender<RenderProjection>,
    presentation_coalesce_rx: &Receiver<RenderProjection>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    let mut candidate = model.clone();
    if !candidate.install_document_for_generation(
        generation,
        opened.descriptor.clone(),
        Vec::new(),
        None,
    ) {
        diagnostics
            .stale_worker_completions
            .fetch_add(1, Ordering::Relaxed);
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(ControlErrorKind::Conflict, superseded_message),
        );
        return;
    }
    if let Err(error) = candidate.install_document_object_layers(opened.resource.object_layers()) {
        reject_actor_request(request, diagnostics, error);
        return;
    }
    *model = candidate;
    *render_document = Some(Arc::new(RenderDocument {
        generation,
        opened: opened.into_control(),
    }));
    publish_projection(
        model,
        render_document.clone(),
        presentation_tx,
        presentation_coalesce_rx,
        wake_ui,
        diagnostics,
    );
    response
        .as_object_mut()
        .expect("alternate open response object")
        .extend([
            ("opened".to_string(), Value::Bool(true)),
            ("mode".to_string(), Value::String("single".to_string())),
            ("model_ready".to_string(), Value::Bool(true)),
            ("resources_ready".to_string(), Value::Bool(true)),
            ("presentation_ready".to_string(), Value::Bool(false)),
        ]);
    finish_request(request, response, diagnostics);
}

fn fail_alternate_document(
    model: &mut AppModel,
    generation: u64,
    request: OdonControlRequest,
    message: String,
    superseded_message: &str,
    diagnostics: &ActorDiagnostics,
) {
    if model.fail_dataset_open_for_generation(generation, &message) {
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
            ControlError::new(ControlErrorKind::Conflict, superseded_message),
        );
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
