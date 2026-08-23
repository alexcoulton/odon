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
        LoadCompletion::MaskImport {
            document_generation,
            mask_generation,
            operation_generation,
            operation_scope,
            request,
            path,
            name,
            editable,
            replace_layer_id,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_mask_io_for_generation(
                    &operation_scope,
                    operation_generation,
                    "Mask import cancelled",
                );
                reject_cancelled_request(request, diagnostics, "mask import");
                return;
            }
            match result {
                Ok(polygons_world) if polygons_world.is_empty() => {
                    model.fail_mask_io_for_generation(
                        &operation_scope,
                        operation_generation,
                        "Mask GeoJSON contains no supported polygon or line geometry",
                    );
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::InvalidParams,
                            "mask GeoJSON contains no supported polygon or line geometry",
                        )
                        .with_data(json!({"path": path.to_string_lossy()})),
                    )
                }
                Ok(polygons_world) => {
                    if let Some(mut response) = model.install_imported_masks_for_generation(
                        document_generation,
                        mask_generation,
                        operation_generation,
                        &operation_scope,
                        name,
                        editable,
                        replace_layer_id,
                        polygons_world,
                        path,
                    ) {
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
                                "mask import was superseded by a newer document or mask edit",
                            ),
                        );
                    }
                }
                Err(error) => {
                    model.fail_mask_io_for_generation(
                        &operation_scope,
                        operation_generation,
                        format!("Failed to import mask GeoJSON: {error}"),
                    );
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::Application,
                            format!("failed to import mask GeoJSON: {error}"),
                        )
                        .with_data(json!({"path": path.to_string_lossy()})),
                    )
                }
            }
        }
        LoadCompletion::MaskExport {
            operation_generation,
            operation_scope,
            request,
            path,
            layer_id,
            layer_count,
            polygon_count,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_mask_io_for_generation(
                    &operation_scope,
                    operation_generation,
                    "Mask export cancelled",
                );
                reject_cancelled_request(request, diagnostics, "mask export");
                return;
            }
            match result {
                Ok(bytes) => {
                    if model.finish_mask_io_for_generation(
                        &operation_scope,
                        operation_generation,
                        "Mask export ready",
                    ) {
                        finish_request(
                            request,
                            json!({
                                "exported": true,
                                "path": path.to_string_lossy(),
                                "layer_id": layer_id,
                                "layer_count": layer_count,
                                "polygon_count": polygon_count,
                                "bytes": bytes,
                                "output_ready": true,
                            }),
                            diagnostics,
                        )
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "mask export was superseded by newer application state",
                            ),
                        );
                    }
                }
                Err(error) => {
                    model.fail_mask_io_for_generation(
                        &operation_scope,
                        operation_generation,
                        format!("Failed to export mask GeoJSON: {error}"),
                    );
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::Application,
                            format!("failed to export mask GeoJSON: {error}"),
                        )
                        .with_data(json!({"path": path.to_string_lossy()})),
                    )
                }
            }
        }
        LoadCompletion::MaskAppend {
            document_generation,
            mask_generation,
            operation_generation,
            operation_scope,
            request,
            path,
            name,
            saved_layers,
            result,
        } => match result {
            Ok(saved) => {
                if let Some(mut response) = model.reconcile_appended_masks_for_generation(
                    document_generation,
                    mask_generation,
                    operation_generation,
                    &operation_scope,
                    &saved_layers,
                    name,
                    saved.polygons_world,
                    path.clone(),
                ) {
                    response["bytes"] = json!(saved.bytes);
                    response["appended_polygon_count"] = json!(saved.appended_polygon_count);
                    response["saved_layer_ids"] = json!(
                        saved_layers
                            .iter()
                            .map(|layer| layer.id)
                            .collect::<Vec<_>>()
                    );
                    if response
                        .get("applied_to_current_document")
                        .and_then(Value::as_bool)
                        == Some(true)
                    {
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
                    diagnostics
                        .stale_worker_completions
                        .fetch_add(1, Ordering::Relaxed);
                    reject_actor_request(
                        request,
                        diagnostics,
                        ControlError::new(
                            ControlErrorKind::Conflict,
                            "mask append completion no longer owns its persistence operation",
                        )
                        .with_data(json!({"path": path.to_string_lossy()})),
                    );
                }
            }
            Err(error) if request_is_cancelled(&request) => {
                model.cancel_mask_io_for_generation(
                    &operation_scope,
                    operation_generation,
                    "Mask append cancelled before file commit",
                );
                reject_cancelled_request(request, diagnostics, "mask append");
            }
            Err(error) => {
                model.fail_mask_io_for_generation(
                    &operation_scope,
                    operation_generation,
                    format!("Failed to append mask GeoJSON: {error}"),
                );
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Application,
                        format!("failed to append mask GeoJSON: {error}"),
                    )
                    .with_data(json!({"path": path.to_string_lossy()})),
                );
            }
        },
        _ => unreachable!("completion domain mismatch"),
    }
}
