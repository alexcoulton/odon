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
                    if let Some(response) = model.install_imported_masks_for_generation(
                        document_generation,
                        mask_generation,
                        operation_generation,
                        &operation_scope,
                        name,
                        editable,
                        polygons_world,
                        path,
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
        _ => unreachable!("completion domain mismatch"),
    }
}
