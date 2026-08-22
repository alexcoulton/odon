use super::*;

pub(super) fn begin_mask_import(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let Some(path) = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
    else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("viewer.masks.import_geojson", "path is required"),
        );
        return;
    };
    let downsample_factor = request
        .command
        .params()
        .get("downsample_factor")
        .and_then(Value::as_f64)
        .unwrap_or(1.0) as f32;
    if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "viewer.masks.import_geojson",
                "downsample_factor must be finite and greater than zero",
            ),
        );
        return;
    }
    let editable = request
        .command
        .params()
        .get("editable")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let name = request
        .command
        .params()
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .or_else(|| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "Imported masks".to_string());
    let (document_generation, mask_generation, operation_generation, operation_scope) =
        match model.begin_mask_import_operation() {
            Ok(generations) => generations,
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        };
    match load_job_tx.try_send(LoadJob::MaskImport {
        document_generation,
        mask_generation,
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        path,
        name,
        editable,
        downsample_factor,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::MaskImport { request, .. } = error.into_inner() else {
                unreachable!("mask-import submission returns its own job")
            };
            model.fail_mask_io_for_generation(
                &operation_scope,
                operation_generation,
                "Mask import worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_mask_export(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let Some(path) = request
        .command
        .params()
        .get("path")
        .and_then(Value::as_str)
        .map(expand_path)
    else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params("viewer.masks.export_geojson", "path is required"),
        );
        return;
    };
    let layer_id = request
        .command
        .params()
        .get("id")
        .or_else(|| request.command.params().get("layer_id"))
        .and_then(Value::as_u64);
    let layers = match model.mask_export_layers(layer_id) {
        Ok(layers) => layers,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    let overwrite = request
        .command
        .params()
        .get("overwrite")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let (operation_generation, operation_scope) = match model.begin_mask_export_operation() {
        Ok(operation) => operation,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    match load_job_tx.try_send(LoadJob::MaskExport {
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        path,
        layer_id,
        layers,
        overwrite,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::MaskExport { request, .. } = error.into_inner() else {
                unreachable!("mask-export submission returns its own job")
            };
            model.fail_mask_io_for_generation(
                &operation_scope,
                operation_generation,
                "Mask export worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
