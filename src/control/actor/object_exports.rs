use super::*;

pub(super) fn begin_object_export(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
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
            ControlError::new(ControlErrorKind::InvalidParams, "path is required"),
        );
        return false;
    };
    let format = match request.command.method() {
        "exports.objects.export_csv" => Some(ObjectExportFormat::Csv),
        "exports.objects.export_geoparquet" => Some(ObjectExportFormat::GeoParquet),
        _ => None,
    };
    let spec = match model.prepare_object_export(request.command.params(), path, format) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::ObjectExport { request, spec }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::ObjectExport { request, spec } = error.into_inner() else {
                unreachable!("object export submission returns its own job")
            };
            model.fail_object_export(&spec, "Object export worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
