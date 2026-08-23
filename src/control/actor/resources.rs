use super::*;

pub(super) fn begin_label_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    render_document: &Option<Arc<RenderDocument>>,
    diagnostics: &ActorDiagnostics,
) {
    let Some(document) = render_document.as_ref().cloned() else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::NotReady,
                "label loading requires an actor-owned OME-Zarr document",
            ),
        );
        return;
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let (document_generation, label_generation, name) =
        match model.begin_label_load(request.command.params()) {
            Ok(values) => values,
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        };
    match load_job_tx.try_send(LoadJob::Labels {
        document_generation,
        label_generation,
        request,
        document,
        name,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::Labels { request, .. } = error.into_inner() else {
                unreachable!("label-load submission returns its own job")
            };
            model.fail_label_load_for_generation(
                document_generation,
                label_generation,
                "label worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_object_resource_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let current = model.current_object_resource_request();
    let (path, downsample_factor) = if request.command.method() == "viewer.objects.source.reload" {
        let Some((path, downsample_factor)) = current else {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(ControlErrorKind::NotReady, "No object source is loaded."),
            );
            return;
        };
        (path, downsample_factor)
    } else {
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
                ControlError::invalid_params("viewer.objects.source.load", "path is required"),
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
                    "viewer.objects.source.load",
                    "downsample_factor must be a positive finite number",
                ),
            );
            return;
        }
        (path, downsample_factor)
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let (document_generation, resource_generation) =
        model.begin_object_resource_load(path.to_string_lossy());
    let options = request.command.params().get("loader_options").cloned();
    match load_job_tx.try_send(LoadJob::ObjectResource {
        document_generation,
        resource_generation,
        request,
        path,
        downsample_factor,
        options,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ObjectResource { request, .. } = error.into_inner() else {
                unreachable!("object-resource submission returns its own job")
            };
            model.fail_object_resource_for_generation(
                document_generation,
                resource_generation,
                "object worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_segmentation_geojson_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let mut params = request.command.params().clone();
    if let Some(path) = request.command.params().get("path").and_then(Value::as_str) {
        params["path"] = json!(expand_path(path).to_string_lossy().into_owned());
    }
    let spec = match model.prepare_segmentation_geojson_load(&params) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::SegmentationGeoJson { request, spec }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::SegmentationGeoJson { request, spec } = error.into_inner() else {
                unreachable!("segmentation GeoJSON submission returns its own job")
            };
            model.fail_segmentation_geojson_load(
                &spec,
                "Segmentation GeoJSON worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}
