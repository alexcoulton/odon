use super::*;

pub(super) fn begin_project_roi_open(
    model: &mut AppModel,
    remote_session: &RemoteSessionState,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let params = request.command.params();
    let roi_query = params
        .get("roi")
        .or_else(|| params.get("id"))
        .or_else(|| params.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let sample_query = params
        .get("sample")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let project = model.project_snapshot();
    let roi = match resolve_roi_target(&project.rois, roi_query, sample_query) {
        Ok(roi) => roi,
        Err(error) => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(ControlErrorKind::ResourceNotFound, error),
            );
            return;
        }
    };
    let Some(mut source) = roi.dataset_source() else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::InvalidParams,
                "project ROI has no dataset source configured",
            ),
        );
        return;
    };
    if let DatasetSource::Local(path) = &mut source
        && path.is_relative()
        && let Some(directory) = project
            .saved_path
            .as_deref()
            .and_then(std::path::Path::parent)
    {
        *path = directory.join(&*path);
    }
    let source_key = source.source_key();
    let saved_view = project
        .state
        .get("roi_views")
        .and_then(|views| views.get(&source_key))
        .cloned();
    let object_path = crate::model::project_roi_segmentation_path(&project, &roi);
    let cached_object = object_path
        .as_ref()
        .and_then(|path| model.cached_project_object_resource(path));
    let (scope, _) = model.project_object_preload_scan();
    let s3_session = match &source {
        DatasetSource::S3 {
            endpoint,
            region,
            bucket,
            ..
        } => match remote_session.credentials() {
            Ok((generation, credentials)) => Some((
                generation,
                crate::data::remote_store::S3SessionCredentials::normalized(
                    endpoint,
                    region,
                    bucket,
                    &credentials.access_key,
                    &credentials.secret_key,
                ),
            )),
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        },
        _ => None,
    };
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let operation_generation = model.begin_project_roi_open(
        &scope,
        format!("Opening project ROI {}", roi.source_display()),
    );
    let spec = ProjectRoiOpenSpec {
        roi,
        source,
        saved_view,
        object_path,
        cached_object,
        s3_session,
        requested_label: None,
    };
    match load_job_tx.try_send(LoadJob::ProjectRoiOpen {
        operation_generation,
        scope: scope.clone(),
        request,
        spec,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ProjectRoiOpen { request, .. } = error.into_inner() else {
                unreachable!("project ROI open submission returns its own job")
            };
            model.fail_project_roi_open(
                &scope,
                operation_generation,
                "project ROI worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
