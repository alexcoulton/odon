use super::*;

pub(super) fn begin_project_object_source_scan(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let (scope, candidates) = model.project_object_preload_scan();
    match load_job_tx.try_send(LoadJob::ProjectObjectSourceScan {
        scope,
        request,
        candidates,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ProjectObjectSourceScan { request, .. } = error.into_inner() else {
                unreachable!("project object source scan returns its own job")
            };
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_project_object_preload(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let params = request.command.params();
    let mode = params
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("full_geometry");
    let settings = match crate::model::ProjectObjectPreloadMode::parse(mode) {
        Ok(mode) => ProjectObjectPreloadSettings {
            mode,
            lazy_properties: params
                .get("lazy_properties")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        },
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    let (generation, scope, candidates) = match model.begin_project_object_preload(settings) {
        Ok(operation) => operation,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    match load_job_tx.try_send(LoadJob::ProjectObjectPreload {
        generation,
        scope: scope.clone(),
        request,
        settings,
        candidates,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::ProjectObjectPreload { request, .. } = error.into_inner() else {
                unreachable!("project object preload submission returns its own job")
            };
            model.fail_project_object_preload(
                &scope,
                generation,
                "project object worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}
