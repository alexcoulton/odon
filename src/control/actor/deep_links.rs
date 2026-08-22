use super::*;

pub(super) fn begin_deep_link_resolution(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let mut deep_link = match AppModel::deep_link_request_from_params(request.command.params()) {
        Ok(request) => request,
        Err(error) => {
            finish_request(request, json!({"error":error.message}), diagnostics);
            return;
        }
    };
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }
    let current_project = model.project_snapshot();
    let needs_example_lookup = deep_link.example.is_some() && deep_link.project_path.is_none();
    let use_current = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path));
    if !needs_example_lookup && use_current {
        let resolution = resolve_roi_target(
            &current_project.rois,
            deep_link.roi.as_deref(),
            deep_link.sample.as_deref(),
        )
        .map(|roi| DeepLinkResolution {
            project_source: "current".to_string(),
            project_path: deep_link.project_path.clone(),
            roi,
        });
        finish_request(
            request,
            deep_link_resolution_response(deep_link, resolution),
            diagnostics,
        );
        return;
    }
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let operation_scope = deep_link.to_url();
    let operation_generation = model.begin_deep_link_resolution(operation_scope.clone());
    match load_job_tx.try_send(LoadJob::DeepLinkResolve {
        operation_generation,
        operation_scope: operation_scope.clone(),
        request,
        deep_link,
        current_project,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::DeepLinkResolve { request, .. } = error.into_inner() else {
                unreachable!("deep-link resolution submission returns its own job")
            };
            model.cancel_deep_link_resolution(
                &operation_scope,
                operation_generation,
                "Deep-link resolver worker queue is unavailable",
            );
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn resolve_deep_link_on_worker(
    mut deep_link: DeepLinkRequest,
    current_project: ProjectModelSnapshot,
) -> DeepLinkResolveWorkerResult {
    if let Some(example) = deep_link.example.clone() {
        apply_example_defaults(&mut deep_link, &example);
        if deep_link.project_path.is_none() {
            deep_link.project_path = resolve_example_project_path(&example);
        }
    }
    if let Some(path) = deep_link.project_path.as_deref() {
        deep_link.project_path = Some(expand_path(&path.to_string_lossy()));
    }
    let use_current = deep_link
        .project_path
        .as_ref()
        .is_none_or(|path| current_project.saved_path.as_ref() == Some(path));
    let resolution = if use_current {
        resolve_roi_target(
            &current_project.rois,
            deep_link.roi.as_deref(),
            deep_link.sample.as_deref(),
        )
        .map(|roi| DeepLinkResolution {
            project_source: "current".to_string(),
            project_path: deep_link.project_path.clone(),
            roi,
        })
    } else {
        let path = deep_link
            .project_path
            .as_ref()
            .expect("non-current deep link has a project path");
        if !path.exists() {
            Err(format!(
                "Deep-link project does not exist: {}",
                path.to_string_lossy()
            ))
        } else {
            read_project_file(path)
                .map_err(|error| format!("Deep-link project could not be loaded: {error}"))
                .and_then(|(config, state)| {
                    crate::model::normalized_loaded_project_snapshot(path.clone(), config, state)
                        .map_err(|error| {
                            format!("Deep-link project could not be loaded: {}", error.message)
                        })
                })
                .and_then(|project| {
                    resolve_roi_target(
                        &project.rois,
                        deep_link.roi.as_deref(),
                        deep_link.sample.as_deref(),
                    )
                })
                .map(|roi| DeepLinkResolution {
                    project_source: "project_file".to_string(),
                    project_path: deep_link.project_path.clone(),
                    roi,
                })
        }
    };
    DeepLinkResolveWorkerResult {
        request: deep_link,
        resolution,
    }
}

pub(super) fn deep_link_resolution_response(
    request: DeepLinkRequest,
    resolution: Result<DeepLinkResolution, String>,
) -> Value {
    match resolution {
        Ok(resolution) => json!({
            "resolved":true,
            "url":request.to_url(),
            "request":request,
            "resolution":resolution,
        }),
        Err(error) => json!({"resolved":false,"error":error}),
    }
}
