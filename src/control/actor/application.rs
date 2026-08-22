use super::*;

pub(super) fn begin_lifecycle_request(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    platform_effect_tx: &Sender<PlatformEffect>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    let quit = request.command.method() == "app.lifecycle.request_quit";
    let decision = request
        .command
        .params()
        .get("save")
        .and_then(Value::as_str)
        .unwrap_or("prompt")
        .to_string();
    if !matches!(decision.as_str(), "prompt" | "save" | "discard") {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::invalid_params(
                "app.lifecycle.request_close",
                "save must be prompt, save, or discard",
            ),
        );
        return;
    }
    let lifecycle = model.lifecycle_state();
    let dirty = lifecycle["dirty"].as_bool().unwrap_or(false);
    let action = if quit { "quit" } else { "close" };
    if dirty && decision == "prompt" {
        finish_request(
            request,
            json!({
                "confirmation_required":true,
                "action":action,
                "lifecycle":lifecycle,
            }),
            diagnostics,
        );
        return;
    }
    let effect = PlatformEffect::CloseWindow { quit };
    if dirty && decision == "save" {
        if load_job_tx.is_full() {
            reject_worker_submission(request, diagnostics);
            return;
        }
        let Some(path) = model.project_snapshot().saved_path else {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(
                    ControlErrorKind::InvalidParams,
                    "cannot save before closing because the project has no saved path",
                ),
            );
            return;
        };
        let (payload, saved_config_generation) = match model.prepare_lifecycle_project_save() {
            Ok(payload) => payload,
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        };
        let generation =
            model.begin_project_operation(format!("Saving {} before {action}", path.display()));
        match load_job_tx.try_send(LoadJob::ProjectSave {
            generation,
            request,
            path,
            payload,
            saved_config_generation,
            platform_effect: Some(effect),
        }) {
            Ok(()) => {
                diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                let LoadJob::ProjectSave { request, .. } = error.into_inner() else {
                    unreachable!("lifecycle save submission returns its own job")
                };
                model.fail_project_operation(
                    generation,
                    "Project worker queue is unavailable before close",
                );
                reject_worker_submission(request, diagnostics);
            }
        }
        return;
    }
    if platform_effect_tx.try_send(effect).is_err() {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::NotReady,
                "platform effect queue is unavailable",
            ),
        );
        return;
    }
    wake_ui();
    finish_request(
        request,
        json!({
            "accepted":true,
            "action":action,
            "discarded":dirty && decision == "discard",
        }),
        diagnostics,
    );
}

pub(super) fn begin_settings_mutation(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let mutation = match request.command.method() {
        "app.settings.set" => model.prepare_settings_set(request.command.params()),
        "app.recent_projects.forget" => {
            let path = request
                .command
                .params()
                .get("path")
                .and_then(Value::as_str)
                .map(expand_path)
                .ok_or_else(|| {
                    ControlError::invalid_params("app.recent_projects.forget", "path is required")
                });
            path.and_then(|path| model.prepare_recent_project_forget(path))
        }
        "app.recent_projects.clear" => model.prepare_recent_projects_clear(),
        _ => unreachable!("settings mutation dispatcher only receives settings methods"),
    };
    let mutation = match mutation {
        Ok(mutation) => mutation,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return;
        }
    };
    let SettingsMutationOutcome::Persist(operation) = mutation else {
        let SettingsMutationOutcome::Immediate(response) = mutation else {
            unreachable!()
        };
        finish_request(request, response, diagnostics);
        return;
    };
    let generation = operation.generation;
    match load_job_tx.try_send(LoadJob::SettingsSave {
        generation,
        request,
        path: operation.path,
        settings: operation.settings,
        response: operation.response,
    }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let LoadJob::SettingsSave { request, .. } = error.into_inner() else {
                unreachable!("settings submission returns its own job")
            };
            model.fail_settings_for_generation(generation, "Settings worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn enqueue_recent_project_persistence(
    model: &mut AppModel,
    project_path: PathBuf,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let mutation = match model.prepare_recent_project_record(project_path) {
        Ok(mutation) => mutation,
        Err(error) => {
            eprintln!("could not record recent project in actor settings: {error}");
            return;
        }
    };
    let SettingsMutationOutcome::Persist(operation) = mutation else {
        return;
    };
    let generation = operation.generation;
    if load_job_tx
        .try_send(LoadJob::SettingsPersist {
            generation,
            path: operation.path,
            settings: operation.settings,
            response: operation.response,
        })
        .is_ok()
    {
        diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
    } else {
        model.fail_settings_for_generation(
            generation,
            "Could not persist recent project because the settings worker queue is unavailable",
        );
    }
}
