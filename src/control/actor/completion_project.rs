use super::completion::{
    CompletionContext, fail_project_worker, reject_cancelled_request, reject_stale_project_worker,
    request_is_cancelled,
};
use super::*;

pub(super) fn finish(completion: LoadCompletion, context: CompletionContext<'_>) {
    let CompletionContext {
        model,
        render_document,
        remote_session: _,
        resource_registry,
        presentation_tx,
        presentation_coalesce_rx,
        platform_effect_tx,
        load_job_tx,
        wake_ui,
        diagnostics,
    } = context;
    match completion {
        LoadCompletion::ProjectOpen {
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
                model.fail_project_operation(generation, "project open was cancelled");
                let _ = request.reply.send(Err(ControlError::new(
                    ControlErrorKind::Cancelled,
                    "project open was cancelled",
                )));
                return;
            }
            match result {
                Ok((config, state)) => {
                    let mut candidate = model.clone();
                    match candidate.install_project_for_generation(
                        generation,
                        path.clone(),
                        config,
                        state,
                    ) {
                        Ok(true) => {
                            let project = candidate.project_snapshot();
                            if let Err(error) = resource_registry.replace_project_manifest(
                                &project.config.control_resources,
                                &project.config.control_layers,
                            ) {
                                model.fail_project_operation(generation, error.message.clone());
                                diagnostics
                                    .rejected_requests
                                    .fetch_add(1, Ordering::Relaxed);
                                diagnostics.record_reply_time(request.command.queue_age());
                                let _ = request.reply.send(Err(error));
                                return;
                            }
                            *model = candidate;
                            enqueue_recent_project_persistence(
                                model,
                                path.clone(),
                                load_job_tx,
                                diagnostics,
                            );
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
                                    "path": path.to_string_lossy(),
                                    "project": {
                                        "roi_count": project.rois.len(),
                                        "view_count": project.view_count,
                                    },
                                    "model_ready": true,
                                    "resources_ready": true,
                                    "presentation_ready": false,
                                }),
                                diagnostics,
                            );
                        }
                        Ok(false) => {
                            diagnostics
                                .stale_worker_completions
                                .fetch_add(1, Ordering::Relaxed);
                            let _ = request.reply.send(Err(ControlError::new(
                                ControlErrorKind::Conflict,
                                "project open result was superseded by a newer persistence request",
                            )));
                        }
                        Err(error) => {
                            model.fail_project_operation(generation, error.message.clone());
                            diagnostics
                                .rejected_requests
                                .fetch_add(1, Ordering::Relaxed);
                            let _ = request.reply.send(Err(error));
                        }
                    }
                }
                Err(error) => {
                    let message = format!("failed to open project: {error}");
                    if model.fail_project_operation(generation, &message) {
                        diagnostics
                            .rejected_requests
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Application,
                            message,
                        )
                        .with_data(json!({"path": path.to_string_lossy()}))));
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "failed project open was superseded by a newer persistence request",
                        )));
                    }
                }
            }
        }
        LoadCompletion::ProjectSave {
            generation,
            request,
            path,
            saved_config_generation,
            platform_effect,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_project_operation(generation, "project save was cancelled");
                reject_cancelled_request(request, diagnostics, "project save");
                return;
            }
            match result {
                Ok(())
                    if model.finish_project_save_for_generation(
                        generation,
                        path.clone(),
                        saved_config_generation,
                    ) =>
                {
                    enqueue_recent_project_persistence(
                        model,
                        path.clone(),
                        load_job_tx,
                        diagnostics,
                    );
                    publish_projection(
                        model,
                        render_document.clone(),
                        presentation_tx,
                        presentation_coalesce_rx,
                        wake_ui,
                        diagnostics,
                    );
                    if let Some(effect) = platform_effect {
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
                    }
                    finish_request(
                        request,
                        if let Some(PlatformEffect::CloseWindow { quit }) = platform_effect {
                            json!({
                                "accepted":true,
                                "action":if quit { "quit" } else { "close" },
                                "saved":true,
                                "path":path.to_string_lossy(),
                            })
                        } else {
                            json!({
                                "saved": true,
                                "path": path.to_string_lossy(),
                                "model_ready": true,
                                "presentation_ready": false,
                            })
                        },
                        diagnostics,
                    );
                }
                Ok(()) => {
                    diagnostics
                        .stale_worker_completions
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = request.reply.send(Err(ControlError::new(
                        ControlErrorKind::Conflict,
                        "project save was superseded by a newer persistence request",
                    )));
                }
                Err(error) => {
                    let message = format!("failed to save project: {error}");
                    if model.fail_project_operation(generation, &message) {
                        diagnostics
                            .rejected_requests
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Application,
                            message,
                        )
                        .with_data(json!({"path": path.to_string_lossy()}))));
                    } else {
                        diagnostics
                            .stale_worker_completions
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = request.reply.send(Err(ControlError::new(
                            ControlErrorKind::Conflict,
                            "failed project save was superseded by a newer persistence request",
                        )));
                    }
                }
            }
        }
        LoadCompletion::SettingsSave {
            generation,
            request,
            settings,
            response,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_settings_for_generation(generation, "Settings save was cancelled");
                reject_cancelled_request(request, diagnostics, "settings save");
                return;
            }
            match result {
                Ok(path) => {
                    if let Some(response) =
                        model.install_settings_for_generation(generation, settings, response)
                    {
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
                                "settings save was superseded by a newer operation",
                            )
                            .with_data(json!({"path":path.to_string_lossy()})),
                        );
                    }
                }
                Err(error) => {
                    let message = format!("settings save failed: {error}");
                    if model.fail_settings_for_generation(generation, &message) {
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
                            ControlError::new(
                                ControlErrorKind::Conflict,
                                "failed settings save was superseded by a newer operation",
                            ),
                        );
                    }
                }
            }
        }
        LoadCompletion::SettingsPersist {
            generation,
            settings,
            response,
            result,
        } => match result {
            Ok(_) => {
                if model
                    .install_settings_for_generation(generation, settings, response)
                    .is_some()
                {
                    publish_projection(
                        model,
                        render_document.clone(),
                        presentation_tx,
                        presentation_coalesce_rx,
                        wake_ui,
                        diagnostics,
                    );
                } else {
                    diagnostics
                        .stale_worker_completions
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(error) => {
                let message = format!("recent-project settings save failed: {error}");
                if !model.fail_settings_for_generation(generation, &message) {
                    diagnostics
                        .stale_worker_completions
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        },
        LoadCompletion::SamplesheetInspect { request, result } => {
            if request_is_cancelled(&request) {
                reject_cancelled_request(request, diagnostics, "samplesheet inspection");
            } else {
                finish_request(request, result, diagnostics);
            }
        }
        LoadCompletion::SamplesheetImport {
            generation,
            request,
            path,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_project_operation(generation, "samplesheet import was cancelled");
                reject_cancelled_request(request, diagnostics, "samplesheet import");
                return;
            }
            match result {
                Ok(rois) => match model
                    .replace_project_rois_from_samplesheet_for_generation(generation, rois)
                {
                    Ok(Some(project)) => {
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
                                "imported": true,
                                "path": path.to_string_lossy(),
                                "project": project,
                                "model_ready": true,
                                "presentation_ready": false,
                            }),
                            diagnostics,
                        );
                    }
                    Ok(None) => {
                        reject_stale_project_worker(request, diagnostics, "samplesheet import")
                    }
                    Err(error) => {
                        model.fail_project_operation(generation, error.message.clone());
                        reject_actor_request(request, diagnostics, error);
                    }
                },
                Err(error) => fail_project_worker(
                    model,
                    generation,
                    request,
                    diagnostics,
                    "import samplesheet",
                    &path,
                    error,
                ),
            }
        }
        LoadCompletion::SamplesheetExport {
            generation,
            request,
            path,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_project_operation(generation, "samplesheet export was cancelled");
                reject_cancelled_request(request, diagnostics, "samplesheet export");
                return;
            }
            match result {
                Ok(bytes) if model.finish_project_operation_for_generation(generation) => {
                    finish_request(
                        request,
                        json!({
                            "exported": true,
                            "path": path.to_string_lossy(),
                            "bytes": bytes,
                            "output_ready": true,
                        }),
                        diagnostics,
                    );
                }
                Ok(_) => reject_stale_project_worker(request, diagnostics, "samplesheet export"),
                Err(error) => fail_project_worker(
                    model,
                    generation,
                    request,
                    diagnostics,
                    "export samplesheet",
                    &path,
                    error,
                ),
            }
        }
        LoadCompletion::ProjectDiscovery {
            generation,
            request,
            root,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.fail_project_operation(generation, "project discovery was cancelled");
                reject_cancelled_request(request, diagnostics, "project discovery");
                return;
            }
            match result {
                Ok(roots) => {
                    match model.add_discovered_project_roots_for_generation(generation, roots) {
                        Ok(Some((added, project))) => {
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
                                    "discovered": true,
                                    "root": root.to_string_lossy(),
                                    "added": added,
                                    "project": project,
                                    "model_ready": true,
                                    "presentation_ready": false,
                                }),
                                diagnostics,
                            );
                        }
                        Ok(None) => {
                            reject_stale_project_worker(request, diagnostics, "project discovery")
                        }
                        Err(error) => {
                            model.fail_project_operation(generation, error.message.clone());
                            reject_actor_request(request, diagnostics, error);
                        }
                    }
                }
                Err(error) => fail_project_worker(
                    model,
                    generation,
                    request,
                    diagnostics,
                    "discover datasets under",
                    &root,
                    error,
                ),
            }
        }
        LoadCompletion::ProjectObjectSourceScan {
            scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                reject_cancelled_request(request, diagnostics, "project object source scan");
                return;
            }
            match result {
                Ok(sources) => {
                    if !model.install_project_object_preload_sources(&scope, sources) {
                        reject_stale_project_worker(
                            request,
                            diagnostics,
                            "project object source scan",
                        );
                        return;
                    }
                    let response = match request.command.method() {
                        "project.objects.preload.get" => model.project_object_preload_snapshot(),
                        "project.objects.preload.list_sources" => {
                            let offset = request
                                .command
                                .params()
                                .get("offset")
                                .and_then(Value::as_u64)
                                .unwrap_or(0) as usize;
                            let limit = request
                                .command
                                .params()
                                .get("limit")
                                .and_then(Value::as_u64)
                                .unwrap_or(200) as usize;
                            model.project_object_preload_sources_snapshot(offset, limit)
                        }
                        _ => unreachable!("source scan has a preload inspection request"),
                    };
                    finish_request(request, response, diagnostics);
                }
                Err(error) => reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::new(
                        ControlErrorKind::Application,
                        format!("project object source scan failed: {error}"),
                    ),
                ),
            }
        }
        LoadCompletion::ProjectObjectPreload {
            generation,
            scope,
            request,
            result,
        } => {
            if request_is_cancelled(&request) {
                model.cancel_project_object_preload(
                    &scope,
                    generation,
                    "Project object preload cancelled",
                );
                reject_cancelled_request(request, diagnostics, "project object preload");
                return;
            }
            match result {
                Ok(result) => {
                    let failed = result.failures.len();
                    let failure_details = result
                        .failures
                        .iter()
                        .map(|(path, error)| json!({"path":path.to_string_lossy(),"error":error}))
                        .collect::<Vec<_>>();
                    if model.finish_project_object_preload(
                        &scope,
                        generation,
                        result.sources,
                        result.resources,
                        failed,
                    ) {
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
                                "started": true,
                                "completed": true,
                                "failures": failure_details,
                                "preload": model.project_object_preload_snapshot(),
                                "model_ready": true,
                                "resources_ready": true,
                                "presentation_ready": false,
                            }),
                            diagnostics,
                        );
                    } else {
                        reject_stale_project_worker(request, diagnostics, "project object preload");
                    }
                }
                Err(error) => {
                    let message = format!("project object preload failed: {error}");
                    if model.fail_project_object_preload(&scope, generation, message.clone()) {
                        reject_actor_request(
                            request,
                            diagnostics,
                            ControlError::new(ControlErrorKind::Application, message),
                        );
                    } else {
                        reject_stale_project_worker(request, diagnostics, "project object preload");
                    }
                }
            }
        }
        _ => unreachable!("completion domain mismatch"),
    }
}
