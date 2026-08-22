use super::*;

pub struct ControlActorChannels {
    pub request_tx: Sender<OdonControlRequest>,
    pub legacy_rx: Receiver<OdonControlRequest>,
    pub presentation_rx: Receiver<RenderProjection>,
    pub platform_effect_rx: Receiver<PlatformEffect>,
    pub model_tx: Sender<ActorModelUpdate>,
    pub task_service: TaskServiceHandle,
    pub diagnostics: Arc<ActorDiagnostics>,
}

pub enum ActorModelUpdate {
    RendererCapabilities {
        gpu_available: bool,
    },
    BootstrapDataset {
        dataset: OmeZarrDataset,
        workspace: Value,
        store: Arc<dyn ReadableStorageTraits>,
        path: PathBuf,
    },
    BootstrapMode(ModelMode),
    BootstrapProject(ProjectModelSnapshot),
    BootstrapSettings {
        settings: AppSettings,
        path: Option<PathBuf>,
        recent_project_exists: Vec<(PathBuf, bool)>,
    },
    RendererWorkspaceObserved {
        workspace: Value,
        based_on_projection_revision: u64,
    },
    PresentationApplied(u64),
    ViewportGeometry {
        viewport_id: String,
        width: f32,
        height: f32,
    },
}

pub fn spawn_control_actor(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
) -> anyhow::Result<ControlActorChannels> {
    spawn_control_actor_with_services(wake_ui, resource_registry, None, None, None)
}

pub fn spawn_control_actor_with_object_loader(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
) -> anyhow::Result<ControlActorChannels> {
    spawn_control_actor_with_services(wake_ui, resource_registry, object_loader, None, None)
}

pub fn spawn_control_actor_with_services(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Option<Arc<dyn DatasetInspector>>,
    task_registry: Option<Arc<TaskRegistry>>,
) -> anyhow::Result<ControlActorChannels> {
    let (request_tx, request_rx) =
        crossbeam_channel::bounded::<OdonControlRequest>(ACTOR_QUEUE_CAPACITY);
    let (legacy_tx, legacy_rx) = crossbeam_channel::bounded(ACTOR_QUEUE_CAPACITY);
    // The renderer needs only the newest immutable projection. The actor keeps a receiver clone
    // solely to replace a stale queued projection when the UI is occluded.
    let (presentation_tx, presentation_rx) = crossbeam_channel::bounded(1);
    let presentation_coalesce_rx = presentation_rx.clone();
    let (load_tx, load_rx) = crossbeam_channel::bounded(WORKER_COMPLETION_CAPACITY);
    let (load_job_tx, load_job_rx) = crossbeam_channel::bounded::<LoadJob>(LOAD_JOB_CAPACITY);
    let (model_tx, model_rx) = crossbeam_channel::bounded(ACTOR_QUEUE_CAPACITY);
    let (platform_effect_tx, platform_effect_rx) = crossbeam_channel::bounded(8);
    let task_registry =
        task_registry.unwrap_or_else(|| TaskRegistry::shared(crate::control::EventHub::shared()));
    let (task_service, task_service_rx) =
        TaskServiceHandle::channel(ACTOR_QUEUE_CAPACITY, Arc::clone(&task_registry));
    let diagnostics = ActorDiagnostics::shared();
    diagnostics.set_alive(true);
    let dataset_inspector: Arc<dyn DatasetInspector> =
        dataset_inspector.unwrap_or_else(|| Arc::new(CoreDatasetInspector));

    spawn_resource_workers(load_job_rx, load_tx, object_loader, dataset_inspector)?;

    thread::Builder::new()
        .name("odon-control-actor".to_string())
        .spawn({
            let diagnostics = Arc::clone(&diagnostics);
            move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut model = AppModel::project();
                    let mut render_document = None;
                    loop {
                        while let Ok(update) = model_rx.try_recv() {
                            apply_model_update(
                                &mut model,
                                &mut render_document,
                                update,
                                &diagnostics,
                            );
                        }
                        crossbeam_channel::select! {
                            recv(task_service_rx) -> task_request => {
                                let Ok(task_request) = task_request else { break; };
                                task_registry.handle_service_request(task_request);
                            }
                            recv(request_rx) -> request => {
                                let Ok(request) = request else { break; };
                                diagnostics.record_queue_wait(request.command.queue_age());
                                dispatch_request(
                                    &mut model,
                                    request,
                                    &legacy_tx,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &render_document,
                                    &resource_registry,
                                    &wake_ui,
                                    &diagnostics,
                                );
                            }
                            recv(load_rx) -> completion => {
                                let Ok(completion) = completion else { break; };
                                finish_load(
                                    &mut model,
                                    &mut render_document,
                                    completion,
                                    &resource_registry,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &wake_ui,
                                    &diagnostics,
                                );
                            }
                            recv(model_rx) -> update => {
                                let Ok(update) = update else { break; };
                                apply_model_update(
                                    &mut model,
                                    &mut render_document,
                                    update,
                                    &diagnostics,
                                );
                            }
                        }
                    }
                }));
                diagnostics.set_alive(false);
                if outcome.is_err() {
                    eprintln!("odon control actor panicked; closing its command mailbox");
                }
            }
        })?;

    Ok(ControlActorChannels {
        request_tx,
        legacy_rx,
        presentation_rx,
        platform_effect_rx,
        model_tx,
        task_service,
        diagnostics,
    })
}

fn apply_model_update(
    model: &mut AppModel,
    render_document: &mut Option<Arc<RenderDocument>>,
    update: ActorModelUpdate,
    diagnostics: &ActorDiagnostics,
) {
    match update {
        ActorModelUpdate::RendererCapabilities { gpu_available } => {
            model.set_renderer_gpu_available(gpu_available);
        }
        ActorModelUpdate::BootstrapDataset {
            dataset,
            workspace,
            store,
            path: _,
        } => {
            if let Err(error) = model.bootstrap_dataset_from_renderer(&dataset, &workspace) {
                eprintln!("could not bootstrap control actor from renderer state: {error}");
                *render_document = None;
            } else {
                *render_document = Some(Arc::new(RenderDocument {
                    generation: model.document_generation(),
                    opened: OpenedDocument {
                        descriptor: crate::data::document::DocumentDescriptor::from_ome_zarr(
                            &dataset,
                        ),
                        resource: OmeZarrDocumentResource { dataset, store },
                    },
                }));
            }
        }
        ActorModelUpdate::BootstrapMode(mode) => {
            if mode != ModelMode::Single {
                *render_document = None;
            }
            model.bootstrap_mode_from_renderer(mode);
        }
        ActorModelUpdate::BootstrapProject(snapshot) => {
            model.bootstrap_project_from_renderer(snapshot);
        }
        ActorModelUpdate::BootstrapSettings {
            settings,
            path,
            recent_project_exists,
        } => {
            model.bootstrap_settings(settings, path, recent_project_exists);
        }
        ActorModelUpdate::RendererWorkspaceObserved {
            workspace,
            based_on_projection_revision,
        } => {
            model.observe_renderer_workspace(&workspace, based_on_projection_revision);
        }
        ActorModelUpdate::PresentationApplied(revision) => {
            model.mark_projection_presented(revision);
            diagnostics.projection_presented(revision);
        }
        ActorModelUpdate::ViewportGeometry {
            viewport_id,
            width,
            height,
        } => model.report_viewport_geometry(&viewport_id, width, height),
    }
}
