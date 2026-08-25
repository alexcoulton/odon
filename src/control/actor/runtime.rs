use super::*;

pub struct ControlActorChannels {
    pub request_tx: Sender<OdonControlRequest>,
    pub presentation_rx: Receiver<RenderProjection>,
    pub presentation_capture_rx: Receiver<PresentationCaptureRequest>,
    pub presentation_completion_tx: Sender<PresentationCaptureCompletion>,
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
        store: Arc<dyn ReadableStorageTraits>,
    },
    BootstrapMode(ModelMode),
    BootstrapMosaic {
        resource: ControlMosaicResource,
    },
    BootstrapProject(ProjectModelSnapshot),
    BootstrapSettings {
        settings: AppSettings,
        path: Option<PathBuf>,
        recent_project_exists: Vec<(PathBuf, bool)>,
    },
    RendererObservation {
        observation: Value,
        based_on_projection_revision: u64,
    },
    PresentationApplied(u64),
    ViewportGeometry {
        viewport_id: String,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
}

impl ActorModelUpdate {
    fn allowed_during_presentation_barrier(&self) -> bool {
        matches!(
            self,
            Self::RendererCapabilities { .. }
                | Self::PresentationApplied(_)
                | Self::ViewportGeometry { .. }
        )
    }
}

pub fn spawn_control_actor(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
) -> anyhow::Result<ControlActorChannels> {
    spawn_control_actor_with_services(wake_ui, resource_registry, None, None, None, None, None)
}

pub fn spawn_control_actor_with_object_loader(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
) -> anyhow::Result<ControlActorChannels> {
    spawn_control_actor_with_services(
        wake_ui,
        resource_registry,
        object_loader,
        None,
        None,
        None,
        None,
    )
}

pub fn spawn_control_actor_with_services(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Option<Arc<dyn DatasetInspector>>,
    task_registry: Option<Arc<TaskRegistry>>,
    remote_backend: Option<Arc<dyn RemoteDatasetBackend>>,
    alternate_backend: Option<Arc<dyn AlternateDatasetBackend>>,
) -> anyhow::Result<ControlActorChannels> {
    spawn_control_actor_with_services_and_ui(
        wake_ui,
        resource_registry,
        object_loader,
        dataset_inspector,
        task_registry,
        remote_backend,
        alternate_backend,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spawn_control_actor_with_services_and_ui(
    wake_ui: UiWake,
    resource_registry: Arc<ResourceRegistry>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Option<Arc<dyn DatasetInspector>>,
    task_registry: Option<Arc<TaskRegistry>>,
    remote_backend: Option<Arc<dyn RemoteDatasetBackend>>,
    alternate_backend: Option<Arc<dyn AlternateDatasetBackend>>,
    ui_registry: Option<Arc<UiRegistry>>,
) -> anyhow::Result<ControlActorChannels> {
    let (request_tx, request_rx) =
        crossbeam_channel::bounded::<OdonControlRequest>(ACTOR_QUEUE_CAPACITY);
    // The renderer needs only the newest immutable projection. The actor keeps a receiver clone
    // solely to replace a stale queued projection when the UI is occluded.
    let (presentation_tx, presentation_rx) = crossbeam_channel::bounded(1);
    let presentation_coalesce_rx = presentation_rx.clone();
    let (presentation_capture_tx, presentation_capture_rx) = crossbeam_channel::bounded(1);
    let (presentation_completion_tx, presentation_completion_rx) =
        crossbeam_channel::bounded(ACTOR_QUEUE_CAPACITY);
    let (load_tx, load_rx) = crossbeam_channel::bounded(WORKER_COMPLETION_CAPACITY);
    let (load_job_tx, load_job_rx) = crossbeam_channel::bounded::<LoadJob>(LOAD_JOB_CAPACITY);
    let (model_tx, model_rx) = crossbeam_channel::bounded::<ActorModelUpdate>(ACTOR_QUEUE_CAPACITY);
    let (platform_effect_tx, platform_effect_rx) = crossbeam_channel::bounded(8);
    let task_registry =
        task_registry.unwrap_or_else(|| TaskRegistry::shared(crate::control::EventHub::shared()));
    let (task_service, task_service_rx) =
        TaskServiceHandle::channel(ACTOR_QUEUE_CAPACITY, Arc::clone(&task_registry));
    let diagnostics = ActorDiagnostics::shared();
    diagnostics.set_alive(true);
    let dataset_inspector: Arc<dyn DatasetInspector> =
        dataset_inspector.unwrap_or_else(|| Arc::new(CoreDatasetInspector));
    let remote_backend: Arc<dyn RemoteDatasetBackend> =
        remote_backend.unwrap_or_else(|| Arc::new(CoreRemoteDatasetBackend));
    let alternate_backend: Arc<dyn AlternateDatasetBackend> =
        alternate_backend.unwrap_or_else(|| Arc::new(UnavailableAlternateDatasetBackend));

    spawn_resource_workers(
        load_job_rx,
        load_tx,
        object_loader,
        dataset_inspector,
        remote_backend,
        alternate_backend,
    )?;

    thread::Builder::new()
        .name("odon-control-actor".to_string())
        .spawn({
            let diagnostics = Arc::clone(&diagnostics);
            move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut model = AppModel::project();
                    let mut render_document = None;
                    let mut remote_session = RemoteSessionState::default();
                    let mut presentation_captures = PresentationCaptureManager::default();
                    let mut deferred_requests = VecDeque::new();
                    let mut deferred_completions = VecDeque::new();
                    let mut deferred_model_updates = VecDeque::new();
                    let maintenance_tick = crossbeam_channel::tick(Duration::from_secs(1));
                    loop {
                        if !presentation_captures.barrier_active() {
                            if let Some(update) = deferred_model_updates.pop_front() {
                                apply_model_update(
                                    &mut model,
                                    &mut render_document,
                                    update,
                                    &load_job_tx,
                                    &diagnostics,
                                );
                                continue;
                            }
                            if let Some(completion) = deferred_completions.pop_front() {
                                finish_load(
                                    &mut model,
                                    &mut render_document,
                                    completion,
                                    &mut remote_session,
                                    &resource_registry,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &wake_ui,
                                    &diagnostics,
                                );
                                enqueue_restored_annotations(
                                    &mut model,
                                    &load_job_tx,
                                    &diagnostics,
                                );
                                continue;
                            }
                            if let Some(request) = deferred_requests.pop_front() {
                                dispatch_request(
                                    &mut model,
                                    request,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &mut presentation_captures,
                                    &presentation_capture_tx,
                                    &render_document,
                                    &mut remote_session,
                                    &resource_registry,
                                    ui_registry.as_deref(),
                                    &wake_ui,
                                    &diagnostics,
                                );
                                continue;
                            }
                        }
                        while let Ok(update) = model_rx.try_recv() {
                            if presentation_captures.barrier_active()
                                && !update.allowed_during_presentation_barrier()
                            {
                                deferred_model_updates.push_back(update);
                                continue;
                            }
                            apply_model_update(
                                &mut model,
                                &mut render_document,
                                update,
                                &load_job_tx,
                                &diagnostics,
                            );
                            presentation_captures.release_presentable(
                                model.presented_projection_revision(),
                                &presentation_capture_tx,
                                &wake_ui,
                            );
                        }
                        if !presentation_captures.barrier_active()
                            && enqueue_auto_contrast_on_open(
                                &mut model,
                                &render_document,
                                &load_job_tx,
                                &diagnostics,
                            )
                        {
                            publish_projection(
                                &mut model,
                                render_document.clone(),
                                &presentation_tx,
                                &presentation_coalesce_rx,
                                &wake_ui,
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
                                if presentation_captures.barrier_active() && request.command.mutates() {
                                    if deferred_requests.len() >= ACTOR_QUEUE_CAPACITY {
                                        reject_actor_request(
                                            request,
                                            &diagnostics,
                                            ControlError::new(
                                                ControlErrorKind::NotReady,
                                                "the actor mutation queue is full while waiting for screenshot presentation",
                                            ),
                                        );
                                    } else {
                                        deferred_requests.push_back(request);
                                    }
                                    continue;
                                }
                                dispatch_request(
                                    &mut model,
                                    request,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &mut presentation_captures,
                                    &presentation_capture_tx,
                                    &render_document,
                                    &mut remote_session,
                                    &resource_registry,
                                    ui_registry.as_deref(),
                                    &wake_ui,
                                    &diagnostics,
                                );
                            }
                            recv(load_rx) -> completion => {
                                let Ok(completion) = completion else { break; };
                                if presentation_captures.barrier_active()
                                    && !completion.allowed_during_presentation_barrier()
                                {
                                    deferred_completions.push_back(completion);
                                    continue;
                                }
                                finish_load(
                                    &mut model,
                                    &mut render_document,
                                    completion,
                                    &mut remote_session,
                                    &resource_registry,
                                    &presentation_tx,
                                    &presentation_coalesce_rx,
                                    &platform_effect_tx,
                                    &load_job_tx,
                                    &wake_ui,
                                    &diagnostics,
                                );
                                enqueue_restored_annotations(
                                    &mut model,
                                    &load_job_tx,
                                    &diagnostics,
                                );
                            }
                            recv(presentation_completion_rx) -> completion => {
                                let Ok(completion) = completion else { break; };
                                presentation_captures.receive_pixels(
                                    &mut model,
                                    completion,
                                    &load_job_tx,
                                    &diagnostics,
                                );
                            }
                            recv(model_rx) -> update => {
                                let Ok(update) = update else { break; };
                                if presentation_captures.barrier_active()
                                    && !update.allowed_during_presentation_barrier()
                                {
                                    deferred_model_updates.push_back(update);
                                    continue;
                                }
                                apply_model_update(
                                    &mut model,
                                    &mut render_document,
                                    update,
                                    &load_job_tx,
                                    &diagnostics,
                                );
                                presentation_captures.release_presentable(
                                    model.presented_projection_revision(),
                                    &presentation_capture_tx,
                                    &wake_ui,
                                );
                            }
                            recv(maintenance_tick) -> _ => {
                                presentation_captures.sweep(&mut model, &diagnostics);
                                enqueue_restored_annotations(
                                    &mut model,
                                    &load_job_tx,
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
        presentation_rx,
        presentation_capture_rx,
        presentation_completion_tx,
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
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    let mut restore_annotations = false;
    match update {
        ActorModelUpdate::RendererCapabilities { gpu_available } => {
            model.set_renderer_gpu_available(gpu_available);
        }
        ActorModelUpdate::BootstrapDataset { dataset, store } => {
            if let Err(error) = model.bootstrap_dataset(&dataset) {
                eprintln!("could not bootstrap control actor dataset: {error}");
                *render_document = None;
            } else {
                *render_document = Some(Arc::new(RenderDocument {
                    generation: model.document_generation(),
                    opened: OpenedDocument {
                        descriptor: crate::data::document::DocumentDescriptor::from_ome_zarr(
                            &dataset,
                        ),
                        resource: OmeZarrDocumentResource {
                            dataset,
                            store,
                            runtime_guard: None,
                        },
                    }
                    .into_control(),
                }));
                restore_annotations = true;
            }
        }
        ActorModelUpdate::BootstrapMode(mode) => {
            if mode != ModelMode::Single {
                *render_document = None;
            }
            model.bootstrap_mode_from_renderer(mode);
        }
        ActorModelUpdate::BootstrapMosaic { resource } => {
            *render_document = None;
            if let Err(error) = model.bootstrap_mosaic(resource) {
                eprintln!("could not bootstrap control actor mosaic: {error:?}");
            } else {
                restore_annotations = true;
            }
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
        ActorModelUpdate::RendererObservation {
            observation,
            based_on_projection_revision,
        } => {
            model.observe_renderer_state(&observation, based_on_projection_revision);
        }
        ActorModelUpdate::PresentationApplied(revision) => {
            model.mark_projection_presented(revision);
            diagnostics.projection_presented(revision);
        }
        ActorModelUpdate::ViewportGeometry {
            viewport_id,
            x,
            y,
            width,
            height,
        } => model.report_viewport_geometry(&viewport_id, x, y, width, height),
    }
    model.apply_startup_shell_layout_if_needed();
    if restore_annotations {
        enqueue_restored_annotations(model, load_job_tx, diagnostics);
    }
}

fn enqueue_restored_annotations(
    model: &mut AppModel,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    for spec in model.prepare_restored_annotation_loads() {
        match load_job_tx.try_send(LoadJob::Annotations {
            request: None,
            spec,
        }) {
            Ok(()) => {
                diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => {
                let LoadJob::Annotations { spec, .. } = error.into_inner() else {
                    unreachable!("restored annotation submission returns its own job")
                };
                model.fail_annotation_load(
                    &spec,
                    "Annotation worker queue is unavailable".to_string(),
                );
            }
        }
    }
}
