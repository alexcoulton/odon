use super::*;

pub(super) fn dispatch_request(
    model: &mut AppModel,
    request: OdonControlRequest,
    presentation_tx: &Sender<RenderProjection>,
    presentation_coalesce_rx: &Receiver<RenderProjection>,
    platform_effect_tx: &Sender<PlatformEffect>,
    load_job_tx: &Sender<LoadJob>,
    presentation_captures: &mut PresentationCaptureManager,
    presentation_capture_tx: &Sender<PresentationCaptureRequest>,
    render_document: &Option<Arc<RenderDocument>>,
    remote_session: &mut RemoteSessionState,
    resource_registry: &ResourceRegistry,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    if let Err(error) = prepare_request(&request) {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let _ = request.reply.send(Err(error));
        return;
    }

    let mode = model.mode().as_str();
    if !request.command.available_in().contains(&mode) {
        diagnostics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        diagnostics.record_reply_time(request.command.queue_age());
        let method = request.command.method();
        let _ = request.reply.send(Err(ControlError::new(
            ControlErrorKind::NotReady,
            format!("{method} is not available while Odon is in {mode} mode"),
        )
        .with_data(json!({
            "method": method,
            "mode": mode,
            "available_in": request.command.available_in(),
            "loading": model.loading_state()["loading"],
        }))));
        return;
    }

    let routes_to_legacy =
        crate::control::registry::method(request.command.method()).is_some_and(|descriptor| {
            crate::control::registry::execution_owner(
                descriptor,
                mode,
                request.command.params(),
                false,
            ) == ExecutionOwner::LegacyUi
        });
    if routes_to_legacy {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::Application,
                "registered application method has no actor execution route",
            ),
        );
        return;
    }

    if matches!(
        request.command.method(),
        "viewer.screenshot.capture"
            | "viewer.workspace.screenshot.capture"
            | "app.screenshot.capture"
            | "project.screenshot.capture"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        presentation_captures.begin(
            model,
            request,
            render_document,
            presentation_tx,
            presentation_coalesce_rx,
            presentation_capture_tx,
            wake_ui,
            diagnostics,
        );
        return;
    }

    if request.command.method() == "project.views.apply" {
        match model.prepare_project_view_apply_resources(request.command.params()) {
            Ok(Some(spec)) => {
                diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
                begin_project_view_apply(
                    model,
                    request,
                    spec,
                    load_job_tx,
                    render_document,
                    diagnostics,
                );
                return;
            }
            Ok(None) => {}
            Err(error) => {
                reject_actor_request(request, diagnostics, error);
                return;
            }
        }
    }

    if matches!(
        request.command.method(),
        "app.settings.set" | "app.recent_projects.forget" | "app.recent_projects.clear"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_settings_mutation(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "app.lifecycle.request_close" | "app.lifecycle.request_quit"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_lifecycle_request(
            model,
            request,
            load_job_tx,
            platform_effect_tx,
            wake_ui,
            diagnostics,
        );
        return;
    }

    match request.command.method() {
        "datasets.s3.get_session" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            finish_request(request, remote_session.snapshot(), diagnostics);
            return;
        }
        "datasets.s3.configure_session" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            let params = request.command.params().clone();
            let response = remote_session.configure(&params, model);
            finish_request(request, response, diagnostics);
            return;
        }
        "datasets.s3.clear_session" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            let response = remote_session.clear(model);
            finish_request(request, response, diagnostics);
            return;
        }
        "datasets.s3.list" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            begin_remote_list(model, remote_session, request, load_job_tx, diagnostics);
            return;
        }
        "datasets.open_http" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            begin_remote_http_open(model, request, load_job_tx, diagnostics);
            return;
        }
        "datasets.open_s3" => {
            diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
            begin_remote_s3_open(model, remote_session, request, load_job_tx, diagnostics);
            return;
        }
        _ => {}
    }

    if request.command.method() == "datasets.open_ome_zarr" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_ome_zarr_load(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "datasets.open_tiff" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_tiff_load(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "datasets.open_spatialdata" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_spatialdata_load(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "datasets.open_xenium" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_xenium_load(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "datasets.inspect" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_dataset_inspection(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "datasets.open_mosaic_samplesheet" | "project.rois.open_selected_mosaic"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_mosaic_open(model, remote_session, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "mosaic.objects.load" | "mosaic.objects.load_selected"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_mosaic_object_load(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "mosaic.objects.cancel_load" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        match model.cancel_mosaic_object_load() {
            Ok(response) => {
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
                    json!({"mode":"mosaic","result":response}),
                    diagnostics,
                );
            }
            Err(error) => reject_actor_request(request, diagnostics, error),
        }
        return;
    }
    if request.command.method() == "deep_links.resolve" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_deep_link_resolution(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "deep_links.apply" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_deep_link_application(
            model,
            remote_session,
            request,
            load_job_tx,
            render_document,
            diagnostics,
        );
        return;
    }
    if request.command.method() == "viewer.channels.intensity_stats" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_channel_intensity(model, request, load_job_tx, render_document, diagnostics);
        return;
    }
    if request.command.method() == "viewer.screenshot.settings.set" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_screenshot_settings_update(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "memory.pin" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        let started = if model.mode() == ModelMode::Mosaic {
            begin_mosaic_memory_pin(model, request, load_job_tx, diagnostics)
        } else {
            begin_memory_pin(model, request, load_job_tx, render_document, diagnostics)
        };
        if started {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "viewer.thresholds.preview.configure" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_threshold_configure(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.thresholds.preview.start" | "viewer.thresholds.preview.refresh"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        let refresh = request.command.method() == "viewer.thresholds.preview.refresh";
        if begin_threshold_load(
            model,
            request,
            load_job_tx,
            render_document,
            refresh,
            diagnostics,
        ) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "viewer.thresholds.preview.apply" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_threshold_apply(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.analysis.histogram"
            | "viewer.analysis.suggest_thresholds"
            | "viewer.analysis.warmup.start"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        let method = request.command.method();
        let kind = match method {
            "viewer.analysis.histogram" => AnalysisComputeKind::Histogram,
            "viewer.analysis.suggest_thresholds" => AnalysisComputeKind::ThresholdSuggestions,
            _ => AnalysisComputeKind::Warmup,
        };
        let present = matches!(kind, AnalysisComputeKind::Warmup);
        if begin_analysis_compute(model, request, load_job_tx, kind, diagnostics) && present {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "viewer.analysis.presets.import" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_analysis_preset_import(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.analysis.presets.export" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_analysis_preset_export(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.measurements.start" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_measurement(model, request, load_job_tx, render_document, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if matches!(
        request.command.method(),
        "exports.objects.start"
            | "exports.objects.export_csv"
            | "exports.objects.export_geoparquet"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_object_export(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "project.open" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_open(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(request.command.method(), "project.save" | "project.save_as") {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_save(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "project.samplesheets.inspect" | "project.samplesheets.validate"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_samplesheet_inspect(request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "project.samplesheets.import" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_samplesheet_import(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "project.samplesheets.export" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_samplesheet_export(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "project.discovery.add_root" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_discovery(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "project.objects.preload.get" | "project.objects.preload.list_sources"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_object_source_scan(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "project.objects.preload.start" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_object_preload(model, request, load_job_tx, diagnostics);
        publish_projection(
            model,
            render_document.clone(),
            presentation_tx,
            presentation_coalesce_rx,
            wake_ui,
            diagnostics,
        );
        return;
    }
    if request.command.method() == "project.objects.preload.clear" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        let (removed, cancelled) = model.clear_project_object_preload();
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
                "cleared": true,
                "removed": removed,
                "cancelled": cancelled,
                "preload": model.project_object_preload_snapshot(),
            }),
            diagnostics,
        );
        return;
    }
    if request.command.method() == "project.rois.open" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_project_roi_open(model, remote_session, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.objects.source.load" | "viewer.objects.source.reload"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_object_resource_load(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.segmentation_geojson.source.load" | "viewer.segmentation_geojson.source.reload"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_segmentation_geojson_load(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }
    if request.command.method() == "viewer.labels.load"
        || (request.command.method() == "viewer.labels.set_visibility"
            && model.labels_require_load(request.command.params()))
    {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_label_load(model, request, load_job_tx, render_document, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.viewports.objects.filter.set"
            | "viewer.objects.set_filter"
            | "viewer.objects.filters.set_model"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_object_filter_evaluation(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.objects.selection.select_filtered"
        && request.command.params().get("filter_query").is_some()
    {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_object_selection_filter_evaluation(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.masks.import_geojson" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_mask_import(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.masks.export_geojson" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_mask_export(model, request, load_job_tx, diagnostics);
        return;
    }
    if request.command.method() == "viewer.masks.persistence.append_geojson" {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        begin_mask_append(model, request, load_job_tx, diagnostics);
        return;
    }
    if matches!(
        request.command.method(),
        "viewer.annotations.source.inspect"
            | "viewer.annotations.source.load"
            | "viewer.annotations.source.reload"
    ) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        if begin_annotation_load(model, request, load_job_tx, diagnostics) {
            publish_projection(
                model,
                render_document.clone(),
                presentation_tx,
                presentation_coalesce_rx,
                wake_ui,
                diagnostics,
            );
        }
        return;
    }

    if is_resource_registry_method(request.command.method()) {
        diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
        dispatch_resource_registry_request(
            model,
            request,
            resource_registry,
            presentation_tx,
            presentation_coalesce_rx,
            render_document,
            wake_ui,
            diagnostics,
        );
        return;
    }

    let method = request.command.method();
    let params = request.command.params().clone();
    let model_started = Instant::now();
    let result = model.dispatch(method, &params);
    diagnostics.record_model_time(model_started.elapsed());
    let Some(result) = result else {
        reject_actor_request(
            request,
            diagnostics,
            ControlError::new(
                ControlErrorKind::Application,
                format!("registered method '{method}' has no actor implementation"),
            ),
        );
        return;
    };
    diagnostics.actor_requests.fetch_add(1, Ordering::Relaxed);
    match result {
        Ok(outcome) => {
            if outcome.present {
                publish_projection(
                    model,
                    render_document.clone(),
                    presentation_tx,
                    presentation_coalesce_rx,
                    wake_ui,
                    diagnostics,
                );
            }
            finish_request(request, outcome.response, diagnostics);
        }
        Err(error) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(error));
        }
    }
}

fn is_resource_registry_method(method: &str) -> bool {
    matches!(
        method,
        "data.resources.register"
            | "data.resources.list"
            | "data.resources.get"
            | "data.resources.remove"
            | "viewer.layers.add"
            | "viewer.layers.list"
            | "viewer.layers.get"
            | "viewer.layers.update"
            | "viewer.layers.remove"
            | "viewer.layers.reorder"
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_resource_registry_request(
    model: &mut AppModel,
    request: OdonControlRequest,
    registry: &ResourceRegistry,
    presentation_tx: &Sender<RenderProjection>,
    presentation_coalesce_rx: &Receiver<RenderProjection>,
    render_document: &Option<Arc<RenderDocument>>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    let method = request.command.method();
    let params = request.command.params();
    let session_id = request.session_id.as_str();
    let model_started = Instant::now();
    let result = (|| -> Result<Value, ControlError> {
        match method {
            "data.resources.register" => {
                serialize_control(registry.register_resource(params.clone(), session_id)?)
            }
            "data.resources.list" => Ok(json!({
                "resources": registry.list_resources(),
                "revision": request.event_hub.revision(),
            })),
            "data.resources.get" => serialize_control(
                registry.get_resource(required_registry_id(method, "resource_id", params)?)?,
            ),
            "data.resources.remove" => {
                let id = required_registry_id(method, "resource_id", params)?;
                registry.remove_resource(id, session_id)?;
                Ok(json!({"resource_id": id, "removed": true}))
            }
            "viewer.layers.add" => {
                serialize_control(registry.add_layer(params.clone(), session_id)?)
            }
            "viewer.layers.list" => Ok(json!({
                "layers": registry.list_layers(),
                "revision": request.event_hub.revision(),
            })),
            "viewer.layers.get" => serialize_control(
                registry.get_layer(required_registry_id(method, "layer_id", params)?)?,
            ),
            "viewer.layers.update" => {
                let id = required_registry_id(method, "layer_id", params)?;
                let mut patch = params.clone();
                patch
                    .as_object_mut()
                    .map(|object| object.remove("layer_id"));
                if let Some(expected) = request.command.if_revision() {
                    patch
                        .as_object_mut()
                        .expect("validated command params are an object")
                        .insert("if_revision".to_string(), json!(expected));
                }
                serialize_control(registry.update_layer(id, &patch, session_id)?)
            }
            "viewer.layers.remove" => {
                let id = required_registry_id(method, "layer_id", params)?;
                registry.remove_layer(id, session_id)?;
                Ok(json!({"layer_id": id, "removed": true}))
            }
            "viewer.layers.reorder" => {
                let order = params
                    .get("order")
                    .and_then(Value::as_array)
                    .ok_or_else(|| ControlError::invalid_params(method, "order is required"))?
                    .iter()
                    .map(|id| {
                        id.as_str().map(str::to_string).ok_or_else(|| {
                            ControlError::invalid_params(method, "layer IDs must be strings")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(json!({
                    "layers": registry.reorder_layers(&order, session_id)?,
                    "revision": request.event_hub.revision(),
                }))
            }
            _ => unreachable!("resource registry method was checked before dispatch"),
        }
    })();
    diagnostics.record_model_time(model_started.elapsed());

    match result {
        Ok(mut response) => {
            if request.command.mutates() {
                let (resources, layers) = registry.project_manifest();
                let project_changed = model.update_project_manifest(resources, layers);
                if project_changed {
                    publish_projection(
                        model,
                        render_document.clone(),
                        presentation_tx,
                        presentation_coalesce_rx,
                        wake_ui,
                        diagnostics,
                    );
                } else {
                    // Session-owned layers still affect the next frame even though they do not
                    // alter the persisted project model.
                    wake_ui();
                }
            }
            if let Some(object) = response.as_object_mut() {
                object.insert(
                    "_control".to_string(),
                    json!({"revision": request.event_hub.revision()}),
                );
            }
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Ok(response));
        }
        Err(error) => {
            diagnostics
                .rejected_requests
                .fetch_add(1, Ordering::Relaxed);
            diagnostics.record_reply_time(request.command.queue_age());
            let _ = request.reply.send(Err(error));
        }
    }
}

fn required_registry_id<'a>(
    method: &str,
    field: &str,
    params: &'a Value,
) -> Result<&'a str, ControlError> {
    params
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params(method, format!("{field} is required")))
}

fn serialize_control(value: impl serde::Serialize) -> Result<Value, ControlError> {
    serde_json::to_value(value).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize control resource: {error}"),
        )
    })
}

fn prepare_request(request: &OdonControlRequest) -> Result<(), ControlError> {
    if let Some(task_id) = request.task_id.as_deref() {
        match request.task_registry.get(task_id) {
            Ok(task) if task.state == TaskState::Cancelled => {
                return Err(
                    ControlError::new(ControlErrorKind::Cancelled, "task was cancelled")
                        .with_data(json!({"task_id": task_id})),
                );
            }
            Ok(_) => {
                request.task_registry.mark_running(task_id)?;
            }
            Err(error) => return Err(error),
        }
    }
    let current = request.event_hub.revision();
    if request.command.method() != "viewer.layers.update"
        && let Some(expected) = request.command.if_revision()
        && expected != current
    {
        return Err(ControlError::new(
            ControlErrorKind::Conflict,
            format!("state revision conflict: expected {expected}, current revision is {current}"),
        )
        .with_data(json!({
            "method": request.command.method(),
            "expected_revision": expected,
            "current_revision": current,
        })));
    }
    Ok(())
}
