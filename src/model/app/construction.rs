//! AppModel construction and renderer/bootstrap installation.

use super::*;
use crate::model::{MosaicMemoryPinResult, MosaicMemoryPinSpec};

impl AppModel {
    pub fn project() -> Self {
        Self {
            mode: ModelMode::Project,
            dataset: None,
            project: ProjectModel::default(),
            project_initialized: false,
            readiness: ReadinessModel::default(),
            projection_revision: 0,
            presented_projection_revision: 0,
            document_generation: 0,
            dataset_inspection_operation_generation: 0,
            remote_listing_operation_generation: 0,
            deep_link_resolve_operation_generation: 0,
            deep_link_apply_operation_generation: 0,
            deep_link_apply_pending: false,
            project_view_apply_generation: 0,
            project_view_apply_pending: false,
            project_operation_generation: 0,
            project_operation_pending: false,
            project_object_preload: ProjectObjectPreloadCatalog::default(),
            project_roi_open_generation: 0,
            project_roi_open_pending: false,
            object_resource_generation: 0,
            installed_object_resource_generation: 0,
            object_resource_pending: false,
            object_filter_operation_generation: 0,
            pending_object_filters: HashMap::new(),
            object_selection_filter_operation_generation: 0,
            pending_object_selection_filters: HashMap::new(),
            mask_io_operation_generation: 0,
            channel_compute: ChannelComputeModel::default(),
            settings: AppSettings::default(),
            recent_project_exists: HashMap::new(),
            settings_path: None,
            settings_status: String::new(),
            settings_operation_generation: 0,
            settings_operation_pending: false,
            screenshot_preferences: ScreenshotPreferences::default(),
            screenshot_settings_generation: 0,
            screenshot_settings_pending: false,
            tile_loading: TileLoadingModel::default(),
            pinned_memory: PinnedMemoryModel::default(),
            memory_projection_cache: None,
            threshold_preview: ThresholdPreviewModel::default(),
            annotations: AnnotationModel::default(),
            analyses: HashMap::from([(ObjectTarget::Primary, AnalysisModel::default())]),
            measurement: MeasurementModel::default(),
            object_export: ObjectExportModel::default(),
            mosaic: MosaicModel::default(),
            mosaic_operation_generation: 0,
            mosaic_operation_pending: false,
            measured_viewports: HashSet::new(),
            renderer_gpu_available: false,
        }
    }

    pub fn mode(&self) -> ModelMode {
        self.mode
    }

    pub(crate) fn projection_revision(&self) -> u64 {
        self.projection_revision
    }

    pub(crate) fn presented_projection_revision(&self) -> u64 {
        self.presented_projection_revision
    }

    pub(crate) fn prepare_project_capture(&mut self) -> Result<Value, ControlError> {
        self.show_project()
    }

    pub(crate) fn capture_viewport_id(
        &self,
        requested: Option<&str>,
    ) -> Result<Option<String>, ControlError> {
        match self.mode {
            ModelMode::Single => {
                let workspace = &self.dataset()?.workspace;
                let id = match requested {
                    Some(value) => {
                        ViewportId::new(value).map_err(|error| invalid(error.to_string()))?
                    }
                    None => workspace.active_id().clone(),
                };
                if workspace.get(&id).is_none() {
                    return Err(ControlError::new(
                        ControlErrorKind::ResourceNotFound,
                        format!("viewport '{id}' was not found"),
                    ));
                }
                Ok(Some(id.to_string()))
            }
            ModelMode::Mosaic => {
                self.mosaic.require_ready()?;
                if requested.is_some() {
                    return Err(invalid(
                        "viewport_id is not supported by the mosaic viewer capture",
                    ));
                }
                Ok(None)
            }
            ModelMode::Project => Err(ControlError::new(
                ControlErrorKind::WrongMode,
                "No dataset viewer is currently open",
            )),
            ModelMode::Transition => Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon is currently transitioning between views",
            )),
        }
    }

    pub(crate) fn capture_default_filename(&self) -> Result<String, ControlError> {
        if self.mode == ModelMode::Mosaic {
            self.mosaic.default_screenshot_filename()
        } else {
            let dataset = self.dataset()?;
            Ok(default_screenshot_filename(
                &dataset.descriptor.source.display_name(),
            ))
        }
    }

    pub(crate) fn begin_presentation_capture(&mut self, generation: u64, status: &str) {
        self.readiness.begin_scoped(
            OperationKind::Presentation,
            generation.to_string(),
            generation,
            status,
        );
    }

    pub(crate) fn finish_presentation_capture(&mut self, generation: u64, status: &str) {
        self.readiness.finish_scoped(
            OperationKind::Presentation,
            &generation.to_string(),
            generation,
            status,
        );
    }

    pub(crate) fn fail_presentation_capture(&mut self, generation: u64, status: &str) {
        self.readiness.fail_scoped(
            OperationKind::Presentation,
            &generation.to_string(),
            generation,
            status,
        );
    }

    pub(crate) fn cancel_presentation_capture(&mut self, generation: u64, status: &str) {
        self.readiness.cancel_scoped(
            OperationKind::Presentation,
            &generation.to_string(),
            generation,
            status,
        );
    }

    pub fn project_snapshot(&self) -> ProjectModelSnapshot {
        self.project.snapshot()
    }

    pub(crate) fn begin_mosaic_open(&mut self, source: impl Into<String>) -> u64 {
        self.mosaic_operation_generation = self.mosaic_operation_generation.wrapping_add(1).max(1);
        self.mosaic_operation_pending = true;
        self.readiness.begin(
            OperationKind::Mosaic,
            self.mosaic_operation_generation,
            format!("Opening mosaic {}", source.into()),
        );
        self.mosaic_operation_generation
    }

    pub fn bootstrap_mosaic(
        &mut self,
        mut resource: ControlMosaicResource,
    ) -> Result<(), ControlError> {
        let generation = resource.generation.max(1);
        resource.generation = generation;
        self.mosaic.install_resource(Arc::new(resource));
        let project = self.project.snapshot();
        self.mosaic
            .restore_project_state(&project.state, &project.config.layer_groups)?;
        let annotations = project
            .state
            .get("mosaic")
            .and_then(|state| state.get("annotation_layers"))
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| invalid(format!("invalid mosaic annotation layers: {error}")))?
            .unwrap_or_default();
        self.restore_annotation_states(annotations)?;
        self.mosaic_operation_generation = generation;
        self.mosaic_operation_pending = false;
        self.mode = ModelMode::Mosaic;
        self.dataset = None;
        self.readiness
            .mark_ready(OperationKind::Mosaic, generation, "Mosaic resources ready");
        Ok(())
    }

    pub(crate) fn install_mosaic_for_generation(
        &mut self,
        generation: u64,
        mut resource: ControlMosaicResource,
    ) -> Result<bool, ControlError> {
        if generation != self.mosaic_operation_generation || !self.mosaic_operation_pending {
            return Ok(false);
        }
        resource.generation = generation;
        self.mosaic.install_resource(Arc::new(resource));
        let project = self.project.snapshot();
        self.mosaic
            .restore_project_state(&project.state, &project.config.layer_groups)?;
        let annotations = project
            .state
            .get("mosaic")
            .and_then(|state| state.get("annotation_layers"))
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| invalid(format!("invalid mosaic annotation layers: {error}")))?
            .unwrap_or_default();
        self.restore_annotation_states(annotations)?;
        self.mosaic_operation_pending = false;
        self.mode = ModelMode::Mosaic;
        self.dataset = None;
        self.readiness
            .finish(OperationKind::Mosaic, generation, "Mosaic resources ready");
        Ok(true)
    }

    pub(crate) fn fail_mosaic_open(&mut self, generation: u64, message: impl Into<String>) -> bool {
        if generation != self.mosaic_operation_generation || !self.mosaic_operation_pending {
            return false;
        }
        self.mosaic_operation_pending = false;
        self.readiness
            .fail(OperationKind::Mosaic, generation, message)
    }

    pub(crate) fn cancel_mosaic_open(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if generation != self.mosaic_operation_generation || !self.mosaic_operation_pending {
            return false;
        }
        self.mosaic_operation_pending = false;
        self.readiness
            .cancel(OperationKind::Mosaic, generation, message)
    }

    pub(crate) fn mosaic_resource(&self) -> Option<Arc<ControlMosaicResource>> {
        self.mosaic.resource()
    }

    pub(crate) fn mosaic_resource_generation(&self) -> u64 {
        self.mosaic.resource_generation()
    }

    pub(crate) fn mosaic_projection_state(&self) -> Value {
        self.mosaic.projection_state()
    }

    pub(crate) fn annotation_projections(&self) -> Vec<ControlAnnotationLayerProjection> {
        self.annotations.projections()
    }

    pub(crate) fn mosaic_object_resources(&self) -> Vec<(usize, Arc<ControlObjectResource>)> {
        self.mosaic.object_resources()
    }

    pub(crate) fn secondary_object_projections(&self) -> Vec<ControlSecondaryObjectProjection> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Vec::new();
        };
        let mut layers = dataset
            .secondary_object_layers
            .values()
            .map(|layer| ControlSecondaryObjectProjection {
                layer_id: layer.layer_id,
                name: layer.name.clone(),
                generation: layer.generation,
                resource: Arc::clone(&layer.resource),
                selection: layer.selection.projection_json(),
                analysis_generation: self
                    .analysis_generation_for_target(ObjectTarget::SpatialShape(layer.layer_id)),
                analysis_state: self
                    .analysis_state_for_target(ObjectTarget::SpatialShape(layer.layer_id))
                    .clone(),
            })
            .collect::<Vec<_>>();
        layers.sort_by_key(|layer| layer.layer_id);
        layers
    }

    pub(crate) fn mosaic_pinned_level_resources(
        &self,
    ) -> Vec<(usize, Arc<ControlPinnedLevelResource>)> {
        self.mosaic.pinned_level_resources()
    }

    pub(crate) fn prepare_mosaic_memory_pin(
        &mut self,
        params: &Value,
    ) -> Result<MosaicMemoryPinSpec, ControlError> {
        let spec = self.mosaic.prepare_memory_pin(params)?;
        self.readiness.begin_scoped(
            OperationKind::MemoryPin,
            mosaic_memory_scope(&spec),
            spec.operation_generation,
            format!(
                "Pinning mosaic level {} for {} ROI(s)",
                spec.level,
                spec.items.len()
            ),
        );
        Ok(spec)
    }

    pub(crate) fn install_mosaic_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        result: MosaicMemoryPinResult,
        system: Option<SystemMemorySnapshot>,
    ) -> Option<Value> {
        let response = self.mosaic.install_memory_pin(spec, result, system)?;
        self.readiness.finish_scoped(
            OperationKind::MemoryPin,
            &mosaic_memory_scope(spec),
            spec.operation_generation,
            "Mosaic pinned level ready",
        );
        Some(response)
    }

    pub(crate) fn finish_mosaic_memory_confirmation(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        system: Option<SystemMemorySnapshot>,
        risk: &str,
        projected_bytes: u64,
        available_bytes: u64,
    ) -> Option<Value> {
        let response = self.mosaic.finish_memory_confirmation(
            spec,
            system,
            risk,
            projected_bytes,
            available_bytes,
        )?;
        self.readiness.finish_scoped(
            OperationKind::MemoryPin,
            &mosaic_memory_scope(spec),
            spec.operation_generation,
            "Mosaic RAM pinning requires confirmation",
        );
        Some(response)
    }

    pub(crate) fn fail_mosaic_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        let message = message.into();
        if !self.mosaic.fail_memory_pin(spec, message.clone()) {
            return false;
        }
        self.readiness.fail_scoped(
            OperationKind::MemoryPin,
            &mosaic_memory_scope(spec),
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn cancel_mosaic_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        let message = message.into();
        if !self.mosaic.cancel_memory_pin(spec, message.clone()) {
            return false;
        }
        self.readiness.cancel_scoped(
            OperationKind::MemoryPin,
            &mosaic_memory_scope(spec),
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn prepare_mosaic_object_load(
        &mut self,
        params: &Value,
    ) -> Result<MosaicObjectLoadSpec, ControlError> {
        let downsample_factor = params
            .get("downsample_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0) as f32;
        let spec = self.mosaic.prepare_object_load(params, downsample_factor)?;
        self.readiness.begin(
            OperationKind::MosaicObjects,
            spec.operation_generation,
            format!("Loading objects for {} mosaic ROI(s)", spec.items.len()),
        );
        Ok(spec)
    }

    pub(crate) fn finish_mosaic_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        result: MosaicObjectLoadResult,
    ) -> Option<Value> {
        let response = self.mosaic.finish_object_load(spec, result)?;
        let cancelled = response["cancelled"].as_bool() == Some(true);
        if cancelled {
            self.readiness.cancel(
                OperationKind::MosaicObjects,
                spec.operation_generation,
                "Mosaic object loading cancelled",
            );
        } else {
            self.readiness.finish(
                OperationKind::MosaicObjects,
                spec.operation_generation,
                "Mosaic object resources ready",
            );
        }
        Some(response)
    }

    pub(crate) fn fail_mosaic_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        message: impl Into<String>,
    ) -> bool {
        let message = message.into();
        if !self.mosaic.fail_object_load(spec, message.clone()) {
            return false;
        }
        self.readiness.fail(
            OperationKind::MosaicObjects,
            spec.operation_generation,
            message,
        )
    }

    pub(crate) fn cancel_mosaic_object_load(&mut self) -> Result<Value, ControlError> {
        let generation = self.mosaic.object_operation_generation();
        let response = self.mosaic.cancel_object_load_response()?;
        if generation > 0 {
            self.readiness.cancel(
                OperationKind::MosaicObjects,
                generation,
                "Mosaic object loading cancelled",
            );
        }
        Ok(response)
    }

    pub(crate) fn deep_link_current_resources(&self) -> Option<DeepLinkCurrentResources> {
        let dataset = self.dataset.as_ref()?;
        Some(DeepLinkCurrentResources {
            source_key: dataset.descriptor.source.source_key(),
            object: dataset.object_resource.clone(),
            label_available: dataset.label_available.clone(),
            label_loaded: dataset.label_loaded.clone(),
            label: dataset.label_resource.clone(),
        })
    }

    pub(crate) fn project_object_preload_scan(
        &mut self,
    ) -> (ProjectObjectPreloadScope, Vec<PathBuf>) {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        (
            self.project_object_preload.scope(),
            project_object_preload_candidates(&project),
        )
    }

    pub(crate) fn install_project_object_preload_sources(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        sources: Vec<ProjectObjectPreloadSource>,
    ) -> bool {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_object_preload.install_sources(scope, sources)
    }

    pub(crate) fn begin_project_object_preload(
        &mut self,
        settings: ProjectObjectPreloadSettings,
    ) -> Result<(u64, ProjectObjectPreloadScope, Vec<PathBuf>), ControlError> {
        let project = self.project_snapshot();
        if project.saved_path.is_none() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "save the project before preloading object segmentations",
            ));
        }
        self.project_object_preload.sync_scope(&project);
        if self.project_object_preload.is_loading() {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "project object preload is already running",
            ));
        }
        let candidates = project_object_preload_candidates(&project);
        if candidates.is_empty() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "project has no preload-eligible Parquet or GeoParquet segmentation paths",
            ));
        }
        let generation = self
            .project_object_preload
            .begin(settings, candidates.len());
        self.readiness.begin(
            OperationKind::ProjectObjectPreload,
            generation,
            format!("Preloading {} project object source(s)", candidates.len()),
        );
        Ok((generation, self.project_object_preload.scope(), candidates))
    }

    pub(crate) fn finish_project_object_preload(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        sources: Vec<ProjectObjectPreloadSource>,
        resources: Vec<(PathBuf, ControlObjectResource)>,
        failed: usize,
    ) -> bool {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        if !self
            .project_object_preload
            .finish(scope, generation, sources, resources, failed)
        {
            return false;
        }
        self.readiness.finish(
            OperationKind::ProjectObjectPreload,
            generation,
            "Project object preload ready",
        );
        true
    }

    pub(crate) fn fail_project_object_preload(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        if !self.project_object_preload.fail(scope, generation) {
            return false;
        }
        self.readiness
            .fail(OperationKind::ProjectObjectPreload, generation, message);
        true
    }

    pub(crate) fn cancel_project_object_preload(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        if !self.project_object_preload.fail(scope, generation) {
            return false;
        }
        self.readiness
            .cancel(OperationKind::ProjectObjectPreload, generation, message);
        true
    }

    pub(crate) fn clear_project_object_preload(&mut self) -> (usize, bool) {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        let (removed, cancelled, generation) = self.project_object_preload.clear();
        if cancelled {
            self.readiness.cancel(
                OperationKind::ProjectObjectPreload,
                generation,
                "Project object preload cleared",
            );
        }
        (removed, cancelled)
    }

    pub(crate) fn project_object_preload_snapshot(&mut self) -> Value {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_object_preload.snapshot()
    }

    pub(crate) fn project_object_preload_sources_snapshot(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Value {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_object_preload.list_sources(offset, limit)
    }

    pub(crate) fn project_object_preload_projection(&mut self) -> ProjectObjectPreloadProjection {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_object_preload.projection()
    }

    pub(crate) fn cached_project_object_resource(
        &mut self,
        path: &PathBuf,
    ) -> Option<Arc<ControlObjectResource>> {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_object_preload.cached_resource(path)
    }

    pub(crate) fn begin_project_roi_open(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        description: impl Into<String>,
    ) -> u64 {
        self.cancel_pending_deep_link_apply("Superseded by project ROI open");
        self.project_roi_open_generation = self.project_roi_open_generation.wrapping_add(1).max(1);
        self.project_roi_open_pending = true;
        self.readiness.begin(
            OperationKind::ProjectRoiOpen,
            self.project_roi_open_generation,
            description,
        );
        // Keep the scope synchronized before the worker starts. A later structural project change
        // changes this identity and makes the completion stale without disturbing the current
        // usable document.
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        debug_assert_eq!(&self.project_object_preload.scope(), scope);
        self.project_roi_open_generation
    }

    pub(crate) fn project_roi_open_is_current(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
    ) -> bool {
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        self.project_roi_open_pending
            && self.project_roi_open_generation == generation
            && self.project_object_preload.scope() == *scope
            && self
                .readiness
                .is_pending(OperationKind::ProjectRoiOpen, generation)
    }

    pub(crate) fn fail_project_roi_open(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if !self.project_roi_open_is_current(scope, generation) {
            return false;
        }
        self.project_roi_open_pending = false;
        self.readiness
            .fail(OperationKind::ProjectRoiOpen, generation, message);
        true
    }

    pub(crate) fn cancel_project_roi_open(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if !self.project_roi_open_is_current(scope, generation) {
            return false;
        }
        self.project_roi_open_pending = false;
        self.readiness
            .cancel(OperationKind::ProjectRoiOpen, generation, message);
        true
    }

    pub(crate) fn supersede_project_roi_open(&mut self, generation: u64, message: &str) {
        if self.project_roi_open_pending && self.project_roi_open_generation == generation {
            self.project_roi_open_pending = false;
            self.readiness
                .cancel(OperationKind::ProjectRoiOpen, generation, message);
        }
    }

    pub(crate) fn install_project_roi_for_generation(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        operation_generation: u64,
        roi: &ProjectRoi,
        descriptor: DocumentDescriptor,
        label_available: Vec<String>,
        label_resource: Option<Arc<ControlLabelResource>>,
        object_resource: Option<Arc<ControlObjectResource>>,
        saved_view: Option<&Value>,
    ) -> Result<Option<u64>, ControlError> {
        if !self.project_roi_open_is_current(scope, operation_generation) {
            return Ok(None);
        }

        // This method is called on a candidate clone. Every fallible resource/view installation
        // completes before the caller swaps the candidate into the actor, so the previous usable
        // document remains authoritative if any step fails.
        self.project_roi_open_pending = false;
        let document_generation = self.begin_dataset_open(roi.source_display());
        if !self.install_document_for_generation(
            document_generation,
            descriptor,
            label_available,
            label_resource,
        ) {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "project ROI document generation changed during installation",
            ));
        }
        {
            let dataset = self.dataset_mut()?;
            dataset.masks.replace(roi.mask_layers.clone(), None);
            Self::sync_mask_native_layers(dataset);
        }
        if let Some(resource) = object_resource {
            if let Some(path) = super::project_roi_segmentation_path(&self.project_snapshot(), roi)
            {
                self.project_object_preload
                    .remember_resource(scope, path, Arc::clone(&resource));
            }
            self.install_object_resource_immediate(resource)?;
        }
        if let Some(view) = saved_view {
            self.restore_project_roi_view(view)?;
        }
        self.project.activate_roi(roi)?;
        self.project_initialized = true;
        self.readiness.finish(
            OperationKind::ProjectRoiOpen,
            operation_generation,
            "Project ROI ready",
        );
        Ok(Some(document_generation))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install_deep_link_apply_for_generation(
        &mut self,
        operation_generation: u64,
        guard: DeepLinkApplyGuard,
        project: ProjectModelSnapshot,
        roi: &ProjectRoi,
        reuse_current: bool,
        descriptor: DocumentDescriptor,
        label_available: Vec<String>,
        label_resource: Option<Arc<ControlLabelResource>>,
        object_resource: Option<Arc<ControlObjectResource>>,
        saved_view: Option<&Value>,
        request: &DeepLinkRequest,
        object_filter: Option<ControlObjectFilterResult>,
    ) -> Result<Option<(u64, Vec<String>)>, ControlError> {
        if !self.deep_link_apply_is_current(operation_generation, guard) {
            return Ok(None);
        }

        // The caller applies this to a clone. Project replacement, document/resource install,
        // saved-view restoration, and all deep-link presentation changes therefore commit as one
        // retained transaction or not at all.
        self.deep_link_apply_pending = false;
        if reuse_current {
            if self.dataset()?.descriptor.source.source_key() != descriptor.source.source_key() {
                return Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    "the current document changed during deep-link reuse",
                ));
            }
            if let Some(resource) = label_resource {
                self.install_label_resource_immediate(resource, label_available)?;
            }
            if let Some(resource) = object_resource {
                let already_installed = self
                    .dataset()?
                    .object_resource
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &resource));
                if !already_installed {
                    self.install_object_resource_immediate(resource)?;
                }
            }
            self.project.activate_roi(roi)?;
            let notes = self.apply_deep_link_to_current_dataset(request, object_filter)?;
            self.sync_current_dataset_view_to_project()?;
            self.readiness.finish(
                OperationKind::DeepLinkApply,
                operation_generation,
                "Deep link applied",
            );
            return Ok(Some((self.document_generation, notes)));
        }
        self.project.replace(project);
        self.project_initialized = true;
        let project = self.project_snapshot();
        self.project_object_preload.sync_scope(&project);
        let scope = self.project_object_preload.scope();
        let document_generation = self.begin_dataset_open(roi.source_display());
        if !self.install_document_for_generation(
            document_generation,
            descriptor,
            label_available,
            label_resource,
        ) {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "deep-link document generation changed during installation",
            ));
        }
        {
            let dataset = self.dataset_mut()?;
            dataset.masks.replace(roi.mask_layers.clone(), None);
            Self::sync_mask_native_layers(dataset);
        }
        if let Some(resource) = object_resource {
            if let Some(path) = super::project_roi_segmentation_path(&project, roi) {
                self.project_object_preload
                    .remember_resource(&scope, path, Arc::clone(&resource));
            }
            self.install_object_resource_immediate(resource)?;
        }
        if let Some(view) = saved_view {
            self.restore_project_roi_view(view)?;
        }
        self.project.activate_roi(roi)?;
        let notes = self.apply_deep_link_to_current_dataset(request, object_filter)?;
        self.sync_current_dataset_view_to_project()?;
        self.readiness.finish(
            OperationKind::DeepLinkApply,
            operation_generation,
            "Deep link applied",
        );
        Ok(Some((document_generation, notes)))
    }

    pub(super) fn install_object_resource_immediate(
        &mut self,
        resource: Arc<ControlObjectResource>,
    ) -> Result<(), ControlError> {
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.installed_object_resource_generation = self.object_resource_generation;
        self.object_resource_pending = false;
        let dataset = self.dataset_mut()?;
        dataset.object_resource = Some(resource);
        dataset.object_selection.reset();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.set_primary_objects(true);
            viewport.state.objects["visible"] = Value::Bool(true);
        }
        self.readiness.mark_ready(
            OperationKind::Objects,
            self.object_resource_generation,
            "Ready",
        );
        Ok(())
    }

    pub(crate) fn install_document_object_layers(
        &mut self,
        layers: &[DocumentObjectLayerResource],
    ) -> Result<(), ControlError> {
        let mut primary_seen = false;
        let mut spatial_ids = HashSet::new();
        for layer in layers {
            if layer.primary {
                if primary_seen || layer.layer_id != "segmentation_objects" {
                    return Err(invalid(
                        "alternate document contains invalid or duplicate primary object layers",
                    ));
                }
                primary_seen = true;
            } else {
                let id = layer
                    .layer_id
                    .strip_prefix("spatial_shape:")
                    .and_then(|value| value.parse::<u64>().ok())
                    .filter(|id| *id > 0)
                    .ok_or_else(|| invalid("alternate object layer has an invalid layer ID"))?;
                if !spatial_ids.insert(id) {
                    return Err(invalid(format!(
                        "alternate document contains duplicate spatial shape layer {id}"
                    )));
                }
            }
        }

        for layer in layers {
            if layer.primary {
                self.install_object_resource_immediate(Arc::clone(&layer.resource))?;
                continue;
            }
            let id = layer
                .layer_id
                .strip_prefix("spatial_shape:")
                .and_then(|value| value.parse::<u64>().ok())
                .expect("validated spatial object layer ID");
            self.object_resource_generation =
                self.object_resource_generation.wrapping_add(1).max(1);
            let generation = self.object_resource_generation;
            self.analyses
                .entry(ObjectTarget::SpatialShape(id))
                .or_default();
            let dataset = self.dataset_mut()?;
            dataset.secondary_object_layers.insert(
                id,
                SecondaryObjectLayerModel {
                    layer_id: id,
                    name: layer.name.clone(),
                    generation,
                    resource: Arc::clone(&layer.resource),
                    selection: ObjectSelectionModel::default(),
                },
            );
            for viewport in dataset.workspace.viewports_mut() {
                viewport
                    .state
                    .secondary_objects
                    .insert(id, SecondaryObjectViewportModel::new());
                viewport.state.native_layers.set_spatial_object_layer(
                    &layer.layer_id,
                    &layer.name,
                    true,
                );
            }
        }
        Ok(())
    }

    pub(super) fn install_label_resource_immediate(
        &mut self,
        resource: Arc<ControlLabelResource>,
        mut available: Vec<String>,
    ) -> Result<(), ControlError> {
        let name = resource.dataset.label_name.clone();
        if !available.contains(&name) {
            available.push(name.clone());
        }
        available.sort();
        available.dedup();
        let label_generation = {
            let dataset = self.dataset_mut()?;
            dataset.label_generation = dataset.label_generation.wrapping_add(1).max(1);
            dataset.label_available = available;
            dataset.label_selected = name.clone();
            dataset.label_loaded = Some(name.clone());
            dataset.label_resource = Some(resource);
            dataset.label_pending = false;
            dataset.label_actor_owned = true;
            dataset.label_status = format!("Loaded labels/{name}.");
            for viewport in dataset.workspace.viewports_mut() {
                viewport.state.segmentation_labels_visible = true;
                viewport
                    .state
                    .native_layers
                    .set_segmentation_labels(true, true);
            }
            dataset.label_generation
        };
        self.readiness
            .mark_ready(OperationKind::Labels, label_generation, "Ready");
        Ok(())
    }
}
