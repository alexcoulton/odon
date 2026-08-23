//! Primary-object, label, and saved-view resource lifecycle.

use super::*;

impl AppModel {
    pub fn begin_object_resource_load(&mut self, source: impl Into<String>) -> (u64, u64) {
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.object_resource_pending = true;
        self.readiness.begin(
            OperationKind::Objects,
            self.object_resource_generation,
            format!("Loading objects: {}", source.into()),
        );
        (self.document_generation, self.object_resource_generation)
    }

    pub fn current_object_resource_request(&self) -> Option<(PathBuf, f32)> {
        let resource = self.dataset.as_ref()?.object_resource.as_ref()?;
        Some((resource.source.clone(), resource.downsample_factor))
    }

    pub fn install_object_resource_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        resource: std::sync::Arc<ControlObjectResource>,
    ) -> bool {
        if self.mode != ModelMode::Single
            || document_generation != self.document_generation
            || resource_generation != self.object_resource_generation
            || !self.object_resource_pending
        {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        dataset.object_resource = Some(resource);
        dataset.object_selection.reset();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.set_primary_objects(true);
            set_object_filter_model(&mut viewport.state.objects, default_object_filter_model());
            viewport.state.object_filter_indices = Arc::new(Vec::new());
            viewport.state.object_filter_active = false;
            viewport.state.object_filter_revision =
                viewport.state.object_filter_revision.wrapping_add(1).max(1);
        }
        self.pending_object_filters.clear();
        self.pending_object_selection_filters.clear();
        self.tile_loading.reset_observation();
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by object resource replacement",
        );
        self.installed_object_resource_generation = resource_generation;
        self.object_resource_pending = false;
        self.readiness
            .finish(OperationKind::Objects, resource_generation, "Ready");
        true
    }

    pub fn fail_object_resource_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if document_generation != self.document_generation
            || resource_generation != self.object_resource_generation
            || !self.object_resource_pending
        {
            return false;
        }
        self.object_resource_pending = false;
        self.readiness
            .fail(OperationKind::Objects, resource_generation, message);
        true
    }

    pub fn clear_object_resource(&mut self) -> Result<Value, ControlError> {
        let dataset = self.dataset_mut()?;
        let previous = dataset.object_resource.take();
        dataset.object_selection.reset();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.set_primary_objects(false);
        }
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.installed_object_resource_generation = self.object_resource_generation;
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filters.clear();
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by clearing the object resource",
        );
        self.readiness.mark_ready(
            OperationKind::Objects,
            self.object_resource_generation,
            "Ready",
        );
        Ok(json!({
            "cleared": previous.is_some(),
            "previous_path": previous.as_ref().map(|resource| resource.source.to_string_lossy()),
            "previous_count": previous.as_ref().map_or(0, |resource| resource.features.len()),
        }))
    }

    pub fn cancel_object_resource_load(&mut self) -> Value {
        let cancelled = self.object_resource_pending;
        if cancelled {
            let cancelled_generation = self.object_resource_generation;
            self.readiness.cancel(
                OperationKind::Objects,
                cancelled_generation,
                "Object load cancelled.",
            );
            self.object_resource_generation =
                self.object_resource_generation.wrapping_add(1).max(1);
            self.object_resource_pending = false;
        }
        json!({"cancelled": cancelled, "state": self.object_resource_state()})
    }

    pub fn object_resource_state(&self) -> Value {
        let resource = self
            .dataset
            .as_ref()
            .and_then(|dataset| dataset.object_resource.as_ref());
        json!({
            "source": resource.map(|resource| resource.source.to_string_lossy()),
            "loading": self.object_resource_pending,
            "status": self.readiness.status_for(OperationKind::Objects).unwrap_or("Ready"),
            "object_count": resource.map_or(0, |resource| resource.features.len()),
            "generation": self.installed_object_resource_generation,
            "request_generation": self.object_resource_generation,
            "available_properties": resource
                .map(|resource| resource.property_names.as_ref().clone())
                .unwrap_or_default(),
        })
    }

    pub fn object_resource(&self) -> Option<std::sync::Arc<ControlObjectResource>> {
        self.dataset
            .as_ref()
            .and_then(|dataset| dataset.object_resource.clone())
    }

    pub fn label_resource(&self) -> Option<Arc<ControlLabelResource>> {
        self.dataset
            .as_ref()
            .and_then(|dataset| dataset.label_resource.clone())
    }

    pub fn labels_require_load(&self, params: &Value) -> bool {
        params.get("visible").and_then(Value::as_bool) == Some(true)
            && self
                .dataset
                .as_ref()
                .is_some_and(|dataset| dataset.label_loaded.is_none())
    }

    pub fn begin_label_load(&mut self, params: &Value) -> Result<(u64, u64, String), ControlError> {
        let document_generation = self.document_generation;
        let dataset = self.dataset_mut()?;
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(dataset.label_selected.as_str())
            .to_string();
        if name.is_empty() {
            return Err(invalid(
                "label name is required because this dataset has no default label group",
            ));
        }
        dataset.label_generation = dataset.label_generation.wrapping_add(1).max(1);
        dataset.label_pending = true;
        dataset.label_status = format!("Loading labels/{name}...");
        let status = dataset.label_status.clone();
        let label_generation = dataset.label_generation;
        self.readiness
            .begin(OperationKind::Labels, label_generation, status);
        Ok((document_generation, label_generation, name))
    }

    pub fn fail_label_load_for_generation(
        &mut self,
        document_generation: u64,
        label_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if document_generation != self.document_generation {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        if dataset.label_generation != label_generation {
            return false;
        }
        dataset.label_pending = false;
        dataset.label_status = message.into();
        self.readiness.fail(
            OperationKind::Labels,
            label_generation,
            dataset.label_status.clone(),
        );
        true
    }

    pub fn install_label_resource_for_generation(
        &mut self,
        document_generation: u64,
        label_generation: u64,
        resource: Arc<ControlLabelResource>,
    ) -> bool {
        if document_generation != self.document_generation {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        if dataset.label_generation != label_generation {
            return false;
        }
        let name = resource.dataset.label_name.clone();
        if !dataset.label_available.contains(&name) {
            dataset.label_available.push(name.clone());
            dataset.label_available.sort();
            dataset.label_available.dedup();
        }
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
        self.readiness
            .finish(OperationKind::Labels, label_generation, "Ready");
        true
    }

    pub(super) fn labels_snapshot(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let viewport = &dataset.workspace.active().state;
        let offset = viewport
            .native_layers
            .get("segmentation_labels")
            .map(|layer| layer.offset_world)
            .unwrap_or([0.0, 0.0]);
        Ok(json!({
            "available":dataset.label_available,
            "selected":dataset.label_selected,
            "loaded":dataset.label_loaded,
            "visible":viewport.segmentation_labels_visible,
            "busy":dataset.label_pending,
            "gpu_available":self.renderer_gpu_available,
            "status":dataset.label_status,
            "offset_world":offset,
            "generation":dataset.label_generation,
            "actor_owned":dataset.label_actor_owned,
        }))
    }

    pub(super) fn unload_labels(&mut self) -> Result<Value, ControlError> {
        let (unloaded, label_generation) = {
            let dataset = self.dataset_mut()?;
            let unloaded = dataset.label_loaded.take();
            dataset.label_resource = None;
            dataset.label_generation = dataset.label_generation.wrapping_add(1).max(1);
            dataset.label_pending = false;
            dataset.label_actor_owned = true;
            dataset.label_status = "Unloaded segmentation labels.".to_string();
            for viewport in dataset.workspace.viewports_mut() {
                viewport.state.segmentation_labels_visible = false;
                viewport
                    .state
                    .native_layers
                    .set_segmentation_labels(false, false);
            }
            (unloaded, dataset.label_generation)
        };
        self.readiness.mark_ready(
            OperationKind::Labels,
            label_generation,
            "Unloaded segmentation labels.",
        );
        Ok(json!({"unloaded":unloaded,"labels":self.labels_snapshot()?}))
    }

    pub(super) fn set_labels_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible must be a boolean"))?;
        if visible && self.dataset()?.label_loaded.is_none() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "the selected label resource must be loaded before it can be shown",
            ));
        }
        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let viewport = &mut dataset.workspace.active_mut().state;
        let mut changed = viewport.segmentation_labels_visible != visible;
        viewport.segmentation_labels_visible = visible;
        if viewport.native_layers.get("segmentation_labels").is_some() {
            changed |= viewport
                .native_layers
                .set_visibility("segmentation_labels", visible)?;
        }
        if changed {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        self.labels_snapshot()
    }

    pub(crate) fn prepare_project_view_apply_resources(
        &mut self,
        params: &Value,
    ) -> Result<Option<ProjectViewApplySpec>, ControlError> {
        let view = self.project.dispatch("project.views.get", params)?;
        let spec = view.get("spec").unwrap_or(&Value::Null);
        let dataset = self.dataset()?;
        let needs_objects = spec
            .get("segmentation_source")
            .and_then(Value::as_str)
            .is_some_and(|source| !source.trim().is_empty())
            && spec.get("load_labels").and_then(Value::as_bool) != Some(true);
        let needs_labels = spec.get("load_labels").and_then(Value::as_bool) == Some(true);
        let load_objects = needs_objects && dataset.object_resource.is_none();
        let load_labels = needs_labels && dataset.label_resource.is_none();
        if !load_objects && !load_labels {
            return Ok(None);
        }

        let project = self.project_snapshot();
        let current_source_key = dataset.descriptor.source.source_key();
        let active_roi = project
            .focused_source_key
            .as_deref()
            .and_then(|focused| {
                project
                    .rois
                    .iter()
                    .find(|roi| roi.source_key().as_deref() == Some(focused))
            })
            .or_else(|| {
                project
                    .rois
                    .iter()
                    .find(|roi| roi.source_key().as_deref() == Some(current_source_key.as_str()))
            });
        let object_path = if load_objects {
            let roi = active_roi.ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::ResourceNotFound,
                    "saved view requests object segmentation but the current dataset has no project ROI",
                )
            })?;
            Some(project_roi_segmentation_path(&project, roi).ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::ResourceNotFound,
                    "saved view requests object segmentation but the current ROI has no segmentation source",
                )
            })?)
        } else {
            None
        };
        let label_name = if load_labels {
            let selected = dataset.label_selected.trim();
            let name = if selected.is_empty() {
                dataset.label_available.first().map(String::as_str)
            } else {
                Some(selected)
            }
            .ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::ResourceNotFound,
                    "saved view requests labels but this dataset has no label groups",
                )
            })?;
            Some(name.to_string())
        } else {
            None
        };

        if self.project_view_apply_pending {
            self.readiness.cancel(
                OperationKind::ProjectViewApply,
                self.project_view_apply_generation,
                "Superseded by a newer saved-view application",
            );
        }
        self.project_view_apply_generation =
            self.project_view_apply_generation.wrapping_add(1).max(1);
        self.project_view_apply_pending = true;
        self.readiness.begin(
            OperationKind::ProjectViewApply,
            self.project_view_apply_generation,
            "Loading saved-view resources",
        );
        Ok(Some(ProjectViewApplySpec {
            operation_generation: self.project_view_apply_generation,
            document_generation: self.document_generation,
            project_config_generation: project.config_generation,
            params: params.clone(),
            object_path,
            label_name,
        }))
    }

    pub(super) fn project_view_apply_is_current(&self, spec: &ProjectViewApplySpec) -> bool {
        self.project_view_apply_pending
            && self.project_view_apply_generation == spec.operation_generation
            && self.document_generation == spec.document_generation
            && self.project_snapshot().config_generation == spec.project_config_generation
            && self
                .readiness
                .is_pending(OperationKind::ProjectViewApply, spec.operation_generation)
    }

    pub(crate) fn install_project_view_apply_resources(
        &mut self,
        spec: &ProjectViewApplySpec,
        object_resource: Option<Arc<ControlObjectResource>>,
        label_resource: Option<Arc<ControlLabelResource>>,
    ) -> Result<Option<Value>, ControlError> {
        if !self.project_view_apply_is_current(spec) {
            return Ok(None);
        }
        if let Some(resource) = object_resource {
            self.install_object_resource_immediate(resource)?;
        }
        if let Some(resource) = label_resource {
            let available = self.dataset()?.label_available.clone();
            self.install_label_resource_immediate(resource, available)?;
        }
        let response = self.apply_project_view(&spec.params)?;
        self.project_view_apply_pending = false;
        self.readiness.finish(
            OperationKind::ProjectViewApply,
            spec.operation_generation,
            "Saved view applied",
        );
        Ok(Some(response))
    }

    pub(crate) fn fail_project_view_apply(
        &mut self,
        spec: &ProjectViewApplySpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.project_view_apply_is_current(spec) {
            return false;
        }
        self.project_view_apply_pending = false;
        self.readiness.fail(
            OperationKind::ProjectViewApply,
            spec.operation_generation,
            message,
        )
    }

    pub(crate) fn cancel_project_view_apply(
        &mut self,
        spec: &ProjectViewApplySpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.project_view_apply_is_current(spec) {
            return false;
        }
        self.project_view_apply_pending = false;
        self.readiness.cancel(
            OperationKind::ProjectViewApply,
            spec.operation_generation,
            message,
        )
    }
}
