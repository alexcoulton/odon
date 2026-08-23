//! Model mode, document transitions, readiness, and renderer observations.

use super::*;

impl AppModel {
    pub fn render_workspace_snapshot(&self) -> Option<Value> {
        self.workspace_snapshot().ok()
    }

    pub fn set_mode(&mut self, mode: ModelMode) {
        self.cancel_pending_deep_link_apply("Superseded by application mode change");
        self.project_roi_open_pending = false;
        self.mode = mode;
        if mode != ModelMode::Single {
            self.dataset = None;
        }
        self.readiness
            .cancel_all_pending("Superseded by application mode change");
    }

    pub fn bootstrap_mode_from_renderer(&mut self, mode: ModelMode) {
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        self.set_mode(mode);
    }

    pub fn mark_projection_dirty(&mut self) -> u64 {
        self.projection_revision = self.projection_revision.wrapping_add(1).max(1);
        self.projection_revision
    }

    pub fn mark_projection_presented(&mut self, revision: u64) {
        self.presented_projection_revision = self.presented_projection_revision.max(revision);
    }

    pub fn report_viewport_geometry(
        &mut self,
        viewport_id: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) {
        if !x.is_finite()
            || !y.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || width <= 0.0
            || height <= 0.0
        {
            return;
        }
        let Ok(id) = ViewportId::new(viewport_id) else {
            return;
        };
        if let Some(dataset) = self.dataset.as_mut()
            && let Some(viewport) = dataset.workspace.get_mut(&id)
        {
            viewport.state.screen_origin = [x, y];
            viewport.state.logical_size = [width, height];
            self.measured_viewports.insert(id);
            if dataset
                .workspace
                .viewports()
                .iter()
                .all(|viewport| self.measured_viewports.contains(&viewport.id))
            {
                dataset.logical_workspace_size = observed_workspace_size(&dataset.workspace);
                dataset.geometry_source = GeometrySource::Observed;
            }
        }
    }

    pub fn begin_dataset_open(&mut self, source: impl Into<String>) -> u64 {
        self.cancel_pending_deep_link_apply("Superseded by newer document request");
        if self.project_roi_open_pending {
            let generation = self.project_roi_open_generation;
            self.project_roi_open_pending = false;
            self.readiness.cancel(
                OperationKind::ProjectRoiOpen,
                generation,
                "Superseded by newer document request",
            );
        }
        for kind in [
            OperationKind::Document,
            OperationKind::Labels,
            OperationKind::Objects,
            OperationKind::ObjectFilter,
            OperationKind::MaskIo,
            OperationKind::MemoryPin,
            OperationKind::ThresholdPreview,
            OperationKind::Analysis,
        ] {
            self.readiness
                .cancel_kind_pending(kind, "Superseded by newer document request");
        }
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filters.clear();
        if let Some(dataset) = self.dataset.as_mut() {
            dataset.label_pending = false;
        }
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        self.mode = ModelMode::Transition;
        self.readiness.begin(
            OperationKind::Document,
            self.document_generation,
            format!("Opening {}", source.into()),
        );
        self.document_generation
    }

    pub fn begin_dataset_inspection(&mut self, path: &Path) -> (u64, String) {
        self.dataset_inspection_operation_generation = self
            .dataset_inspection_operation_generation
            .wrapping_add(1)
            .max(1);
        let generation = self.dataset_inspection_operation_generation;
        let scope = path.to_string_lossy().into_owned();
        self.readiness.begin_scoped(
            OperationKind::DatasetInspection,
            scope.clone(),
            generation,
            format!("Inspecting {scope}"),
        );
        (generation, scope)
    }

    pub fn finish_dataset_inspection(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .finish_scoped(OperationKind::DatasetInspection, scope, generation, status)
    }

    pub fn begin_remote_listing(&mut self, scope: String) -> u64 {
        self.remote_listing_operation_generation = self
            .remote_listing_operation_generation
            .wrapping_add(1)
            .max(1);
        let generation = self.remote_listing_operation_generation;
        self.readiness.begin_scoped(
            OperationKind::RemoteListing,
            scope,
            generation,
            "Listing remote S3 prefix",
        );
        generation
    }

    pub fn finish_remote_listing(&mut self, scope: &str, generation: u64) -> bool {
        self.readiness.finish_scoped(
            OperationKind::RemoteListing,
            scope,
            generation,
            "Remote S3 listing complete",
        )
    }

    pub fn fail_remote_listing(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .fail_scoped(OperationKind::RemoteListing, scope, generation, status)
    }

    pub fn cancel_remote_listing(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::RemoteListing, scope, generation, status)
    }

    pub fn cancel_pending_remote_listings(&mut self, status: impl Into<String>) {
        let status = status.into();
        self.readiness
            .cancel_kind_pending(OperationKind::RemoteListing, &status);
    }

    pub fn cancel_dataset_inspection(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::DatasetInspection, scope, generation, status)
    }

    pub fn begin_deep_link_resolution(&mut self, scope: String) -> u64 {
        self.deep_link_resolve_operation_generation = self
            .deep_link_resolve_operation_generation
            .wrapping_add(1)
            .max(1);
        let generation = self.deep_link_resolve_operation_generation;
        self.readiness.begin_scoped(
            OperationKind::DeepLinkResolve,
            scope,
            generation,
            "Resolving deep link",
        );
        generation
    }

    pub fn finish_deep_link_resolution(&mut self, scope: &str, generation: u64) -> bool {
        self.readiness.finish_scoped(
            OperationKind::DeepLinkResolve,
            scope,
            generation,
            "Deep link resolved",
        )
    }

    pub fn cancel_deep_link_resolution(
        &mut self,
        scope: &str,
        generation: u64,
        status: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::DeepLinkResolve, scope, generation, status)
    }

    pub(crate) fn begin_deep_link_apply(
        &mut self,
        description: impl Into<String>,
    ) -> (u64, DeepLinkApplyGuard) {
        if self.deep_link_apply_pending {
            self.readiness.cancel(
                OperationKind::DeepLinkApply,
                self.deep_link_apply_operation_generation,
                "Superseded by a newer deep link",
            );
        }
        self.deep_link_apply_operation_generation = self
            .deep_link_apply_operation_generation
            .wrapping_add(1)
            .max(1);
        self.deep_link_apply_pending = true;
        self.readiness.begin(
            OperationKind::DeepLinkApply,
            self.deep_link_apply_operation_generation,
            description,
        );
        let project = self.project_snapshot();
        (
            self.deep_link_apply_operation_generation,
            DeepLinkApplyGuard {
                project_load_generation: project.load_generation,
                project_config_generation: project.config_generation,
                document_generation: self.document_generation,
            },
        )
    }

    pub(crate) fn deep_link_apply_is_current(
        &self,
        generation: u64,
        guard: DeepLinkApplyGuard,
    ) -> bool {
        let project = self.project_snapshot();
        self.deep_link_apply_pending
            && self.deep_link_apply_operation_generation == generation
            && project.load_generation == guard.project_load_generation
            && project.config_generation == guard.project_config_generation
            && self.document_generation == guard.document_generation
            && self
                .readiness
                .is_pending(OperationKind::DeepLinkApply, generation)
    }

    pub(crate) fn fail_deep_link_apply(
        &mut self,
        generation: u64,
        guard: DeepLinkApplyGuard,
        message: impl Into<String>,
    ) -> bool {
        if !self.deep_link_apply_is_current(generation, guard) {
            return false;
        }
        self.deep_link_apply_pending = false;
        self.readiness
            .fail(OperationKind::DeepLinkApply, generation, message);
        true
    }

    pub(crate) fn cancel_deep_link_apply(
        &mut self,
        generation: u64,
        guard: DeepLinkApplyGuard,
        message: impl Into<String>,
    ) -> bool {
        if !self.deep_link_apply_is_current(generation, guard) {
            return false;
        }
        self.deep_link_apply_pending = false;
        self.readiness
            .cancel(OperationKind::DeepLinkApply, generation, message);
        true
    }

    pub(crate) fn supersede_deep_link_apply(&mut self, generation: u64, message: &str) {
        if self.deep_link_apply_pending && self.deep_link_apply_operation_generation == generation {
            self.deep_link_apply_pending = false;
            self.readiness
                .cancel(OperationKind::DeepLinkApply, generation, message);
        }
    }

    pub(crate) fn cancel_pending_deep_link_apply(&mut self, message: &str) {
        if self.deep_link_apply_pending {
            let generation = self.deep_link_apply_operation_generation;
            self.deep_link_apply_pending = false;
            self.readiness
                .cancel(OperationKind::DeepLinkApply, generation, message);
        }
    }

    pub fn fail_dataset_open(&mut self, message: impl Into<String>) {
        self.mode = ModelMode::Project;
        self.dataset = None;
        self.readiness
            .fail(OperationKind::Document, self.document_generation, message);
    }

    pub fn fail_dataset_open_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if generation != self.document_generation {
            return false;
        }
        self.fail_dataset_open(message);
        true
    }

    pub fn install_dataset(&mut self, dataset: &OmeZarrDataset) {
        let available = dataset
            .is_root_label_mask()
            .then(|| vec![LabelZarrDataset::root_label_name(dataset)])
            .unwrap_or_default();
        self.install_dataset_with_labels(dataset, available, None);
    }

    pub fn install_dataset_with_labels(
        &mut self,
        dataset: &OmeZarrDataset,
        mut label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) {
        if dataset.is_root_label_mask() {
            let root = LabelZarrDataset::root_label_name(dataset);
            if !label_available.contains(&root) {
                label_available.push(root);
            }
        }
        self.install_document_descriptor_with_labels(
            DocumentDescriptor::from_ome_zarr(dataset),
            label_available,
            root_label_resource,
        );
    }

    pub fn install_document_descriptor_with_labels(
        &mut self,
        descriptor: DocumentDescriptor,
        mut label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) {
        let channel_count = descriptor.channels.len();
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        self.installed_object_resource_generation = self.object_resource_generation;
        self.object_resource_pending = false;
        self.pending_object_filters.clear();
        self.pending_object_selection_filters.clear();
        self.readiness.cancel_kind_pending(
            OperationKind::ObjectFilter,
            "Object filters superseded by dataset replacement",
        );
        let retained_geometry = self
            .dataset
            .as_ref()
            .map(|dataset| (dataset.logical_workspace_size, dataset.geometry_source));
        let level = descriptor.levels.first();
        let world_size = level
            .and_then(|level| {
                Some([
                    *level.shape.get(descriptor.dims.x)? as f32,
                    *level.shape.get(descriptor.dims.y)? as f32,
                ])
            })
            .unwrap_or([1.0, 1.0]);
        let plane_extents = level
            .map(|level| {
                [
                    descriptor
                        .dims
                        .z
                        .and_then(|dimension| level.shape.get(dimension).copied())
                        .unwrap_or(1),
                    level.shape.get(descriptor.dims.y).copied().unwrap_or(1),
                    level.shape.get(descriptor.dims.x).copied().unwrap_or(1),
                ]
            })
            .unwrap_or([1, 1, 1]);
        let mut viewport = ViewportModel::new(&descriptor.channels);
        let (logical_workspace_size, geometry_source) =
            retained_geometry.unwrap_or((DEFAULT_LOGICAL_CANVAS, GeometrySource::Bootstrap));
        viewport.logical_size = logical_workspace_size;
        fit_camera(&mut viewport, world_size);
        label_available.sort();
        label_available.dedup();
        let label_selected = if label_available.iter().any(|name| name == "cells") {
            "cells".to_string()
        } else {
            label_available.first().cloned().unwrap_or_default()
        };
        let loaded_root = root_label_resource
            .as_ref()
            .map(|resource| resource.dataset.label_name.clone());
        if root_label_resource.is_some() {
            viewport.native_layers.set_segmentation_labels(true, true);
            viewport.segmentation_labels_visible = true;
        }
        let source_key = descriptor.source.source_key();
        self.dataset = Some(DatasetModel {
            orthogonal_planes: descriptor.dims.z.is_some(),
            descriptor,
            world_size,
            plane_extents,
            logical_workspace_size,
            geometry_source,
            show_left_panel: true,
            show_right_panel: true,
            left_tab: "layers".to_string(),
            right_tab: "properties".to_string(),
            shared_resources: default_shared_resources(source_key),
            performance: default_performance_snapshot(),
            object_resource: None,
            label_available,
            label_selected: loaded_root.clone().unwrap_or(label_selected),
            label_loaded: loaded_root.clone(),
            label_resource: root_label_resource,
            label_status: loaded_root
                .map(|name| format!("Opened top-level label mask '{name}'."))
                .unwrap_or_default(),
            label_generation: 1,
            label_pending: false,
            label_actor_owned: true,
            object_selection: ObjectSelectionModel::default(),
            secondary_object_layers: HashMap::new(),
            masks: MaskModel::default(),
            workspace: ViewportWorkspace::new(viewport),
        });
        self.pinned_memory.clear_for_document(channel_count);
        let default_threshold_level = self
            .dataset
            .as_ref()
            .and_then(|dataset| {
                dataset.descriptor.levels.iter().find_map(|level| {
                    level
                        .shape
                        .get(dataset.descriptor.dims.x)
                        .copied()
                        .zip(level.shape.get(dataset.descriptor.dims.y).copied())
                        .and_then(|(width, height)| width.checked_mul(height))
                        .filter(|pixels| *pixels <= THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS)
                        .map(|_| level.index)
                })
            })
            .unwrap_or_else(|| {
                self.dataset.as_ref().map_or(0, |dataset| {
                    dataset.descriptor.levels.len().saturating_sub(1)
                })
            });
        self.threshold_preview.reset(default_threshold_level);
        self.analyses.clear();
        self.analyses
            .insert(ObjectTarget::Primary, AnalysisModel::default());
        self.measurement.reset();
        self.object_export.reset();
        self.measured_viewports
            .retain(|id| id.as_str() == "viewport-1");
        self.mode = ModelMode::Single;
        self.readiness
            .mark_ready(OperationKind::Document, self.document_generation, "Ready");
    }

    pub fn install_dataset_for_generation(
        &mut self,
        generation: u64,
        dataset: &OmeZarrDataset,
        label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) -> bool {
        let mut label_available = label_available;
        if dataset.is_root_label_mask() {
            let root = LabelZarrDataset::root_label_name(dataset);
            if !label_available.contains(&root) {
                label_available.push(root);
            }
        }
        self.install_document_for_generation(
            generation,
            DocumentDescriptor::from_ome_zarr(dataset),
            label_available,
            root_label_resource,
        )
    }

    pub fn install_document_for_generation(
        &mut self,
        generation: u64,
        descriptor: DocumentDescriptor,
        label_available: Vec<String>,
        root_label_resource: Option<Arc<ControlLabelResource>>,
    ) -> bool {
        if generation != self.document_generation {
            return false;
        }
        self.install_document_descriptor_with_labels(
            descriptor,
            label_available,
            root_label_resource,
        );
        true
    }

    pub fn set_renderer_gpu_available(&mut self, available: bool) {
        self.renderer_gpu_available = available;
    }

    pub fn bootstrap_dataset(&mut self, dataset: &OmeZarrDataset) -> Result<(), ControlError> {
        self.document_generation = self.document_generation.wrapping_add(1).max(1);
        let source_key = dataset.source.source_key();
        self.install_dataset(dataset);
        if let Some(masks) = self
            .project
            .mask_layers_for_source_key(&source_key)
            .map(|layers| layers.to_vec())
        {
            let dataset = self.dataset_mut()?;
            dataset.masks.replace(masks, None);
            Self::sync_mask_native_layers(dataset);
        }
        if let Some(view) = self.project.roi_view_state_json(&source_key).cloned() {
            self.restore_project_roi_view(&view)?;
        }
        Ok(())
    }

    /// Merge renderer-owned compatibility data into the canonical model.
    ///
    /// The renderer is a projection consumer, not an authority for fields handled by
    /// [`Self::dispatch`]. In particular, channel metadata, transforms, presentation, panels,
    /// cameras, planes, and workspace structure must come back as typed native commands. A
    /// delayed renderer snapshot may be based on an older projection revision, so copying those
    /// fields here would allow a frame rendered after an occlusion to undo newer Python work.
    pub fn observe_renderer_state(
        &mut self,
        observation: &Value,
        based_on_projection_revision: u64,
    ) -> bool {
        if based_on_projection_revision > self.projection_revision {
            return false;
        }
        let Some(dataset) = self.dataset.as_mut() else {
            return false;
        };
        let observed_source = observation
            .get("shared_resources")
            .and_then(|resources| resources.get("dataset_source"))
            .and_then(Value::as_str);
        let canonical_source = dataset
            .shared_resources
            .get("dataset_source")
            .and_then(Value::as_str);
        if observed_source.is_some()
            && canonical_source.is_some()
            && observed_source != canonical_source
        {
            return false;
        }

        // These fields are still produced by renderer-side compatibility domains. They do not
        // overlap any actor-owned mutation and are therefore safe to merge even when the
        // observation is based on an older projection revision.
        if let Some(shared_resources) = observation.get("shared_resources") {
            dataset.shared_resources = shared_resources.clone();
        }
        if let Some(performance) = observation.get("performance") {
            dataset.performance = performance.clone();
        }
        if let Some(tile_loading) = observation.get("tile_loading_observation") {
            self.tile_loading.observe(tile_loading);
        }
        if let Some(projected) = observation
            .get("native_layer_observations")
            .and_then(Value::as_array)
        {
            for value in projected {
                let Some(id) = value
                    .get("viewport_id")
                    .and_then(Value::as_str)
                    .and_then(|id| ViewportId::new(id).ok())
                else {
                    continue;
                };
                let Some(native_layers) = value.get("native_layers") else {
                    continue;
                };
                if let Some(viewport) = dataset.workspace.get_mut(&id) {
                    // Compatibility renderers may discover a resource that has not yet moved to
                    // the actor. Adopt only missing descriptors: a delayed frame must never
                    // overwrite presentation already committed by Python or native commands.
                    let _ = viewport.state.native_layers.merge_missing(native_layers);
                }
            }
        }
        true
    }

    pub(super) fn restore_workspace_snapshot(
        &mut self,
        snapshot: &Value,
    ) -> Result<(), ControlError> {
        if let Some(left_tab) = snapshot
            .get("ui")
            .and_then(|ui| ui.get("left_tab"))
            .and_then(Value::as_str)
        {
            self.dataset_mut()?.left_tab = left_tab.to_string();
        }
        if let Some(right_tab) = snapshot
            .get("ui")
            .and_then(|ui| ui.get("right_tab"))
            .and_then(Value::as_str)
        {
            self.dataset_mut()?.right_tab = right_tab.to_string();
        }
        if let Some(labels) = snapshot.get("labels") {
            let available = labels
                .get("available")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>();
            let dataset = self.dataset_mut()?;
            dataset.label_available = available;
            dataset.label_selected = labels
                .get("selected")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            dataset.label_loaded = labels
                .get("loaded")
                .and_then(Value::as_str)
                .map(str::to_string);
            dataset.label_status = labels
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            dataset.label_pending = labels.get("busy").and_then(Value::as_bool).unwrap_or(false);
            dataset.label_generation = labels
                .get("generation")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            dataset.label_actor_owned = labels
                .get("actor_owned")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(gpu_available) = labels.get("gpu_available").and_then(Value::as_bool) {
                self.renderer_gpu_available = gpu_available;
            }
        }
        if let Some(masks) = snapshot.get("masks") {
            self.dataset_mut()?.masks.restore_projection(masks)?;
        }
        if let Some(selection) = snapshot.get("object_selection") {
            let object_count = self
                .dataset()?
                .object_resource
                .as_ref()
                .map_or(0, |resource| resource.features.len());
            self.dataset_mut()?
                .object_selection
                .restore_projection(selection, object_count)?;
        }
        let projected = snapshot
            .get("viewports")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("workspace snapshot has no viewport array"))?;
        if projected.is_empty() || projected.len() > crate::viewports::MAX_VIEWPORTS {
            return Err(invalid("workspace snapshot has an invalid viewport count"));
        }
        let default_channels = self.dataset()?.workspace.active().state.channels.clone();
        let mut measured = HashSet::new();
        let mut slots = Vec::with_capacity(projected.len());
        for value in projected {
            let id = value
                .get("viewport_id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("workspace viewport has no ID"))
                .and_then(|id| ViewportId::new(id).map_err(|error| invalid(error.to_string())))?;
            let title = value
                .get("title")
                .and_then(Value::as_str)
                .filter(|title| !title.trim().is_empty())
                .ok_or_else(|| invalid(format!("workspace viewport '{id}' has no title")))?
                .to_string();
            let mut state = ViewportModel {
                center: [0.0, 0.0],
                zoom: 1.0,
                logical_size: DEFAULT_LOGICAL_CANVAS,
                screen_origin: [0.0, 0.0],
                plane_mode: "xy".to_string(),
                plane_slices: [0, 0, 0],
                channels: default_channels.clone(),
                active_channel: 0,
                channel_order: default_channels
                    .iter()
                    .map(|channel| channel.index)
                    .collect(),
                channel_sort: "manual".to_string(),
                channel_search: String::new(),
                channel_groups: ProjectLayerGroups::default(),
                smooth_pixels: true,
                show_scale_bar: true,
                show_hud: true,
                show_tile_debug: false,
                objects: default_object_snapshot(),
                segmentation_labels_visible: true,
                segmentation_geojson_visible: false,
                object_filter_indices: Arc::new(Vec::new()),
                object_filter_active: false,
                object_filter_revision: 1,
                secondary_objects: HashMap::new(),
                native_layers: NativeLayersModel::channels(
                    &default_channels
                        .iter()
                        .map(|channel| ChannelInfo {
                            index: channel.index,
                            name: channel.name.clone(),
                            visible: channel.visible,
                            color_rgb: channel.color_rgb,
                            window: channel.window,
                            note: channel.note.clone(),
                        })
                        .collect::<Vec<_>>(),
                ),
            };
            apply_workspace_viewport(&mut state, value)?;
            if renderer_viewport_size(value).is_some() {
                measured.insert(id.clone());
            }
            slots.push(crate::viewports::ViewportSlot {
                id,
                title,
                state,
                navigation_revision: value
                    .get("navigation_revision")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
                presentation_revision: value
                    .get("presentation_revision")
                    .and_then(Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
            });
        }
        let active = snapshot
            .get("active_viewport_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("workspace snapshot has no active viewport ID"))
            .and_then(|id| ViewportId::new(id).map_err(|error| invalid(error.to_string())))?;
        let layout = snapshot
            .get("layout")
            .and_then(Value::as_str)
            .and_then(ViewportLayout::parse)
            .ok_or_else(|| invalid("workspace snapshot has an invalid layout"))?;
        let links = snapshot
            .get("links")
            .map(|links| ViewportLinks {
                camera: links.get("camera").and_then(Value::as_bool).unwrap_or(true),
                plane: links.get("plane").and_then(Value::as_bool).unwrap_or(true),
                selection: links
                    .get("selection")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            })
            .unwrap_or_default();
        let ratio = snapshot.get("ratio").and_then(Value::as_f64).unwrap_or(0.5) as f32;
        let revision = snapshot
            .get("revision")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        let workspace =
            ViewportWorkspace::restore_projection(slots, active, layout, links, ratio, revision)
                .map_err(|error| invalid(error.to_string()))?;
        let mut workspace = workspace;
        apply_workspace_channel_metadata(&mut workspace, snapshot);
        apply_workspace_channel_transforms(&mut workspace, snapshot);
        if let Some(presentation) = snapshot.get("channel_presentation") {
            let active = workspace.active_mut();
            if let Some(search) = presentation.get("search").and_then(Value::as_str) {
                active.state.channel_search = search.to_string();
            }
            if let Some(sort) = presentation
                .get("sort")
                .and_then(Value::as_str)
                .and_then(canonical_channel_sort)
            {
                active.state.channel_sort = sort.to_string();
            }
        }
        if let Some(panels) = snapshot.get("panels") {
            let dataset = self.dataset_mut()?;
            if let Some(left) = panels.get("left").and_then(Value::as_bool) {
                dataset.show_left_panel = left;
            }
            if let Some(right) = panels.get("right").and_then(Value::as_bool) {
                dataset.show_right_panel = right;
            }
        }
        let all_measured = workspace
            .viewports()
            .iter()
            .all(|viewport| measured.contains(&viewport.id));
        let dataset = self.dataset_mut()?;
        dataset.workspace = workspace;
        if let Some(shared_resources) = snapshot.get("shared_resources") {
            dataset.shared_resources = shared_resources.clone();
        }
        if let Some(performance) = snapshot.get("performance") {
            dataset.performance = performance.clone();
        }
        if all_measured {
            dataset.logical_workspace_size = observed_workspace_size(&dataset.workspace);
            dataset.geometry_source = GeometrySource::Observed;
        } else {
            update_logical_geometry(dataset);
        }
        self.measured_viewports = measured;
        Ok(())
    }

    pub(super) fn restore_project_roi_view(&mut self, view: &Value) -> Result<(), ControlError> {
        let channel_count = self.dataset()?.workspace.active().state.channels.len();
        let base = self.workspace_snapshot()?;
        let snapshot = project_roi_view_workspace_snapshot(view, channel_count, &base)?;
        self.restore_workspace_snapshot(&snapshot)?;
        let dataset = self.dataset_mut()?;
        let has_objects = dataset.object_resource.is_some();
        let has_labels = dataset.label_resource.is_some();
        for viewport in dataset.workspace.viewports_mut() {
            viewport
                .state
                .native_layers
                .set_primary_objects(has_objects);
            viewport
                .state
                .native_layers
                .set_segmentation_labels(has_labels, has_labels);
        }
        Self::sync_mask_native_layers(dataset);
        let annotations = view
            .get("annotation_layers")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| invalid(format!("invalid project annotation layers: {error}")))?
            .unwrap_or_default();
        self.restore_annotation_states(annotations)?;
        Ok(())
    }

    pub fn loading_state(&self) -> Value {
        let canvas_ready = self.dataset.as_ref().is_some_and(|dataset| {
            !dataset.workspace.viewports().is_empty()
                && dataset
                    .workspace
                    .viewports()
                    .iter()
                    .all(|viewport| self.measured_viewports.contains(&viewport.id))
        });
        let geometry = self.dataset.as_ref().map(|dataset| {
            json!({
                "source": dataset.geometry_source.as_str(),
                "confidence": match dataset.geometry_source {
                    GeometrySource::Observed => "exact",
                    GeometrySource::Derived => "stable_estimate",
                    GeometrySource::Bootstrap => "bootstrap",
                },
                "workspace_size_points": dataset.logical_workspace_size,
            })
        });
        let resources_busy = self.readiness.any_pending();
        let loading = json!({
            "busy": resources_busy,
            "status": self.readiness.aggregate_status(),
            "model_ready": self.mode != ModelMode::Transition,
            "resources_ready": !resources_busy,
            "geometry_ready": self.dataset.is_some(),
            "presentation_ready": self.presented_projection_revision >= self.projection_revision,
            "canvas_ready": canvas_ready,
            "geometry": geometry,
            "projection_revision": self.projection_revision,
            "presented_projection_revision": self.presented_projection_revision,
            "operations": self.readiness.snapshot(),
        });
        let mut response = json!({
            "mode": self.mode.as_str(),
            "pending_deep_link": false,
            "loading": loading,
        });
        let object = response
            .as_object_mut()
            .expect("loading state response is an object");
        match self.mode {
            ModelMode::Project => {
                object.insert("busy".to_string(), Value::Bool(false));
                object.insert(
                    "note".to_string(),
                    Value::String("No dataset viewer is currently open.".to_string()),
                );
            }
            ModelMode::Transition => {
                object.insert("busy".to_string(), Value::Bool(true));
                object.insert("reasons".to_string(), json!(["transition"]));
            }
            ModelMode::Single | ModelMode::Mosaic => {}
        }
        response
    }

    pub(super) fn application_state(&self) -> Value {
        let view = (self.mode == ModelMode::Single)
            .then(|| self.dataset.as_ref())
            .flatten()
            .map(|dataset| {
                let viewport = dataset.workspace.active();
                let active_channel = viewport
                    .state
                    .channels
                    .get(viewport.state.active_channel)
                    .map(|channel| json!({"index":channel.index,"name":channel.name}));
                let level0 = dataset.descriptor.levels.first();
                json!({
                    "dataset": dataset.descriptor.source.source_key(),
                    "dataset_descriptor": {
                        "source": dataset.descriptor.source.source_key(),
                        "kind": dataset.descriptor.kind,
                        "axes": dataset.descriptor.axes.iter().map(|axis| json!({
                            "name":axis.name,
                            "unit":axis.unit,
                        })).collect::<Vec<_>>(),
                        "shape": level0.map(|level| level.shape.clone()),
                        "chunks": level0.map(|level| level.chunks.clone()),
                        "dtype": level0.map(|level| level.dtype.clone()),
                        "scale": level0.map(|level| level.scale.clone()),
                        "translation": level0.map(|level| level.translation.clone()),
                        "pyramid_levels": dataset.descriptor.levels.len(),
                        "render_kind": match dataset.descriptor.render_kind {
                            DatasetRenderKind::Image => "image",
                            DatasetRenderKind::LabelMask => "labels",
                        },
                    },
                    "active_channel": active_channel,
                    "channel_count": viewport.state.channels.len(),
                    "visible_channels": viewport.state.channels.iter()
                        .filter(|channel| channel.visible)
                        .map(|channel| channel.name.clone())
                        .collect::<Vec<_>>(),
                })
            });
        json!({
            "mode":self.mode.as_str(),
            "view":view,
            "project":self.project.rois_json(),
        })
    }

    pub(super) fn rendering_state(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        Ok(json!({
            "mode":"single",
            "gpu_available":self.renderer_gpu_available,
            "renderer":if self.renderer_gpu_available { "opengl" } else { "cpu" },
            "compositing":"additive",
            "smooth_pixels":dataset.workspace.active().state.smooth_pixels,
            "deterministic_capture":{
                "method":"viewer.screenshot.capture",
                "readiness":self.loading_state(),
            },
        }))
    }

    pub(super) fn show_project(&mut self) -> Result<Value, ControlError> {
        if self.mode == ModelMode::Transition {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "Odon is currently transitioning between views",
            ));
        }
        let changed = self.mode != ModelMode::Project;
        self.mode = ModelMode::Project;
        Ok(json!({"mode":"project","changed":changed}))
    }
}
