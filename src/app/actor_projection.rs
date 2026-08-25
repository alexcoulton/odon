use super::*;

impl OmeZarrViewerApp {
    pub fn apply_control_actor_channel_compute(
        &mut self,
        generation: u64,
        state: &serde_json::Value,
    ) {
        if generation <= self.control_actor_channel_compute_generation {
            return;
        }
        let histogram = state.get("histogram").unwrap_or(state);
        let request_id = histogram
            .get("request_id")
            .and_then(serde_json::Value::as_u64);
        self.hist_request_pending = histogram
            .get("pending")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            && request_id.is_none_or(|request_id| request_id == self.hist_request_id);
        if let (Some(request_id), Some(projected)) = (request_id, histogram.get("histogram")) {
            let bins = projected
                .get("bins")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_u64)
                .map(|count| count.min(u64::from(u32::MAX)) as u32)
                .collect::<Vec<_>>();
            let stats = projected.get("stats").and_then(|stats| {
                Some(crate::imaging::histogram::HistogramStats {
                    min: stats.get("min")?.as_f64()? as f32,
                    q1: stats.get("q1")?.as_f64()? as f32,
                    median: stats.get("median")?.as_f64()? as f32,
                    q3: stats.get("q3")?.as_f64()? as f32,
                    max: stats.get("max")?.as_f64()? as f32,
                    n: stats
                        .get("n")?
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())?,
                })
            });
            if request_id == self.hist_request_id && !bins.is_empty() {
                self.hist = Some(crate::imaging::histogram::HistogramResponse {
                    request_id,
                    bins,
                    stats,
                });
                self.hist_request_pending = false;
            }
        }
        if histogram.get("error").is_some() {
            self.hist_request_pending = false;
        }
        self.control_actor_channel_compute_generation = generation;
    }

    pub fn install_control_actor_segmentation_geojson_resource(
        &mut self,
        state: &serde_json::Value,
        resource: Option<&odon::model::ControlSegmentationGeoJsonResource>,
    ) -> Result<(), String> {
        let generation = state
            .get("generation")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        self.seg_geojson
            .install_control_resource(generation, resource, state)
    }

    pub fn apply_control_actor_object_export_state(
        &mut self,
        generation: u64,
        state: &serde_json::Value,
    ) {
        if generation <= self.control_actor_object_export_generation {
            return;
        }
        self.seg_objects.apply_control_actor_export_state(state);
        self.control_actor_object_export_generation = generation;
    }

    pub fn apply_control_actor_measurement_state(
        &mut self,
        generation: u64,
        state: &serde_json::Value,
    ) -> Result<(), String> {
        if generation <= self.control_actor_measurement_generation {
            return Ok(());
        }
        self.seg_objects
            .apply_control_actor_measurement_state(state)?;
        self.control_actor_measurement_generation = generation;
        Ok(())
    }

    pub fn apply_control_actor_analysis_state(
        &mut self,
        generation: u64,
        state: &serde_json::Value,
    ) -> Result<(), String> {
        if generation <= self.control_actor_analysis_generation {
            return Ok(());
        }
        let state =
            serde_json::from_value::<crate::objects::ObjectProjectAnalysisState>(state.clone())
                .map_err(|error| format!("actor analysis state is invalid: {error}"))?;
        let active_channel = self
            .channels
            .get(self.selected_channel)
            .map(|channel| channel.name.as_str());
        self.seg_objects
            .apply_project_analysis_state(&state, active_channel);
        self.control_actor_analysis_generation = generation;
        Ok(())
    }

    pub fn apply_control_actor_threshold_preview(
        &mut self,
        ctx: &egui::Context,
        projection_generation: u64,
        pending: bool,
        resource: Option<&Arc<odon::model::ControlThresholdPreviewResource>>,
        state: &serde_json::Value,
    ) -> Result<(), String> {
        if let Some(scope) = state
            .get("configured_scope")
            .and_then(serde_json::Value::as_str)
        {
            self.threshold_region_scope = match scope {
                "visible" => ThresholdRegionScope::VisibleRegion,
                "entire_image" => ThresholdRegionScope::EntireImage,
                _ => return Err(format!("actor threshold scope '{scope}' is invalid")),
            };
        }
        if let Some(level) = state
            .get("configured_full_level")
            .and_then(serde_json::Value::as_u64)
            .and_then(|level| usize::try_from(level).ok())
        {
            self.threshold_region_full_level = level;
        }
        if let Some(minimum) = state
            .get("configured_min_component_pixels")
            .and_then(serde_json::Value::as_u64)
            .and_then(|minimum| usize::try_from(minimum).ok())
        {
            self.threshold_region_min_pixels = minimum.max(1);
        }
        if let Some(status) = state.get("status").and_then(serde_json::Value::as_str) {
            self.threshold_region_status = status.to_string();
        }
        self.threshold_region_draft = ThresholdRegionDraft {
            min_pixels: self.threshold_region_min_pixels,
            scope: self.threshold_region_scope,
            full_level: self.threshold_region_full_level,
        };
        if let Some(resource) = resource {
            if resource.generation() <= self.control_actor_threshold_generation {
                return Ok(());
            }
            let [width, height] = resource.size();
            let plane = Array2::from_shape_vec((height, width), resource.values().as_ref().clone())
                .map_err(|error| format!("actor threshold pixels are invalid: {error}"))?;
            if resource.included().len() != width.saturating_mul(height) {
                return Err("actor threshold mask dimensions are invalid".to_string());
            }
            let [x0, y0] = resource.origin();
            let scope = match resource.scope() {
                odon::model::ThresholdScope::Visible => ThresholdRegionScope::VisibleRegion,
                odon::model::ThresholdScope::EntireImage => ThresholdRegionScope::EntireImage,
            };
            let mut preview = ThresholdRegionPreview {
                generation: resource.generation(),
                channel_index: resource.channel_index(),
                channel_name: resource.channel_name().to_string(),
                scope,
                level_index: resource.level(),
                downsample: resource.downsample(),
                x0,
                y0,
                raw_values: Arc::clone(resource.values()),
                plane,
                threshold: resource.threshold(),
                min_component_pixels: resource.min_component_pixels(),
                mask: ThresholdRegionMask {
                    width,
                    height,
                    included: resource.included().as_ref().clone(),
                },
                texture: None,
            };
            if !self.uses_gpu_threshold_region_preview(&preview) {
                Self::recompute_threshold_region_preview_cpu_data(ctx, &mut preview);
            }
            self.threshold_region_min_pixels = preview.min_component_pixels;
            self.threshold_region_scope = preview.scope;
            self.threshold_region_full_level = preview.level_index;
            self.threshold_region_status = Self::threshold_region_preview_status_message(
                &preview,
                self.uses_gpu_threshold_region_preview(&preview),
            );
            self.threshold_region_preview = Some(preview);
            self.control_actor_threshold_generation = resource.generation();
            self.bump_render_id();
        } else if !pending && projection_generation > self.control_actor_threshold_generation {
            self.threshold_region_preview = None;
            self.threshold_region_status.clear();
            self.control_actor_threshold_generation = projection_generation;
            self.bump_render_id();
        }
        Ok(())
    }

    pub fn apply_control_actor_memory(
        &mut self,
        state: &serde_json::Value,
        resources: &[Arc<odon::model::ControlPinnedLevelResource>],
    ) {
        self.control_actor_memory_state = state.clone();
        self.pinned_levels
            .replace_control_actor_resources(resources);
    }

    pub fn control_actor_dataset(&self) -> OmeZarrDataset {
        self.dataset.clone()
    }

    pub fn control_actor_store(&self) -> Arc<dyn zarrs::storage::ReadableStorageTraits> {
        Arc::clone(&self.store)
    }

    pub fn control_actor_source_key(&self) -> String {
        self.dataset.source.source_key()
    }

    pub fn install_control_actor_object_resource(
        &mut self,
        generation: u64,
        resource: &odon::model::ControlObjectResource,
    ) -> bool {
        if generation <= self.control_actor_object_generation
            || !self.seg_objects.install_control_resource(resource)
        {
            return false;
        }
        self.control_actor_object_generation = generation;
        self.rebuild_layer_orders();
        true
    }

    pub fn install_control_actor_secondary_object_resources(
        &mut self,
        layers: &[odon::model::ControlSecondaryObjectProjection],
    ) -> Result<(), String> {
        let wanted = layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect::<HashSet<_>>();
        self.control_actor_secondary_object_generations
            .retain(|layer_id, _| wanted.contains(layer_id));
        self.control_actor_secondary_object_selection_generations
            .retain(|layer_id, _| wanted.contains(layer_id));
        self.control_actor_secondary_object_analysis_generations
            .retain(|layer_id, _| wanted.contains(layer_id));
        let active_channel = self
            .channels
            .get(self.selected_channel)
            .map(|channel| channel.name.clone());
        for projected in layers {
            let installed = self
                .control_actor_secondary_object_generations
                .get(&projected.layer_id)
                .copied()
                .unwrap_or(0);
            let layer = self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|layer| layer.id == projected.layer_id)
                .ok_or_else(|| {
                    format!(
                        "actor spatial shape layer {} ({}) is unavailable in the renderer",
                        projected.layer_id, projected.name
                    )
                })?;
            let objects = layer.object_layer_mut().ok_or_else(|| {
                format!(
                    "actor spatial shape layer {} has no object renderer",
                    projected.layer_id
                )
            })?;
            if projected.generation > installed {
                if !objects.install_control_resource(&projected.resource) {
                    return Err(format!(
                        "actor spatial shape layer {} rejected its prepared resource",
                        projected.layer_id
                    ));
                }
                self.control_actor_secondary_object_generations
                    .insert(projected.layer_id, projected.generation);
                self.control_actor_secondary_object_analysis_generations
                    .remove(&projected.layer_id);
            }
            let selected_indices = projected
                .selection
                .get("selected_indices")
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_u64)
                .filter_map(|value| usize::try_from(value).ok())
                .collect::<Vec<_>>();
            let primary_index = projected
                .selection
                .get("primary_index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok());
            objects.install_control_selection(&selected_indices, primary_index)?;
            let selection_generation = projected
                .selection
                .get("generation")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1)
                .max(1);
            self.control_actor_secondary_object_selection_generations
                .insert(projected.layer_id, selection_generation);
            let installed_analysis = self
                .control_actor_secondary_object_analysis_generations
                .get(&projected.layer_id)
                .copied()
                .unwrap_or(0);
            if projected.analysis_generation > installed_analysis {
                let state = serde_json::from_value::<crate::objects::ObjectProjectAnalysisState>(
                    projected.analysis_state.clone(),
                )
                .map_err(|error| {
                    format!(
                        "actor spatial shape layer {} analysis state is invalid: {error}",
                        projected.layer_id
                    )
                })?;
                objects.apply_project_analysis_state(&state, active_channel.as_deref());
                self.control_actor_secondary_object_analysis_generations
                    .insert(projected.layer_id, projected.analysis_generation);
            }
        }
        Ok(())
    }

    pub fn install_control_actor_label_resource(
        &mut self,
        generation: u64,
        resource: &odon::model::ControlLabelResource,
    ) -> Result<bool, String> {
        // Actor-owned label state is already an explicit decision. Never leave the native
        // discovery prompt open over a remotely controlled or restored label projection.
        self.seg_label_prompt_open = false;
        self.seg_label_prompt_always = false;
        if generation <= self.control_actor_label_generation {
            return Ok(false);
        }
        self.labels_gl
            .get_or_insert_with(|| LabelsGl::new(1024))
            .reset();
        let labels = resource.dataset.clone();
        self.spatial_label_transform = self.spatial_label_transform_for_name(&labels.label_name);
        self.label_loader = Some(
            spawn_label_tile_loader(
                Arc::clone(&resource.store),
                labels.levels.clone(),
                labels.dims.clone(),
            )
            .map_err(|error| format!("could not start actor label loader: {error}"))?,
        );
        self.label_cells_xform = Some(compute_label_to_world_xforms(
            &self.dataset,
            &labels,
            self.spatial_label_transform,
        ));
        self.seg_label_selected = labels.label_name.clone();
        self.seg_label_input = labels.label_name.clone();
        self.seg_label_status = format!("Loaded labels/{}.", labels.label_name);
        self.label_cells = Some(labels);
        self.cells_outlines_visible = true;
        self.control_actor_label_generation = generation;
        self.rebuild_layer_orders();
        Ok(true)
    }

    pub fn unload_control_actor_label_resource(&mut self, generation: u64) -> bool {
        // An actor-owned unload is also explicit; reopening the discovery prompt would turn a
        // deterministic remote command into an unattended native-modal workflow.
        self.seg_label_prompt_open = false;
        self.seg_label_prompt_always = false;
        if generation <= self.control_actor_label_generation {
            return false;
        }
        self.label_cells = None;
        self.label_loader = None;
        self.label_cells_xform = None;
        self.cells_outlines_visible = false;
        if let Some(labels) = self.labels_gl.as_ref() {
            labels.reset();
        }
        self.seg_label_status = "Unloaded segmentation labels.".to_string();
        self.control_actor_label_generation = generation;
        self.rebuild_layer_orders();
        true
    }

    pub(super) fn apply_control_actor_object_selection_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
        let generation = projection
            .get("generation")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if generation <= self.control_actor_object_selection_generation {
            return Ok(());
        }
        let selected_indices = projection
            .get("selected_indices")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "actor object selection has no selected_indices".to_string())?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| "actor object selection index is invalid".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let primary_index = projection
            .get("primary_index")
            .filter(|value| !value.is_null())
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| "actor object selection primary is invalid".to_string())
            })
            .transpose()?;
        self.seg_objects
            .install_control_selection(&selected_indices, primary_index)?;
        self.control_actor_object_selection_generation = generation;
        Ok(())
    }

    pub(super) fn apply_control_actor_masks_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
        let generation = projection
            .get("generation")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if generation <= self.control_actor_mask_generation {
            return Ok(());
        }
        if self.mask_polygon_gesture_active() {
            let pending_generation = self
                .pending_control_actor_mask_projection
                .as_ref()
                .and_then(|pending| pending.get("generation"))
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if generation > pending_generation {
                self.pending_control_actor_mask_projection = Some(projection.clone());
            }
            return Ok(());
        }
        let layers = projection
            .get("layers")
            .cloned()
            .map(serde_json::from_value::<Vec<crate::data::project_config::ProjectMaskLayer>>)
            .transpose()
            .map_err(|error| format!("actor mask projection is invalid: {error}"))?
            .unwrap_or_default();
        self.mask_layers = layers.iter().map(MaskLayer::from_project).collect();
        self.active_layer = projection
            .get("active_layer_id")
            .and_then(serde_json::Value::as_u64)
            .filter(|id| self.mask_layers.iter().any(|layer| layer.id == *id))
            .map(LayerId::Mask)
            .unwrap_or_else(|| {
                if self.channels.is_empty() {
                    LayerId::Points
                } else {
                    LayerId::Channel(self.selected_channel.min(self.channels.len() - 1))
                }
            });
        self.selected_mask_polygon = None;
        self.selected_mask_vertex = None;
        if let Some(selection) = projection.get("selection").filter(|value| !value.is_null()) {
            let layer_id = selection
                .get("layer_id")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| "actor mask selection has no layer_id".to_string())?;
            let polygon_idx = selection
                .get("polygon_index")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "actor mask selection has no polygon_index".to_string())?;
            let polygon = self
                .mask_layers
                .iter()
                .find(|layer| layer.id == layer_id)
                .and_then(|layer| layer.polygons_world.get(polygon_idx))
                .ok_or_else(|| "actor mask selection references an unknown polygon".to_string())?;
            let vertex_index = selection
                .get("vertex_index")
                .filter(|value| !value.is_null())
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|index| *index < Self::mask_polygon_unique_vertex_count(polygon))
                        .ok_or_else(|| {
                            "actor mask selection references an unknown vertex".to_string()
                        })
                })
                .transpose()?;
            self.selected_mask_polygon = Some(MaskPolygonSelection {
                layer_id,
                polygon_idx,
            });
            self.selected_mask_vertex = vertex_index;
        }
        if self.drawing_mask_layer.is_some_and(|id| {
            !self
                .mask_layers
                .iter()
                .any(|layer| layer.id == id && layer.editable)
        }) {
            self.drawing_mask_layer = None;
            self.drawing_mask_polygon.clear();
        }
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = None;
        self.control_actor_mask_undo_available = projection
            .get("undo_available")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        self.control_actor_mask_generation = generation;
        self.pending_control_actor_mask_projection = None;
        self.rebuild_layer_orders();
        Ok(())
    }

    pub fn apply_control_actor_workspace_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
        if let Some(labels) = projection.get("labels") {
            let generation = labels
                .get("generation")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if self.control_actor_label_state_generation != Some(generation) {
                self.seg_label_names = labels
                    .get("available")
                    .and_then(serde_json::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect();
                self.seg_label_selected = labels
                    .get("selected")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.seg_label_input = self.seg_label_selected.clone();
                self.seg_label_status = labels
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let loaded = labels.get("loaded").is_some_and(|value| !value.is_null());
                let actor_owned = labels
                    .get("actor_owned")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                self.seg_label_prompt_open = false;
                if !loaded && !actor_owned && !self.seg_label_names.is_empty() {
                    match self.seg_label_prompt_preference {
                        LabelPromptSessionPreference::Ask => {
                            self.seg_label_prompt_open = self.tiles_gl.is_some();
                        }
                        LabelPromptSessionPreference::AlwaysLoad
                            if !self.seg_label_selected.is_empty() =>
                        {
                            self.native_command_ingress.push(NativeControlIntent {
                                method: "viewer.labels.load",
                                params: serde_json::json!({"name":self.seg_label_selected}),
                            });
                        }
                        LabelPromptSessionPreference::AlwaysSkip
                        | LabelPromptSessionPreference::AlwaysLoad => {}
                    }
                }
                self.control_actor_label_state_generation = Some(generation);
            }
        }
        if let Some(left_tab) = projection
            .get("ui")
            .and_then(|ui| ui.get("left_tab"))
            .and_then(serde_json::Value::as_str)
        {
            self.left_tab = LeftTab::from_storage_key(left_tab)
                .ok_or_else(|| format!("actor left tab '{left_tab}' is invalid"))?;
        }
        if let Some(right_tab) = projection
            .get("ui")
            .and_then(|ui| ui.get("right_tab"))
            .and_then(serde_json::Value::as_str)
        {
            self.right_tab = RightTab::from_storage_key(right_tab)
                .ok_or_else(|| format!("actor right tab '{right_tab}' is invalid"))?;
        }
        if let Some(selection) = projection.get("object_selection") {
            self.apply_control_actor_object_selection_projection(selection)?;
        }
        if let Some(masks) = projection.get("masks") {
            self.apply_control_actor_masks_projection(masks)?;
        }
        if let Some(panels) = projection.get("panels") {
            if let Some(left) = panels.get("left").and_then(serde_json::Value::as_bool) {
                self.show_left_panel = left;
            }
            if let Some(right) = panels.get("right").and_then(serde_json::Value::as_bool) {
                self.show_right_panel = right;
            }
        }
        if let Some(resource) = projection.get("object_resource") {
            let generation = resource
                .get("generation")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            if generation > self.control_actor_object_generation {
                if resource
                    .get("source")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
                {
                    self.seg_objects.clear();
                    self.control_actor_object_generation = generation;
                    self.rebuild_layer_orders();
                } else {
                    return Err(format!(
                        "actor object resource generation {generation} has no shared renderer payload"
                    ));
                }
            }
        }
        if let Some(metadata) = projection
            .get("channel_metadata")
            .and_then(serde_json::Value::as_array)
        {
            for item in metadata {
                let Some(index) = item
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                else {
                    continue;
                };
                if let (Some(channel), Some(note)) = (
                    self.channels.get_mut(index),
                    item.get("note").and_then(serde_json::Value::as_str),
                ) {
                    channel.note = note.to_string();
                }
            }
        }
        if let Some(transforms) = projection
            .get("channel_transforms")
            .and_then(serde_json::Value::as_array)
        {
            for transform in transforms {
                let Some(index) = transform
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                else {
                    continue;
                };
                if index >= self.channels.len() {
                    continue;
                }
                let pair = |name: &str| {
                    transform
                        .get(name)
                        .and_then(serde_json::Value::as_array)
                        .filter(|values| values.len() == 2)
                        .and_then(|values| {
                            Some(egui::vec2(
                                values[0].as_f64()? as f32,
                                values[1].as_f64()? as f32,
                            ))
                        })
                        .filter(|value| value.x.is_finite() && value.y.is_finite())
                };
                if let Some(offset) = pair("offset_world") {
                    self.channel_offsets_world[index] = offset;
                }
                if let Some(scale) = pair("scale") {
                    self.channel_scales[index] = scale;
                }
                if let Some(rotation) = transform
                    .get("rotation_rad")
                    .and_then(serde_json::Value::as_f64)
                    .map(|value| value as f32)
                    .filter(|value| value.is_finite())
                {
                    self.channel_rotations_rad[index] = rotation;
                }
            }
        }
        if let Some(presentation) = projection.get("channel_presentation") {
            if let Some(search) = presentation
                .get("search")
                .and_then(serde_json::Value::as_str)
            {
                self.channel_list_search = search.to_string();
            }
            if let Some(sort) = presentation
                .get("sort")
                .and_then(serde_json::Value::as_str)
                .and_then(ChannelSortMode::from_storage_key)
            {
                self.channel_sort_mode = sort;
            }
        }
        let projected_viewports = projection
            .get("viewports")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "actor projection has no viewport array".to_string())?;
        if projected_viewports.is_empty()
            || projected_viewports.len() > crate::viewports::MAX_VIEWPORTS
        {
            return Err("actor projection has an invalid viewport count".to_string());
        }

        let mut current = self.viewport_workspace.take();
        let current_active = current
            .as_ref()
            .map(|workspace| workspace.active_id().clone());
        let previous_viewport_ids = current
            .as_ref()
            .into_iter()
            .flat_map(|workspace| workspace.viewports())
            .map(|viewport| viewport.id.clone())
            .collect::<HashSet<_>>();
        if let Some(workspace) = current.as_mut() {
            let active_id = workspace.active_id().clone();
            if let Some(active) = workspace.get_mut(&active_id) {
                active.state.capture_runtime(self);
            }
        }
        let fallback = current
            .as_ref()
            .map(|workspace| workspace.active().state.clone())
            .unwrap_or_else(|| ViewerViewportState::capture(self));
        let projected_masks = self
            .mask_layers
            .iter()
            .map(|layer| MaskViewportPresentation {
                id: layer.id,
                visible: layer.visible,
                opacity: layer.opacity,
                width_screen_px: layer.width_screen_px,
                display_mode: layer.display_mode,
                color_rgb: layer.color_rgb,
            })
            .collect::<Vec<_>>();

        let mut slots = Vec::with_capacity(projected_viewports.len());
        for projected in projected_viewports {
            let id = projected
                .get("viewport_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "actor projection viewport has no ID".to_string())
                .and_then(|id| ViewportId::new(id).map_err(|error| error.to_string()))?;
            let title = projected
                .get("title")
                .and_then(serde_json::Value::as_str)
                .filter(|title| !title.trim().is_empty())
                .ok_or_else(|| format!("actor projection viewport '{id}' has no title"))?
                .to_string();
            let mut state = current
                .as_ref()
                .and_then(|workspace| workspace.get(&id))
                .map(|viewport| viewport.state.clone())
                .unwrap_or_else(|| fallback.clone());
            state.masks.clone_from(&projected_masks);
            Self::apply_actor_viewport_projection(&mut state, projected)?;
            self.apply_actor_native_layer_topology(&mut state, projected)?;
            slots.push(ViewportSlot {
                id,
                title,
                state,
                navigation_revision: projected
                    .get("navigation_revision")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
                presentation_revision: projected
                    .get("presentation_revision")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(1)
                    .max(1),
            });
        }

        let active = projection
            .get("active_viewport_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "actor projection has no active viewport ID".to_string())
            .and_then(|id| ViewportId::new(id).map_err(|error| error.to_string()))?;
        let projected_viewport_ids = slots
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<HashSet<_>>();
        if current_active.as_ref() != Some(&active)
            || previous_viewport_ids != projected_viewport_ids
        {
            self.cancel_viewport_transient_gestures();
        }
        self.screenshot_capture
            .pending
            .retain(|pending| projected_viewport_ids.contains(&pending.viewport_id));
        let layout = projection
            .get("layout")
            .and_then(serde_json::Value::as_str)
            .and_then(ViewportLayout::parse)
            .ok_or_else(|| "actor projection has an invalid viewport layout".to_string())?;
        let links = projection
            .get("links")
            .map(|links| ViewportLinks {
                camera: links
                    .get("camera")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                plane: links
                    .get("plane")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
                selection: links
                    .get("selection")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(true),
            })
            .unwrap_or_default();
        let ratio = projection
            .get("ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.5) as f32;
        let revision = projection
            .get("revision")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1);
        let mut workspace =
            ViewportWorkspace::restore_projection(slots, active, layout, links, ratio, revision)
                .map_err(|error| error.to_string())?;

        // Recompute renderer generations from each projected presentation. This invalidates stale
        // black/off-image tile work while retaining each viewport's last measured canvas rectangle.
        let ids = workspace
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for id in &ids {
            let projected_state = workspace
                .get(id)
                .expect("projected viewport remains present")
                .state
                .clone();
            projected_state.apply(self);
            if let Some(native_layers) = projected_viewports
                .iter()
                .find(|projected| {
                    projected
                        .get("viewport_id")
                        .and_then(serde_json::Value::as_str)
                        == Some(id.as_str())
                })
                .and_then(|projected| projected.get("native_layers"))
            {
                self.apply_control_actor_native_layers_projection(native_layers)?;
            }
            self.bump_render_id();
            workspace
                .get_mut(id)
                .expect("projected viewport remains present")
                .state = ViewerViewportState::capture(self);
        }
        workspace.active().state.apply(self);
        self.loader
            .set_active_render_ids(Self::workspace_cpu_render_ids(&workspace));
        self.hist_dirty = true;
        self.hist_navigation_dirty_since = Some(Instant::now());
        self.viewport_workspace = Some(workspace);
        self.control_actor_workspace_revision = revision.max(1);
        Ok(())
    }

    fn apply_actor_native_layer_topology(
        &self,
        state: &mut ViewerViewportState,
        projected: &serde_json::Value,
    ) -> Result<(), String> {
        let Some(layers) = projected
            .get("native_layers")
            .and_then(serde_json::Value::as_array)
        else {
            return Ok(());
        };
        let mut overlays = Vec::new();
        let mut active = None;
        for layer in layers {
            if layer.get("available").and_then(serde_json::Value::as_bool) == Some(false) {
                continue;
            }
            let raw = layer
                .get("layer_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "actor native layer has no layer_id".to_string())?;
            let id = self
                .parse_layer_id_storage_key(raw)
                .ok_or_else(|| format!("unknown actor native layer '{raw}'"))?;
            if layer.get("active").and_then(serde_json::Value::as_bool) == Some(true) {
                active = Some(id);
            }
            if layer.get("stack").and_then(serde_json::Value::as_str) == Some("overlays") {
                let order = layer
                    .get("order")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(u64::MAX);
                overlays.push((order, id));
            }
        }
        overlays.sort_by_key(|(order, _)| *order);
        state.overlay_layer_order = overlays.into_iter().map(|(_, id)| id).collect();
        if let Some(active) = active {
            state.active_layer = active;
        }
        Ok(())
    }

    pub(super) fn apply_actor_viewport_projection(
        state: &mut ViewerViewportState,
        projected: &serde_json::Value,
    ) -> Result<(), String> {
        if let Some(camera) = projected.get("camera") {
            if let Some(center) = camera
                .get("center_world_lvl0")
                .and_then(serde_json::Value::as_array)
                .filter(|center| center.len() == 2)
            {
                let x = center[0]
                    .as_f64()
                    .ok_or_else(|| "actor camera center x is invalid".to_string())?
                    as f32;
                let y = center[1]
                    .as_f64()
                    .ok_or_else(|| "actor camera center y is invalid".to_string())?
                    as f32;
                if !x.is_finite() || !y.is_finite() {
                    return Err("actor camera center is not finite".to_string());
                }
                state.camera.center_world_lvl0 = egui::pos2(x, y);
            }
            if let Some(zoom) = camera
                .get("zoom_screen_per_lvl0_px")
                .and_then(serde_json::Value::as_f64)
            {
                let zoom = zoom as f32;
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err("actor camera zoom is invalid".to_string());
                }
                state.camera.zoom_screen_per_lvl0_px = zoom;
            }
        }

        if let Some(plane) = projected.get("plane") {
            if let Some(mode) = plane.get("mode").and_then(serde_json::Value::as_str) {
                state.view_plane_mode = match mode.to_ascii_lowercase().as_str() {
                    "xy" => ViewPlaneMode::Xy,
                    "xz" => ViewPlaneMode::Xz,
                    "yz" => ViewPlaneMode::Yz,
                    _ => return Err(format!("actor plane mode '{mode}' is invalid")),
                };
            }
            if let Some(slice) = plane.get("slice").and_then(serde_json::Value::as_u64) {
                match state.view_plane_mode {
                    ViewPlaneMode::Xy => state.current_z_level0 = slice,
                    ViewPlaneMode::Xz => state.current_y_level0 = slice,
                    ViewPlaneMode::Yz => state.current_x_level0 = slice,
                }
                state.transient.draft_view_slice_level0 = None;
            }
        }

        if let Some(channels) = projected
            .get("channels")
            .and_then(serde_json::Value::as_array)
        {
            for projected_channel in channels {
                let Some(index) = projected_channel
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                else {
                    continue;
                };
                let Some(channel) = state
                    .channels
                    .iter_mut()
                    .find(|channel| channel.index == index)
                else {
                    continue;
                };
                if let Some(visible) = projected_channel
                    .get("visible")
                    .and_then(serde_json::Value::as_bool)
                {
                    channel.visible = visible;
                }
                if let Some(color) = projected_channel
                    .get("color_rgb")
                    .and_then(ViewerViewportState::rgb_from_json)
                {
                    channel.color_rgb = color;
                }
                channel.window = projected_channel.get("window").and_then(|window| {
                    if window.is_null() {
                        return None;
                    }
                    if let Some(values) = window.as_array().filter(|values| values.len() == 2) {
                        return Some((values[0].as_f64()? as f32, values[1].as_f64()? as f32));
                    }
                    Some((
                        window.get("min")?.as_f64()? as f32,
                        window.get("max")?.as_f64()? as f32,
                    ))
                });
                if projected_channel
                    .get("selected")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
                {
                    state.selected_channel = index;
                }
            }
        }

        if let Some(order) = projected
            .get("channel_order")
            .and_then(serde_json::Value::as_array)
        {
            let order = order
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|index| usize::try_from(index).ok())
                        .filter(|index| *index < state.channels.len())
                        .ok_or_else(|| "actor channel order contains an invalid index".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let unique = order.iter().copied().collect::<HashSet<_>>();
            if order.len() != state.channels.len() || unique.len() != order.len() {
                return Err(
                    "actor channel order must contain every channel exactly once".to_string(),
                );
            }
            state.channel_layer_order = order;
        }
        if let Some(sort) = projected
            .get("channel_sort")
            .and_then(serde_json::Value::as_str)
        {
            state.channel_sort_mode = ChannelSortMode::from_storage_key(sort)
                .ok_or_else(|| format!("actor channel sort mode '{sort}' is invalid"))?;
        }
        if let Some(groups) = projected
            .get("channel_groups")
            .and_then(serde_json::Value::as_array)
        {
            let mut layer_groups = state.layer_groups.clone();
            layer_groups.channel_groups.clear();
            layer_groups.channel_members.clear();
            for group in groups {
                let id = group
                    .get("id")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| "actor channel group has no valid id".to_string())?;
                let name = group
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| "actor channel group has no valid name".to_string())?
                    .to_string();
                let color_rgb = group
                    .get("color_rgb")
                    .and_then(ViewerViewportState::rgb_from_json)
                    .ok_or_else(|| "actor channel group has an invalid color".to_string())?;
                layer_groups.channel_groups.push(ProjectChannelGroup {
                    id,
                    name,
                    expanded: group
                        .get("expanded")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(true),
                    color_rgb,
                });
                if let Some(members) = group.get("members").and_then(serde_json::Value::as_array) {
                    for member in members {
                        let name = member
                            .get("name")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| "actor channel group member has no name".to_string())?;
                        layer_groups.channel_members.insert(
                            name.to_string(),
                            ProjectChannelGroupMember {
                                group_id: id,
                                inherit_color: member
                                    .get("inherit_color")
                                    .and_then(serde_json::Value::as_bool)
                                    .unwrap_or(true),
                            },
                        );
                    }
                }
            }
            state.layer_groups = layer_groups;
        }

        if let Some(objects) = projected.get("objects") {
            if let Some(value) = objects.get("visible").and_then(serde_json::Value::as_bool) {
                state.object_visible = value;
            }
            if let Some(value) = objects.get("opacity").and_then(serde_json::Value::as_f64) {
                state.object_opacity = (value as f32).clamp(0.0, 1.0);
            }
            if let Some(value) = objects
                .get("width_screen_px")
                .and_then(serde_json::Value::as_f64)
            {
                state.object_width_screen_px = (value as f32).clamp(f32::EPSILON, 100.0);
            }
            if let Some(value) = objects
                .get("color_rgb")
                .and_then(ViewerViewportState::rgb_from_json)
            {
                state.object_color_rgb = value;
            }
            if let Some(value) = objects
                .get("fill_cells")
                .and_then(serde_json::Value::as_bool)
            {
                state.object_display.fill_cells = value;
            }
            if let Some(value) = objects
                .get("fill_opacity")
                .and_then(serde_json::Value::as_f64)
            {
                state.object_display.fill_opacity = (value as f32).clamp(0.0, 1.0);
            }
            if let Some(value) = objects
                .get("selected_fill_opacity")
                .and_then(serde_json::Value::as_f64)
            {
                state.object_display.selected_fill_opacity = (value as f32).clamp(0.0, 1.0);
            }
            if let Some(value) = objects
                .get("show_selection_overlay")
                .and_then(serde_json::Value::as_bool)
            {
                state.object_show_selection_overlay = value;
            }
            if let Some(value) = objects
                .get("fast_rendering")
                .and_then(serde_json::Value::as_bool)
            {
                state.object_display.fast_rendering = value;
            }
            if let Some(value) = objects.get("color_property") {
                state.object_display.color_property_key = value
                    .as_str()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
            }
            if let Some(value) = objects.get("color_mapping") {
                state.object_display.color_mapping =
                    Some(serde_json::from_value(value.clone()).map_err(|error| {
                        format!("actor object color mapping is invalid: {error}")
                    })?);
            }
            if let Some(value) = objects.get("color_level_overrides") {
                state.object_display.color_level_overrides = serde_json::from_value(value.clone())
                    .map_err(|error| {
                        format!("actor object legend overrides are invalid: {error}")
                    })?;
            }
            if let Some(filter) = objects.get("filter") {
                state.object_filter = ObjectViewportFilterState::from_project_json(filter)
                    .map_err(|error| format!("actor object filter is invalid: {error}"))?;
                state.transient.object_filter_cache = ObjectViewportFilterCacheState::empty();
            }
        }

        if let Some(overlays) = projected.get("object_overlay_visibility") {
            if let Some(value) = overlays
                .get("segmentation_labels")
                .and_then(serde_json::Value::as_bool)
            {
                state.cells_outlines_visible = value;
            }
            if let Some(value) = overlays
                .get("segmentation_geojson")
                .and_then(serde_json::Value::as_bool)
            {
                state.seg_geojson_visible = value;
            }
        }

        if let Some(rendering) = projected.get("rendering") {
            if let Some(value) = rendering
                .get("smooth_pixels")
                .and_then(serde_json::Value::as_bool)
            {
                state.smooth_pixels = value;
            }
            if let Some(value) = rendering
                .get("show_scale_bar")
                .and_then(serde_json::Value::as_bool)
            {
                state.show_scale_bar = value;
            }
            if let Some(value) = rendering
                .get("show_hud")
                .and_then(serde_json::Value::as_bool)
            {
                state.show_hud = value;
            }
            if let Some(value) = rendering
                .get("show_tile_debug")
                .and_then(serde_json::Value::as_bool)
            {
                state.show_tile_debug = value;
            }
        }
        Ok(())
    }

    pub fn control_actor_viewport_geometry(&self) -> Vec<(String, f32, f32, f32, f32)> {
        self.viewport_workspace
            .as_ref()
            .into_iter()
            .flat_map(|workspace| workspace.viewports())
            .filter_map(|viewport| {
                let rect = viewport.state.render.last_canvas_rect?;
                (rect.width().is_finite()
                    && rect.height().is_finite()
                    && rect.width() > 0.0
                    && rect.height() > 0.0)
                    .then(|| {
                        (
                            viewport.id.as_str().to_string(),
                            rect.min.x,
                            rect.min.y,
                            rect.width(),
                            rect.height(),
                        )
                    })
            })
            .collect()
    }
}
