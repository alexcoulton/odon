use super::*;

impl OmeZarrViewerApp {
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

    pub fn install_control_actor_label_resource(
        &mut self,
        generation: u64,
        resource: &odon::model::ControlLabelResource,
    ) -> Result<bool, String> {
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
        let layers = projection
            .get("layers")
            .cloned()
            .map(serde_json::from_value::<Vec<crate::data::project_config::ProjectMaskLayer>>)
            .transpose()
            .map_err(|error| format!("actor mask projection is invalid: {error}"))?
            .unwrap_or_default();
        self.mask_layers = layers.iter().map(MaskLayer::from_project).collect();
        self.next_mask_layer_id = self
            .mask_layers
            .iter()
            .map(|layer| layer.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
            .max(1);
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
        self.undo_stack.clear();
        self.drawing_mask_layer = None;
        self.drawing_mask_polygon.clear();
        self.dragging_mask_vertex = None;
        self.moving_mask_polygon = None;
        self.mask_layers_project_dirty = projection
            .get("dirty")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        self.control_actor_mask_undo_available = projection
            .get("undo_available")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        self.control_actor_mask_generation = generation;
        self.rebuild_layer_orders();
        Ok(())
    }

    pub(super) fn apply_control_actor_native_layers_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
        let layers = projection
            .as_array()
            .ok_or_else(|| "actor native_layers projection is not an array".to_string())?;
        for layer in layers {
            if layer.get("available").and_then(serde_json::Value::as_bool) == Some(false) {
                continue;
            }
            let layer_id = layer
                .get("layer_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "actor native layer has no layer_id".to_string())?;
            let mut params = serde_json::json!({"layer_id":layer_id});
            if let Some(presentation) = layer.get("presentation") {
                let mut presentation = presentation.clone();
                if presentation
                    .get("window")
                    .is_some_and(serde_json::Value::is_null)
                {
                    presentation
                        .as_object_mut()
                        .expect("native layer presentation is an object")
                        .remove("window");
                }
                params["presentation"] = presentation;
                let response = self.control_set_native_layer_presentation(&params);
                if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
                    return Err(format!(
                        "actor native layer '{layer_id}' presentation failed: {error}"
                    ));
                }
            } else if let Some(visible) = layer.get("visible") {
                params["visible"] = visible.clone();
                let response = self.control_set_native_layer_visibility(&params);
                if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
                    return Err(format!(
                        "actor native layer '{layer_id}' visibility failed: {error}"
                    ));
                }
            }
            if let Some(offset) = layer.get("offset_world") {
                params["offset_world"] = offset.clone();
                let response = self.control_set_native_layer_offset(&params);
                if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
                    return Err(format!(
                        "actor native layer '{layer_id}' offset failed: {error}"
                    ));
                }
            }
        }
        for stack in ["channels", "overlays"] {
            let ordered = layers
                .iter()
                .filter(|layer| {
                    layer.get("stack").and_then(serde_json::Value::as_str) == Some(stack)
                })
                .filter_map(|layer| {
                    Some((
                        layer.get("order").and_then(serde_json::Value::as_u64)?,
                        layer
                            .get("layer_id")
                            .and_then(serde_json::Value::as_str)?
                            .to_string(),
                    ))
                })
                .collect::<Vec<_>>();
            if ordered.is_empty() {
                continue;
            }
            let mut ordered = ordered;
            ordered.sort_by_key(|(order, _)| *order);
            let mut ordered = ordered.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
            // During incremental migration the renderer can discover compatibility-only layers
            // before the actor has observed their descriptors. Preserve those layers after the
            // actor-owned order instead of rejecting the whole projection.
            for layer in self
                .control_native_layer_snapshot_list()
                .as_array()
                .into_iter()
                .flatten()
                .filter(|layer| {
                    layer.get("stack").and_then(serde_json::Value::as_str) == Some(stack)
                })
            {
                if let Some(layer_id) = layer.get("layer_id").and_then(serde_json::Value::as_str)
                    && !ordered.iter().any(|candidate| candidate == layer_id)
                {
                    ordered.push(layer_id.to_string());
                }
            }
            let response = self.control_set_native_layer_order(&serde_json::json!({
                "stack":stack,
                "layers":ordered,
            }));
            if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
                return Err(format!("actor native {stack} order failed: {error}"));
            }
        }
        if let Some(active) = layers
            .iter()
            .find(|layer| layer.get("active").and_then(serde_json::Value::as_bool) == Some(true))
            .and_then(|layer| layer.get("layer_id"))
            .and_then(serde_json::Value::as_str)
        {
            let response =
                self.control_set_active_native_layer(&serde_json::json!({"layer_id":active}));
            if let Some(error) = response.get("error").and_then(serde_json::Value::as_str) {
                return Err(format!("actor active native layer failed: {error}"));
            }
        }
        Ok(())
    }

    pub fn apply_control_actor_workspace_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
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
                self.control_actor_object_generation = generation;
                if let Some(source) = resource.get("source").and_then(serde_json::Value::as_str) {
                    let downsample_factor = resource
                        .get("downsample_factor")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(1.0) as f32;
                    self.seg_objects
                        .load_path(PathBuf::from(source), downsample_factor.max(f32::EPSILON));
                } else {
                    self.seg_objects.clear();
                    self.rebuild_layer_orders();
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

        let mut current = self
            .viewport_workspace
            .take()
            .unwrap_or_else(|| ViewportWorkspace::new(ViewerViewportState::capture(self)));
        let current_active = current.active_id().clone();
        if let Some(active) = current.get_mut(&current_active) {
            active.state = ViewerViewportState::capture(self);
        }
        let fallback = current.active().state.clone();
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
                .get(&id)
                .map(|viewport| viewport.state.clone())
                .unwrap_or_else(|| fallback.clone());
            state.masks.clone_from(&projected_masks);
            Self::apply_actor_viewport_projection(&mut state, projected)?;
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
                state.draft_view_slice_level0 = None;
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
            if let Some(value) = objects.get("color_level_overrides") {
                state.object_display.color_level_overrides = serde_json::from_value(value.clone())
                    .map_err(|error| {
                        format!("actor object legend overrides are invalid: {error}")
                    })?;
            }
            if let Some(filter) = objects.get("filter") {
                state.object_filter = ObjectViewportFilterState::from_project_json(filter)
                    .map_err(|error| format!("actor object filter is invalid: {error}"))?;
                state.object_filter_cache = ObjectViewportFilterCacheState::empty();
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

    pub fn control_actor_viewport_geometry(&self) -> Vec<(String, f32, f32)> {
        self.viewport_workspace
            .as_ref()
            .into_iter()
            .flat_map(|workspace| workspace.viewports())
            .filter_map(|viewport| {
                let rect = viewport.state.last_canvas_rect?;
                (rect.width().is_finite()
                    && rect.height().is_finite()
                    && rect.width() > 0.0
                    && rect.height() > 0.0)
                    .then(|| {
                        (
                            viewport.id.as_str().to_string(),
                            rect.width(),
                            rect.height(),
                        )
                    })
            })
            .collect()
    }
}
