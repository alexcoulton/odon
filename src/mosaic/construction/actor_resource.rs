use super::*;

impl MosaicViewerApp {
    pub(crate) fn apply_control_actor_annotation_layers(
        &mut self,
        projections: &[odon::model::ControlAnnotationLayerProjection],
    ) {
        let mut existing = self
            .annotation_layers
            .drain(..)
            .map(|layer| (layer.id, layer))
            .collect::<HashMap<_, _>>();
        self.annotation_layers = projections
            .iter()
            .map(|projection| {
                let mut layer = existing.remove(&projection.state.id).unwrap_or_else(|| {
                    AnnotationPointsLayer::new(projection.state.id, projection.state.name.clone())
                });
                layer.apply_control_projection(projection);
                layer
            })
            .collect();
        let annotation_ids = self
            .annotation_layers
            .iter()
            .map(|layer| layer.id)
            .collect::<std::collections::HashSet<_>>();
        self.overlay_layer_order.retain(
            |layer| !matches!(layer, MosaicLayerId::Annotation(id) if !annotation_ids.contains(id)),
        );
        for id in annotation_ids {
            let layer = MosaicLayerId::Annotation(id);
            if !self.overlay_layer_order.contains(&layer) {
                self.overlay_layer_order.push(layer);
            }
        }
        if matches!(self.active_layer, MosaicLayerId::Annotation(id) if !self.annotation_layers.iter().any(|layer| layer.id == id))
        {
            self.active_layer = self
                .channel_layer_order
                .first()
                .copied()
                .map(MosaicLayerId::Channel)
                .unwrap_or(MosaicLayerId::TextLabels);
        }
    }

    pub fn from_control_resource(
        ctx: &egui::Context,
        gpu_available: bool,
        resource: &odon::model::ControlMosaicResource,
    ) -> anyhow::Result<Self> {
        apply_napari_like_dark(ctx);
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }
        if resource.items.is_empty() {
            anyhow::bail!("actor mosaic resource contains no items");
        }

        let mut items = Vec::with_capacity(resource.items.len());
        let mut stores = Vec::with_capacity(resource.items.len());
        let mut remote_runtimes = Vec::new();
        for item in resource.items.iter() {
            let dataset = item.document.resource.dataset().clone();
            stores.push(Arc::clone(item.document.resource.store()));
            if let Some(runtime) = item.document.resource.runtime_guard() {
                remote_runtimes.push(runtime);
            }
            items.push(MosaicItem {
                id: item.id,
                sample_id: item.roi_id.clone(),
                meta: item.metadata.clone(),
                dataset,
                offset: egui::vec2(0.0, 0.0),
                scale: 1.0,
                placed_size: egui::vec2(1.0, 1.0),
            });
        }

        let mut seg_geojson = MosaicGeoJsonSegmentationOverlay::default();
        seg_geojson.set_samplesheet_dir(resource.base_dir.clone());
        for item in resource.items.iter() {
            let mut metadata = item.metadata.clone();
            if let Some(path) = item.segmentation_path.as_ref() {
                metadata.insert("segpath".to_string(), path.to_string_lossy().into_owned());
            }
            seg_geojson.discover_from_meta(item.id, &metadata);
        }

        let abs_max = items
            .iter()
            .map(|item| item.dataset.abs_max)
            .fold(0.0_f32, f32::max)
            .max(1.0);
        let channels = build_global_channels(items.iter().map(|item| &item.dataset));
        let columns = resource
            .initial_columns
            .filter(|columns| *columns > 0)
            .unwrap_or_else(|| ((items.len() as f32).sqrt().ceil() as usize).max(1));
        let grid_pad = 64.0;
        let (cell_width, cell_height) = max_level0_size_items(&items);
        let cell_width = cell_width.max(1.0);
        let cell_height = cell_height.max(1.0);
        let (mosaic_bounds, group_blocks) = layout_items_grouped(
            &mut items,
            columns,
            cell_width,
            cell_height,
            grid_pad,
            None,
            0.0,
            MosaicLayoutMode::FitCells,
        );
        let sources = Arc::new(
            items
                .iter()
                .zip(stores.iter())
                .map(|(item, store)| MosaicSource {
                    source: item.dataset.source.clone(),
                    store: Arc::clone(store),
                    levels: item.dataset.levels.clone(),
                    dims: item.dataset.dims.clone(),
                    channel_map: build_channel_map(&channels, &item.dataset),
                })
                .collect::<Vec<_>>(),
        );
        let threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(8)
            .clamp(2, 16);
        let pinned_levels = MosaicPinnedLevels::new();
        let loader = spawn_mosaic_raw_tile_loader(
            Arc::clone(&sources),
            pinned_levels.clone(),
            threads,
            8192,
        )?;
        let mut camera = Camera::default();
        camera.center_world_lvl0 = mosaic_bounds.center();
        camera.zoom_screen_per_lvl0_px = 0.01;

        Ok(Self::from_prepared_construction(
            PreparedMosaicConstruction {
                items,
                sources,
                pinned_levels,
                loader,
                remote_runtimes,
                camera,
                mosaic_bounds,
                abs_max,
                channels,
                metadata_columns: resource.metadata_columns.as_ref().clone(),
                group_blocks,
                grid_cols: columns,
                renderer_status: "Ready.".to_string(),
                show_return_navigation: true,
                seg_geojson,
                consumed_mosaic_resource_generation: resource.generation,
            },
        ))
    }

    pub fn apply_control_actor_state(
        &mut self,
        state: &serde_json::Value,
        memory_state: &serde_json::Value,
        object_resources: &[(usize, Arc<odon::model::ControlObjectResource>)],
        pinned_levels: &[(usize, Arc<odon::model::ControlPinnedLevelResource>)],
    ) -> Result<(), String> {
        let generation = state
            .get("generation")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "actor mosaic projection has no generation".to_string())?;
        if generation != self.consumed_mosaic_resource_generation {
            return Err(format!(
                "actor mosaic generation {generation} does not match renderer generation {}",
                self.consumed_mosaic_resource_generation
            ));
        }

        if let Some(projected_items) = state.get("items").and_then(serde_json::Value::as_array) {
            let order = projected_items
                .iter()
                .enumerate()
                .filter_map(|(position, item)| {
                    item.get("id")
                        .and_then(serde_json::Value::as_u64)
                        .map(|id| (id as usize, position))
                })
                .collect::<HashMap<_, _>>();
            if order.len() != self.items.len()
                || self.items.iter().any(|item| !order.contains_key(&item.id))
            {
                return Err("actor mosaic item identity differs from renderer resource".to_string());
            }
            self.items.sort_by_key(|item| order[&item.id]);
            self.selected_core_ids.clear();
            self.focused_core_id = None;
            for projected in projected_items {
                let id = projected["id"].as_u64().unwrap_or_default() as usize;
                let item = self
                    .items
                    .iter_mut()
                    .find(|item| item.id == id)
                    .ok_or_else(|| format!("actor mosaic item {id} is missing"))?;
                if let Some(offset) = json_vec2(projected.get("offset_world")) {
                    item.offset = egui::vec2(offset[0], offset[1]);
                }
                if let Some(scale) = projected.get("scale").and_then(serde_json::Value::as_f64) {
                    item.scale = scale as f32;
                }
                if let Some(size) = json_vec2(projected.get("placed_size")) {
                    item.placed_size = egui::vec2(size[0], size[1]);
                }
                if projected["selected"].as_bool() == Some(true) {
                    self.selected_core_ids.insert(id);
                }
                if projected["focused"].as_bool() == Some(true) {
                    self.focused_core_id = Some(id);
                }
            }
        }
        if let Some(bounds) = state.get("bounds") {
            let min = json_vec2(bounds.get("min"));
            let max = json_vec2(bounds.get("max"));
            if let (Some(min), Some(max)) = (min, max) {
                self.mosaic_bounds = egui::Rect::from_min_max(
                    egui::pos2(min[0], min[1]),
                    egui::pos2(max[0], max[1]),
                );
            }
        }
        if let Some(camera) = state.get("camera") {
            if let Some(center) = json_vec2(camera.get("center_world_lvl0")) {
                self.camera.center_world_lvl0 = egui::pos2(center[0], center[1]);
            }
            if let Some(zoom) = camera
                .get("zoom_screen_per_lvl0_px")
                .and_then(serde_json::Value::as_f64)
            {
                self.camera.zoom_screen_per_lvl0_px = zoom as f32;
            }
        }
        if let Some(tab) = state
            .get("left_tab")
            .and_then(serde_json::Value::as_str)
            .and_then(LeftTab::from_storage_key)
        {
            self.left_tab = tab;
        }
        if let Some(tab) = state
            .get("right_tab")
            .and_then(serde_json::Value::as_str)
            .and_then(RightTab::from_storage_key)
        {
            self.right_tab = tab;
        }
        if let Some(layout) = state.get("layout") {
            if let Some(value) = layout.get("group_by").and_then(serde_json::Value::as_str) {
                self.group_by = value.to_string();
            }
            if let Some(value) = layout.get("sort_by").and_then(serde_json::Value::as_str) {
                self.sort_by = value.to_string();
            }
            if let Some(value) = layout
                .get("sort_secondary_enabled")
                .and_then(serde_json::Value::as_bool)
            {
                self.sort_secondary_enabled = value;
            }
            if let Some(value) = layout
                .get("sort_by_secondary")
                .and_then(serde_json::Value::as_str)
            {
                self.sort_by_secondary = value.to_string();
            }
            if let Some(value) = layout
                .get("layout")
                .and_then(serde_json::Value::as_str)
                .and_then(MosaicLayoutMode::from_storage_key)
            {
                self.layout_mode = value;
            }
            if let Some(value) = layout.get("columns").and_then(serde_json::Value::as_u64) {
                self.grid_cols = (value as usize).max(1);
            }
            if let Some(value) = layout.get("group_gap").and_then(serde_json::Value::as_f64) {
                self.group_gap = value as f32;
            }
            if let Some(value) = layout
                .get("show_group_labels")
                .and_then(serde_json::Value::as_bool)
            {
                self.show_group_labels = value;
            }
            if let Some(value) = layout
                .get("show_text_labels")
                .and_then(serde_json::Value::as_bool)
            {
                self.show_text_labels = value;
            }
            if let Some(values) = layout
                .get("label_columns")
                .and_then(serde_json::Value::as_array)
            {
                self.label_columns = values
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect();
            }
        }
        if let Some(panels) = state.get("panels") {
            if let Some(value) = panels.get("left").and_then(serde_json::Value::as_bool) {
                self.show_left_panel = value;
            }
            if let Some(value) = panels.get("right").and_then(serde_json::Value::as_bool) {
                self.show_right_panel = value;
            }
        }
        if let Some(value) = state
            .get("smooth_pixels")
            .and_then(serde_json::Value::as_bool)
        {
            self.smooth_pixels = value;
            self.tiles_gl.set_smooth_pixels(value);
        }
        if let Some(value) = state
            .get("show_tile_debug")
            .and_then(serde_json::Value::as_bool)
        {
            self.show_tile_debug = value;
        }
        if let Some(value) = state
            .get("objects_visible")
            .and_then(serde_json::Value::as_bool)
        {
            self.seg_geojson.visible = value;
        }
        if let Some(value) = state
            .get("fast_object_rendering")
            .and_then(serde_json::Value::as_bool)
        {
            self.seg_geojson.set_fast_object_rendering(value);
        }
        if let Some(style) = state.get("object_style") {
            self.seg_geojson.apply_control_style(style)?;
        }
        if let Some(channels) = state.get("channels").and_then(serde_json::Value::as_array) {
            for projected in channels {
                let Some(index) = projected
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .map(|index| index as usize)
                else {
                    continue;
                };
                let Some(channel) = self.channels.get_mut(index) else {
                    continue;
                };
                if let Some(value) = projected
                    .get("visible")
                    .and_then(serde_json::Value::as_bool)
                {
                    channel.visible = value;
                }
                if let Some(values) = projected
                    .get("color_rgb")
                    .and_then(serde_json::Value::as_array)
                    .filter(|values| values.len() == 3)
                {
                    channel.color_rgb = [
                        values[0].as_u64().unwrap_or(channel.color_rgb[0] as u64) as u8,
                        values[1].as_u64().unwrap_or(channel.color_rgb[1] as u64) as u8,
                        values[2].as_u64().unwrap_or(channel.color_rgb[2] as u64) as u8,
                    ];
                }
                if let Some(values) = projected
                    .get("window")
                    .and_then(serde_json::Value::as_array)
                    .filter(|values| values.len() == 2)
                {
                    channel.window = values[0]
                        .as_f64()
                        .zip(values[1].as_f64())
                        .map(|(minimum, maximum)| (minimum as f32, maximum as f32));
                }
                if let Some(value) = projected.get("note").and_then(serde_json::Value::as_str) {
                    channel.note = value.to_string();
                }
                if projected.get("active").and_then(serde_json::Value::as_bool) == Some(true) {
                    self.selected_channel = index;
                }
            }
        }
        if let Some(order) = state
            .get("channel_order")
            .and_then(serde_json::Value::as_array)
        {
            self.channel_layer_order = order
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|index| index as usize)
                .filter(|index| *index < self.channels.len())
                .collect();
        }
        if let Some(presentation) = state.get("channel_presentation") {
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
        if let Some(groups) = state.get("layer_groups") {
            self.layer_groups = serde_json::from_value(groups.clone())
                .map_err(|error| format!("invalid actor mosaic channel groups: {error}"))?;
        }
        if let Some(layers) = state
            .get("native_layers")
            .and_then(serde_json::Value::as_array)
        {
            let channel_order = layers
                .iter()
                .filter(|layer| layer["stack"].as_str() == Some("channels"))
                .filter_map(|layer| {
                    layer["layer_id"]
                        .as_str()?
                        .strip_prefix("channel:")?
                        .parse::<usize>()
                        .ok()
                })
                .collect::<Vec<_>>();
            if channel_order.len() == self.channels.len() {
                self.channel_layer_order = channel_order;
            }
            let overlay_order = layers
                .iter()
                .filter(|layer| layer["stack"].as_str() == Some("overlays"))
                .filter_map(|layer| self.parse_layer_id_storage_key(layer["layer_id"].as_str()?))
                .filter(|layer| self.layer_available(*layer))
                .collect::<Vec<_>>();
            if overlay_order.len() == self.overlay_layer_order.len() {
                self.overlay_layer_order = overlay_order;
            }
            for layer in layers {
                let Some(id) = layer["layer_id"]
                    .as_str()
                    .and_then(|id| self.parse_layer_id_storage_key(id))
                else {
                    continue;
                };
                if let Some(visible) = layer["visible"].as_bool() {
                    self.apply_layer_visibility_projection(id, visible);
                }
                if layer["active"].as_bool() == Some(true) && self.layer_available(id) {
                    self.apply_active_layer_projection(id);
                }
            }
        }

        let object_generation = state["objects"]["generation"].as_u64().unwrap_or(0);
        if object_generation > self.consumed_mosaic_object_generation {
            for (item_id, resource) in object_resources {
                self.seg_geojson
                    .install_control_resource(*item_id, resource.as_ref());
            }
            self.consumed_mosaic_object_generation = object_generation;
        }
        if let Some(selections) = state.get("object_selections") {
            self.seg_geojson.apply_control_selections(selections)?;
        }
        self.seg_geojson
            .reconcile_actor_load_state(&state["objects"]);
        self.control_actor_memory_state = memory_state.clone();
        self.pinned_levels.replace_control_resources(pinned_levels);
        self.tile_request_generation = self.tile_request_generation.wrapping_add(1).max(1);
        self.last_tile_request_signature = None;
        Ok(())
    }

    pub fn consumed_mosaic_resource_generation(&self) -> u64 {
        self.consumed_mosaic_resource_generation
    }

    pub fn control_actor_signature(&self) -> String {
        let sources = self
            .items
            .iter()
            .map(|item| item.dataset.source.source_key())
            .collect::<Vec<_>>()
            .join("|");
        format!("mosaic:{sources}")
    }

    pub fn control_actor_resource(&self) -> odon::model::ControlMosaicResource {
        let generation = self.consumed_mosaic_resource_generation.max(1);
        let runtime_by_item = (self._remote_runtimes.len() == self.items.len())
            .then_some(self._remote_runtimes.as_slice());
        let items = self
            .items
            .iter()
            .map(|item| {
                let store = self
                    .sources
                    .get(item.id)
                    .map(|source| Arc::clone(&source.store))
                    .expect("mosaic item has a matching source");
                let document = odon::data::document::OpenedDocument {
                    descriptor: odon::data::document::DocumentDescriptor::from_ome_zarr(
                        &item.dataset,
                    ),
                    resource: odon::data::document::DocumentResource::OmeZarr(
                        odon::data::document::OmeZarrDocumentResource {
                            dataset: item.dataset.clone(),
                            store,
                            runtime_guard: runtime_by_item
                                .and_then(|runtimes| runtimes.get(item.id))
                                .cloned(),
                        },
                    ),
                };
                odon::model::ControlMosaicItemResource {
                    id: item.id,
                    roi_id: item.sample_id.clone(),
                    metadata: item.meta.clone(),
                    document,
                    segmentation_path: self.seg_geojson.segmentation_path(item.id),
                }
            })
            .collect::<Vec<_>>();
        odon::model::ControlMosaicResource {
            generation,
            source: self
                .project_space
                .saved_project_path()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_else(|| "native mosaic".to_string()),
            base_dir: self.seg_geojson.samplesheet_dir(),
            initial_columns: Some(self.grid_cols),
            metadata_columns: Arc::new(self.metadata_columns.clone()),
            items: Arc::new(items),
        }
    }
}
