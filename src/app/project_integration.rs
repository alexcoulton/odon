use super::*;

impl OmeZarrViewerApp {
    pub fn set_project_object_cache_ui_state(&mut self, state: ProjectObjectCacheUiState) {
        self.project_space.set_object_cache_ui_state(state);
    }

    pub fn is_viewing_project_roi(&self, roi: &ProjectRoi) -> bool {
        let Some(source) = roi.dataset_source() else {
            return false;
        };
        match (source, &self.dataset.source) {
            (
                crate::data::dataset_source::DatasetSource::Local(path),
                crate::data::dataset_source::DatasetSource::Local(active),
            ) => path == *active || path.to_string_lossy() == active.to_string_lossy(),
            (source, active) => source == *active,
        }
    }

    pub fn control_current_project_view_spec(&mut self) -> ProjectViewSpec {
        let display = self.seg_objects.project_display_state();
        let analysis = self.seg_objects.project_analysis_state();
        let (cell_color_by, visible_cell_types, hidden_cell_types) = self
            .seg_objects
            .active_color_value_visibility_snapshot()
            .map(|(property_key, visible_values, hidden_values)| {
                let visible_values = if hidden_values.is_empty() {
                    Vec::new()
                } else {
                    visible_values
                };
                (Some(property_key), visible_values, hidden_values)
            })
            .unwrap_or_else(|| (display.color_property_key.clone(), Vec::new(), Vec::new()));
        let uses_object_segmentation =
            self.seg_objects.object_count() > 0 || cell_color_by.is_some() || display.fill_cells;
        let channel_ref =
            self.channels
                .get(self.selected_channel)
                .map(|channel| ProjectViewChannelRef {
                    label: channel.name.clone(),
                    alias: suggest_channel_alias(&channel.name),
                });
        let visible_channel_refs = self
            .channels
            .iter()
            .filter(|channel| channel.visible)
            .map(|channel| ProjectViewChannelRef {
                label: channel.name.clone(),
                alias: suggest_channel_alias(&channel.name),
            })
            .collect::<Vec<_>>();

        ProjectViewSpec {
            channel: None,
            channel_ref,
            visible_channels: Vec::new(),
            visible_channel_refs,
            hidden_channels: Vec::new(),
            segmentation_source: uses_object_segmentation.then(|| "geoparquet".to_string()),
            load_labels: uses_object_segmentation.then_some(false),
            cell_color_by,
            visible_cell_types,
            hidden_cell_types,
            fill_cells: uses_object_segmentation.then_some(display.fill_cells),
            show_selection_overlay: uses_object_segmentation
                .then_some(analysis.show_selection_overlay),
            camera: Some(self.project_camera_state()),
        }
    }

    pub(super) fn handle_project_space_action(&mut self, action: ProjectSpaceAction) {
        if self.project_space.submit_action_control_intent(&action) {
            return;
        }
        match action {
            ProjectSpaceAction::CaptureCurrentView => {
                let spec = self.control_current_project_view_spec();
                self.project_space.set_view_preset_draft(spec);
            }
            ProjectSpaceAction::OpenRemoteDialog => {
                self.pending_request = Some(ViewerRequest::OpenRemoteDialog);
            }
            ProjectSpaceAction::ShowHelp(topic) => {
                self.active_help_topic = Some(topic);
            }
            _ => unreachable!("actor-owned project action was not accepted by its command outbox"),
        }
    }

    pub(super) fn ui_help_heading(
        &mut self,
        ui: &mut egui::Ui,
        title: &str,
        topic: crate::ui::help::HelpTopic,
    ) {
        ui.horizontal(|ui| {
            ui.heading(title);
            if crate::ui::help::help_button(ui, topic) {
                self.active_help_topic = Some(topic);
            }
        });
    }

    #[cfg(test)]
    pub fn set_project_space(&mut self, project_space: ProjectSpace) {
        self.project_space = project_space;
        self.apply_view_state_from_project_space();
        self.control_actor_project_config_generation = u64::MAX;
        self.restore_mask_layers_from_project_space();
        self.restore_loaded_layer_offsets_from_current_project_view_or_capture();
        self.auto_load_project_roi_segmentation();
    }

    /// Install the actor's projected project shell without starting renderer-owned resource
    /// loads. The same render projection immediately installs saved workspace, masks, labels, and
    /// objects from actor-owned immutable handles.
    pub fn set_project_space_from_actor(&mut self, project_space: ProjectSpace) {
        self.project_space = project_space;
        self.control_actor_project_config_generation = u64::MAX;
    }

    pub fn set_remote_runtime(&mut self, runtime: Option<Arc<tokio::runtime::Runtime>>) {
        self.remote_runtime = runtime;
    }

    pub fn attach_spatialdata_layers(
        &mut self,
        spatial_root: PathBuf,
        image_transform: SpatialDataTransform2,
        extra_images: Vec<SpatialDataElement>,
        labels: Option<SpatialDataElement>,
        tables: Vec<SpatialDataElement>,
        shapes: Vec<SpatialDataElement>,
        points: Option<(SpatialDataElement, usize)>,
    ) {
        self.spatial_image_layers.clear();
        self.spatial_layers.clear();
        self.xenium_layers.clear();
        self.spatial_layers.set_root(spatial_root.clone());
        self.spatial_layers.set_tables(tables);
        self.spatial_image_transform = image_transform;
        self.spatial_label_transform = labels
            .as_ref()
            .map(|l| l.transform.relative_to(image_transform))
            .unwrap_or_default();
        self.spatial_root = Some(spatial_root.clone());
        self.spatial_label_store = zarrs::filesystem::FilesystemStore::new(&spatial_root)
            .ok()
            .map(|s| Arc::new(s) as Arc<dyn zarrs::storage::ReadableStorageTraits>);
        self.xenium_cells_offset_world = egui::Vec2::ZERO;
        self.xenium_transcripts_offset_world = egui::Vec2::ZERO;
        self.spatial_points_offset_world = egui::Vec2::ZERO;
        self.seg_objects_offset_world = egui::Vec2::ZERO;
        self.seg_objects.clear();
        self.label_cells = None;
        self.label_loader = None;
        self.label_cells_xform = None;
        self.seg_label_names.clear();
        self.seg_label_selected.clear();
        self.seg_label_input = self.seg_label_selected.clone();
        self.seg_label_prompt_open = false;

        for image in &extra_images {
            let mut image = image.clone();
            image.transform = image.transform.relative_to(image_transform);
            if let Err(err) = self.spatial_image_layers.load_image(
                &spatial_root,
                &image,
                self.tiles_gl.is_some(),
                self.smooth_pixels,
            ) {
                eprintln!(
                    "failed to load SpatialData image layer {}: {err}",
                    image.name
                );
            }
        }

        for sh in &shapes {
            let mut sh = sh.clone();
            sh.transform = sh.transform.relative_to(image_transform);
            if sh.name == "cell_boundaries" {
                if let Some(rel) = sh.rel_parquet.as_ref() {
                    self.seg_objects.load_spatialdata_shapes(
                        spatial_root.join(rel),
                        sh.transform,
                        sh.name.as_str(),
                    );
                }
            } else {
                let id = self.spatial_layers.load_shapes(&sh);
                if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                    && let Some(objects) = layer.object_layer_mut()
                {
                    objects.fast_rendering = self.fast_object_rendering;
                }
            }
        }
        if let Some((pt, max_points)) = points.as_ref() {
            let mut pt = pt.clone();
            pt.transform = pt.transform.relative_to(image_transform);
            let shape0 = self.dataset.levels.get(0).map(|l| l.shape.clone());
            let image_size = shape0.and_then(|s| {
                let x = s.get(self.dataset.dims.x).copied()? as f32;
                let y = s.get(self.dataset.dims.y).copied()? as f32;
                Some([x, y])
            });
            self.spatial_layers
                .load_points_with_image_size(&pt, *max_points, image_size);
        }
        self.rebuild_layer_orders();
        self.bump_render_id();
    }

    pub fn attach_prepared_spatialdata_layers(
        &mut self,
        spatial_root: PathBuf,
        image_transform: SpatialDataTransform2,
        extra_images: Vec<crate::spatialdata::PreparedSpatialImage>,
        labels: Option<SpatialDataElement>,
        tables: Vec<SpatialDataElement>,
        shapes: Vec<SpatialDataElement>,
        points: Option<(SpatialDataElement, usize)>,
    ) {
        self.attach_spatialdata_layers(
            spatial_root,
            image_transform,
            Vec::new(),
            labels,
            tables,
            shapes,
            points,
        );
        for mut image in extra_images {
            image.element.transform = image.element.transform.relative_to(image_transform);
            if let Err(error) = self.spatial_image_layers.load_prepared_image(
                image,
                self.tiles_gl.is_some(),
                self.smooth_pixels,
            ) {
                eprintln!("failed to realize prepared SpatialData image layer: {error}");
            }
        }
        self.rebuild_layer_orders();
        self.bump_render_id();
    }

    pub fn sync_control_external_layers(
        &mut self,
        layers: &[LayerSnapshot],
        resources: &[DataResourceSnapshot],
    ) {
        let transform_for = |layer: &LayerSnapshot| {
            let Some(resource) = resources
                .iter()
                .find(|resource| resource.resource_id == layer.data_resource_id)
            else {
                return (egui::Vec2::ONE, egui::Vec2::ZERO);
            };
            let coordinate = &resource.coordinate_space;
            let axis_value = |values: &[f64], axis: &str, default: f32| {
                coordinate
                    .axes
                    .iter()
                    .position(|name| name.eq_ignore_ascii_case(axis))
                    .and_then(|index| values.get(index))
                    .copied()
                    .map(|value| value as f32)
                    .unwrap_or(default)
            };
            (
                egui::vec2(
                    axis_value(&coordinate.scale, "x", 1.0),
                    axis_value(&coordinate.scale, "y", 1.0),
                ),
                egui::vec2(
                    axis_value(&coordinate.translation, "x", 0.0),
                    axis_value(&coordinate.translation, "y", 0.0),
                ),
            )
        };
        let apply_style = |loaded: &mut crate::spatialdata::SpatialImageLayer,
                           layer: &LayerSnapshot,
                           resource: &DataResourceSnapshot| {
            let default_color = if layer.kind == "labels" {
                Some([255, 190, 40])
            } else {
                None
            };
            let style_color = layer
                .style
                .get("color_rgb")
                .and_then(serde_json::Value::as_array)
                .filter(|values| values.len() == 3)
                .and_then(|values| {
                    Some([
                        u8::try_from(values[0].as_u64()?).ok()?,
                        u8::try_from(values[1].as_u64()?).ok()?,
                        u8::try_from(values[2].as_u64()?).ok()?,
                    ])
                })
                .or(default_color);
            let style_window = layer.style.get("contrast").and_then(|value| {
                Some((
                    value.get("min")?.as_f64()? as f32,
                    value.get("max")?.as_f64()? as f32,
                ))
            });
            let resource_window = resource
                .metadata
                .get("value_max")
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|maximum| (0.0, maximum as f32));
            for channel in &mut loaded.channels {
                if let Some(color) = style_color {
                    channel.color_rgb = color;
                }
                if let Some(window) = style_window.or(resource_window) {
                    channel.window = Some(window);
                }
            }
        };
        let wanted = layers
            .iter()
            .filter(|layer| matches!(layer.kind.as_str(), "image" | "labels"))
            .map(|layer| layer.layer_id.as_str())
            .collect::<HashSet<_>>();
        self.spatial_image_layers.images.retain(|layer| {
            layer
                .external_id
                .as_deref()
                .is_none_or(|external_id| wanted.contains(external_id))
        });

        for layer in layers
            .iter()
            .filter(|layer| matches!(layer.kind.as_str(), "image" | "labels"))
        {
            let Some(resource) = resources
                .iter()
                .find(|resource| resource.resource_id == layer.data_resource_id)
            else {
                continue;
            };
            if let Some(existing_index) = self
                .spatial_image_layers
                .images
                .iter()
                .position(|existing| existing.external_id.as_deref() == Some(&layer.layer_id))
            {
                let existing = &mut self.spatial_image_layers.images[existing_index];
                if existing.external_resource_id.as_deref() == Some(&layer.data_resource_id) {
                    existing.name = layer.name.clone();
                    existing.visible = layer.visible;
                    existing.opacity = layer.opacity as f32;
                    (existing.scale_world, existing.offset_world) = transform_for(layer);
                    apply_style(existing, layer, resource);
                    continue;
                }
                self.spatial_image_layers.images.remove(existing_index);
            }
            let Ok(url) = url::Url::parse(&resource.uri) else {
                continue;
            };
            let Ok(path) = url.to_file_path() else {
                continue;
            };
            match self.spatial_image_layers.load_external_image(
                layer.layer_id.clone(),
                layer.data_resource_id.clone(),
                layer.name.clone(),
                path,
                self.tiles_gl.is_some(),
                self.smooth_pixels,
            ) {
                Ok(id) => {
                    if let Some(loaded) = self
                        .spatial_image_layers
                        .images
                        .iter_mut()
                        .find(|loaded| loaded.id == id)
                    {
                        loaded.visible = layer.visible;
                        loaded.opacity = layer.opacity as f32;
                        (loaded.scale_world, loaded.offset_world) = transform_for(layer);
                        apply_style(loaded, layer, resource);
                    }
                }
                Err(error) => {
                    crate::log_warn!(
                        "failed to attach external Odon layer {}: {}",
                        layer.layer_id,
                        error
                    );
                }
            }
        }

        let spatial_transform_for = |resource: &DataResourceSnapshot| {
            let coordinate = &resource.coordinate_space;
            let axis_value = |values: &[f64], axis: &str, default: f32| {
                coordinate
                    .axes
                    .iter()
                    .position(|name| name.eq_ignore_ascii_case(axis))
                    .and_then(|index| values.get(index))
                    .copied()
                    .map(|value| value as f32)
                    .unwrap_or(default)
            };
            crate::spatialdata::SpatialDataTransform2 {
                scale: [
                    axis_value(&coordinate.scale, "x", 1.0),
                    axis_value(&coordinate.scale, "y", 1.0),
                ],
                translation: [
                    axis_value(&coordinate.translation, "x", 0.0),
                    axis_value(&coordinate.translation, "y", 0.0),
                ],
            }
        };
        let apply_shape_style = |loaded: &mut crate::spatialdata::SpatialShapesLayer,
                                 layer: &LayerSnapshot| {
            loaded.name = layer.name.clone();
            *loaded.visible_mut() = layer.visible;
            loaded.opacity = layer.opacity as f32;
            if let Some(objects) = loaded.object_layer_mut() {
                objects.opacity = layer.opacity as f32;
            }
            if let Some(width) = layer
                .style
                .get("width")
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
            {
                loaded.width_screen_px = width as f32;
                if let Some(objects) = loaded.object_layer_mut() {
                    objects.width_screen_px = width as f32;
                }
            }
            if let Some(color) = layer
                .style
                .get("color_rgb")
                .and_then(serde_json::Value::as_array)
                .filter(|values| values.len() == 3)
                .and_then(|values| {
                    Some([
                        u8::try_from(values[0].as_u64()?).ok()?,
                        u8::try_from(values[1].as_u64()?).ok()?,
                        u8::try_from(values[2].as_u64()?).ok()?,
                    ])
                })
            {
                loaded.color_rgb = color;
                if let Some(objects) = loaded.object_layer_mut() {
                    objects.color_rgb = color;
                }
            }
        };
        let wanted_shapes = layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.kind.as_str(),
                    "objects" | "points" | "shapes" | "mask" | "annotations"
                )
            })
            .filter_map(|layer| {
                resources
                    .iter()
                    .find(|resource| resource.resource_id == layer.data_resource_id)
                    .filter(|resource| matches!(resource.format.as_str(), "parquet" | "geoparquet"))
                    .map(|_| layer.layer_id.as_str())
            })
            .collect::<HashSet<_>>();
        self.spatial_layers.shapes.retain(|layer| {
            layer
                .external_id
                .as_deref()
                .is_none_or(|external_id| wanted_shapes.contains(external_id))
        });
        for layer in layers.iter().filter(|layer| {
            matches!(
                layer.kind.as_str(),
                "objects" | "points" | "shapes" | "mask" | "annotations"
            )
        }) {
            let Some(resource) = resources.iter().find(|resource| {
                resource.resource_id == layer.data_resource_id
                    && matches!(resource.format.as_str(), "parquet" | "geoparquet")
            }) else {
                continue;
            };
            let transform = spatial_transform_for(resource);
            if let Some(existing_index) = self
                .spatial_layers
                .shapes
                .iter()
                .position(|existing| existing.external_id.as_deref() == Some(&layer.layer_id))
            {
                let existing = &mut self.spatial_layers.shapes[existing_index];
                if existing.external_resource_id.as_deref() == Some(&layer.data_resource_id)
                    && existing.transform == transform
                {
                    apply_shape_style(existing, layer);
                    continue;
                }
                self.spatial_layers.shapes.remove(existing_index);
            }
            let Ok(url) = url::Url::parse(&resource.uri) else {
                continue;
            };
            let Ok(path) = url.to_file_path() else {
                continue;
            };
            let id = self.spatial_layers.load_external_shapes(
                layer.layer_id.clone(),
                layer.data_resource_id.clone(),
                layer.name.clone(),
                path,
                transform,
            );
            if let Some(loaded) = self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|loaded| loaded.id == id)
            {
                apply_shape_style(loaded, layer);
            }
        }
        self.rebuild_layer_orders();
    }

    pub fn attach_prepared_xenium_layers(
        &mut self,
        dataset_root: PathBuf,
        cells: Option<crate::xenium::PreparedXeniumCells>,
        transcripts: Option<crate::xenium::PreparedXeniumTranscripts>,
        pixel_size_um: f32,
    ) {
        self.xenium_layers.clear();
        self.xenium_layers
            .attach_prepared(dataset_root, cells, transcripts, pixel_size_um);
        self.xenium_cells_offset_world = egui::Vec2::ZERO;
        self.xenium_transcripts_offset_world = egui::Vec2::ZERO;
        self.bump_render_id();
    }

    pub fn current_local_dataset_root(&self) -> Option<PathBuf> {
        self.dataset.source.local_path().map(|p| p.to_path_buf())
    }

    pub(super) fn current_project_roi(&self) -> Option<&ProjectRoi> {
        let source_key = self.dataset.source.source_key();
        self.project_space
            .rois()
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(source_key.as_str()))
    }

    pub(super) fn current_dataset_source_display(&self) -> String {
        match &self.dataset.source {
            crate::data::dataset_source::DatasetSource::Local(path) => {
                path.to_string_lossy().to_string()
            }
            crate::data::dataset_source::DatasetSource::Http { base_url } => base_url.clone(),
            crate::data::dataset_source::DatasetSource::S3 { bucket, prefix, .. } => {
                if prefix.trim().is_empty() {
                    format!("s3://{bucket}")
                } else {
                    format!("s3://{bucket}/{}", prefix.trim_matches('/'))
                }
            }
        }
    }

    pub(super) fn current_project_path_display(&self) -> String {
        self.project_space
            .current_project_path()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_else(|| "<unsaved project>".to_string())
    }

    pub(super) fn current_roi_compact_label(&self) -> String {
        self.current_project_roi()
            .and_then(|roi| {
                roi.display_name
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .map(str::to_string)
                    .or_else(|| (!roi.id.trim().is_empty()).then(|| roi.id.clone()))
            })
            .unwrap_or_else(|| self.dataset.source.display_name())
    }

    pub(super) fn resolved_project_roi_segpath(&self, roi: &ProjectRoi) -> Option<PathBuf> {
        let segpath = roi.segpath.as_ref()?.clone();
        if segpath.is_absolute() {
            return Some(segpath);
        }
        self.project_space
            .project_dir()
            .map(|dir| dir.join(&segpath))
            .or(Some(segpath))
    }

    pub(super) fn current_roi_hover_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("ROI: {}", self.current_roi_compact_label()));
        if let Some(roi) = self.current_project_roi() {
            if !roi.id.trim().is_empty() {
                lines.push(format!("ID: {}", roi.id));
            }
            if let Some(dataset) = roi
                .dataset
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                lines.push(format!("Dataset: {dataset}"));
            }
            lines.push(format!("Image: {}", roi.source_display()));
            lines.push(format!(
                "Segmentation: {}",
                self.resolved_project_roi_segpath(roi)
                    .map(|path| path.to_string_lossy().to_string())
                    .unwrap_or_else(|| "<none>".to_string())
            ));
        } else {
            lines.push(format!("Image: {}", self.current_dataset_source_display()));
            lines.push(
                "Current dataset is not matched to an ROI entry in the Project panel.".to_string(),
            );
        }
        lines.push(format!("Project: {}", self.current_project_path_display()));
        lines.join("\n")
    }

    pub(super) fn ui_current_roi_field(ui: &mut egui::Ui, label: &str, value: &str) {
        ui.label(label);
        ui.add(egui::Label::new(egui::RichText::new(value).monospace()).wrap());
        ui.end_row();
    }

    pub(super) fn ui_current_roi_summary(&self, ui: &mut egui::Ui) {
        ui.label("Current ROI");
        egui::Grid::new("current-roi-summary-grid")
            .num_columns(2)
            .spacing([12.0, 4.0])
            .show(ui, |ui| {
                Self::ui_current_roi_field(ui, "Label", &self.current_roi_compact_label());
                if let Some(roi) = self.current_project_roi() {
                    if !roi.id.trim().is_empty() {
                        Self::ui_current_roi_field(ui, "ID", &roi.id);
                    }
                    if let Some(dataset) = roi
                        .dataset
                        .as_deref()
                        .filter(|value| !value.trim().is_empty())
                    {
                        Self::ui_current_roi_field(ui, "Dataset", dataset);
                    }
                    Self::ui_current_roi_field(ui, "Image", &roi.source_display());
                    Self::ui_current_roi_field(
                        ui,
                        "Segmentation",
                        &self
                            .resolved_project_roi_segpath(roi)
                            .map(|path| path.to_string_lossy().to_string())
                            .unwrap_or_else(|| "<none>".to_string()),
                    );
                } else {
                    Self::ui_current_roi_field(ui, "Image", &self.current_dataset_source_display());
                    Self::ui_current_roi_field(ui, "Project ROI", "<not matched>");
                }
                Self::ui_current_roi_field(ui, "Project", &self.current_project_path_display());
            });

        if let Some(roi) = self.current_project_roi() {
            if roi.meta.is_empty() {
                ui.label("Metadata: none");
            } else {
                let title = format!("Metadata ({})", roi.meta.len());
                ui.collapsing(title, |ui| {
                    let mut meta_keys = roi.meta.keys().cloned().collect::<Vec<_>>();
                    meta_keys.sort();
                    egui::Grid::new("current-roi-meta-grid")
                        .num_columns(2)
                        .spacing([12.0, 4.0])
                        .show(ui, |ui| {
                            for key in meta_keys {
                                let value = roi.meta.get(&key).cloned().unwrap_or_default();
                                Self::ui_current_roi_field(ui, &key, &value);
                            }
                        });
                });
            }
        }
    }

    pub(super) fn spatial_label_transform_for_name(
        &self,
        label_name: &str,
    ) -> SpatialDataTransform2 {
        let Some(root) = self.spatial_root.as_ref() else {
            return self.spatial_label_transform;
        };
        let Ok(discovery) = discover_spatialdata(root) else {
            return self.spatial_label_transform;
        };
        discovery
            .labels
            .iter()
            .find(|label| label.name == label_name)
            .map(|label| label.transform.relative_to(self.spatial_image_transform))
            .unwrap_or(self.spatial_label_transform)
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.roi_selector_ui.set_status(status);
    }

    pub fn open_mapping_settings(&mut self) {
        let opened = if self.seg_objects.has_data() {
            self.seg_objects.open_analysis_channel_mapping_popup();
            true
        } else if let LayerId::SpatialShape(id) = self.active_layer {
            if let Some(layer) = self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|shape| shape.id == id)
                && let Some(objects) = layer.object_layer_mut()
                && objects.has_data()
            {
                objects.open_analysis_channel_mapping_popup();
                true
            } else {
                false
            }
        } else {
            false
        };

        if opened {
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.panels.set",
                params: serde_json::json!({
                    "left":self.show_left_panel,
                    "right":true,
                }),
            });
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.ui.set_right_tab",
                params: serde_json::json!({"tab":"analysis"}),
            });
        } else {
            self.set_status(
                "Mapping settings are available for segmentation objects and object-backed SpatialData shape layers.",
            );
        }
    }

    pub(super) fn ui_mapping_settings_dialogs(&mut self, ctx: &egui::Context) {
        self.seg_objects.ui_analysis_channel_mapping_popup(
            ctx,
            &self.channels,
            self.selected_channel,
        );
        for layer in &mut self.spatial_layers.shapes {
            if let Some(objects) = layer.object_layer_mut() {
                objects.ui_analysis_channel_mapping_popup(
                    ctx,
                    &self.channels,
                    self.selected_channel,
                );
            }
        }
    }

    pub(super) fn ui_object_export_dialogs(&mut self, ctx: &egui::Context) {
        if let Some(intent) = self
            .seg_objects
            .ui_export_dialog(ctx, self.control_actor_object_export_generation > 0)
        {
            self.native_control_intents.push(NativeControlIntent {
                method: intent.method,
                params: intent.params,
            });
        }
        for layer in &mut self.spatial_layers.shapes {
            if let Some(objects) = layer.object_layer_mut() {
                let _ = objects.ui_export_dialog(ctx, false);
            }
        }
    }
}
