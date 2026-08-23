use super::*;

impl OmeZarrViewerApp {
    pub(super) fn step_selected_channel_visibility(&mut self, step: i32) {
        if self.channels.is_empty() || self.channel_layer_order.is_empty() {
            return;
        }
        let cur_idx = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let cur_pos = self
            .channel_layer_order
            .iter()
            .position(|&idx| idx == cur_idx)
            .unwrap_or(0);
        let n = self.channel_layer_order.len() as i32;
        let next_pos = ((cur_pos as i32) + step).rem_euclid(n) as usize;
        let next_idx =
            self.channel_layer_order[next_pos].min(self.channels.len().saturating_sub(1));

        if self.native_viewport_actor_owned() {
            let mut state = self.control_native_layer_snapshot_list();
            for layer in state.as_array_mut().into_iter().flatten() {
                let Some(index) = layer
                    .get("layer_id")
                    .and_then(serde_json::Value::as_str)
                    .and_then(|id| id.strip_prefix("channel:"))
                    .and_then(|index| index.parse::<usize>().ok())
                else {
                    layer["active"] = serde_json::json!(false);
                    continue;
                };
                if index == cur_idx {
                    layer["visible"] = serde_json::json!(false);
                    layer["presentation"]["visible"] = serde_json::json!(false);
                }
                if index == next_idx {
                    layer["visible"] = serde_json::json!(true);
                    layer["presentation"]["visible"] = serde_json::json!(true);
                }
                layer["active"] = serde_json::json!(index == next_idx);
            }
            self.submit_native_layer_state_replace(state);
            if let Some(ch) = self.channels.get(next_idx) {
                let _ = self.cell_thresholds.sync_marker_from_channel_name(&ch.name);
            }
            return;
        }

        if let Some(cur) = self.channels.get_mut(cur_idx) {
            cur.visible = false;
        }
        if let Some(next) = self.channels.get_mut(next_idx) {
            next.visible = true;
        }

        self.selected_channel = next_idx;
        self.active_layer = LayerId::Channel(next_idx);
        self.hist_dirty = true;
        self.bump_render_id();

        if let Some(ch) = self.channels.get(next_idx) {
            let _ = self.cell_thresholds.sync_marker_from_channel_name(&ch.name);
        }
    }

    pub(super) fn sync_analysis_follow_active_channel_state(&mut self) {
        self.seg_objects
            .ensure_object_property_analysis_warmup_started(&self.channels, self.selected_channel);
        self.seg_objects
            .sync_analysis_follow_active_channel(&self.channels, self.selected_channel);

        let active_shape_id = match self.active_layer {
            LayerId::SpatialShape(id) => Some(id),
            _ => None,
        };
        if let Some(id) = active_shape_id
            && let Some(layer) = self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|shape| shape.id == id)
            && let Some(objects) = layer.object_layer_mut()
        {
            objects.ensure_object_property_analysis_warmup_started(
                &self.channels,
                self.selected_channel,
            );
            objects.sync_analysis_follow_active_channel(&self.channels, self.selected_channel);
        }
    }

    pub(super) fn auto_load_project_roi_segmentation(&mut self) {
        if self.seg_objects.loaded_geojson.is_some()
            || self.seg_geojson.loaded_geojson.is_some()
            || self.seg_objects.is_loading()
            || self.seg_geojson.is_busy()
        {
            return;
        }

        let Some(roi) = self
            .project_space
            .config()
            .rois
            .iter()
            .find(|roi| match (roi.dataset_source(), &self.dataset.source) {
                (
                    Some(crate::data::dataset_source::DatasetSource::Local(path)),
                    crate::data::dataset_source::DatasetSource::Local(active),
                ) => path == *active || path.to_string_lossy() == active.to_string_lossy(),
                (Some(source), active) => source == *active,
                (None, _) => false,
            })
            .cloned()
        else {
            return;
        };

        let Some(segpath) = roi.segpath else {
            return;
        };

        let segpath = if segpath.is_relative() {
            self.project_space
                .project_dir()
                .map(|dir| dir.join(&segpath))
                .unwrap_or(segpath)
        } else {
            segpath
        };

        let Some(ext) = segpath.extension().and_then(|s| s.to_str()) else {
            self.roi_selector.set_status(format!(
                "Project segmentation path has no supported extension: {}",
                segpath.to_string_lossy()
            ));
            return;
        };

        if !segpath.exists() {
            self.roi_selector.set_status(format!(
                "Project segmentation path was not found: {}",
                segpath.to_string_lossy()
            ));
            return;
        }

        match ext.to_ascii_lowercase().as_str() {
            "geojson" | "json" | "geoparquet" | "parquet" => {
                self.seg_objects
                    .load_path(segpath.clone(), self.seg_objects.downsample_factor);
                self.set_active_layer(LayerId::SegmentationObjects);
                self.roi_selector.set_status(format!(
                    "Loading segmentation: {}",
                    segpath
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("segmentation")
                ));
            }
            _ => {
                self.roi_selector.set_status(format!(
                    "Project segmentation format is not supported for single view: {}",
                    segpath.to_string_lossy()
                ));
            }
        }
    }

    pub(super) fn handle_roi_selector_action(
        &mut self,
        _ctx: &egui::Context,
        action: RoiSelectorAction,
    ) {
        match action {
            RoiSelectorAction::OpenRoi(roi) => {
                if roi.dataset_source().is_none() {
                    self.roi_selector
                        .set_status("Open ROI failed: ROI has no dataset source.".to_string());
                    return;
                }
                assert!(
                    self.project_space
                        .submit_action_control_intent(&ProjectSpaceAction::Open(roi)),
                    "actor-owned ROI action was not accepted by its command outbox"
                );
            }
            RoiSelectorAction::LoadLabels => {
                if let Err(err) = self.ensure_segmentation_labels_loaded() {
                    self.roi_selector
                        .set_status(format!("Load Labels failed: {err}"));
                } else {
                    self.roi_selector.set_status(format!(
                        "Loaded labels/{}.",
                        self.seg_label_selected.as_str()
                    ));
                }
            }
            RoiSelectorAction::LoadMasks => match self.request_exclusion_masks_reload() {
                Ok(n) => {
                    self.roi_selector
                        .set_status(format!("Loaded masks ({n} shapes)."));
                }
                Err(err) => {
                    self.roi_selector
                        .set_status(format!("Load Masks failed: {err}"));
                }
            },
            RoiSelectorAction::SaveMasks => {
                let Some(local_root) = self.dataset.source.local_path() else {
                    self.roi_selector
                        .set_status("Save Masks is supported for local datasets only.".to_string());
                    return;
                };
                if !self.drawing_mask_polygon.is_empty() {
                    self.roi_selector.set_status(
                        "Finish polygon (Enter/double-click) or cancel (Esc) before saving."
                            .to_string(),
                    );
                    return;
                }

                // Legacy "Save Masks" appends editable (non-file-backed) mask polygons to the
                // napari-style masks file path inferred from the Project config.
                let mut polys: Vec<Vec<egui::Pos2>> = Vec::new();
                for l in &self.mask_layers {
                    if !l.editable || l.source_geojson.is_some() {
                        continue;
                    }
                    for poly in &l.polygons_world {
                        if poly.len() < 3 {
                            continue;
                        }
                        polys.push(
                            poly.iter()
                                .copied()
                                .map(|p| p + l.offset_world)
                                .collect::<Vec<_>>(),
                        );
                    }
                }
                if polys.is_empty() {
                    self.roi_selector
                        .set_status("No drawn masks to save.".to_string());
                    return;
                }

                let Some(cfg) = self.roi_selector.masks_config_for_roi(local_root) else {
                    self.roi_selector.set_status(
                        "Save Masks failed: no matching dataset in Project config.".to_string(),
                    );
                    return;
                };
                let entry = self.roi_selector.roi_entry_for_path(local_root);

                match resolve_masks_geojson_path_and_downsample(local_root, &cfg, entry.as_ref()) {
                    Ok(resolved) => {
                        if self.mask_actor_owned() {
                            let mut params = serde_json::Map::new();
                            params.insert(
                                "path".to_string(),
                                serde_json::json!(resolved.geojson_path.to_string_lossy()),
                            );
                            params.insert("name".to_string(), serde_json::json!("Exclusion masks"));
                            params.insert(
                                "downsample_factor".to_string(),
                                serde_json::json!(resolved.downsample_factor),
                            );
                            params.insert(
                                "roi_root".to_string(),
                                serde_json::json!(local_root.to_string_lossy()),
                            );
                            self.submit_native_mask_command(
                                "viewer.masks.persistence.append_geojson",
                                params,
                            );
                            self.roi_selector.set_status(format!(
                                "Saving drawn masks -> {}",
                                resolved.geojson_path.to_string_lossy()
                            ));
                            return;
                        }
                        let result: anyhow::Result<()> = (|| {
                            let ds = resolved.downsample_factor.max(1e-6);
                            let path = &resolved.geojson_path;
                            if let Some(parent) = path.parent() {
                                fs::create_dir_all(parent).with_context(|| {
                                    format!("failed to create {}", parent.to_string_lossy())
                                })?;
                            }

                            let mut root: serde_json::Value = if path.exists() {
                                let text = fs::read_to_string(path).with_context(|| {
                                    format!("failed to read {}", path.to_string_lossy())
                                })?;
                                serde_json::from_str(&text)
                                    .context("failed to parse existing GeoJSON")?
                            } else {
                                serde_json::json!({"type":"FeatureCollection","features":[]})
                            };

                            let feats = root
                                .get_mut("features")
                                .and_then(|v| v.as_array_mut())
                                .ok_or_else(|| {
                                    anyhow::anyhow!("GeoJSON missing 'features' array")
                                })?;

                            for (idx, poly) in polys.iter().enumerate() {
                                if poly.len() < 3 {
                                    continue;
                                }
                                let mut ring: Vec<Vec<f64>> = Vec::with_capacity(poly.len() + 1);
                                for &p in poly {
                                    ring.push(vec![
                                        (p.x as f64) / (ds as f64),
                                        (p.y as f64) / (ds as f64),
                                    ]);
                                }
                                if ring.first() != ring.last() {
                                    if let Some(first) = ring.first().cloned() {
                                        ring.push(first);
                                    }
                                }

                                feats.push(serde_json::json!({
                                    "type": "Feature",
                                    "geometry": { "type": "Polygon", "coordinates": [ ring ] },
                                    "properties": {
                                        "layer": "odon_masks",
                                        "shape_index": idx as i64,
                                        "roi_root": local_root.to_string_lossy(),
                                    }
                                }));
                            }

                            let text = serde_json::to_string_pretty(&root)
                                .context("failed to encode GeoJSON")?;
                            fs::write(path, text).with_context(|| {
                                format!("failed to write {}", path.to_string_lossy())
                            })?;
                            Ok(())
                        })();
                        if let Err(err) = result {
                            self.roi_selector
                                .set_status(format!("Save Masks failed: {err}"));
                            return;
                        }

                        // Clear saved (editable, non-file-backed) layers.
                        for l in &mut self.mask_layers {
                            if l.editable && l.source_geojson.is_none() {
                                l.clear();
                            }
                        }
                        self.mark_mask_layers_project_dirty();

                        // Refresh the read-only layer from disk so appended shapes show up there too.
                        if let Err(err) = self.ensure_exclusion_masks_loaded() {
                            self.roi_selector
                                .set_status(format!("Saved masks, but reload failed: {err}"));
                        } else {
                            self.roi_selector.set_status(format!(
                                "Saved masks (appended) -> {}",
                                resolved.geojson_path.to_string_lossy()
                            ));
                        }
                    }
                    Err(err) => {
                        self.roi_selector
                            .set_status(format!("Save Masks failed: {err}"));
                    }
                }
            }
        }
    }

    pub(super) fn switch_dataset_with_store(
        &mut self,
        ctx: &egui::Context,
        mut dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        remote_runtime: Option<Arc<tokio::runtime::Runtime>>,
    ) -> anyhow::Result<()> {
        if dataset.source == self.dataset.source {
            return Ok(());
        }
        // Persist editable, project-backed state before replacing the dataset, then rebuild the
        // viewer around the new source while preserving the user's channel preferences and an
        // approximate camera position in normalized image coordinates.
        self.sync_current_view_state_into_project_space();

        let prev_channels = self.channels.clone();
        let prev_selected_name = self
            .channels
            .get(self.selected_channel)
            .map(|c| c.name.clone());

        let old_world_w = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let old_world_h = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        let fx = if old_world_w > 0.0 {
            (self.camera.center_world_lvl0.x / old_world_w).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old_world_h > 0.0 {
            (self.camera.center_world_lvl0.y / old_world_h).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let old_zoom = self.camera.zoom_screen_per_lvl0_px;
        let reload_exclusion_masks = self
            .mask_layers
            .iter()
            .any(|l| l.name == "Exclusion masks" && !l.editable && l.visible);

        let mut new_channels = dataset.channels.clone();
        apply_preserved_channel_settings(&prev_channels, &mut new_channels);
        for ch in &mut new_channels {
            if let Some(w) = self.channel_window_overrides.get(&ch.name).copied() {
                ch.window = Some(w);
            }
        }
        dataset.channels = new_channels.clone();

        let loader = spawn_tile_loader(
            store.clone(),
            dataset.levels.clone(),
            dataset.dims.clone(),
            self.tile_loader_threads,
        )?;

        let raw_loader = if self.tiles_gl.is_some() {
            Some(
                spawn_raw_tile_loader(
                    store.clone(),
                    dataset.levels.clone(),
                    dataset.dims.clone(),
                    self.tile_loader_threads,
                )
                .ok(),
            )
            .flatten()
        } else {
            None
        };

        let hist_loader =
            spawn_histogram_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())?;

        let chanmax_level = update::choose_default_max_level(&dataset);
        let chanmax_loader =
            spawn_channel_max_loader(store.clone(), dataset.levels.clone(), dataset.dims.clone())?;

        let label_cells: Option<LabelZarrDataset> = None;
        let label_loader: Option<LabelTileLoaderHandle> = None;
        let label_cells_xform: Option<Vec<LabelToWorld>> = None;
        let labels_gl = if self.tiles_gl.is_some() {
            Some(
                self.labels_gl
                    .clone()
                    .unwrap_or_else(|| LabelsGl::new(1024)),
            )
        } else {
            None
        };

        let seg_label_names = dataset
            .source
            .local_path()
            .map(discover_label_names_local)
            .unwrap_or_default();
        let mut seg_label_selected = self.seg_label_selected.clone();
        if seg_label_selected.is_empty() {
            seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else if let Some(first) = seg_label_names.first() {
                first.clone()
            } else {
                "cells".to_string()
            };
        }
        if !seg_label_names.is_empty() && !seg_label_names.iter().any(|n| n == &seg_label_selected)
        {
            seg_label_selected = if seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else {
                seg_label_names[0].clone()
            };
        }
        let seg_label_input = seg_label_selected.clone();
        let seg_label_status = String::new();

        // Commit the switch (after all fallible operations succeed).
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.reset();
        }
        if let Some(labels_gl) = labels_gl.as_ref() {
            labels_gl.reset();
        }

        self.dataset = dataset;
        self.store = store;
        self.remote_runtime = remote_runtime;
        self.channels = new_channels;
        self.channel_offsets_world = vec![egui::Vec2::ZERO; self.channels.len()];
        self.channel_scales = vec![egui::Vec2::splat(1.0); self.channels.len()];
        self.channel_rotations_rad = vec![0.0; self.channels.len()];
        self.points_offset_world = egui::Vec2::ZERO;
        self.spatial_points_offset_world = egui::Vec2::ZERO;
        self.mask_layers.clear();
        self.next_mask_layer_id = 1;
        self.drawing_mask_layer = None;
        self.seg_labels_offset_world = egui::Vec2::ZERO;
        self.seg_geojson_offset_world = egui::Vec2::ZERO;
        self.seg_objects_offset_world = egui::Vec2::ZERO;
        self.seg_geojson.clear();
        self.seg_objects.clear();
        self.spatial_layers.clear();
        self.spatial_root = None;
        self.spatial_label_store = None;
        self.loader = loader;
        self.raw_loader = raw_loader;
        self.hist_loader = hist_loader;
        self.chanmax_loader = chanmax_loader;
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        self.chanmax_level = chanmax_level;
        self.chanmax_pending = vec![false; self.channels.len()];
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        self.label_cells = label_cells;
        self.label_loader = label_loader;
        self.label_cells_xform = label_cells_xform;
        self.labels_gl = labels_gl;
        self.seg_label_names = seg_label_names;
        self.seg_label_selected = seg_label_selected;
        self.seg_label_input = seg_label_input;
        self.seg_label_status = seg_label_status;
        self.seg_label_prompt_open = self.tiles_gl.is_some() && !self.seg_label_names.is_empty();
        self.cells_outlines_visible = false;
        self.tiff_plane_state = None;
        self.configure_root_label_dataset_if_needed();

        if let Some(name) = prev_selected_name {
            if let Some(ch) = self.channels.iter().find(|c| c.name == name) {
                self.selected_channel = ch.index;
            } else {
                self.selected_channel = self
                    .selected_channel
                    .min(self.channels.len().saturating_sub(1));
            }
        } else {
            self.selected_channel = self
                .selected_channel
                .min(self.channels.len().saturating_sub(1));
        }
        if matches!(self.active_layer, LayerId::Channel(_)) {
            self.active_layer = LayerId::Channel(self.selected_channel);
        }

        let new_world_w = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.x] as f32)
            .unwrap_or(0.0);
        let new_world_h = self
            .dataset
            .levels
            .get(0)
            .map(|l| l.shape[self.dataset.dims.y] as f32)
            .unwrap_or(0.0);
        self.camera.center_world_lvl0 = egui::pos2(new_world_w * fx, new_world_h * fy);
        self.camera.zoom_screen_per_lvl0_px = old_zoom;
        self.apply_view_state_from_project_space();
        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        self.chanmax_pending = vec![false; self.channels.len()];
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();

        self.cache = TileCache::new(256);
        self.pending.clear();
        self.previous_render_id = None;
        self.previous_view_selection = None;
        self.previous_displayed_view_selection = None;
        self.active_render_id = self.compute_render_id();
        self.last_render_view_selection = self.committed_view_selection();
        self.restore_mask_layers_from_project_space();
        self.restore_loaded_layer_offsets_from_current_project_view_or_capture();

        self.hist = None;
        self.hist_request_id = 0;
        self.hist_request_pending = false;
        self.hist_dirty = true;
        self.hist_navigation_dirty_since = None;
        self.hist_last_sent = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or_else(Instant::now);
        self.pinned_levels = PinnedLevels::new();
        self.pending_memory_load = None;
        self.memory_status.clear();
        self.memory_selected_channels = (0..self.channels.len()).collect();

        self.maybe_apply_auto_contrast_on_open();
        self.roi_selector
            .sync_to_dataset_source(&self.dataset.source);
        if let Some(local_root) = self.dataset.source.local_path() {
            self.cell_thresholds.set_dataset_root(
                local_root,
                self.dataset.multiscale.name.as_deref(),
                &mut self.cell_points,
            );
        }
        self.auto_load_project_roi_segmentation();
        self.drawing_mask_polygon.clear();
        self.clear_mask_polygon_selection();
        self.undo_stack.clear();
        if reload_exclusion_masks {
            if self.dataset.source.local_path().is_some() {
                if let Err(err) = self.ensure_exclusion_masks_loaded() {
                    self.roi_selector
                        .set_status(format!("Load Masks failed: {err}"));
                }
            } else {
                self.roi_selector
                    .set_status("Masks are supported for local datasets only.".to_string());
            }
        }

        // Best-effort fit if the new ROI is wildly different in size.
        if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
            if self.camera.zoom_screen_per_lvl0_px <= 0.0 {
                self.camera.fit_to_world_rect(
                    viewport,
                    egui::Rect::from_min_size(
                        egui::pos2(0.0, 0.0),
                        egui::vec2(new_world_w.max(1.0), new_world_h.max(1.0)),
                    ),
                );
            }
        }

        Ok(())
    }

    pub(super) fn ensure_exclusion_masks_loaded(&mut self) -> anyhow::Result<usize> {
        let Some(local_root) = self.dataset.source.local_path() else {
            anyhow::bail!("exclusion masks are supported for local datasets only");
        };
        let Some(cfg) = self.roi_selector.masks_config_for_roi(local_root) else {
            anyhow::bail!("no matching dataset entry in Project config");
        };
        let entry = self.roi_selector.roi_entry_for_path(local_root);

        let resolved = resolve_masks_geojson_path_and_downsample(local_root, &cfg, entry.as_ref())?;
        let polylines = load_geojson_polylines_world(
            &resolved.geojson_path,
            resolved.downsample_factor,
            PolygonRingMode::AllRings,
        )
        .with_context(|| {
            format!(
                "failed to load masks: {}",
                resolved.geojson_path.to_string_lossy()
            )
        })?;

        let existing_idx = self.mask_layers.iter().position(|l| {
            !l.editable
                && l.source_geojson
                    .as_ref()
                    .is_some_and(|p| p == &resolved.geojson_path)
        });

        let idx = match existing_idx {
            Some(i) => i,
            None => {
                let id = self.next_mask_layer_id.max(1);
                self.next_mask_layer_id = id.saturating_add(1);
                self.mask_layers.push(MaskLayer {
                    id,
                    name: "Exclusion masks".to_string(),
                    visible: true,
                    opacity: 0.85,
                    width_screen_px: 1.5,
                    display_mode: MaskDisplayMode::default_new_layer(),
                    color_rgb: [50, 220, 255],
                    offset_world: egui::Vec2::ZERO,
                    editable: false,
                    polygons_world: Vec::new(),
                    raster_display: None,
                    source_geojson: Some(resolved.geojson_path.clone()),
                });
                self.mark_mask_layers_project_dirty();
                self.mask_layers.len().saturating_sub(1)
            }
        };

        if let Some(l) = self.mask_layers.get_mut(idx) {
            l.polygons_world = polylines;
            l.raster_display = None;
            l.source_geojson = Some(resolved.geojson_path);
            l.visible = true;
            l.editable = false;
            self.mark_mask_layers_project_dirty();
        }

        Ok(self
            .mask_layers
            .get(idx)
            .map(|l| l.polygons_world.len())
            .unwrap_or(0))
    }

    pub(super) fn request_exclusion_masks_reload(&mut self) -> anyhow::Result<usize> {
        if !self.mask_actor_owned() {
            return self.ensure_exclusion_masks_loaded();
        }
        let Some(local_root) = self.dataset.source.local_path() else {
            anyhow::bail!("exclusion masks are supported for local datasets only");
        };
        let Some(cfg) = self.roi_selector.masks_config_for_roi(local_root) else {
            anyhow::bail!("no matching dataset entry in Project config");
        };
        let entry = self.roi_selector.roi_entry_for_path(local_root);
        let resolved = resolve_masks_geojson_path_and_downsample(local_root, &cfg, entry.as_ref())?;
        let existing = self.mask_layers.iter().find(|layer| {
            !layer.editable
                && layer
                    .source_geojson
                    .as_ref()
                    .is_some_and(|path| path == &resolved.geojson_path)
        });
        let current_count = existing.map_or(0, |layer| layer.polygons_world.len());
        let mut params = serde_json::Map::new();
        params.insert(
            "path".to_string(),
            serde_json::json!(resolved.geojson_path.to_string_lossy()),
        );
        params.insert("name".to_string(), serde_json::json!("Exclusion masks"));
        params.insert("editable".to_string(), serde_json::json!(false));
        params.insert(
            "downsample_factor".to_string(),
            serde_json::json!(resolved.downsample_factor),
        );
        if let Some(layer) = existing {
            params.insert("replace_layer_id".to_string(), serde_json::json!(layer.id));
        }
        self.submit_native_mask_command("viewer.masks.import_geojson", params);
        Ok(current_count)
    }

    pub(super) fn refresh_seg_label_names_for_current_roi(&mut self) {
        if self.dataset.is_root_label_mask() {
            self.seg_label_names.clear();
            self.seg_label_selected = LabelZarrDataset::root_label_name(&self.dataset);
            self.seg_label_input = self.seg_label_selected.clone();
            self.seg_label_prompt_open = false;
            return;
        }

        self.seg_label_names = self
            .spatial_root
            .as_deref()
            .or_else(|| self.dataset.source.local_path())
            .map(discover_label_names_local)
            .unwrap_or_default();
        if self.seg_label_selected.trim().is_empty() {
            self.seg_label_selected = if self.seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else if let Some(first) = self.seg_label_names.first() {
                first.clone()
            } else {
                "cells".to_string()
            };
        }
        if !self.seg_label_names.is_empty()
            && !self
                .seg_label_names
                .iter()
                .any(|n| n == &self.seg_label_selected)
        {
            self.seg_label_selected = if self.seg_label_names.iter().any(|n| n == "cells") {
                "cells".to_string()
            } else {
                self.seg_label_names[0].clone()
            };
        }
        if self.seg_label_input.trim().is_empty() || self.seg_label_input == self.seg_label_selected
        {
            self.seg_label_input = self.seg_label_selected.clone();
        }
    }

    pub(super) fn ensure_segmentation_labels_loaded(&mut self) -> anyhow::Result<()> {
        let name = self.seg_label_selected.trim().to_string();
        if name.is_empty() {
            anyhow::bail!("label name is empty");
        }
        self.load_segmentation_labels(name.as_str())
    }

    pub(super) fn load_segmentation_labels(&mut self, label_name: &str) -> anyhow::Result<()> {
        if self.dataset.is_root_label_mask() {
            return self.load_root_segmentation_labels();
        }
        if self.tiles_gl.is_none() {
            anyhow::bail!("segmentation overlay requires the GPU path");
        }
        self.labels_gl
            .get_or_insert_with(|| LabelsGl::new(1024))
            .reset();

        let label_store = self
            .spatial_label_store
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.store.clone());
        match LabelZarrDataset::try_open(label_store.clone(), label_name)? {
            Some(lbl) => {
                self.spatial_label_transform = self.spatial_label_transform_for_name(label_name);
                self.label_loader =
                    spawn_label_tile_loader(label_store, lbl.levels.clone(), lbl.dims.clone()).ok();
                self.label_cells_xform = Some(compute_label_to_world_xforms(
                    &self.dataset,
                    &lbl,
                    self.spatial_label_transform,
                ));
                self.label_cells = Some(lbl);
                self.cells_outlines_visible = true;
                self.seg_label_selected = label_name.to_string();
                self.seg_label_input = self.seg_label_selected.clone();
                self.seg_label_prompt_open = false;
                self.rebuild_layer_orders();
                Ok(())
            }
            None => {
                self.label_cells = None;
                self.label_loader = None;
                self.label_cells_xform = None;
                anyhow::bail!("no labels/{label_name} found in this ROI")
            }
        }
    }

    pub(super) fn ui_seg_label_prompt(&mut self, ctx: &egui::Context) {
        if !self.seg_label_prompt_open {
            return;
        }
        if self.tiles_gl.is_none() {
            self.seg_label_prompt_open = false;
            return;
        }
        if self.seg_label_names.is_empty() {
            self.seg_label_prompt_open = false;
            return;
        }

        match self.seg_label_prompt_preference {
            LabelPromptSessionPreference::AlwaysSkip => {
                self.seg_label_status.clear();
                self.seg_label_prompt_open = false;
                return;
            }
            LabelPromptSessionPreference::AlwaysLoad => {
                let name = self.seg_label_selected.trim().to_string();
                if name.is_empty() {
                    self.seg_label_prompt_preference = LabelPromptSessionPreference::Ask;
                    self.seg_label_prompt_always = false;
                } else {
                    match self.load_segmentation_labels(name.as_str()) {
                        Ok(()) => {
                            self.seg_label_status.clear();
                            self.set_active_layer(LayerId::SegmentationLabels);
                            self.bump_render_id();
                            self.seg_label_prompt_open = false;
                            return;
                        }
                        Err(err) => {
                            self.seg_label_status = format!("Load labels/{name} failed: {err}");
                            self.seg_label_prompt_preference = LabelPromptSessionPreference::Ask;
                            self.seg_label_prompt_always = false;
                        }
                    }
                }
            }
            LabelPromptSessionPreference::Ask => {}
        }

        let mut open = true;
        let mut request_close = false;
        egui::Window::new("Load labels?")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label(format!(
                    "Found {} label group(s) under labels/.",
                    self.seg_label_names.len()
                ));
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label("Label");
                    egui::ComboBox::from_id_salt("seg_label_prompt_select")
                        .selected_text(self.seg_label_selected.clone())
                        .show_ui(ui, |ui| {
                            for name in self.seg_label_names.clone() {
                                ui.selectable_value(
                                    &mut self.seg_label_selected,
                                    name.clone(),
                                    name,
                                );
                            }
                        });
                });

                if !self.seg_label_status.trim().is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.seg_label_status.clone());
                }

                ui.add_space(8.0);
                ui.checkbox(&mut self.seg_label_prompt_always, "Always");

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.button("Skip").clicked() {
                        self.seg_label_prompt_preference = if self.seg_label_prompt_always {
                            LabelPromptSessionPreference::AlwaysSkip
                        } else {
                            LabelPromptSessionPreference::Ask
                        };
                        self.seg_label_status.clear();
                        request_close = true;
                    }
                    if ui.button("Load labels").clicked() {
                        let name = self.seg_label_selected.trim().to_string();
                        if name.is_empty() {
                            self.seg_label_status = "Label name is empty.".to_string();
                        } else {
                            match self.load_segmentation_labels(name.as_str()) {
                                Ok(()) => {
                                    self.seg_label_prompt_preference =
                                        if self.seg_label_prompt_always {
                                            LabelPromptSessionPreference::AlwaysLoad
                                        } else {
                                            LabelPromptSessionPreference::Ask
                                        };
                                    self.seg_label_status.clear();
                                    self.set_active_layer(LayerId::SegmentationLabels);
                                    self.bump_render_id();
                                    request_close = true;
                                }
                                Err(err) => {
                                    self.seg_label_status =
                                        format!("Load labels/{name} failed: {err}");
                                }
                            }
                        }
                    }
                });
            });

        if request_close {
            open = false;
        }
        if !open {
            self.seg_label_prompt_open = false;
        }
    }
}
