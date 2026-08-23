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
                let name = self.seg_label_selected.trim().to_string();
                if name.is_empty() {
                    self.roi_selector.set_status(
                        "Load Labels failed: the actor has no selected label group.".to_string(),
                    );
                } else {
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.labels.load",
                        params: serde_json::json!({"name":name}),
                    });
                    self.roi_selector
                        .set_status(format!("Loading labels/{name}..."));
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

                if !self.mask_layers.iter().any(|layer| {
                    layer.editable
                        && layer.source_geojson.is_none()
                        && !layer.polygons_world.is_empty()
                }) {
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
                    }
                    Err(err) => {
                        self.roi_selector
                            .set_status(format!("Save Masks failed: {err}"));
                    }
                }
            }
        }
    }

    pub(super) fn request_exclusion_masks_reload(&mut self) -> anyhow::Result<usize> {
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
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.labels.load",
                        params: serde_json::json!({"name":name}),
                    });
                    self.seg_label_status = format!("Loading labels/{name}...");
                    self.seg_label_prompt_open = false;
                    return;
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
                            self.seg_label_prompt_preference = if self.seg_label_prompt_always {
                                LabelPromptSessionPreference::AlwaysLoad
                            } else {
                                LabelPromptSessionPreference::Ask
                            };
                            self.native_control_intents.push(NativeControlIntent {
                                method: "viewer.labels.load",
                                params: serde_json::json!({"name":name}),
                            });
                            self.seg_label_status = format!("Loading labels/{name}...");
                            request_close = true;
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
