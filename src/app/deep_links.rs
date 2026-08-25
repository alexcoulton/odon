use super::*;

impl OmeZarrViewerApp {
    pub fn set_label_prompt_preference(&mut self, preference: LabelPromptSessionPreference) {
        self.seg_label_prompt_preference = preference;
        if preference == LabelPromptSessionPreference::Ask {
            self.seg_label_prompt_always = false;
        }
    }

    pub fn label_prompt_preference(&self) -> LabelPromptSessionPreference {
        self.seg_label_prompt_preference
    }

    #[cfg(test)]
    pub fn apply_deep_link_request(&mut self, request: &DeepLinkRequest) {
        let mut notes = Vec::new();
        let segmentation_source = request
            .segmentation_source
            .as_deref()
            .or_else(|| request.segmentation.as_deref())
            .map(normalize_deep_link_name);
        let object_segmentation_requested = segmentation_source.as_deref().is_some_and(|source| {
            matches!(
                source,
                "objects"
                    | "object"
                    | "geoparquet"
                    | "parquet"
                    | "project"
                    | "project_objects"
                    | "cells_geoparquet"
            )
        }) || !request.object_filters.is_empty()
            || request.object_query.is_some()
            || !request.object_level_colors.is_empty();
        let bundled_labels_requested = segmentation_source
            .as_deref()
            .is_none_or(|source| !object_segmentation_requested && source != "none");
        let load_bundled_labels = request
            .load_segmentation_labels
            .unwrap_or(bundled_labels_requested);
        let suppress_bundled_label_prompt = !load_bundled_labels
            || object_segmentation_requested
            || segmentation_source.as_deref() == Some("none");
        if suppress_bundled_label_prompt {
            self.seg_label_prompt_open = false;
        }

        let mut channel_visibility_changed = false;
        let channel_terms = if request.channel_alternatives.is_empty() {
            request.channel.iter().cloned().collect::<Vec<_>>()
        } else {
            request.channel_alternatives.clone()
        };
        if !channel_terms.is_empty() {
            match self.find_channel_index_for_link_terms(&channel_terms) {
                Some(idx) => {
                    self.selected_channel = idx;
                    if let Some(channel) = self.channels.get_mut(idx) {
                        channel_visibility_changed |= !channel.visible;
                        channel.visible = true;
                    }
                    self.set_active_layer(LayerId::Channel(idx));
                }
                None => notes.push(format!(
                    "channel '{}' was not found",
                    channel_terms.join("' or '")
                )),
            }
        }

        let visible_channel_groups = deep_link_channel_groups(
            &request.visible_channels,
            &request.visible_channel_alternatives,
        );
        if !visible_channel_groups.is_empty() {
            for channel in &mut self.channels {
                channel_visibility_changed |= channel.visible;
                channel.visible = false;
            }
            let mut visible_channel_indices = Vec::new();
            for channel_terms in &visible_channel_groups {
                match self.find_channel_index_for_link_terms(channel_terms) {
                    Some(idx) => {
                        if !visible_channel_indices.contains(&idx) {
                            visible_channel_indices.push(idx);
                        }
                        if let Some(channel) = self.channels.get_mut(idx) {
                            channel_visibility_changed |= !channel.visible;
                            channel.visible = true;
                        }
                    }
                    None => notes.push(format!(
                        "visible channel '{}' was not found",
                        channel_terms.join("' or '")
                    )),
                }
            }
            if request.group_visible_channels || request.visible_channel_group.is_some() {
                if visible_channel_indices.is_empty() {
                    notes.push("no visible channels were available to group".to_string());
                } else {
                    let group_name = request
                        .visible_channel_group
                        .as_deref()
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .unwrap_or("Deep link channels");
                    self.group_channel_indices_for_deep_link(
                        group_name,
                        &visible_channel_indices,
                        request.visible_channel_group_color,
                    );
                }
            }
            if request.channel_order == Some(DeepLinkChannelOrder::Listed) {
                if visible_channel_indices.is_empty() {
                    notes.push("no visible channels were available to order".to_string());
                } else {
                    self.move_channels_to_top_for_deep_link(&visible_channel_indices);
                }
            }
        }

        let hidden_channel_groups = deep_link_channel_groups(
            &request.hidden_channels,
            &request.hidden_channel_alternatives,
        );
        for channel_terms in &hidden_channel_groups {
            match self.find_channel_index_for_link_terms(channel_terms) {
                Some(idx) => {
                    if let Some(channel) = self.channels.get_mut(idx) {
                        channel_visibility_changed |= channel.visible;
                        channel.visible = false;
                    }
                }
                None => notes.push(format!(
                    "hidden channel '{}' was not found",
                    channel_terms.join("' or '")
                )),
            }
        }
        if channel_visibility_changed {
            self.bump_render_id();
        }

        for channel_color in &request.channel_colors {
            match self
                .find_channel_index_for_link_terms(std::slice::from_ref(&channel_color.channel))
            {
                Some(idx) => {
                    if let Some(channel) = self.channels.get_mut(idx) {
                        channel.color_rgb = channel_color.color_rgb;
                    }
                    self.set_channel_group_color_inheritance(idx, false);
                    self.bump_render_id();
                }
                None => notes.push(format!(
                    "channel colour target '{}' was not found",
                    channel_color.channel
                )),
            }
        }

        if request.contrast_min.is_some() || request.contrast_max.is_some() {
            let idx = self
                .selected_channel
                .min(self.channels.len().saturating_sub(1));
            if self.channels.get(idx).is_some() {
                let abs_max = self.dataset.abs_max.max(1.0);
                let (existing_lo, existing_hi) =
                    self.channels[idx].window.unwrap_or((0.0, abs_max));
                let lo = request.contrast_min.unwrap_or(existing_lo);
                let hi = request.contrast_max.unwrap_or(existing_hi);
                if !self.set_channel_window_for_link(idx, lo, hi) {
                    let channel_name = self.channels[idx].name.clone();
                    notes.push(format!(
                        "contrast limits for channel '{channel_name}' were invalid"
                    ));
                }
            }
        }

        for contrast in &request.channel_contrasts {
            match self.find_channel_index_for_link(&contrast.channel) {
                Some(idx) => {
                    if !self.set_channel_window_for_link(idx, contrast.min, contrast.max) {
                        notes.push(format!(
                            "contrast limits for channel '{}' were invalid",
                            contrast.channel
                        ));
                    }
                }
                None => notes.push(format!(
                    "contrast channel '{}' was not found",
                    contrast.channel
                )),
            }
        }

        if let Some(label_name) = request.segmentation.as_deref()
            && load_bundled_labels
            && bundled_labels_requested
        {
            if self.seg_label_names.is_empty()
                || self.seg_label_names.iter().any(|n| n == label_name)
            {
                self.seg_label_selected = label_name.to_string();
                self.seg_label_input = self.seg_label_selected.clone();
                if self.tiles_gl.is_some() {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "viewer.labels.load",
                        params: serde_json::json!({"name":label_name}),
                    });
                    self.seg_label_prompt_open = false;
                }
            } else {
                notes.push(format!("labels/{label_name} was not found"));
            }
        } else if suppress_bundled_label_prompt {
            self.cells_outlines_visible = false;
        }

        if object_segmentation_requested {
            self.auto_load_project_roi_segmentation();
            self.set_active_layer(LayerId::SegmentationObjects);
        }

        if let Some(color_by) = request.cell_color_by.as_deref() {
            let mut display = self.seg_objects.project_display_state();
            display.color_property_key = Some(color_by.to_string());
            display.color_mapping = Some(odon::model::ObjectColorMapping::categorical(color_by));
            display.fill_cells = request.fill_cells.unwrap_or(true);
            self.seg_objects.apply_project_display_state(&display);
            self.set_active_layer(LayerId::SegmentationObjects);
        } else if let Some(fill_cells) = request.fill_cells {
            let mut display = self.seg_objects.project_display_state();
            display.fill_cells = fill_cells;
            self.seg_objects.apply_project_display_state(&display);
        }
        if let Some(mapping) = request.object_color_mapping.as_ref() {
            let mut display = self.seg_objects.project_display_state();
            display.color_property_key = mapping.property().map(str::to_string);
            display.color_mapping = Some(mapping.clone());
            display.fill_cells = request.fill_cells.unwrap_or(true);
            self.seg_objects.apply_project_display_state(&display);
            self.set_active_layer(LayerId::SegmentationObjects);
        }

        if let Some(show_selection_overlay) = request.show_selection_overlay {
            let mut analysis = self.seg_objects.project_analysis_state();
            analysis.show_selection_overlay = show_selection_overlay;
            let active_channel_name = self
                .channels
                .get(self.selected_channel)
                .map(|channel| channel.name.as_str());
            self.seg_objects
                .apply_project_analysis_state(&analysis, active_channel_name);
        }

        if let Some(fast_object_rendering) = request.fast_object_rendering {
            self.set_fast_object_rendering(fast_object_rendering);
        }

        if !request.object_level_colors.is_empty() {
            let colors = request
                .object_level_colors
                .iter()
                .map(|level| (level.value.clone(), level.color_rgb))
                .collect::<Vec<_>>();
            self.seg_objects
                .set_color_value_colors(request.cell_color_by.as_deref(), &colors);
            self.set_active_layer(LayerId::SegmentationObjects);
        }

        if !request.visible_cell_types.is_empty() || !request.hidden_cell_types.is_empty() {
            self.seg_objects.set_color_value_visibility(
                request.cell_color_by.as_deref(),
                &request.visible_cell_types,
                &request.hidden_cell_types,
            );
        }

        if let Some(logic) = request.object_filter_logic {
            let logic = match logic {
                DeepLinkObjectFilterLogic::All => ObjectFilterLogic::All,
                DeepLinkObjectFilterLogic::Any => ObjectFilterLogic::Any,
            };
            self.seg_objects.set_filter_logic(logic);
        }

        if !request.object_filters.is_empty() {
            let filter_pairs = request
                .object_filters
                .iter()
                .map(|clause| (clause.property_key.clone(), clause.query.clone()))
                .collect::<Vec<_>>();
            self.seg_objects
                .set_filter_clauses_from_pairs(&filter_pairs);
            self.set_active_layer(LayerId::SegmentationObjects);
        }

        if let Some(query) = request.object_query.as_deref() {
            self.seg_objects.set_filter_query_from_text(query);
            self.set_active_layer(LayerId::SegmentationObjects);
        }

        if let Some(center) = request.center_world {
            self.camera.center_world_lvl0 = egui::pos2(center[0], center[1]);
        }
        if let Some(zoom) = request.zoom {
            self.camera.zoom_screen_per_lvl0_px = zoom;
        }

        if notes.is_empty() {
            self.roi_selector_ui.set_status("Opened Odon deep link.");
        } else {
            self.roi_selector_ui
                .set_status(format!("Opened Odon deep link; {}", notes.join("; ")));
        }
    }

    pub fn install_preloaded_project_segmentation(&mut self, preloaded: &PreloadedObjectLayer) {
        self.seg_objects.install_preloaded(preloaded);
        self.restore_project_object_state_after_segmentation_load();
        self.set_active_layer(LayerId::SegmentationObjects);
        self.roi_selector_ui
            .set_status("Loaded cached project segmentation.");
    }

    pub(super) fn restore_project_object_state_after_segmentation_load(&mut self) {
        if let Some(view) = self.project_space.roi_view_state(&self.dataset.source) {
            if let Some(object_display) = view
                .segmentation
                .as_ref()
                .and_then(|segmentation| segmentation.object_display.as_ref())
            {
                self.seg_objects
                    .apply_project_display_state_preserving_color_visibility(object_display);
            } else {
                self.seg_objects.clear_project_display_state();
            }
            self.seg_objects.fast_rendering = self.fast_object_rendering;
            if let Some(analysis) = view.analysis.as_ref() {
                let active_channel_name = self
                    .channels
                    .get(self.selected_channel)
                    .map(|channel| channel.name.as_str());
                self.seg_objects
                    .apply_project_analysis_state(analysis, active_channel_name);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn find_channel_index_for_link(&self, channel_name: &str) -> Option<usize> {
        let needle = normalize_deep_link_name(channel_name);
        if needle.is_empty() {
            return None;
        }

        if let Some(idx) = self
            .channels
            .iter()
            .position(|channel| normalize_deep_link_name(&channel.name) == needle)
        {
            return Some(idx);
        }

        if let Some(idx) = self.channels.iter().position(|channel| {
            normalize_deep_link_name(marker_name_from_channel_label(&channel.name)) == needle
        }) {
            return Some(idx);
        }

        let marker_matches = self
            .channels
            .iter()
            .enumerate()
            .filter_map(|(idx, channel)| {
                marker_alias_matches(channel_name, marker_name_from_channel_label(&channel.name))
                    .then_some(idx)
            })
            .collect::<Vec<_>>();
        if marker_matches.len() == 1 {
            return marker_matches.first().copied();
        }

        let contains_matches = self
            .channels
            .iter()
            .enumerate()
            .filter_map(|(idx, channel)| {
                normalize_deep_link_name(&channel.name)
                    .contains(&needle)
                    .then_some(idx)
            })
            .collect::<Vec<_>>();
        (contains_matches.len() == 1).then(|| contains_matches[0])
    }

    #[cfg(test)]
    pub(super) fn find_channel_index_for_link_terms(&self, terms: &[String]) -> Option<usize> {
        for term in terms {
            if let Some(idx) = self.find_channel_index_for_link(term) {
                return Some(idx);
            }
        }
        None
    }

    #[cfg(test)]
    pub(super) fn set_channel_window_for_link(&mut self, idx: usize, lo: f32, hi: f32) -> bool {
        if !lo.is_finite() || !hi.is_finite() {
            return false;
        }
        let abs_max = self.dataset.abs_max.max(1.0);
        let lo = lo.clamp(0.0, abs_max);
        let hi = hi.clamp(0.0, abs_max);
        if hi <= lo {
            return false;
        }
        let Some(channel) = self.channels.get_mut(idx) else {
            return false;
        };
        channel.window = Some((lo, hi));
        self.channel_window_overrides
            .insert(channel.name.clone(), (lo, hi));
        self.bump_render_id();
        true
    }
}
