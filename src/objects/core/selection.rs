//! Object selection identities, saved selection elements, queries, and filter evaluation.

use super::*;

impl ObjectsLayer {
    pub fn selected_object_details(
        &self,
        local_to_world_offset: egui::Vec2,
    ) -> Option<SelectedObjectDetails> {
        let idx = self.selected_object_index?;
        let obj = self.objects.as_ref()?.get(idx)?;
        let properties = self.loaded_property_display_pairs(idx, obj);
        let scale = egui::vec2(
            self.display_transform.scale[0].max(1e-6),
            self.display_transform.scale[1].max(1e-6),
        );
        let offset = egui::vec2(
            local_to_world_offset.x + self.display_transform.translation[0],
            local_to_world_offset.y + self.display_transform.translation[1],
        );
        Some(SelectedObjectDetails {
            id: obj.id.clone(),
            area_px: obj.area_px,
            perimeter_px: obj.perimeter_px,
            centroid_world: egui::pos2(
                obj.centroid_world.x * scale.x + offset.x,
                obj.centroid_world.y * scale.y + offset.y,
            ),
            properties,
        })
    }

    pub fn select_objects_by_ids(&mut self, ids: &std::collections::HashSet<String>) -> usize {
        let indices = self.object_indices_matching_ids(ids);
        self.apply_object_selection_mode(&indices, "replace");
        self.selected_object_indices.len()
    }

    pub(crate) fn object_indices_matching_ids(
        &self,
        ids: &std::collections::HashSet<String>,
    ) -> Vec<usize> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let mut indices = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            let mut matched = ids.contains(&obj.id);
            if !matched {
                for key in ["cell_id", "id", "object_id", "label", "name"] {
                    if let Some(value) = self.object_property_label(idx, obj, key)
                        && ids.contains(&value)
                    {
                        matched = true;
                        break;
                    }
                }
            }
            if matched {
                indices.push(idx);
            }
        }
        indices
    }

    pub(in crate::objects) fn apply_object_selection_mode(
        &mut self,
        indices: &[usize],
        mode: &str,
    ) -> bool {
        let before = self.selected_object_indices.clone();
        match mode {
            "replace" => self.selected_object_indices = indices.iter().copied().collect(),
            "add" => self.selected_object_indices.extend(indices.iter().copied()),
            "remove" => self
                .selected_object_indices
                .retain(|index| !indices.contains(index)),
            "toggle" => {
                for index in indices {
                    if !self.selected_object_indices.insert(*index) {
                        self.selected_object_indices.remove(index);
                    }
                }
            }
            _ => return false,
        }
        self.selected_object_index = self.selected_object_indices.iter().min().copied();
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
        before != self.selected_object_indices
    }

    #[cfg(test)]
    pub fn control_select_ids_json(
        &mut self,
        ids: &std::collections::HashSet<String>,
        mode: &str,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return serde_json::json!({"error": "selection mode must be replace, add, remove, or toggle"});
        }
        let indices = self.object_indices_matching_ids(ids);
        let missing = ids
            .iter()
            .filter(|id| {
                self.object_indices_matching_ids(&std::collections::HashSet::from([(*id).clone()]))
                    .is_empty()
            })
            .cloned()
            .collect::<Vec<_>>();
        let changed = self.apply_object_selection_mode(&indices, mode);
        serde_json::json!({
            "changed": changed,
            "matched_count": indices.len(),
            "missing_ids": missing,
            "selection": self.selection_snapshot_json(local_to_world_offset, limit),
        })
    }

    #[cfg(test)]
    pub fn control_select_filtered_json(
        &mut self,
        mode: &str,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        self.ensure_filter_cache();
        let indices = self
            .filtered_ordered_indices
            .as_ref()
            .map(|indices| indices.as_ref().clone())
            .unwrap_or_else(|| {
                self.objects
                    .as_ref()
                    .map(|objects| (0..objects.len()).collect())
                    .unwrap_or_default()
            });
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return serde_json::json!({"error": "selection mode must be replace, add, remove, or toggle"});
        }
        let changed = self.apply_object_selection_mode(&indices, mode);
        serde_json::json!({
            "changed": changed,
            "matched_count": indices.len(),
            "filter_revision": self.filter_generation,
            "selection": self.selection_snapshot_json(local_to_world_offset, limit),
        })
    }

    pub fn install_control_selection(
        &mut self,
        selected_indices: &[usize],
        primary_index: Option<usize>,
    ) -> Result<(), String> {
        let object_count = self.objects.as_ref().map_or(0, |objects| objects.len());
        if selected_indices.iter().any(|index| *index >= object_count) {
            return Err("actor object selection contains an out-of-range index".to_string());
        }
        let selected = selected_indices.iter().copied().collect::<HashSet<_>>();
        if primary_index.is_some_and(|index| !selected.contains(&index)) {
            return Err("actor object selection primary is not selected".to_string());
        }
        self.selected_object_indices = selected;
        self.selected_object_index = primary_index;
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
        Ok(())
    }

    pub(super) fn current_selection_object_ids(&self) -> Vec<String> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let mut indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices
            .into_iter()
            .filter_map(|idx| objects.get(idx).map(|obj| obj.id.clone()))
            .collect()
    }

    pub(super) fn create_selection_element_from_current_selection(&mut self) -> usize {
        let object_ids = self.current_selection_object_ids();
        if object_ids.is_empty() {
            return 0;
        }
        let next_idx = self.selection_elements.len() + 1;
        let name = if self.selection_element_name_draft.trim().is_empty() {
            format!("Selection Element {next_idx}")
        } else {
            self.selection_element_name_draft.trim().to_string()
        };
        self.selection_elements
            .push(SelectionElement { name, object_ids });
        self.selection_element_selected = Some(self.selection_elements.len() - 1);
        self.selection_element_name_draft =
            format!("Selection Element {}", self.selection_elements.len() + 1);
        self.selection_elements
            .last()
            .map(|element| element.object_ids.len())
            .unwrap_or(0)
    }

    pub(super) fn create_selection_element_from_ids(
        &mut self,
        name: Option<String>,
        object_ids: Vec<String>,
    ) -> usize {
        if object_ids.is_empty() {
            return 0;
        }
        let next_idx = self.selection_elements.len() + 1;
        let name = name
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("Selection Element {next_idx}"));
        self.selection_elements
            .push(SelectionElement { name, object_ids });
        self.selection_element_selected = Some(self.selection_elements.len() - 1);
        self.selection_element_name_draft =
            format!("Selection Element {}", self.selection_elements.len() + 1);
        self.selection_elements
            .last()
            .map(|element| element.object_ids.len())
            .unwrap_or(0)
    }

    pub(super) fn select_selection_element(&mut self, idx: usize) -> usize {
        let Some(element) = self.selection_elements.get(idx) else {
            return 0;
        };
        let ids = element.object_ids.iter().cloned().collect::<HashSet<_>>();
        self.selection_element_selected = Some(idx);
        self.select_objects_by_ids(&ids)
    }

    pub fn selection_elements_snapshot(&self) -> Vec<(usize, String, usize)> {
        self.selection_elements
            .iter()
            .enumerate()
            .map(|(idx, element)| (idx, element.name.clone(), element.object_ids.len()))
            .collect()
    }

    pub fn create_selection_element_from_current_selection_with_name(
        &mut self,
        name: Option<String>,
    ) -> usize {
        let object_ids = self.current_selection_object_ids();
        let count = self.create_selection_element_from_ids(name, object_ids);
        if count > 0 {
            self.status = format!("Saved selection element with {count} object(s).");
        }
        count
    }

    pub fn add_current_selection_to_element(&mut self, idx: usize) -> usize {
        let selected_ids = self.current_selection_object_ids();
        let Some(element) = self.selection_elements.get_mut(idx) else {
            return 0;
        };
        let mut existing = element.object_ids.iter().cloned().collect::<HashSet<_>>();
        let mut added = 0usize;
        for id in selected_ids {
            if existing.insert(id.clone()) {
                element.object_ids.push(id);
                added += 1;
            }
        }
        self.selection_element_selected = Some(idx);
        self.status = format!("Added {added} object(s) to selection element.");
        added
    }

    pub(in crate::objects) fn ui_selection_elements_editor(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Selection Elements", |ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(
                    egui::TextEdit::singleline(&mut self.selection_element_name_draft)
                        .desired_width(180.0),
                );
                if ui
                    .add_enabled(
                        self.selection_count() > 0,
                        egui::Button::new("New from selection"),
                    )
                    .clicked()
                {
                    let count = self.create_selection_element_from_current_selection();
                    if count > 0 {
                        self.status = format!("Saved selection element with {count} object(s).");
                    }
                }
            });
            if self.selection_elements.is_empty() {
                ui.label("No saved selection elements.");
            } else {
                let mut clicked_idx = None;
                let mut delete_idx = None;
                egui::ScrollArea::vertical()
                    .id_salt("seg_objects_selection_elements")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        for (idx, element) in self.selection_elements.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                let selected = self.selection_element_selected == Some(idx);
                                let label =
                                    format!("{} ({})", element.name, element.object_ids.len());
                                if ui.selectable_label(selected, label).clicked() {
                                    clicked_idx = Some(idx);
                                }
                                ui.add(
                                    egui::TextEdit::singleline(&mut element.name)
                                        .desired_width(140.0),
                                );
                                if ui.button("Delete").clicked() {
                                    delete_idx = Some(idx);
                                }
                            });
                        }
                    });
                if let Some(idx) = clicked_idx {
                    let count = self.select_selection_element(idx);
                    self.status = format!("Selected {count} object(s) from saved element.");
                }
                if let Some(idx) = delete_idx {
                    self.selection_elements.remove(idx);
                    self.selection_element_selected = match self.selection_elements.is_empty() {
                        true => None,
                        false => self
                            .selection_element_selected
                            .map(|selected| selected.min(self.selection_elements.len() - 1)),
                    };
                }
            }
        });
    }

    pub fn is_analyzing(&self) -> bool {
        self.analysis_selection_rx.is_some()
    }

    pub(super) fn request_load(
        &mut self,
        path: PathBuf,
        downsample_factor: f32,
        load_options: Option<ObjectLoadOptions>,
    ) {
        // The worker thread produces a single fully prepared `LoadResult` that already contains
        // the expensive derived structures (bins, LODs, point payloads). `tick` then installs it
        // atomically on the UI thread.
        self.cancel_current_load();
        self.object_load_request_id = self.object_load_request_id.wrapping_add(1).max(1);
        let request_id = self.object_load_request_id;
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = cancel.clone();
        let (tx, rx) = crossbeam_channel::bounded::<Result<LoadResult, String>>(1);
        self.object_load_cancel = Some(cancel);
        self.load_rx = Some(rx);
        self.property_load_rx = None;
        self.property_load_key = None;
        self.status = format!("Loading objects: {}", path.to_string_lossy());

        std::thread::Builder::new()
            .name("seg-objects-loader".to_string())
            .spawn(move || {
                let msg = load_in_thread(
                    path,
                    downsample_factor,
                    load_options,
                    request_id,
                    &cancel_worker,
                )
                .map_err(|error| error.to_string());
                let _ = tx.send(msg);
            })
            .ok();
    }

    pub(super) fn invalidate_filter_cache(&mut self) {
        self.filtered_ordered_indices = None;
        self.filtered_mask = None;
        self.filtered_render_lods = None;
        self.filtered_point_positions_world = None;
        self.filtered_point_values = None;
        self.filtered_point_lods = None;
        self.filtered_color_groups = None;
        self.filter_generation = self.filter_generation.wrapping_add(1).max(1);
        self.color_legend_cache = None;
        self.mark_live_analysis_selection_dirty();
        self.invalidate_object_property_analysis_cache();
        self.invalidate_table_cache();
    }

    pub(in crate::objects) fn ensure_filter_cache(&mut self) {
        self.reconcile_filter_clauses();
        if !self.has_active_filter() {
            if self.filtered_mask.is_some()
                || self.filtered_render_lods.is_some()
                || self.filtered_color_groups.is_some()
            {
                self.invalidate_filter_cache();
            }
            return;
        }
        if self.filtered_mask.is_some() {
            return;
        }
        if let Some(key) = self.unloaded_active_query_property_key() {
            self.ensure_property_loaded(&key);
            return;
        }
        let unloaded_key = (self.filter_mode == ObjectFilterMode::Simple)
            .then(|| {
                self.active_filter_clauses().find_map(|clause| {
                    (clause.property_key != "id"
                        && self.property_column_available_but_unloaded(&clause.property_key))
                    .then(|| clause.property_key.clone())
                })
            })
            .flatten();
        if let Some(key) = unloaded_key {
            self.ensure_property_loaded(&key);
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            return;
        };

        // Filtering materializes a subset snapshot plus the derived render/point/color products
        // for that subset. The rest of the layer reads from these caches instead of re-evaluating
        // the filter predicate on every paint or analysis pass.
        let prepared_clauses = self.prepare_filter_clauses();
        let mut ordered_indices = Vec::new();
        let mut mask = vec![false; objects.len()];
        let mut subset = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if !self.object_matches_active_filter(idx, obj, &prepared_clauses) {
                continue;
            }
            ordered_indices.push(idx);
            if let Some(slot) = mask.get_mut(idx) {
                *slot = true;
            }
            subset.push(obj.clone());
        }

        self.filtered_ordered_indices = Some(Arc::new(ordered_indices));
        self.filtered_mask = Some(Arc::new(mask));
        if subset.is_empty() {
            self.filtered_render_lods = None;
            self.filtered_point_positions_world = None;
            self.filtered_point_values = None;
            self.filtered_point_lods = None;
            self.filtered_color_groups = None;
        } else {
            self.filtered_render_lods = build_render_lods(&subset).ok();
            let (positions, values, lods) =
                build_object_point_payload(&subset, self.display_transform);
            self.filtered_point_positions_world = Some(positions);
            self.filtered_point_values = Some(values);
            self.filtered_point_lods = Some(lods);
            self.filtered_color_groups = if self.color_mode == ObjectColorMode::ByProperty
                && !self.color_property_key.is_empty()
            {
                let labels = objects
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| {
                        self.filtered_mask
                            .as_ref()
                            .and_then(|mask| mask.get(*idx))
                            .copied()
                            .unwrap_or(false)
                    })
                    .filter_map(|(idx, obj)| {
                        self.object_property_label(idx, obj, &self.color_property_key)
                            .map(|label| (idx, obj, label))
                    })
                    .collect::<Vec<_>>();
                build_color_groups_for_property_labels(labels, &self.color_property_key).ok()
            } else {
                None
            };
        }

        // Selection identity belongs to the shared object document. A filter
        // controls what this presentation draws, but must not delete selected
        // IDs merely because they are hidden in one viewport.
        self.visible_selected_render_cache = None;
        self.mark_live_analysis_selection_dirty();
        self.invalidate_table_cache();
    }

    pub(in crate::objects) fn ensure_color_groups(&mut self) {
        // Color grouping is lazily built against either the full set or the filtered subset,
        // depending on which view is currently active. This keeps legend/group generation aligned
        // with what the user actually sees.
        if self.has_active_filter() {
            self.ensure_filter_cache();
            if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty()
            {
                self.filtered_color_groups = None;
                return;
            }
            if self
                .filtered_color_groups
                .as_ref()
                .is_some_and(|g| g.property_key == self.color_property_key)
            {
                return;
            }
            self.ensure_filter_cache();
            let (Some(objects), Some(filtered_mask)) =
                (self.objects.as_ref(), self.filtered_mask.as_ref())
            else {
                self.filtered_color_groups = None;
                return;
            };
            let labels = objects
                .iter()
                .enumerate()
                .filter(|(idx, _)| filtered_mask.get(*idx).copied().unwrap_or(false))
                .filter_map(|(idx, obj)| {
                    self.object_property_label(idx, obj, &self.color_property_key)
                        .map(|label| (idx, obj, label))
                })
                .collect::<Vec<_>>();
            self.filtered_color_groups =
                build_color_groups_for_property_labels(labels, &self.color_property_key).ok();
            return;
        }
        self.filtered_color_groups = None;
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return;
        }
        if self
            .color_groups
            .as_ref()
            .is_some_and(|g| g.property_key == self.color_property_key)
        {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            return;
        };
        let labels = objects
            .iter()
            .enumerate()
            .filter_map(|(idx, obj)| {
                self.object_property_label(idx, obj, &self.color_property_key)
                    .map(|label| (idx, obj, label))
            })
            .collect::<Vec<_>>();
        self.color_groups =
            build_color_groups_for_property_labels(labels, &self.color_property_key).ok();
    }

    pub(in crate::objects) fn active_color_groups(&self) -> Option<&ObjectColorGroups> {
        if self.has_active_filter() {
            self.filtered_color_groups.as_ref()
        } else {
            self.color_groups.as_ref()
        }
    }

    pub(in crate::objects) fn has_active_filter(&self) -> bool {
        match self.filter_mode {
            ObjectFilterMode::Simple => self.active_filter_clauses().next().is_some(),
            ObjectFilterMode::Query => self.filter_query_expr.is_some(),
        }
    }

    pub(in crate::objects) fn render_cache_generation(&self) -> u64 {
        if self.has_active_filter() {
            self.generation ^ self.filter_generation.rotate_left(17) ^ 0x9e37_79b9_7f4a_7c15
        } else {
            self.generation
        }
    }

    pub(in crate::objects) fn filtered_mask_contains(&self, idx: usize) -> bool {
        match self.filtered_mask.as_ref() {
            Some(mask) => mask.get(idx).copied().unwrap_or(false),
            None => true,
        }
    }

    pub(super) fn prepare_filter_clauses(&self) -> Vec<PreparedObjectFilterClause<'_>> {
        if self.filter_mode != ObjectFilterMode::Simple {
            return Vec::new();
        }
        self.active_filter_clauses()
            .map(|clause| {
                let column = self.property_store.loaded_columns.get(&clause.property_key);
                PreparedObjectFilterClause {
                    property_key: clause.property_key.as_str(),
                    needle: clause.query.trim().to_ascii_lowercase(),
                    column,
                    column_matcher: column.map(|column| column.contains_matcher(&clause.query)),
                }
            })
            .collect()
    }

    pub(super) fn unloaded_active_query_property_key(&self) -> Option<String> {
        if self.filter_mode != ObjectFilterMode::Query {
            return None;
        }
        self.filter_query_expr.as_ref().and_then(|expr| {
            expr.referenced_properties()
                .into_iter()
                .find(|key| self.property_column_available_but_unloaded(key))
        })
    }

    pub(super) fn object_matches_active_filter(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        prepared_clauses: &[PreparedObjectFilterClause<'_>],
    ) -> bool {
        match self.filter_mode {
            ObjectFilterMode::Simple => Self::object_matches_prepared_filter(
                object_index,
                obj,
                prepared_clauses,
                self.filter_logic,
            ),
            ObjectFilterMode::Query => self
                .filter_query_expr
                .as_ref()
                .is_none_or(|expr| expr.matches(object_index, obj, &self.property_store)),
        }
    }

    pub(super) fn object_matches_prepared_filter(
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        clauses: &[PreparedObjectFilterClause<'_>],
        logic: ObjectFilterLogic,
    ) -> bool {
        if clauses.is_empty() {
            return true;
        }

        let clause_matches = |clause: &PreparedObjectFilterClause<'_>| {
            if clause.property_key == "id" {
                return obj.id.to_ascii_lowercase().contains(&clause.needle);
            }

            if let (Some(column), Some(matcher)) = (clause.column, clause.column_matcher.as_ref()) {
                return column.matches_contains(object_index, matcher);
            }

            let Some(value) = obj.inline_properties.get(clause.property_key) else {
                return false;
            };
            value_to_display_text(value)
                .to_ascii_lowercase()
                .contains(&clause.needle)
        };

        match logic {
            ObjectFilterLogic::All => clauses.iter().all(clause_matches),
            ObjectFilterLogic::Any => clauses.iter().any(clause_matches),
        }
    }

    pub(in crate::objects) fn color_value_visible_for_label(
        &self,
        property_key: &str,
        value_label: &str,
    ) -> bool {
        if let Some(pending) = self.pending_color_value_visibility.as_ref()
            && pending.property_key == property_key
        {
            let value = normalize_color_value_label(value_label);
            let visible_values = pending
                .visible_values
                .iter()
                .map(|value| normalize_color_value_label(value))
                .collect::<HashSet<_>>();
            let hidden_values = pending
                .hidden_values
                .iter()
                .map(|value| normalize_color_value_label(value))
                .collect::<HashSet<_>>();
            if !visible_values.is_empty() && !visible_values.contains(&value) {
                return false;
            }
            if hidden_values.contains(&value) {
                return false;
            }
        }
        if self.color_level_overrides_property_key == property_key
            && let Some(style) = self.color_level_overrides.get(value_label)
            && !style.visible
        {
            return false;
        }
        true
    }

    pub(super) fn object_color_value_visible(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
    ) -> bool {
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return true;
        }
        let Some(value_label) =
            self.object_property_label(object_index, obj, &self.color_property_key)
        else {
            return true;
        };
        self.color_value_visible_for_label(&self.color_property_key, &value_label)
    }

    pub(in crate::objects) fn is_index_visible(&self, idx: usize) -> bool {
        if !self.filtered_mask_contains(idx) {
            return false;
        }
        let Some(objects) = self.objects.as_ref() else {
            return true;
        };
        objects
            .get(idx)
            .map(|obj| self.object_color_value_visible(idx, obj))
            .unwrap_or(true)
    }
}
