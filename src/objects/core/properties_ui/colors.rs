use super::*;

impl ObjectsLayer {
    pub fn active_color_legend_entries(&mut self) -> Option<Vec<ObjectColorLegendEntry>> {
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return None;
        }
        if let Some(cache) = self.color_legend_cache.as_ref()
            && cache.property_key == self.color_property_key
            && cache.generation == self.generation
        {
            return Some(cache.entries.clone());
        }

        use std::collections::BTreeMap;

        let objects = self.objects.as_ref()?;
        let mut counts = BTreeMap::<String, usize>::new();
        if let Some(filtered_ordered_indices) = self.filtered_ordered_indices.as_ref() {
            for idx in filtered_ordered_indices.iter() {
                let Some(obj) = objects.get(*idx) else {
                    continue;
                };
                let Some(value_label) =
                    self.object_property_label(*idx, obj, &self.color_property_key)
                else {
                    continue;
                };
                *counts.entry(value_label).or_default() += 1;
            }
        } else {
            for (idx, obj) in objects.iter().enumerate() {
                let Some(value_label) =
                    self.object_property_label(idx, obj, &self.color_property_key)
                else {
                    continue;
                };
                *counts.entry(value_label).or_default() += 1;
            }
        }

        if counts.is_empty() {
            return None;
        }

        let entries = counts
            .into_iter()
            .map(|(value_label, count)| ObjectColorLegendEntry {
                color_rgb: hashed_color_rgb(&self.color_property_key, &value_label),
                count,
                value_label,
            })
            .collect::<Vec<_>>();
        self.color_legend_cache = Some(ObjectColorLegendCache {
            property_key: self.color_property_key.clone(),
            generation: self.generation,
            entries: entries.clone(),
        });
        Some(entries)
    }

    pub(crate) fn active_color_value_visibility_snapshot(
        &mut self,
    ) -> Option<(String, Vec<String>, Vec<String>)> {
        let property_key = self.color_property_key.clone();
        let entries = self.active_color_legend_entries()?;
        let mut visible_values = Vec::new();
        let mut hidden_values = Vec::new();
        for entry in entries {
            if self.color_value_visible_for_label(&property_key, &entry.value_label) {
                visible_values.push(entry.value_label);
            } else {
                hidden_values.push(entry.value_label);
            }
        }
        Some((property_key, visible_values, hidden_values))
    }

    pub fn set_color_by_property(&mut self, property_key: Option<String>) {
        let (next_mode, next_key) = match property_key {
            Some(property_key) if !property_key.is_empty() => {
                (ObjectColorMode::ByProperty, property_key)
            }
            _ => (ObjectColorMode::Single, String::new()),
        };
        let needs_property_load = next_mode == ObjectColorMode::ByProperty
            && self.property_column_available_but_unloaded(next_key.as_str());
        if self.color_mode == next_mode
            && self.color_property_key == next_key
            && (!needs_property_load
                || self.property_load_key.as_deref() == Some(next_key.as_str()))
        {
            return;
        }
        // A legend can be inspected before the render path has materialized
        // its color groups. Build the current unfiltered groups before
        // switching properties so another viewport can restore them without
        // recomputing or losing the property's cached presentation.
        if self.color_mode == ObjectColorMode::ByProperty
            && !self.color_property_key.is_empty()
            && self.color_groups.is_none()
            && !self.has_active_filter()
        {
            self.ensure_color_groups();
        }
        if let Some(groups) = self.color_groups.take() {
            self.color_groups_cache
                .insert(groups.property_key.clone(), groups);
        }
        self.color_mode = next_mode;
        self.color_property_key = next_key;
        if self.color_mode == ObjectColorMode::ByProperty {
            let key = self.color_property_key.clone();
            self.ensure_property_loaded(key.as_str());
            if !self.is_loading() {
                self.reconcile_active_color_property();
            }
            self.apply_pending_color_value_colors();
            self.apply_pending_color_value_visibility();
        }
        self.color_groups = (self.color_mode == ObjectColorMode::ByProperty)
            .then(|| self.color_groups_cache.remove(&self.color_property_key))
            .flatten();
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub(crate) fn project_display_state(&self) -> ObjectProjectDisplayState {
        let color_property_key = (self.color_mode == ObjectColorMode::ByProperty)
            .then(|| self.color_property_key.clone())
            .filter(|key| !key.is_empty());
        let color_level_overrides = if color_property_key.as_deref()
            == Some(self.color_level_overrides_property_key.as_str())
        {
            self.color_level_overrides.clone()
        } else {
            BTreeMap::new()
        };
        ObjectProjectDisplayState {
            color_property_key,
            color_level_overrides,
            fill_cells: self.fill_cells,
            fill_opacity: self.fill_opacity,
            selected_fill_opacity: self.selected_fill_opacity,
            fast_rendering: self.fast_rendering,
        }
    }

    pub(crate) fn apply_project_display_state(&mut self, state: &ObjectProjectDisplayState) {
        self.set_color_by_property(state.color_property_key.clone());
        // Project and control-actor presentation is declarative. Preserve a requested property
        // even when its object column has not materialized yet; the resource loader can satisfy
        // it later without losing the canonical presentation while the renderer is catching up.
        if let Some(property_key) = state
            .color_property_key
            .as_deref()
            .filter(|property_key| !property_key.is_empty())
        {
            self.color_mode = ObjectColorMode::ByProperty;
            self.color_property_key = property_key.to_string();
        }
        self.color_level_overrides_property_key =
            state.color_property_key.clone().unwrap_or_default();
        self.color_level_overrides = state.color_level_overrides.clone();
        self.fill_cells = state.fill_cells;
        self.fill_opacity = state.fill_opacity.clamp(0.0, 1.0);
        self.selected_fill_opacity = state.selected_fill_opacity.clamp(0.0, 1.0);
        self.fast_rendering = state.fast_rendering;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub(crate) fn viewport_filter_state(&self) -> ObjectViewportFilterState {
        ObjectViewportFilterState {
            mode: self.filter_mode,
            clauses: self.filter_clauses.clone(),
            logic: self.filter_logic,
            query_text: self.filter_query_text.clone(),
            query_expr: self.filter_query_expr.clone(),
            query_error: self.filter_query_error.clone(),
        }
    }

    pub(crate) fn viewport_filter_cache_state(&self) -> ObjectViewportFilterCacheState {
        ObjectViewportFilterCacheState {
            filtered_ordered_indices: self.filtered_ordered_indices.clone(),
            filtered_mask: self.filtered_mask.clone(),
            filtered_render_lods: self.filtered_render_lods.clone(),
            filtered_point_positions_world: self.filtered_point_positions_world.clone(),
            filtered_point_values: self.filtered_point_values.clone(),
            filtered_point_lods: self.filtered_point_lods.clone(),
            filtered_color_groups: self.filtered_color_groups.clone(),
            filter_generation: self.filter_generation,
        }
    }

    pub(crate) fn apply_viewport_filter_cache_state(
        &mut self,
        state: &ObjectViewportFilterCacheState,
    ) {
        self.filtered_ordered_indices
            .clone_from(&state.filtered_ordered_indices);
        self.filtered_mask.clone_from(&state.filtered_mask);
        self.filtered_render_lods
            .clone_from(&state.filtered_render_lods);
        self.filtered_point_positions_world
            .clone_from(&state.filtered_point_positions_world);
        self.filtered_point_values
            .clone_from(&state.filtered_point_values);
        self.filtered_point_lods
            .clone_from(&state.filtered_point_lods);
        self.filtered_color_groups
            .clone_from(&state.filtered_color_groups);
        self.filter_generation = state.filter_generation;
        self.visible_selected_render_cache = None;
    }

    pub(crate) fn apply_viewport_filter_state(&mut self, state: &ObjectViewportFilterState) {
        if self.filter_mode == state.mode
            && self.filter_clauses == state.clauses
            && self.filter_logic == state.logic
            && self.filter_query_text == state.query_text
            && self.filter_query_expr == state.query_expr
            && self.filter_query_error == state.query_error
        {
            return;
        }
        self.filter_mode = state.mode;
        self.filter_clauses.clone_from(&state.clauses);
        self.filter_logic = state.logic;
        self.filter_query_text.clone_from(&state.query_text);
        self.filter_query_expr.clone_from(&state.query_expr);
        self.filter_query_error.clone_from(&state.query_error);
        self.ensure_filter_clause_row();
        self.ensure_active_filter_properties_loaded();
        self.invalidate_filter_cache();
    }

    pub(crate) fn apply_project_display_state_preserving_color_visibility(
        &mut self,
        state: &ObjectProjectDisplayState,
    ) {
        let runtime_color_key = self.color_property_key.clone();
        let runtime_overrides_key = self.color_level_overrides_property_key.clone();
        let runtime_overrides = self.color_level_overrides.clone();
        let preserve_runtime_overrides = !runtime_color_key.is_empty()
            && runtime_overrides_key == runtime_color_key
            && state.color_property_key.as_deref() == Some(runtime_color_key.as_str())
            && state.color_level_overrides.is_empty()
            && runtime_overrides
                .values()
                .any(|style| !style.visible || style.color_rgb.is_some());

        self.apply_project_display_state(state);

        if preserve_runtime_overrides {
            crate::log_warn!(
                "objects: preserving runtime Color by overrides for '{}' after project display restore",
                runtime_color_key
            );
            self.color_level_overrides_property_key = runtime_overrides_key;
            self.color_level_overrides = runtime_overrides;
            self.color_groups = None;
            self.filtered_color_groups = None;
            self.color_legend_cache = None;
            self.ensure_color_groups();
            self.generation = self.generation.wrapping_add(1).max(1);
        }
    }

    pub(crate) fn clear_project_display_state(&mut self) {
        self.set_color_by_property(None);
        self.color_level_overrides_property_key.clear();
        self.color_level_overrides.clear();
        self.pending_color_value_colors = None;
        self.pending_color_value_visibility = None;
        self.fill_cells = false;
        self.fill_opacity = 0.30;
        self.selected_fill_opacity = 0.70;
        self.fast_rendering = true;
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
    }

    pub fn set_color_level_overrides(
        &mut self,
        property_key: Option<&str>,
        overrides: &std::collections::HashMap<String, ObjectColorLevelOverride>,
    ) {
        let property_key = property_key.unwrap_or_default();
        self.color_level_overrides_property_key = property_key.to_string();
        self.color_level_overrides.clear();
        if !property_key.is_empty() {
            self.color_level_overrides
                .extend(overrides.iter().map(|(key, value)| (key.clone(), *value)));
        }
    }

    #[cfg(test)]
    pub(crate) fn set_color_value_visibility(
        &mut self,
        property_key: Option<&str>,
        visible_values: &[String],
        hidden_values: &[String],
    ) {
        let property_key = property_key
            .filter(|key| !key.trim().is_empty())
            .unwrap_or(self.color_property_key.as_str())
            .trim();
        if property_key.is_empty() || (visible_values.is_empty() && hidden_values.is_empty()) {
            return;
        }
        self.pending_color_value_visibility = Some(PendingColorValueVisibility {
            property_key: property_key.to_string(),
            visible_values: visible_values.to_vec(),
            hidden_values: hidden_values.to_vec(),
        });
        self.apply_pending_color_value_visibility();
    }

    #[cfg(test)]
    pub(crate) fn set_color_value_colors(
        &mut self,
        property_key: Option<&str>,
        colors: &[(String, [u8; 3])],
    ) {
        let property_key = property_key
            .filter(|key| !key.trim().is_empty())
            .unwrap_or(self.color_property_key.as_str())
            .trim();
        if property_key.is_empty() || colors.is_empty() {
            return;
        }
        self.pending_color_value_colors = Some(PendingColorValueColors {
            property_key: property_key.to_string(),
            colors: colors.to_vec(),
        });
        self.color_level_overrides_property_key = property_key.to_string();
        for (value, color_rgb) in colors {
            self.color_level_overrides
                .entry(value.clone())
                .or_default()
                .color_rgb = Some(*color_rgb);
        }
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.color_legend_cache = None;
        self.generation = self.generation.wrapping_add(1).max(1);
        self.apply_pending_color_value_colors();
    }

    pub(in crate::objects::core) fn apply_pending_color_value_colors(&mut self) {
        let Some(pending) = self.pending_color_value_colors.clone() else {
            return;
        };
        if self.color_mode != ObjectColorMode::ByProperty
            || self.color_property_key != pending.property_key
        {
            return;
        }

        let Some(entries) = self.active_color_legend_entries() else {
            return;
        };

        let requested_colors = pending
            .colors
            .iter()
            .map(|(value, color_rgb)| (normalize_color_value_label(value), *color_rgb))
            .collect::<HashMap<_, _>>();

        self.color_level_overrides_property_key = pending.property_key.clone();
        let mut applied_count = 0usize;
        for entry in entries {
            let normalized = normalize_color_value_label(&entry.value_label);
            let Some(color_rgb) = requested_colors.get(&normalized).copied() else {
                continue;
            };
            let override_style = self
                .color_level_overrides
                .entry(entry.value_label)
                .or_default();
            override_style.color_rgb = (color_rgb != entry.color_rgb).then_some(color_rgb);
            applied_count += 1;
        }
        crate::log_warn!(
            "objects: applied Color by colours for '{}' ({} legend value(s))",
            pending.property_key,
            applied_count
        );
        self.pending_color_value_colors = None;
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.ensure_color_groups();
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub(in crate::objects::core) fn apply_pending_color_value_visibility(&mut self) {
        let Some(pending) = self.pending_color_value_visibility.clone() else {
            return;
        };
        if self.color_mode != ObjectColorMode::ByProperty
            || self.color_property_key != pending.property_key
        {
            return;
        }

        let Some(entries) = self.active_color_legend_entries() else {
            return;
        };

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

        self.color_level_overrides_property_key = pending.property_key.clone();
        let mut hidden_count = 0usize;
        for entry in entries {
            let normalized = normalize_color_value_label(&entry.value_label);
            let mut visible = if visible_values.is_empty() {
                true
            } else {
                visible_values.contains(&normalized)
            };
            if hidden_values.contains(&normalized) {
                visible = false;
            }
            self.color_level_overrides
                .entry(entry.value_label)
                .or_default()
                .visible = visible;
            if !visible {
                hidden_count += 1;
            }
        }
        crate::log_warn!(
            "objects: applied Color by visibility for '{}' ({} hidden legend value(s))",
            pending.property_key,
            hidden_count
        );
        self.pending_color_value_visibility = None;
        self.color_groups = None;
        self.filtered_color_groups = None;
        self.ensure_color_groups();
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub(in crate::objects) fn property_column_available_but_unloaded(
        &self,
        property_key: &str,
    ) -> bool {
        if self.property_store.has_loaded(property_key) {
            return false;
        }
        self.lazy_parquet_source.as_ref().is_some_and(|source| {
            source
                .available_property_columns
                .iter()
                .any(|key| key == property_key)
                && !source.loaded_property_columns.contains(property_key)
        })
    }

    pub(in crate::objects) fn ensure_property_loaded(&mut self, property_key: &str) {
        let Some(source) = self.lazy_parquet_source.as_ref() else {
            return;
        };
        if source.loaded_property_columns.contains(property_key) {
            return;
        }
        if !source
            .available_property_columns
            .iter()
            .any(|key| key == property_key)
        {
            return;
        }
        let Some(path) = self.loaded_geojson.as_ref() else {
            return;
        };
        if self.property_load_key.as_deref() == Some(property_key)
            && self.property_load_rx.is_some()
        {
            return;
        };

        let property_key_owned = property_key.to_string();
        let path = path.clone();
        let (tx, rx) = crossbeam_channel::bounded::<PropertyLoadResult>(1);
        self.property_load_rx = Some(rx);
        self.property_load_key = Some(property_key_owned.clone());
        self.status = format!("Loading property '{property_key}'...");

        std::thread::Builder::new()
            .name(format!("seg-objects-property-loader-{property_key}"))
            .spawn(move || {
                if let Ok(values_by_row) = load_parquet_property_values_for_loaded_objects(
                    &path,
                    property_key_owned.as_str(),
                ) {
                    let _ = tx.send(PropertyLoadResult {
                        property_key: property_key_owned,
                        values_by_row,
                    });
                }
            })
            .ok();
    }

    pub(in crate::objects::core) fn apply_loaded_property_values(
        &mut self,
        property_key: &str,
        property_values: &HashMap<usize, serde_json::Value>,
    ) {
        let Some(objects) = self.objects.as_ref() else {
            return;
        };
        let column = ObjectPropertyColumn::from_values_by_row(objects, property_values);
        let is_categorical = column.is_categorical(24);
        self.property_store
            .insert_column(property_key.to_string(), column);
        if let Some(source) = self.lazy_parquet_source.as_mut() {
            source
                .loaded_property_columns
                .insert(property_key.to_string());
        }
        if is_categorical
            && !self
                .color_property_keys
                .iter()
                .any(|key| key == property_key)
        {
            self.color_property_keys.push(property_key.to_string());
            self.color_property_keys.sort();
        }
        self.color_legend_cache = None;
        self.color_groups = None;
        self.invalidate_filter_cache();
        self.reset_object_property_analysis_cache();
        self.generation = self.generation.wrapping_add(1).max(1);
        self.reconcile_active_color_property();
        self.apply_pending_color_value_visibility();
        let n = self.object_count();
        self.status = format!("Loaded {n} object(s).");
    }

    pub(super) fn reconcile_active_color_property(&mut self) {
        if self.color_mode != ObjectColorMode::ByProperty || self.color_property_key.is_empty() {
            return;
        }
        if self
            .color_property_keys
            .iter()
            .any(|loaded| loaded == &self.color_property_key)
            || self
                .property_store
                .loaded_column_is_categorical(self.color_property_key.as_str(), 24)
        {
            return;
        }
        if self.property_column_available_but_unloaded(self.color_property_key.as_str())
            || self.property_load_key.as_deref() == Some(self.color_property_key.as_str())
        {
            return;
        }
        self.status = format!(
            "Property '{}' has too many distinct values for Color by.",
            self.color_property_key
        );
        self.color_mode = ObjectColorMode::Single;
        self.color_property_key.clear();
    }
}
