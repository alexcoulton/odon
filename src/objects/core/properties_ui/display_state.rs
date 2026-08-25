use super::*;

impl ObjectsLayer {
    pub(crate) fn project_display_state(&self) -> ObjectProjectDisplayState {
        let color_property_key = (self.color_mode != ObjectColorMode::Single)
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
            color_mapping: Some(self.color_mapping.clone()),
            color_level_overrides,
            fill_cells: self.fill_cells,
            fill_opacity: self.fill_opacity,
            selected_fill_opacity: self.selected_fill_opacity,
            fast_rendering: self.fast_rendering,
        }
    }

    pub(crate) fn apply_project_display_state(&mut self, state: &ObjectProjectDisplayState) {
        let mapping = state.color_mapping.clone().unwrap_or_else(|| {
            state
                .color_property_key
                .clone()
                .map(ObjectColorMapping::categorical)
                .unwrap_or(ObjectColorMapping::Single)
        });
        if let Err(error) = self.set_color_mapping(mapping) {
            self.status = error;
            self.set_color_by_property(None);
        }
        // Preserve declarative actor/project presentation while lazy properties materialize.
        if let Some(requested) = state.color_mapping.as_ref() {
            self.color_mapping = requested.clone();
            match requested {
                ObjectColorMapping::Single => {
                    self.color_mode = ObjectColorMode::Single;
                    self.color_property_key.clear();
                }
                ObjectColorMapping::Categorical { property } => {
                    self.color_mode = ObjectColorMode::ByProperty;
                    self.color_property_key = property.clone();
                }
                ObjectColorMapping::Continuous { property, .. } => {
                    self.color_mode = ObjectColorMode::Continuous;
                    self.color_property_key = property.clone();
                }
            }
        } else if let Some(property_key) = state
            .color_property_key
            .as_deref()
            .filter(|property_key| !property_key.is_empty())
        {
            self.color_mode = ObjectColorMode::ByProperty;
            self.color_property_key = property_key.to_string();
            self.color_mapping = ObjectColorMapping::categorical(property_key);
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
        self.resolved_continuous_domain = None;
        self.continuous_color_payload = None;
    }
}
