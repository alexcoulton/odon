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
        let next_mapping = if next_mode == ObjectColorMode::ByProperty {
            ObjectColorMapping::categorical(next_key.clone())
        } else {
            ObjectColorMapping::Single
        };
        let needs_property_load = next_mode == ObjectColorMode::ByProperty
            && self.property_column_available_but_unloaded(next_key.as_str());
        if self.color_mode == next_mode
            && self.color_property_key == next_key
            && (!needs_property_load
                || self.property_load_key.as_deref() == Some(next_key.as_str()))
        {
            self.color_mapping = next_mapping;
            self.resolved_continuous_domain = None;
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
        self.color_mapping = next_mapping;
        self.resolved_continuous_domain = None;
        self.continuous_color_payload = None;
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

    pub fn set_color_mapping(&mut self, mapping: ObjectColorMapping) -> Result<(), String> {
        mapping.validate()?;
        if self.color_mapping == mapping {
            if let ObjectColorMapping::Continuous { property, .. } = &mapping {
                self.ensure_property_loaded(property);
                self.resolve_continuous_domain();
            }
            return Ok(());
        }
        match &mapping {
            ObjectColorMapping::Single => {
                self.set_color_by_property(None);
                return Ok(());
            }
            ObjectColorMapping::Categorical { property } => {
                self.set_color_by_property(Some(property.clone()));
                return Ok(());
            }
            ObjectColorMapping::Continuous { property, .. } => {
                let property = property.clone();
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
                self.color_mode = ObjectColorMode::Continuous;
                self.color_property_key = property.clone();
                self.color_mapping = mapping;
                if self
                    .status
                    .ends_with("has too many distinct values for Color by.")
                {
                    self.status = format!("Loaded {} object(s).", self.object_count());
                }
                self.ensure_property_loaded(&property);
                self.resolve_continuous_domain();
                self.continuous_color_payload = None;
                self.filtered_color_groups = None;
                self.color_legend_cache = None;
                self.generation = self.generation.wrapping_add(1).max(1);
            }
        }
        Ok(())
    }

    pub fn color_mapping(&self) -> &ObjectColorMapping {
        &self.color_mapping
    }

    pub fn resolved_continuous_domain(&self) -> Option<[f64; 2]> {
        self.resolved_continuous_domain
    }

    pub(crate) fn numeric_property_domain(&mut self, property: &str) -> Option<[f64; 2]> {
        if self.property_column_available_but_unloaded(property) {
            self.ensure_property_loaded(property);
            return None;
        }
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        if let Some(values) = self.property_store.numeric_pairs(property) {
            for (_, value) in values {
                let value = f64::from(value);
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
        } else if let Some(objects) = self.objects.as_ref() {
            for value in objects.iter().filter_map(|object| {
                object
                    .inline_properties
                    .get(property)
                    .and_then(numeric_json_value)
                    .map(f64::from)
            }) {
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
        }
        (minimum.is_finite() && maximum.is_finite()).then_some([minimum, maximum])
    }

    pub(in crate::objects) fn ensure_continuous_color_payload(
        &mut self,
    ) -> Option<&ObjectContinuousColorPayload> {
        if self.color_mode != ObjectColorMode::Continuous {
            self.continuous_color_payload = None;
            return None;
        }
        if self.resolved_continuous_domain.is_none() {
            self.resolve_continuous_domain();
        }
        let resolved_domain = self.resolved_continuous_domain?;
        if self
            .continuous_color_payload
            .as_ref()
            .is_some_and(|payload| {
                payload.mapping == self.color_mapping && payload.resolved_domain == resolved_domain
            })
        {
            return self.continuous_color_payload.as_ref();
        }
        let property = self.color_mapping.property()?.to_string();
        let config = self.color_mapping.continuous_config()?;
        let objects = self.objects.as_ref()?;
        let mut colors_rgba = Vec::with_capacity(objects.len());
        let mut numeric_count = 0usize;
        for (index, object) in objects.iter().enumerate() {
            let value = self
                .property_store
                .numeric_at(&property, index)
                .or_else(|| {
                    object.inline_properties.get(&property).and_then(|value| {
                        value
                            .as_f64()
                            .or_else(|| value.as_str().and_then(|value| value.parse::<f64>().ok()))
                            .filter(|value| value.is_finite())
                    })
                });
            numeric_count += usize::from(value.is_some());
            colors_rgba.push(config.color_rgba(value, resolved_domain));
        }
        self.continuous_color_generation = self.continuous_color_generation.wrapping_add(1).max(1);
        self.continuous_color_payload = Some(ObjectContinuousColorPayload {
            mapping: self.color_mapping.clone(),
            resolved_domain,
            colors_rgba: Arc::new(colors_rgba),
            numeric_count,
            missing_count: objects.len().saturating_sub(numeric_count),
            generation: self.continuous_color_generation,
        });
        self.continuous_color_payload.as_ref()
    }

    pub(super) fn resolve_continuous_domain(&mut self) {
        let ObjectColorMapping::Continuous {
            property,
            domain,
            scale,
            ..
        } = &self.color_mapping
        else {
            self.resolved_continuous_domain = None;
            return;
        };
        if let Some(domain) = domain.fixed() {
            self.resolved_continuous_domain = Some(domain);
            return;
        }
        let property = property.clone();
        let values = if let Some(values) = self.property_store.numeric_pairs(&property) {
            values
                .into_iter()
                .map(|(_, value)| f64::from(value))
                .collect::<Vec<_>>()
        } else {
            self.objects
                .as_ref()
                .into_iter()
                .flat_map(|objects| objects.iter())
                .filter_map(|object| {
                    object
                        .inline_properties
                        .get(&property)
                        .and_then(numeric_json_value)
                        .map(f64::from)
                })
                .collect::<Vec<_>>()
        };
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for value in values
            .into_iter()
            .filter(|value| value.is_finite())
            .filter(|value| *scale != ContinuousScale::Log10 || *value > 0.0)
        {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        self.resolved_continuous_domain =
            (minimum.is_finite() && maximum.is_finite()).then_some([minimum, maximum]);
        self.continuous_color_payload = None;
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
        self.continuous_color_payload = None;
        if self.color_mode == ObjectColorMode::Continuous {
            self.resolve_continuous_domain();
        }
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
        self.color_mapping = ObjectColorMapping::Single;
        self.resolved_continuous_domain = None;
    }
}
