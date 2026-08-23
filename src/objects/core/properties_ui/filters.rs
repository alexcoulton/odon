use super::*;

impl ObjectsLayer {
    pub(in crate::objects::core) fn object_property_label(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        property_key: &str,
    ) -> Option<String> {
        self.property_store
            .label_at(property_key, object_index)
            .or_else(|| {
                obj.inline_properties
                    .get(property_key)
                    .and_then(property_scalar_value)
            })
    }

    pub(super) fn object_property_display(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
        property_key: &str,
    ) -> Option<String> {
        self.property_store
            .loaded_columns
            .get(property_key)
            .and_then(|column| column.display_at(object_index))
            .or_else(|| {
                obj.inline_properties
                    .get(property_key)
                    .map(value_to_display_text)
                    .filter(|value| !value.trim().is_empty())
            })
    }

    pub(in crate::objects) fn loaded_property_display_pairs(
        &self,
        object_index: usize,
        obj: &GeoJsonObjectFeature,
    ) -> Vec<(String, String)> {
        let mut keys = obj
            .inline_properties
            .keys()
            .cloned()
            .collect::<HashSet<_>>();
        keys.extend(self.property_store.loaded_keys());
        let mut properties = keys
            .into_iter()
            .filter_map(|key| {
                self.object_property_display(object_index, obj, &key)
                    .map(|value| (key, value))
            })
            .collect::<Vec<_>>();
        properties.sort_by(|a, b| a.0.cmp(&b.0));
        properties
    }

    pub(super) fn filter_property_candidates(&self) -> Vec<(String, bool)> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for key in &self.scalar_property_keys {
            if seen.insert(key.clone()) {
                out.push((key.clone(), false));
            }
        }
        for key in self.property_store.loaded_keys() {
            if seen.insert(key.clone()) {
                out.push((key, false));
            }
        }
        if let Some(source) = self.lazy_parquet_source.as_ref() {
            for key in &source.available_property_columns {
                if !seen.contains(key)
                    && !source.loaded_property_columns.contains(key)
                    && !self.property_store.has_loaded(key)
                {
                    out.push((key.clone(), true));
                    seen.insert(key.clone());
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub(super) fn ensure_filter_clause_row(&mut self) {
        if self.filter_clauses.is_empty() {
            self.filter_clauses.push(ObjectFilterClause::default());
        }
    }

    pub(super) fn filter_property_key_available(&self, property_key: &str) -> bool {
        property_key == "id"
            || self
                .scalar_property_keys
                .iter()
                .any(|key| key == property_key)
            || self.property_store.has_loaded(property_key)
            || self.lazy_parquet_source.as_ref().is_some_and(|source| {
                source
                    .available_property_columns
                    .iter()
                    .any(|key| key == property_key)
            })
    }

    pub(in crate::objects::core) fn reconcile_filter_clauses(&mut self) {
        if self.filter_clauses.is_empty() {
            self.filter_clauses.push(ObjectFilterClause::default());
        }
        let validity = self
            .filter_clauses
            .iter()
            .map(|clause| self.filter_property_key_available(&clause.property_key))
            .collect::<Vec<_>>();
        for (clause, valid) in self.filter_clauses.iter_mut().zip(validity) {
            if !valid {
                clause.property_key = "id".to_string();
            }
        }
    }

    pub(in crate::objects::core) fn active_filter_clauses(
        &self,
    ) -> impl Iterator<Item = &ObjectFilterClause> {
        self.filter_clauses
            .iter()
            .filter(|clause| clause.enabled && !clause.query.trim().is_empty())
    }

    pub(super) fn ensure_active_filter_properties_loaded(&mut self) {
        if self.filter_mode == ObjectFilterMode::Query {
            if let Some(key) = self.unloaded_active_query_property_key() {
                self.ensure_property_loaded(&key);
            }
            return;
        }
        let key = self
            .active_filter_clauses()
            .filter_map(|clause| (clause.property_key != "id").then(|| clause.property_key.clone()))
            .find(|key| self.property_column_available_but_unloaded(key));
        if let Some(key) = key {
            self.ensure_property_loaded(&key);
        }
    }

    pub(super) fn filter_value_options_by_key(
        &self,
        max_options: usize,
    ) -> HashMap<String, Vec<String>> {
        let mut out = HashMap::new();
        for (key, _) in self.filter_property_candidates() {
            if let Some(options) = self.property_store.filter_value_options(&key, max_options) {
                out.insert(key, options);
            }
        }
        out
    }

    pub(super) fn apply_filter_query_text(&mut self) -> bool {
        let query = self.filter_query_text.trim();
        if query.is_empty() {
            let changed = self.filter_query_expr.is_some() || self.filter_query_error.is_some();
            self.filter_query_expr = None;
            self.filter_query_error = None;
            return changed;
        }

        let expr = match ObjectFilterQueryExpr::parse(query) {
            Ok(expr) => expr,
            Err(err) => {
                self.filter_query_error = Some(err.to_string());
                return false;
            }
        };
        let missing = expr
            .referenced_properties()
            .into_iter()
            .filter(|key| !self.filter_property_key_available(key))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            self.filter_query_error = Some(format!(
                "Unknown object propert{}: {}",
                if missing.len() == 1 { "y" } else { "ies" },
                missing.join(", ")
            ));
            return false;
        }

        self.filter_query_error = None;
        self.filter_query_expr = Some(expr);
        true
    }

    #[cfg(test)]
    pub(crate) fn set_filter_clauses_from_pairs(&mut self, pairs: &[(String, String)]) {
        self.filter_mode = ObjectFilterMode::Simple;
        self.filter_clauses = pairs
            .iter()
            .filter_map(|(property_key, query)| {
                let property_key = property_key.trim();
                let query = query.trim();
                (!property_key.is_empty() && !query.is_empty()).then(|| ObjectFilterClause {
                    enabled: true,
                    property_key: property_key.to_string(),
                    query: query.to_string(),
                })
            })
            .collect();
        self.ensure_filter_clause_row();

        if self.objects.is_some()
            || self.lazy_parquet_source.is_some()
            || !self.object_property_keys.is_empty()
            || !self.scalar_property_keys.is_empty()
        {
            self.reconcile_filter_clauses();
            self.ensure_active_filter_properties_loaded();
            self.invalidate_filter_cache();
            self.ensure_filter_cache();
            self.ensure_color_groups();
        } else {
            self.invalidate_filter_cache();
        }
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    #[cfg(test)]
    pub(crate) fn set_filter_logic(&mut self, logic: ObjectFilterLogic) {
        let mode_changed = self.filter_mode != ObjectFilterMode::Simple;
        if self.filter_logic == logic && !mode_changed {
            return;
        }
        self.filter_mode = ObjectFilterMode::Simple;
        self.filter_logic = logic;
        self.invalidate_filter_cache();
        self.ensure_filter_cache();
        self.ensure_color_groups();
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    #[cfg(test)]
    pub(crate) fn set_filter_query_from_text(&mut self, query: &str) {
        self.filter_mode = ObjectFilterMode::Query;
        self.filter_query_text = query.trim().to_string();
        let changed = self.apply_filter_query_text();
        if changed {
            self.ensure_active_filter_properties_loaded();
            self.invalidate_filter_cache();
            self.ensure_filter_cache();
            self.ensure_color_groups();
            self.generation = self.generation.wrapping_add(1).max(1);
        } else if self.filter_query_error.is_some() {
            self.invalidate_filter_cache();
            self.generation = self.generation.wrapping_add(1).max(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_filter(&mut self) {
        self.filter_mode = ObjectFilterMode::Simple;
        self.filter_logic = ObjectFilterLogic::All;
        self.filter_clauses = vec![ObjectFilterClause::default()];
        self.filter_query_text.clear();
        self.filter_query_expr = None;
        self.filter_query_error = None;
        self.invalidate_filter_cache();
        self.ensure_filter_cache();
        self.ensure_color_groups();
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    #[cfg(test)]
    pub(crate) fn filter_snapshot_json(&mut self) -> serde_json::Value {
        self.ensure_filter_cache();
        self.ensure_color_groups();
        let total_count = self
            .objects
            .as_ref()
            .map(|objects| objects.len())
            .unwrap_or(0);
        let active = self.has_active_filter();
        let visible_count = if active {
            self.filtered_ordered_indices
                .as_ref()
                .map(|indices| indices.len())
                .unwrap_or(0)
        } else {
            total_count
        };
        let mode = match self.filter_mode {
            ObjectFilterMode::Simple => "simple",
            ObjectFilterMode::Query => "query",
        };
        let logic = match self.filter_logic {
            ObjectFilterLogic::All => "all",
            ObjectFilterLogic::Any => "any",
        };
        let clauses = self
            .filter_clauses
            .iter()
            .map(|clause| {
                serde_json::json!({
                    "enabled": clause.enabled,
                    "property": clause.property_key.as_str(),
                    "query": clause.query.as_str(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "revision": self.filter_generation,
            "active": active,
            "mode": mode,
            "logic": logic,
            "total_count": total_count,
            "visible_count": visible_count,
            "hidden_count": total_count.saturating_sub(visible_count),
            "simple": {
                "logic": logic,
                "clauses": clauses,
            },
            "query": {
                "text": self.filter_query_text.as_str(),
                "applied": self.filter_query_expr.is_some(),
                "error": self.filter_query_error.as_deref(),
            },
        })
    }
}
