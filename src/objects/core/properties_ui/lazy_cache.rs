use super::*;

impl ObjectsLayer {
    /// Bound only columns hydrated on demand from the lazy GeoParquet source.
    /// Geometry, inline values, and columns installed with the source stay resident.
    pub(crate) fn set_lazy_property_cache_capacity(&mut self, capacity: Option<usize>) {
        self.lazy_property_cache_capacity = capacity;
        self.enforce_lazy_property_cache_capacity();
    }

    pub(crate) fn lazy_property_cache_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "policy": if self.lazy_property_cache_capacity.is_some() { "lru" } else { "unbounded" },
            "capacity": self.lazy_property_cache_capacity,
            "resident_lazy_columns": self.lazy_property_lru.len(),
            "evictions": self.lazy_property_cache_evictions,
        })
    }

    pub(super) fn touch_lazy_property_cache_key(&mut self, property_key: &str) {
        if !self
            .lazy_property_lru
            .iter()
            .any(|loaded| loaded == property_key)
        {
            return;
        }
        self.lazy_property_lru
            .retain(|loaded| loaded != property_key);
        self.lazy_property_lru.push_back(property_key.to_string());
    }

    fn pinned_lazy_property_keys(&self) -> HashSet<String> {
        let mut pinned = HashSet::new();
        if !self.color_property_key.is_empty() {
            pinned.insert(self.color_property_key.clone());
        }
        if let Some(property_key) = self.property_load_key.as_ref() {
            pinned.insert(property_key.clone());
        }
        match self.filter_mode {
            ObjectFilterMode::Simple => {
                pinned.extend(
                    self.active_filter_clauses()
                        .filter(|clause| clause.property_key != "id")
                        .map(|clause| clause.property_key.clone()),
                );
            }
            ObjectFilterMode::Query => {
                if let Some(expr) = self.filter_query_expr.as_ref() {
                    pinned.extend(expr.referenced_properties());
                }
            }
        }
        pinned.extend(
            self.analysis_property_thresholds
                .iter()
                .map(|rule| rule.column_key.clone()),
        );
        pinned
    }

    pub(super) fn enforce_lazy_property_cache_capacity(&mut self) {
        let Some(capacity) = self.lazy_property_cache_capacity else {
            return;
        };
        self.lazy_property_lru
            .retain(|key| self.property_store.has_loaded(key));
        let pinned = self.pinned_lazy_property_keys();
        let pinned_resident = self
            .lazy_property_lru
            .iter()
            .filter(|key| pinned.contains(key.as_str()))
            .count();
        let target = capacity.max(pinned_resident);
        let mut evicted = Vec::new();
        while self.lazy_property_lru.len() > target {
            let Some(index) = self
                .lazy_property_lru
                .iter()
                .position(|key| !pinned.contains(key.as_str()))
            else {
                break;
            };
            let Some(property_key) = self.lazy_property_lru.remove(index) else {
                break;
            };
            if self.property_store.remove_column(&property_key) {
                if let Some(source) = self.lazy_parquet_source.as_mut() {
                    source.loaded_property_columns.remove(&property_key);
                }
                self.color_groups_cache.remove(&property_key);
                self.lazy_property_cache_evictions =
                    self.lazy_property_cache_evictions.saturating_add(1);
                evicted.push(property_key);
            }
        }
        if evicted.is_empty() {
            return;
        }
        self.color_legend_cache = None;
        self.continuous_color_payload = None;
        self.invalidate_filter_cache();
        self.reset_object_property_analysis_cache();
        self.generation = self.generation.wrapping_add(1).max(1);
        crate::log_debug!(
            "objects: evicted {} lazy property column(s): {}",
            evicted.len(),
            evicted.join(", ")
        );
    }
}
