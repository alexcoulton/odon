//! Camera-independent per-object presentation payloads shared by fill and outline renderers.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::objects) struct ObjectRenderStatePayload {
    pub values: Arc<Vec<u8>>,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisibilityStateKey {
    resource_cache_id: u64,
    geometry_generation: u64,
    filter_generation: u64,
    object_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ContinuousOutlineStateKey {
    visibility: VisibilityStateKey,
    selection_generation: u64,
    show_selection_overlay: bool,
}

#[derive(Debug, Clone)]
pub(in crate::objects) struct ObjectPresentationStateCache {
    visibility_key: Option<VisibilityStateKey>,
    visibility: ObjectRenderStatePayload,
    continuous_outline_key: Option<ContinuousOutlineStateKey>,
    continuous_outline: ObjectRenderStatePayload,
    next_generation: u64,
    visibility_rebuilds: u64,
    continuous_outline_rebuilds: u64,
}

impl Default for ObjectPresentationStateCache {
    fn default() -> Self {
        Self {
            visibility_key: None,
            visibility: ObjectRenderStatePayload {
                values: Arc::new(Vec::new()),
                generation: 0,
            },
            continuous_outline_key: None,
            continuous_outline: ObjectRenderStatePayload {
                values: Arc::new(Vec::new()),
                generation: 0,
            },
            next_generation: 1,
            visibility_rebuilds: 0,
            continuous_outline_rebuilds: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct ObjectPresentationStateStats {
    pub visibility_bytes: usize,
    pub continuous_outline_bytes: usize,
    pub visibility_rebuilds: u64,
    pub continuous_outline_rebuilds: u64,
    pub visibility_generation: u64,
    pub continuous_outline_generation: u64,
}

impl ObjectPresentationStateStats {
    pub fn merge(&mut self, other: Self) {
        self.visibility_bytes = self.visibility_bytes.saturating_add(other.visibility_bytes);
        self.continuous_outline_bytes = self
            .continuous_outline_bytes
            .saturating_add(other.continuous_outline_bytes);
        self.visibility_rebuilds = self
            .visibility_rebuilds
            .saturating_add(other.visibility_rebuilds);
        self.continuous_outline_rebuilds = self
            .continuous_outline_rebuilds
            .saturating_add(other.continuous_outline_rebuilds);
        self.visibility_generation = self.visibility_generation.max(other.visibility_generation);
        self.continuous_outline_generation = self
            .continuous_outline_generation
            .max(other.continuous_outline_generation);
    }
}

impl ObjectsLayer {
    pub(super) fn cached_visibility_state(&mut self) -> ObjectRenderStatePayload {
        let object_count = self.object_count();
        let key = VisibilityStateKey {
            resource_cache_id: self.render_resource_cache_id,
            geometry_generation: self.geometry_generation,
            filter_generation: self.filter_generation,
            object_count,
        };
        if self.presentation_state_cache.visibility_key == Some(key) {
            return self.presentation_state_cache.visibility.clone();
        }

        let mut values = vec![0u8; object_count];
        for (index, state) in values.iter_mut().enumerate() {
            if self.is_index_visible(index) {
                *state = 255;
            }
        }
        let generation = self.next_presentation_state_generation();
        self.presentation_state_cache.visibility_key = Some(key);
        self.presentation_state_cache.visibility = ObjectRenderStatePayload {
            values: Arc::new(values),
            generation,
        };
        self.presentation_state_cache.visibility_rebuilds = self
            .presentation_state_cache
            .visibility_rebuilds
            .saturating_add(1);
        self.presentation_state_cache.visibility.clone()
    }

    pub(super) fn cached_continuous_outline_state(&mut self) -> ObjectRenderStatePayload {
        let visibility = self.cached_visibility_state();
        let visibility_key = self
            .presentation_state_cache
            .visibility_key
            .expect("visibility payload key is installed with its values");
        let key = ContinuousOutlineStateKey {
            visibility: visibility_key,
            selection_generation: self.selection_generation,
            show_selection_overlay: self.show_selection_overlay,
        };
        if self.presentation_state_cache.continuous_outline_key == Some(key) {
            return self.presentation_state_cache.continuous_outline.clone();
        }

        let mut values = visibility
            .values
            .iter()
            .map(|state| if *state == 0 { 0 } else { 64 })
            .collect::<Vec<_>>();
        if self.show_selection_overlay {
            for index in &self.selected_object_indices {
                if let Some(state) = values.get_mut(*index)
                    && *state != 0
                {
                    *state = 128;
                }
            }
            if let Some(index) = self.selected_object_index
                && let Some(state) = values.get_mut(index)
                && *state != 0
            {
                *state = 255;
            }
        }
        let generation = self.next_presentation_state_generation();
        self.presentation_state_cache.continuous_outline_key = Some(key);
        self.presentation_state_cache.continuous_outline = ObjectRenderStatePayload {
            values: Arc::new(values),
            generation,
        };
        self.presentation_state_cache.continuous_outline_rebuilds = self
            .presentation_state_cache
            .continuous_outline_rebuilds
            .saturating_add(1);
        self.presentation_state_cache.continuous_outline.clone()
    }

    fn next_presentation_state_generation(&mut self) -> u64 {
        let generation = self.presentation_state_cache.next_generation.max(1);
        self.presentation_state_cache.next_generation = generation.wrapping_add(1).max(1);
        generation
    }

    pub(crate) fn presentation_state_stats(&self) -> ObjectPresentationStateStats {
        ObjectPresentationStateStats {
            visibility_bytes: self.presentation_state_cache.visibility.values.capacity(),
            continuous_outline_bytes: self
                .presentation_state_cache
                .continuous_outline
                .values
                .capacity(),
            visibility_rebuilds: self.presentation_state_cache.visibility_rebuilds,
            continuous_outline_rebuilds: self.presentation_state_cache.continuous_outline_rebuilds,
            visibility_generation: self.presentation_state_cache.visibility.generation,
            continuous_outline_generation: self
                .presentation_state_cache
                .continuous_outline
                .generation,
        }
    }
}
