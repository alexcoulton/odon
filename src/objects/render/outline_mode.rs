//! Adaptive low-zoom texture/vector outline selection and bounded frame diagnostics.

use super::*;

const TEXTURE_ENTER_MAX_SCREEN_PER_LOCAL_PIXEL: f32 = 0.35;
const TEXTURE_LEAVE_MAX_SCREEN_PER_LOCAL_PIXEL: f32 = 0.55;
const TEXTURE_ENTER_MIN_VISIBLE_RECORDS: usize = 100_000;
const TEXTURE_LEAVE_MIN_VISIBLE_RECORDS: usize = 50_000;
const HEAVY_VECTOR_VISIBLE_RECORDS: usize = 1_000_000;
const HEAVY_VECTOR_ENTER_MAX_SCREEN_PER_LOCAL_PIXEL: f32 = 0.75;
const HEAVY_VECTOR_LEAVE_MAX_SCREEN_PER_LOCAL_PIXEL: f32 = 1.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObjectOutlineMode {
    #[default]
    None,
    Proxy,
    Texture,
    Vector,
    Mixed,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObjectOutlineModeReason {
    #[default]
    None,
    ProxyPolicy,
    TextureWorkload,
    TextureHysteresis,
    VectorDetail,
    VectorTilesPending,
    VectorTextureUnavailable,
    Mixed,
}

#[derive(Debug, Clone, Default)]
pub(in crate::objects) struct ObjectOutlineModeRuntime {
    prefers_texture: bool,
    transitions: u64,
}

impl ObjectOutlineModeRuntime {
    pub(super) fn resolve_preference(
        &mut self,
        local_screen_per_pixel: f32,
        visible_records: usize,
        texture_eligible: bool,
    ) -> (bool, ObjectOutlineModeReason) {
        if !texture_eligible {
            if self.prefers_texture {
                self.prefers_texture = false;
                self.transitions = self.transitions.saturating_add(1);
            }
            return (false, ObjectOutlineModeReason::VectorTextureUnavailable);
        }

        let scale = if local_screen_per_pixel.is_finite() {
            local_screen_per_pixel.max(0.0)
        } else {
            f32::INFINITY
        };
        let enter = (scale <= TEXTURE_ENTER_MAX_SCREEN_PER_LOCAL_PIXEL
            && visible_records >= TEXTURE_ENTER_MIN_VISIBLE_RECORDS)
            || (scale <= HEAVY_VECTOR_ENTER_MAX_SCREEN_PER_LOCAL_PIXEL
                && visible_records >= HEAVY_VECTOR_VISIBLE_RECORDS);
        let stay = (scale <= TEXTURE_LEAVE_MAX_SCREEN_PER_LOCAL_PIXEL
            && visible_records >= TEXTURE_LEAVE_MIN_VISIBLE_RECORDS)
            || (scale <= HEAVY_VECTOR_LEAVE_MAX_SCREEN_PER_LOCAL_PIXEL
                && visible_records >= HEAVY_VECTOR_VISIBLE_RECORDS / 2);
        let next = if self.prefers_texture { stay } else { enter };
        if next != self.prefers_texture {
            self.prefers_texture = next;
            self.transitions = self.transitions.saturating_add(1);
        }
        if next {
            (
                true,
                if enter {
                    ObjectOutlineModeReason::TextureWorkload
                } else {
                    ObjectOutlineModeReason::TextureHysteresis
                },
            )
        } else {
            (false, ObjectOutlineModeReason::VectorDetail)
        }
    }

    pub(super) fn transitions(&self) -> u64 {
        self.transitions
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub(crate) struct ObjectOutlineFrameStats {
    pub mode: ObjectOutlineMode,
    pub reason: ObjectOutlineModeReason,
    pub layer_count: usize,
    pub local_screen_per_pixel: f32,
    pub visible_bins: usize,
    pub visible_records: usize,
    pub submitted_vector_records: usize,
    pub submitted_vector_draw_calls: usize,
    pub texture_border_draw_calls: usize,
    pub texture_coverage: bool,
    pub tile_frame_planned: bool,
    pub mode_transitions: u64,
    pub visibility_state_plan_ms: f64,
    pub fill_plan_ms: f64,
    pub outline_plan_ms: f64,
    pub proxy_layers: usize,
    pub texture_layers: usize,
    pub vector_layers: usize,
}

impl ObjectOutlineFrameStats {
    pub(super) fn for_layer(
        local_screen_per_pixel: f32,
        visible_bins: usize,
        visible_records: usize,
        transitions: u64,
    ) -> Self {
        Self {
            layer_count: 1,
            local_screen_per_pixel,
            visible_bins,
            visible_records,
            mode_transitions: transitions,
            ..Self::default()
        }
    }

    pub(super) fn set_mode(&mut self, mode: ObjectOutlineMode, reason: ObjectOutlineModeReason) {
        self.mode = mode;
        self.reason = reason;
        self.proxy_layers = usize::from(mode == ObjectOutlineMode::Proxy);
        self.texture_layers = usize::from(mode == ObjectOutlineMode::Texture);
        self.vector_layers = usize::from(mode == ObjectOutlineMode::Vector);
        self.submitted_vector_records = if mode == ObjectOutlineMode::Vector {
            self.visible_records
        } else {
            0
        };
        self.submitted_vector_draw_calls = if mode == ObjectOutlineMode::Vector {
            self.visible_bins
        } else {
            0
        };
    }

    pub fn merge(&mut self, other: Self) {
        if other.layer_count == 0 {
            return;
        }
        let had_layers = self.layer_count > 0;
        self.layer_count = self.layer_count.saturating_add(other.layer_count);
        self.local_screen_per_pixel = self
            .local_screen_per_pixel
            .max(other.local_screen_per_pixel);
        self.visible_bins = self.visible_bins.saturating_add(other.visible_bins);
        self.visible_records = self.visible_records.saturating_add(other.visible_records);
        self.submitted_vector_records = self
            .submitted_vector_records
            .saturating_add(other.submitted_vector_records);
        self.submitted_vector_draw_calls = self
            .submitted_vector_draw_calls
            .saturating_add(other.submitted_vector_draw_calls);
        self.texture_border_draw_calls = self
            .texture_border_draw_calls
            .saturating_add(other.texture_border_draw_calls);
        self.texture_coverage |= other.texture_coverage;
        self.tile_frame_planned |= other.tile_frame_planned;
        self.mode_transitions = self.mode_transitions.saturating_add(other.mode_transitions);
        self.visibility_state_plan_ms += other.visibility_state_plan_ms;
        self.fill_plan_ms += other.fill_plan_ms;
        self.outline_plan_ms += other.outline_plan_ms;
        self.proxy_layers = self.proxy_layers.saturating_add(other.proxy_layers);
        self.texture_layers = self.texture_layers.saturating_add(other.texture_layers);
        self.vector_layers = self.vector_layers.saturating_add(other.vector_layers);
        if !had_layers {
            self.mode = other.mode;
            self.reason = other.reason;
        } else {
            if self.mode != other.mode {
                self.mode = ObjectOutlineMode::Mixed;
            }
            if self.reason != other.reason {
                self.reason = ObjectOutlineModeReason::Mixed;
            }
        }
    }
}

pub(super) fn visible_object_outline_work(
    bins: &ObjectLineSegmentsBins,
    visible_local: egui::Rect,
) -> (usize, usize) {
    let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(visible_local);
    let mut visible_bins = 0usize;
    let mut visible_records = 0usize;
    for by in by0..=by1 {
        for bx in bx0..=bx1 {
            let slice = bins.bin_slice(by * bins.bins_w + bx);
            if slice.is_empty() {
                continue;
            }
            visible_bins = visible_bins.saturating_add(1);
            visible_records = visible_records.saturating_add(slice.len());
        }
    }
    (visible_bins, visible_records)
}

pub(super) fn visible_line_outline_work(
    bins: &LineSegmentsBins,
    visible_local: egui::Rect,
) -> (usize, usize) {
    let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(visible_local);
    let mut visible_bins = 0usize;
    let mut visible_records = 0usize;
    for by in by0..=by1 {
        for bx in bx0..=bx1 {
            let slice = bins.bin_slice(by * bins.bins_w + bx);
            if slice.is_empty() {
                continue;
            }
            visible_bins = visible_bins.saturating_add(1);
            visible_records = visible_records.saturating_add(slice.len());
        }
    }
    (visible_bins, visible_records)
}
