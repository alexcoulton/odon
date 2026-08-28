use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct ObjectLineBinsGlStats {
    pub bin_entries: usize,
    pub bin_capacity_entries: usize,
    pub bin_bytes: usize,
    pub state_entries: usize,
    pub state_capacity_entries: usize,
    pub state_bytes: usize,
    pub color_entries: usize,
    pub color_capacity_entries: usize,
    pub color_bytes: usize,
    pub queued_buffer_bytes: usize,
    pub queued_texture_bytes: usize,
    pub bin_uploads: u64,
    pub state_uploads: u64,
    pub color_uploads: u64,
    pub bin_evictions: u64,
    pub texture_evictions: u64,
    pub buffer_deletions: u64,
    pub texture_deletions: u64,
    pub last_frame_missing_bins: usize,
}

impl ObjectLineBinsGlStats {
    pub fn merge(&mut self, other: Self) {
        macro_rules! add {
            ($($field:ident),+ $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)+
            };
        }
        add!(
            bin_entries,
            bin_capacity_entries,
            bin_bytes,
            state_entries,
            state_capacity_entries,
            state_bytes,
            color_entries,
            color_capacity_entries,
            color_bytes,
            queued_buffer_bytes,
            queued_texture_bytes,
            bin_uploads,
            state_uploads,
            color_uploads,
            bin_evictions,
            texture_evictions,
            buffer_deletions,
            texture_deletions,
            last_frame_missing_bins,
        );
    }
}

impl ObjectLineInner {
    pub(super) fn stats(&self) -> ObjectLineBinsGlStats {
        let bin_bytes = self
            .bins
            .iter()
            .map(|(_, buffer)| buffer.bytes)
            .sum::<usize>();
        let state_bytes = self
            .states
            .iter()
            .map(|(_, texture)| {
                (texture.width.max(0) as usize).saturating_mul(texture.height.max(0) as usize)
            })
            .sum::<usize>();
        let color_bytes = self
            .colors
            .iter()
            .map(|(_, texture)| {
                (texture.width.max(0) as usize)
                    .saturating_mul(texture.height.max(0) as usize)
                    .saturating_mul(4)
            })
            .sum::<usize>();
        ObjectLineBinsGlStats {
            bin_entries: self.bins.len(),
            bin_capacity_entries: self.bins.cap().get(),
            bin_bytes,
            state_entries: self.states.len(),
            state_capacity_entries: self.states.cap().get(),
            state_bytes,
            color_entries: self.colors.len(),
            color_capacity_entries: self.colors.cap().get(),
            color_bytes,
            queued_buffer_bytes: self
                .buffers_to_delete
                .iter()
                .map(|buffer| buffer.bytes)
                .sum(),
            queued_texture_bytes: self
                .textures_to_delete
                .iter()
                .map(|texture| texture.bytes)
                .sum(),
            bin_uploads: self.bin_uploads,
            state_uploads: self.state_uploads,
            color_uploads: self.color_uploads,
            bin_evictions: self.bin_evictions,
            texture_evictions: self.texture_evictions,
            buffer_deletions: self.buffer_deletions,
            texture_deletions: self.texture_deletions,
            last_frame_missing_bins: self.last_frame_missing_bins,
        }
    }
}
