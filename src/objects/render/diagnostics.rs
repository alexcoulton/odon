//! Object fill renderer diagnostics exposed through workspace observations.

use super::*;

impl ObjectsLayer {
    pub(in crate::objects) fn outline_geometry_cache_generation(&self) -> u64 {
        self.geometry_generation
    }

    pub(crate) fn outline_gpu_stats(&self) -> crate::render::line_bins_gl::ObjectLineBinsGlStats {
        self.gl_object_selection.stats()
    }

    pub(crate) fn outline_frame_stats(&self) -> ObjectOutlineFrameStats {
        self.outline_frame_stats
    }

    pub(crate) fn object_presentation_state_stats(&self) -> ObjectPresentationStateStats {
        self.presentation_state_stats()
    }

    pub(crate) fn render_diagnostics_json(&self) -> serde_json::Value {
        let gpu = self.gl_object_fill.stats();
        let outline_gpu = self.outline_gpu_stats();
        let (full_mesh_bytes, binned_mesh_bytes, spatial_bin_count) = self
            .object_fill_mesh
            .as_ref()
            .map(|mesh| {
                let vertex_bytes = std::mem::size_of::<[f32; 3]>();
                (
                    mesh.vertices_local.len().saturating_mul(vertex_bytes),
                    mesh.bin_vertices.iter().fold(0usize, |total, vertices| {
                        total.saturating_add(vertices.len().saturating_mul(vertex_bytes))
                    }),
                    mesh.bin_vertices.len(),
                )
            })
            .unwrap_or((0, 0, 0));
        let continuous_color_bytes = self.continuous_color_payload.as_ref().map_or(0, |payload| {
            payload
                .colors_rgba
                .len()
                .saturating_mul(std::mem::size_of::<[u8; 4]>())
        });

        let gpu = serde_json::to_value(gpu).unwrap_or_default();
        serde_json::json!({
            "cpu": {
                "full_mesh_bytes": full_mesh_bytes,
                "binned_mesh_bytes": binned_mesh_bytes,
                "spatial_bin_count": spatial_bin_count,
                "continuous_color_bytes": continuous_color_bytes,
            },
            "gpu": gpu,
            "outline_gpu": outline_gpu,
            "outline_frame": self.outline_frame_stats(),
            "presentation_state": self.object_presentation_state_stats(),
        })
    }
}
