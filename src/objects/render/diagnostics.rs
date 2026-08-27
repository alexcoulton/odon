//! Object fill renderer diagnostics exposed through workspace observations.

use super::*;

impl ObjectsLayer {
    pub(crate) fn render_diagnostics_json(&self) -> serde_json::Value {
        let gpu = self.gl_object_fill.stats();
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
        })
    }
}
