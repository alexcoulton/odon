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

        serde_json::json!({
            "cpu": {
                "full_mesh_bytes": full_mesh_bytes,
                "binned_mesh_bytes": binned_mesh_bytes,
                "spatial_bin_count": spatial_bin_count,
                "continuous_color_bytes": continuous_color_bytes,
            },
            "gpu": {
                "mesh_entries": gpu.mesh_entries,
                "state_entries": gpu.state_entries,
                "color_entries": gpu.color_entries,
                "tile_entries": gpu.tile_entries,
                "mesh_bytes": gpu.mesh_bytes,
                "state_bytes": gpu.state_bytes,
                "color_bytes": gpu.color_bytes,
                "tile_bytes": gpu.tile_bytes,
                "tile_pending_bytes": gpu.tile_pending_bytes,
                "mesh_budget_bytes": gpu.mesh_budget_bytes,
                "texture_budget_bytes": gpu.texture_budget_bytes,
                "tile_budget_bytes": gpu.tile_budget_bytes,
                "mesh_uploads": gpu.mesh_uploads,
                "state_uploads": gpu.state_uploads,
                "color_uploads": gpu.color_uploads,
                "mesh_evictions": gpu.mesh_evictions,
                "texture_evictions": gpu.texture_evictions,
                "tile_requests": gpu.tile_requests,
                "tile_request_generation": gpu.tile_request_generation,
                "tile_visible": gpu.tile_visible,
                "tile_hits": gpu.tile_hits,
                "tile_generations": gpu.tile_generations,
                "tile_discarded": gpu.tile_discarded,
                "tile_evictions": gpu.tile_evictions,
                "tile_pending": gpu.tile_pending,
                "tile_peak_pending": gpu.tile_peak_pending,
                "last_tile_raster_vertices": gpu.last_tile_raster_vertices,
                "last_tile_raster_draw_calls": gpu.last_tile_raster_draw_calls,
                "last_tile_compose_draw_calls": gpu.last_tile_compose_draw_calls,
                "total_tile_raster_vertices": gpu.total_tile_raster_vertices,
                "last_tile_raster_ms": gpu.last_tile_raster_ms,
                "last_tile_compose_ms": gpu.last_tile_compose_ms,
                "tile_supported": gpu.tile_supported,
                "last_draw_calls": gpu.last_draw_calls,
                "last_triangles": gpu.last_triangles,
                "total_draw_calls": gpu.total_draw_calls,
                "total_triangles": gpu.total_triangles,
                "last_paint_ms": gpu.last_paint_ms,
            },
        })
    }
}
