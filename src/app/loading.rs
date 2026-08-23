use super::*;

impl OmeZarrViewerApp {
    pub(super) fn is_loading_scene(&self) -> bool {
        let mut busy = false;
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            busy |= tiles_gl.is_busy();
        }
        busy |= self.cache.is_busy();
        if let Some(labels_gl) = self.labels_gl.as_ref() {
            busy |= labels_gl.is_busy();
        }
        busy |= self.seg_geojson.is_busy();
        busy |= self.seg_objects.is_busy();
        busy |= self.spatial_image_layers.is_busy();
        busy |= self.spatial_layers.is_busy();
        busy |= self.projected_memory_running();
        busy
    }

    pub(super) fn scene_busy_debug_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if self
            .tiles_gl
            .as_ref()
            .is_some_and(|tiles_gl| tiles_gl.is_busy())
        {
            reasons.push("tiles_gl");
        }
        if self.cache.is_busy() {
            reasons.push("tile_cache");
        }
        if self
            .labels_gl
            .as_ref()
            .is_some_and(|labels_gl| labels_gl.is_busy())
        {
            reasons.push("labels_gl");
        }
        if self.seg_geojson.is_busy() {
            reasons.push("seg_geojson");
        }
        if self.seg_objects.is_busy() {
            reasons.push("seg_objects");
        }
        if self.spatial_image_layers.is_busy() {
            reasons.push("spatial_images");
        }
        if self.spatial_layers.is_busy() {
            reasons.push("spatial_layers");
        }
        if self.projected_memory_running() {
            reasons.push("pinned_levels");
        }
        reasons
    }

    pub(super) fn async_ui_debug_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        let properties_hist_active = self.show_right_panel
            && self.right_tab == RightTab::Properties
            && matches!(self.active_layer, LayerId::Channel(_));
        if properties_hist_active && self.hist_dirty {
            reasons.push("hist_dirty");
        }
        if properties_hist_active && self.hist_request_pending {
            reasons.push("hist_pending");
        }
        if properties_hist_active && self.hist_navigation_dirty_since.is_some() {
            reasons.push("hist_nav_debounce");
        }
        reasons
    }

    pub(super) fn image_tile_request_count(&self) -> usize {
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.in_flight_len()
        } else {
            self.cache.in_flight_len()
        }
    }

    pub(super) fn desired_raw_tile_cache_capacity(&self, visible_target_raw_tiles: usize) -> usize {
        let headroom = visible_target_raw_tiles
            .saturating_div(4)
            .max(RAW_TILE_CACHE_HEADROOM_TILES);
        visible_target_raw_tiles
            .saturating_add(headroom)
            .max(RAW_TILE_CACHE_CAPACITY_TILES)
            .min(RAW_TILE_CACHE_MAX_CAPACITY_TILES)
    }

    pub(super) fn maybe_grow_raw_tile_cache(
        &self,
        tiles_gl: &TilesGl,
        visible_target_raw_tiles: usize,
    ) {
        let desired = self.desired_raw_tile_cache_capacity(visible_target_raw_tiles);
        let current = tiles_gl.capacity();
        if desired <= current {
            return;
        }
        crate::log_info!(
            "growing raw tile cache from {} to {} tiles for visible target set {}",
            current,
            desired,
            visible_target_raw_tiles
        );
        tiles_gl.grow_capacity(desired);
    }

    pub(super) fn loading_indicator_text(&self) -> Option<&'static str> {
        if self.seg_objects.is_loading() {
            Some("Loading segmentation objects...")
        } else if self.spatial_image_layers.is_busy() {
            Some("Loading SpatialData images...")
        } else if self.spatial_layers.is_loading_shapes() {
            Some("Loading SpatialData shapes...")
        } else if self.spatial_layers.is_loading_points() {
            Some("Loading SpatialData points...")
        } else if self.seg_objects.is_analyzing() {
            Some("Running object analysis...")
        } else if self.spatial_layers.is_busy() {
            Some("Running SpatialData layer analysis...")
        } else if self.seg_geojson.is_busy() {
            Some("Loading segmentation...")
        } else if self.projected_memory_running() {
            Some("Pinning image level into RAM...")
        } else if self.is_loading_scene() {
            Some("Loading image tiles...")
        } else {
            None
        }
    }

    pub(super) fn tile_debug_overlay_text(&self) -> String {
        let busy = self.scene_busy_debug_reasons();
        let async_reasons = self.async_ui_debug_reasons();
        let stats = &self.mask_draw_debug_stats;
        let busy_text = if busy.is_empty() {
            "none".to_string()
        } else {
            busy.join(",")
        };
        let async_text = if async_reasons.is_empty() {
            "none".to_string()
        } else {
            async_reasons.join(",")
        };
        format!(
            "Debug\nbusy: {busy_text}\nasync: {async_text}\nimage tiles: {}\nmask layers: {}\nmask painted: {} polys / {} verts\nmask on-screen: {} polys / {} verts\nmask fill: {} polys / {} verts\nmask raster: {} layers / {} px\nmask draw: {:.2} ms",
            self.image_tile_request_count(),
            stats.visible_layers,
            stats.painted_polygons,
            stats.painted_vertices,
            stats.screen_polygons,
            stats.screen_vertices,
            stats.fill_polygons,
            stats.fill_vertices,
            stats.raster_layers,
            stats.raster_pixels,
            stats.draw_time.as_secs_f64() * 1000.0,
        )
    }
}
