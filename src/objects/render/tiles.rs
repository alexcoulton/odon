//! World-aligned multiresolution fill-tile planning.

use super::*;

pub(in crate::objects) const OBJECT_FILL_TILE_SIZE_PX: u32 = 512;
const MAX_OBJECT_FILL_TILE_LEVEL: u8 = 24;
pub(super) const MAX_VISIBLE_OBJECT_FILL_TILES: i64 = 256;
const MAX_EXACT_FLOAT_OBJECT_INDEX: usize = 16_777_215;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::objects) struct ObjectFillTileSpec {
    pub level: u8,
    pub tile_x: i32,
    pub tile_y: i32,
    pub bounds_local: egui::Rect,
}

pub(in crate::objects) struct ObjectFillTileFrame {
    pub request_items: Vec<ObjectFillTileDrawItem>,
    pub draw_items: Vec<ObjectFillTileDrawItem>,
    pub styles: Vec<ObjectFillTileStyle>,
    pub selection_overlay: bool,
    pub params: ObjectFillTileGlParams,
}

impl ObjectFillTileFrame {
    pub fn keys(&self) -> Vec<ObjectFillTileKey> {
        self.draw_items.iter().map(|item| item.key).collect()
    }
}

impl ObjectsLayer {
    pub(in crate::objects) fn plan_object_fill_tile_frame(
        &self,
        fill_mesh: &ObjectFillMesh,
        visible_local: egui::Rect,
        camera: &crate::camera::Camera,
        display_offset: egui::Vec2,
        display_scale: egui::Vec2,
        active_color_groups: Option<&ObjectColorGroups>,
        continuous_colors: Option<&ObjectContinuousColorPayload>,
        render_generation: u64,
    ) -> Option<ObjectFillTileFrame> {
        const MIN_TILE_VERTEX_COUNT: usize = 500_000;
        const MAX_VECTOR_DETAIL_SCREEN_PER_LOCAL_PIXEL: f32 = 2.0;
        if fill_mesh.vertices_local.len() < MIN_TILE_VERTEX_COUNT
            || !object_fill_tile_object_count_supported(fill_mesh.object_count)
        {
            return None;
        }
        let local_screen_per_pixel = camera.zoom_screen_per_lvl0_px
            * display_scale.x.abs().max(display_scale.y.abs()).max(1.0e-9);
        if local_screen_per_pixel > MAX_VECTOR_DETAIL_SCREEN_PER_LOCAL_PIXEL {
            return None;
        }

        let target_specs = plan_object_fill_tiles(
            visible_local,
            fill_mesh.bounds_local,
            local_screen_per_pixel,
        );
        if target_specs.is_empty() {
            return None;
        }
        let build_items = |specs: Vec<ObjectFillTileSpec>| {
            specs
                .into_iter()
                .map(|spec| {
                    let geometry = fill_mesh
                        .spatial_slices_for_local_rect(spec.bounds_local)
                        .into_iter()
                        .map(|slice| ObjectFillTileGeometry {
                            cache_id: object_render_cache_id_usize(0x4ab0, slice.bin_index),
                            generation: self.geometry_generation,
                            bounds_local: slice.bounds_local,
                            vertices_local: slice.vertices_local,
                        })
                        .collect::<Vec<_>>();
                    ObjectFillTileDrawItem {
                        key: object_fill_tile_key(self.geometry_generation, spec),
                        bounds_local: spec.bounds_local,
                        geometry,
                    }
                })
                .collect::<Vec<_>>()
        };
        let draw_items = build_items(target_specs);
        let fallback_specs = plan_object_fill_tiles(
            visible_local,
            fill_mesh.bounds_local,
            local_screen_per_pixel * 0.25,
        );
        let mut request_items = build_items(fallback_specs);
        let fallback_keys = request_items
            .iter()
            .map(|item| item.key)
            .collect::<std::collections::HashSet<_>>();
        request_items.extend(
            draw_items
                .iter()
                .filter(|item| !fallback_keys.contains(&item.key))
                .cloned(),
        );

        let fill_alpha = (self.fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let selection_overlay = self.object_fill_selection_tile_style(fill_mesh.object_count);
        let mut styles = Vec::new();
        if let Some(color_groups) = active_color_groups {
            styles.reserve(color_groups.groups.len());
            for (group_index, group) in color_groups.groups.iter().enumerate() {
                let Some(rgb) = self.effective_color_group_rgb(&color_groups.property_key, group)
                else {
                    continue;
                };
                let color =
                    egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
                styles.push(ObjectFillTileStyle {
                    state_cache_id: object_property_render_cache_id(
                        0x4a21,
                        &color_groups.property_key,
                        group_index,
                    ),
                    object_count: fill_mesh.object_count,
                    state_generation: group.fill_generation,
                    object_state: Arc::clone(&group.fill_state),
                    color_cache_id: 0,
                    color_generation: 0,
                    object_colors_rgba: None,
                    selected_color: color,
                    primary_color: color,
                    object_color_opacity: 1.0,
                    selection_overlay: selection_overlay.clone(),
                });
            }
        } else {
            let mut visible_state = vec![0u8; fill_mesh.object_count];
            for (index, state) in visible_state.iter_mut().enumerate() {
                if self.is_index_visible(index) {
                    *state = 255;
                }
            }
            let rgb = self.color_rgb;
            let color = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
            styles.push(ObjectFillTileStyle {
                state_cache_id: object_render_cache_id(0x4a23, 0),
                object_count: fill_mesh.object_count,
                state_generation: render_generation,
                object_state: Arc::new(visible_state),
                color_cache_id: object_render_cache_id(0x4a24, 0),
                color_generation: continuous_colors.map_or(0, |payload| payload.generation),
                object_colors_rgba: continuous_colors
                    .map(|payload| Arc::clone(&payload.colors_rgba)),
                selected_color: color,
                primary_color: color,
                object_color_opacity: self.fill_opacity,
                selection_overlay: selection_overlay.clone(),
            });
        }

        (!styles.is_empty()).then_some(ObjectFillTileFrame {
            request_items,
            draw_items,
            styles,
            selection_overlay: selection_overlay.is_some(),
            params: ObjectFillTileGlParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                visible: self.visible,
                local_to_world_offset: display_offset,
                local_to_world_scale: display_scale,
            },
        })
    }

    pub(in crate::objects) fn object_fill_selection_tile_style(
        &self,
        object_count: usize,
    ) -> Option<ObjectFillTileSelectionStyle> {
        if !self.show_selection_overlay
            || self.selected_fill_opacity <= 0.0
            || self.selected_object_indices.is_empty()
            || self.selection_fill_state.len() != object_count
        {
            return None;
        }
        let color = egui::Color32::from_rgba_unmultiplied(
            255,
            245,
            140,
            (self.selected_fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
        );
        Some(ObjectFillTileSelectionStyle {
            state_cache_id: object_render_cache_id(0x4a31, 0),
            state_generation: self.selection_generation,
            object_state: Arc::clone(&self.selection_fill_state),
            selected_color: color,
            primary_color: color,
        })
    }
}

pub(in crate::objects) fn object_fill_tile_key(
    geometry_generation: u64,
    spec: ObjectFillTileSpec,
) -> ObjectFillTileKey {
    ObjectFillTileKey {
        resource_cache_id: object_render_cache_id(0x4ac0, 0),
        geometry_generation,
        level: spec.level,
        tile_x: spec.tile_x,
        tile_y: spec.tile_y,
    }
}

pub(in crate::objects) fn object_fill_tile_object_count_supported(object_count: usize) -> bool {
    object_count <= MAX_EXACT_FLOAT_OBJECT_INDEX
}

pub(in crate::objects) fn choose_object_fill_tile_level(local_screen_per_pixel: f32) -> u8 {
    if !local_screen_per_pixel.is_finite() || local_screen_per_pixel <= 0.0 {
        return MAX_OBJECT_FILL_TILE_LEVEL;
    }
    (-local_screen_per_pixel.log2())
        .round()
        .clamp(0.0, MAX_OBJECT_FILL_TILE_LEVEL as f32) as u8
}

pub(in crate::objects) fn plan_object_fill_tiles(
    visible_local: egui::Rect,
    object_bounds_local: egui::Rect,
    local_screen_per_pixel: f32,
) -> Vec<ObjectFillTileSpec> {
    if !visible_local.is_positive() || !object_bounds_local.intersects(visible_local) {
        return Vec::new();
    }
    let visible = visible_local.intersect(object_bounds_local);
    let mut level = choose_object_fill_tile_level(local_screen_per_pixel);
    let (tile_span, x0, y0, x1, y1, tile_count) = loop {
        let downsample = 2.0f32.powi(level as i32);
        let tile_span = OBJECT_FILL_TILE_SIZE_PX as f32 * downsample;
        if !tile_span.is_finite() || tile_span <= 0.0 {
            return Vec::new();
        }
        let x0 = (visible.min.x / tile_span).floor() as i32;
        let y0 = (visible.min.y / tile_span).floor() as i32;
        let x1 = (visible.max.x / tile_span).ceil() as i32 - 1;
        let y1 = (visible.max.y / tile_span).ceil() as i32 - 1;
        if x1 < x0 || y1 < y0 {
            return Vec::new();
        }
        let tile_count =
            (i64::from(x1) - i64::from(x0) + 1).saturating_mul(i64::from(y1) - i64::from(y0) + 1);
        if tile_count <= MAX_VISIBLE_OBJECT_FILL_TILES || level >= MAX_OBJECT_FILL_TILE_LEVEL {
            break (tile_span, x0, y0, x1, y1, tile_count);
        }
        level = level.saturating_add(1);
    };

    let tile_count = tile_count.clamp(0, MAX_VISIBLE_OBJECT_FILL_TILES) as usize;
    let mut specs = Vec::with_capacity(tile_count);
    for tile_y in y0..=y1 {
        for tile_x in x0..=x1 {
            let min = egui::pos2(tile_x as f32 * tile_span, tile_y as f32 * tile_span);
            let bounds_local = egui::Rect::from_min_size(min, egui::vec2(tile_span, tile_span));
            if bounds_local.intersects(object_bounds_local) && bounds_local.intersects(visible) {
                specs.push(ObjectFillTileSpec {
                    level,
                    tile_x,
                    tile_y,
                    bounds_local,
                });
            }
        }
    }
    let center = visible.center();
    specs.sort_by(|left, right| {
        left.bounds_local
            .center()
            .distance_sq(center)
            .total_cmp(&right.bounds_local.center().distance_sq(center))
            .then_with(|| left.tile_y.cmp(&right.tile_y))
            .then_with(|| left.tile_x.cmp(&right.tile_x))
    });
    specs.truncate(MAX_VISIBLE_OBJECT_FILL_TILES as usize);
    specs
}
