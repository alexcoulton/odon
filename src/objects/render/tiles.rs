//! World-aligned multiresolution fill-tile planning.

use super::*;

pub(in crate::objects) const OBJECT_FILL_TILE_SIZE_PX: u32 = 512;
const MAX_OBJECT_FILL_TILE_LEVEL: u8 = 24;
pub(super) const MAX_VISIBLE_OBJECT_FILL_TILES: i64 = 256;
const MAX_EXACT_FLOAT_OBJECT_INDEX: usize = 16_777_215;
const TEXTURE_OUTLINE_TILE_SUPERSAMPLE: f32 = 4.0;

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
        texture_outline_requested: bool,
        visibility_state: &ObjectRenderStatePayload,
        frame_generation: u64,
    ) -> Option<ObjectFillTileFrame> {
        let local_screen_per_pixel = camera.zoom_screen_per_lvl0_px
            * display_scale.x.abs().max(display_scale.y.abs()).max(1.0e-9);
        if !object_fill_tile_path_eligible(fill_mesh, local_screen_per_pixel) {
            return None;
        }

        let (target_screen_per_pixel, fallback_screen_per_pixel) =
            object_fill_tile_planning_scales(local_screen_per_pixel, texture_outline_requested);
        let target_specs = plan_object_fill_tiles(
            visible_local,
            fill_mesh.bounds_local,
            target_screen_per_pixel,
        );
        if target_specs.is_empty() {
            return None;
        }
        let build_items = |specs: Vec<ObjectFillTileSpec>| {
            specs
                .into_iter()
                .map(|spec| {
                    let raster_bounds_local = object_fill_tile_raster_bounds(spec.bounds_local);
                    let geometry = fill_mesh
                        .spatial_slices_for_local_rect(raster_bounds_local)
                        .into_iter()
                        .map(|slice| ObjectFillTileGeometry {
                            cache_id: object_render_cache_id_usize(0x4ab0, slice.bin_index),
                            generation: self.geometry_generation,
                            bounds_local: slice.bounds_local,
                            vertices_local: slice.vertices_local,
                        })
                        .collect::<Vec<_>>();
                    ObjectFillTileDrawItem {
                        key: object_fill_tile_key(
                            self.render_resource_cache_id,
                            self.geometry_generation,
                            spec,
                        ),
                        bounds_local: spec.bounds_local,
                        raster_bounds_local,
                        geometry,
                    }
                })
                .collect::<Vec<_>>()
        };
        let draw_items = build_items(target_specs);
        let fallback_specs = plan_object_fill_tiles(
            visible_local,
            fill_mesh.bounds_local,
            fallback_screen_per_pixel,
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
        let selection_overlay = self
            .object_fill_selection_tile_style(fill_mesh.object_count, texture_outline_requested);
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
                let border_color = egui::Color32::from_rgba_unmultiplied(
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
                );
                styles.push(ObjectFillTileStyle {
                    style_cache_id: self.render_style_cache_id,
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
                    border: self.object_fill_tile_border_style(
                        texture_outline_requested,
                        border_color,
                        false,
                    ),
                });
            }
        } else {
            let rgb = self.color_rgb;
            let color = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
            styles.push(ObjectFillTileStyle {
                style_cache_id: self.render_style_cache_id,
                state_cache_id: object_render_cache_id(0x4a23, 0),
                object_count: fill_mesh.object_count,
                state_generation: visibility_state.generation,
                object_state: Arc::clone(&visibility_state.values),
                color_cache_id: object_render_cache_id(0x4a24, 0),
                color_generation: continuous_colors.map_or(0, |payload| payload.generation),
                object_colors_rgba: continuous_colors
                    .map(|payload| Arc::clone(&payload.colors_rgba)),
                selected_color: color,
                primary_color: color,
                object_color_opacity: self.fill_opacity,
                selection_overlay: selection_overlay.clone(),
                border: self.object_fill_tile_border_style(
                    texture_outline_requested,
                    egui::Color32::from_rgba_unmultiplied(
                        rgb[0],
                        rgb[1],
                        rgb[2],
                        (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
                    ),
                    continuous_colors.is_some(),
                ),
            });
        }

        (!styles.is_empty()).then_some(ObjectFillTileFrame {
            request_items,
            draw_items,
            styles,
            selection_overlay: selection_overlay.is_some(),
            params: ObjectFillTileGlParams {
                frame_generation,
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
        include_outline: bool,
    ) -> Option<ObjectFillTileSelectionStyle> {
        if !self.show_selection_overlay
            || (self.selected_fill_opacity <= 0.0 && !include_outline)
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

    fn object_fill_tile_border_style(
        &self,
        enabled: bool,
        base_color: egui::Color32,
        use_object_colors: bool,
    ) -> ObjectFillTileBorderStyle {
        ObjectFillTileBorderStyle {
            enabled,
            width_points: self.width_screen_px.max(0.0),
            base_color,
            use_object_colors,
            object_color_opacity: self.opacity.clamp(0.0, 1.0),
            selected_color: egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
            primary_color: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
        }
    }
}

pub(in crate::objects) fn object_fill_tile_raster_bounds(bounds: egui::Rect) -> egui::Rect {
    let gutter_scale = 2.0 / OBJECT_FILL_TILE_SIZE_PX as f32;
    bounds.expand2(egui::vec2(
        bounds.width() * gutter_scale,
        bounds.height() * gutter_scale,
    ))
}

pub(in crate::objects) fn object_fill_tile_key(
    resource_cache_id: u64,
    geometry_generation: u64,
    spec: ObjectFillTileSpec,
) -> ObjectFillTileKey {
    ObjectFillTileKey {
        resource_cache_id,
        geometry_generation,
        level: spec.level,
        tile_x: spec.tile_x,
        tile_y: spec.tile_y,
    }
}

pub(in crate::objects) fn object_fill_tile_object_count_supported(object_count: usize) -> bool {
    object_count <= MAX_EXACT_FLOAT_OBJECT_INDEX
}

pub(in crate::objects) fn object_fill_tile_path_eligible(
    fill_mesh: &ObjectFillMesh,
    local_screen_per_pixel: f32,
) -> bool {
    const MIN_TILE_VERTEX_COUNT: usize = 500_000;
    const MAX_VECTOR_DETAIL_SCREEN_PER_LOCAL_PIXEL: f32 = 2.0;
    fill_mesh.vertices_local.len() >= MIN_TILE_VERTEX_COUNT
        && object_fill_tile_object_count_supported(fill_mesh.object_count)
        && local_screen_per_pixel <= MAX_VECTOR_DETAIL_SCREEN_PER_LOCAL_PIXEL
}

pub(in crate::objects) fn object_fill_tile_planning_scales(
    local_screen_per_pixel: f32,
    texture_outline_requested: bool,
) -> (f32, f32) {
    if texture_outline_requested {
        (
            local_screen_per_pixel * TEXTURE_OUTLINE_TILE_SUPERSAMPLE,
            local_screen_per_pixel * (TEXTURE_OUTLINE_TILE_SUPERSAMPLE * 0.5),
        )
    } else {
        (local_screen_per_pixel, local_screen_per_pixel * 0.25)
    }
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
