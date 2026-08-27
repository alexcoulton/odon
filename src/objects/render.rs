use super::core::{build_object_point_payload, object_proxy_position_world, rect_bins};
use super::*;
use anyhow::Context;
use lyon_path::Path;
use lyon_path::math::point;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod diagnostics;
mod fills;
mod lods;
mod mesh;
mod selection;
mod selection_geometry;
#[cfg(test)]
mod tests;
mod tiles;
mod transforms;

pub(in crate::objects) use lods::*;
pub(in crate::objects) use mesh::*;
pub(in crate::objects) use selection_geometry::*;

impl ObjectsLayer {
    pub(super) const SELECTED_RENDER_LOD_LIMIT: usize = 200_000;
    pub(super) const MAX_DIRECT_FILL_VERTICES: usize = 1_000_000;

    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
    ) {
        let Some(_) = self.objects.as_ref() else {
            return;
        };
        if !self.visible {
            return;
        }
        self.ensure_filter_cache();
        self.ensure_color_groups();
        let continuous_colors = self.ensure_continuous_color_payload().cloned();
        if self.color_mode == ObjectColorMode::Continuous && continuous_colors.is_none() {
            return;
        }

        let visible_local = self.world_to_local_rect(visible_world, local_to_world_offset);
        let display_scale = self.display_scale();
        let display_offset = self.display_offset(local_to_world_offset);
        if self.display_mode == ObjectDisplayMode::Points {
            self.draw_points(
                ui,
                camera,
                viewport,
                visible_world,
                local_to_world_offset,
                gpu_available,
            );
            return;
        }

        const SELECTED_FILL_MESH_LIMIT: usize = 4096;
        let use_selected_only_fill_mesh = self.show_selection_overlay
            && self.selected_fill_opacity > 0.0
            && !self.selected_object_indices.is_empty()
            && (!gpu_available || self.selected_object_indices.len() <= SELECTED_FILL_MESH_LIMIT);
        let Some(base_render_lods) = self.render_lods.as_ref() else {
            return;
        };
        let render_lods = if self.has_active_filter() {
            let Some(filtered) = self.filtered_render_lods.as_ref() else {
                return;
            };
            filtered.clone()
        } else {
            base_render_lods.clone()
        };
        if render_lods.is_empty() {
            return;
        }

        let dataset_long_side_screen_px = self
            .bounds_local
            .map(|r| {
                (r.width() * display_scale.x)
                    .max(r.height() * display_scale.y)
                    .max(1e-6)
                    * camera.zoom_screen_per_lvl0_px.max(1e-9)
            })
            .unwrap_or_else(|| viewport.width().max(viewport.height()).max(1.0));
        let lod_idx = choose_lod_index(&render_lods, dataset_long_side_screen_px);
        let (use_fast_proxy_points, use_fill_proxy_points, lod_empty) = {
            let lod = &render_lods[lod_idx];
            (
                self.should_use_fast_proxy_points(dataset_long_side_screen_px),
                self.should_use_fill_proxy_points(lod),
                lod.bins.segments.is_empty(),
            )
        };
        if !use_fast_proxy_points && !use_fill_proxy_points && lod_empty {
            return;
        }
        let render_generation = self.render_cache_generation();

        let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let c = self.color_rgb;
        let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
        let selected_fill = egui::Color32::from_rgba_unmultiplied(
            255,
            245,
            140,
            (self.selected_fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
        );

        let mut selection_fill_composited_by_tiles = false;
        if use_fast_proxy_points || use_fill_proxy_points {
            let proxy_alpha = if use_fill_proxy_points {
                self.fill_opacity
            } else {
                self.opacity
            };
            self.draw_proxy_points(
                ui,
                camera,
                viewport,
                visible_world,
                local_to_world_offset,
                gpu_available,
                proxy_alpha,
                self.generation ^ 0x4641535450524f58,
            );
            let drew_selected_outline = gpu_available
                && self.draw_selected_outline_overlay_gpu(
                    ui,
                    camera,
                    viewport,
                    visible_local,
                    dataset_long_side_screen_px,
                    display_offset,
                    display_scale,
                    true,
                );
            if !drew_selected_outline && self.selected_point_positions_world.is_none() {
                self.draw_visible_selected_outline_overlay(
                    ui,
                    camera,
                    viewport,
                    visible_local,
                    local_to_world_offset,
                    dataset_long_side_screen_px,
                    gpu_available,
                );
            }
            self.draw_point_selection_overlay(
                ui,
                camera,
                viewport,
                local_to_world_offset,
                gpu_available,
            );
        } else {
            let color_groups_binding = self.active_color_groups();
            let active_color_groups = match self.color_mode {
                ObjectColorMode::Single => None,
                ObjectColorMode::ByProperty => color_groups_binding
                    .as_ref()
                    .filter(|g| g.property_key == self.color_property_key),
                ObjectColorMode::Continuous => None,
            };

            selection_fill_composited_by_tiles = self.draw_object_fills(
                ui,
                camera,
                viewport,
                visible_local,
                local_to_world_offset,
                display_offset,
                display_scale,
                gpu_available,
                active_color_groups.copied(),
                continuous_colors.as_ref(),
                render_generation,
            );

            if gpu_available {
                let lod = &render_lods[lod_idx];
                if active_color_groups.is_none()
                    && (continuous_colors.is_some()
                        || (self.show_selection_overlay
                            && !self.selected_object_indices.is_empty()))
                    && let Some(selection_lods) = self.object_selection_lods.as_ref()
                    && let Some(selection_lod) =
                        selection_lods.get(choose_object_selection_lod_index(
                            selection_lods,
                            dataset_long_side_screen_px,
                        ))
                    && let Some(object_count) = self.objects.as_ref().map(|objects| objects.len())
                {
                    let selection_state = if continuous_colors.is_some() {
                        let mut state = vec![0u8; object_count];
                        for (index, slot) in state.iter_mut().enumerate() {
                            if self.is_index_visible(index) {
                                *slot = 64;
                            }
                        }
                        if self.show_selection_overlay {
                            for index in &self.selected_object_indices {
                                if self.is_index_visible(*index)
                                    && let Some(slot) = state.get_mut(*index)
                                {
                                    *slot = 128;
                                }
                            }
                            if let Some(index) = self.selected_object_index
                                && self.is_index_visible(index)
                                && let Some(slot) = state.get_mut(index)
                            {
                                *slot = 255;
                            }
                        }
                        Arc::new(state)
                    } else {
                        Arc::clone(&self.selection_fill_state)
                    };
                    let item = ObjectLineBinsGlDrawItem {
                        data: ObjectLineBinsGlDrawData {
                            cache_id: object_render_cache_id(0x4a90, selection_lod.lod as u64),
                            state_cache_id: object_render_cache_id(0x4a91, 0),
                            generation: self.generation,
                            bins: Arc::clone(&selection_lod.bins),
                            selection_generation: if continuous_colors.is_some() {
                                render_generation
                                    ^ continuous_colors
                                        .as_ref()
                                        .map_or(0, |payload| payload.generation)
                            } else {
                                self.selection_generation
                            },
                            selection_state,
                            object_count,
                            color_cache_id: object_render_cache_id(0x4a92, 0),
                            color_generation: continuous_colors
                                .as_ref()
                                .map_or(0, |payload| payload.generation),
                            object_colors_rgba: continuous_colors
                                .as_ref()
                                .map(|payload| Arc::clone(&payload.colors_rgba)),
                        },
                        params: ObjectLineBinsGlDrawParams {
                            center_world: camera.center_world_lvl0,
                            zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                            base_width_points: self.width_screen_px.max(0.0),
                            selected_width_points: (self.width_screen_px + 1.0).max(1.25),
                            primary_width_points: (self.width_screen_px + 2.0).max(2.0),
                            base_color: color,
                            selected_color: egui::Color32::from_rgba_unmultiplied(
                                255, 245, 140, 210,
                            ),
                            primary_color: egui::Color32::from_rgba_unmultiplied(
                                255, 255, 255, 235,
                            ),
                            draw_unselected: continuous_colors.is_none(),
                            visible: self.visible,
                            local_to_world_offset: display_offset,
                            local_to_world_scale: display_scale,
                            object_color_opacity: self.opacity,
                        },
                        visible_world: visible_local,
                    };
                    let renderer = self.gl_object_selection.clone();
                    let cb = egui_glow::CallbackFn::new(move |info, painter| {
                        renderer.paint_many(info, painter, &[item.clone()]);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect: viewport,
                        callback: Arc::new(cb),
                    });
                } else {
                    let mut items = Vec::new();
                    if let Some(color_groups) = active_color_groups {
                        items.reserve(color_groups.groups.len());
                        for (group_idx, group) in color_groups.groups.iter().enumerate() {
                            let Some(c) =
                                self.effective_color_group_rgb(&color_groups.property_key, group)
                            else {
                                continue;
                            };
                            let group_lod = &group.lods
                                [choose_lod_index(&group.lods, dataset_long_side_screen_px)];
                            items.push(LineBinsGlDrawItem {
                                data: LineBinsGlDrawData {
                                    cache_id: object_property_render_cache_id(
                                        0x4a00 | u32::from(lod.lod),
                                        &color_groups.property_key,
                                        group_idx,
                                    ),
                                    generation: group.fill_generation,
                                    bins: Arc::clone(&group_lod.bins),
                                },
                                params: LineBinsGlDrawParams {
                                    center_world: camera.center_world_lvl0,
                                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                                    width_points: self.width_screen_px.max(0.0),
                                    color: egui::Color32::from_rgba_unmultiplied(
                                        c[0], c[1], c[2], a,
                                    ),
                                    visible: self.visible,
                                    local_to_world_offset: display_offset,
                                    local_to_world_scale: display_scale,
                                },
                                visible_world: visible_local,
                            });
                        }
                    } else {
                        items.push(LineBinsGlDrawItem {
                            data: LineBinsGlDrawData {
                                cache_id: object_render_cache_id(0x4a08, lod.lod as u64),
                                generation: render_generation,
                                bins: Arc::clone(&lod.bins),
                            },
                            params: LineBinsGlDrawParams {
                                center_world: camera.center_world_lvl0,
                                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                                width_points: self.width_screen_px.max(0.0),
                                color,
                                visible: self.visible,
                                local_to_world_offset: display_offset,
                                local_to_world_scale: display_scale,
                            },
                            visible_world: visible_local,
                        });
                    }

                    let renderer = self.gl.clone();
                    let cb = egui_glow::CallbackFn::new(move |info, painter| {
                        renderer.paint_many(info, painter, &items);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect: viewport,
                        callback: Arc::new(cb),
                    });

                    let drew_selected_outline = self.draw_selected_outline_overlay_gpu(
                        ui,
                        camera,
                        viewport,
                        visible_local,
                        dataset_long_side_screen_px,
                        display_offset,
                        display_scale,
                        true,
                    );
                    if !drew_selected_outline {
                        self.draw_visible_selected_outline_overlay(
                            ui,
                            camera,
                            viewport,
                            visible_local,
                            local_to_world_offset,
                            dataset_long_side_screen_px,
                            gpu_available,
                        );
                    }
                }
            } else if let Some(color_groups) = active_color_groups {
                for group in &color_groups.groups {
                    let Some(c) = self.effective_color_group_rgb(&color_groups.property_key, group)
                    else {
                        continue;
                    };
                    let group_lod =
                        &group.lods[choose_lod_index(&group.lods, dataset_long_side_screen_px)];
                    let stroke = egui::Stroke::new(
                        self.width_screen_px.max(0.0),
                        egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a),
                    );
                    let (bx0, by0, bx1, by1) =
                        group_lod.bins.bin_range_for_world_rect(visible_local);
                    for by in by0..=by1 {
                        for bx in bx0..=bx1 {
                            let bin_index = by * group_lod.bins.bins_w + bx;
                            for seg in group_lod.bins.bin_slice(bin_index) {
                                let a = camera.world_to_screen(
                                    self.local_to_world_point(
                                        egui::pos2(seg[0], seg[1]),
                                        local_to_world_offset,
                                    ),
                                    viewport,
                                );
                                let b = camera.world_to_screen(
                                    self.local_to_world_point(
                                        egui::pos2(seg[2], seg[3]),
                                        local_to_world_offset,
                                    ),
                                    viewport,
                                );
                                ui.painter().line_segment([a, b], stroke);
                            }
                        }
                    }
                }
            } else if let Some(payload) = continuous_colors.as_ref()
                && let Some(selection_lods) = self.object_selection_lods.as_ref()
                && let Some(selection_lod) = selection_lods.get(choose_object_selection_lod_index(
                    selection_lods,
                    dataset_long_side_screen_px,
                ))
            {
                let (bx0, by0, bx1, by1) =
                    selection_lod.bins.bin_range_for_world_rect(visible_local);
                for by in by0..=by1 {
                    for bx in bx0..=bx1 {
                        let bin_index = by * selection_lod.bins.bins_w + bx;
                        for seg in selection_lod.bins.bin_slice(bin_index) {
                            let object_index = seg[4].round().max(0.0) as usize;
                            if !self.is_index_visible(object_index) {
                                continue;
                            }
                            let Some(rgba) = payload.colors_rgba.get(object_index) else {
                                continue;
                            };
                            let alpha =
                                ((rgba[3] as f32) * self.opacity.clamp(0.0, 1.0)).round() as u8;
                            if alpha == 0 {
                                continue;
                            }
                            let stroke = egui::Stroke::new(
                                self.width_screen_px.max(0.0),
                                egui::Color32::from_rgba_unmultiplied(
                                    rgba[0], rgba[1], rgba[2], alpha,
                                ),
                            );
                            let a = camera.world_to_screen(
                                self.local_to_world_point(
                                    egui::pos2(seg[0], seg[1]),
                                    local_to_world_offset,
                                ),
                                viewport,
                            );
                            let b = camera.world_to_screen(
                                self.local_to_world_point(
                                    egui::pos2(seg[2], seg[3]),
                                    local_to_world_offset,
                                ),
                                viewport,
                            );
                            ui.painter().line_segment([a, b], stroke);
                        }
                    }
                }
            } else {
                let lod = &render_lods[lod_idx];
                let stroke = egui::Stroke::new(self.width_screen_px.max(0.0), color);
                let (bx0, by0, bx1, by1) = lod.bins.bin_range_for_world_rect(visible_local);
                for by in by0..=by1 {
                    for bx in bx0..=bx1 {
                        let bin_index = by * lod.bins.bins_w + bx;
                        for seg in lod.bins.bin_slice(bin_index) {
                            let a = camera.world_to_screen(
                                self.local_to_world_point(
                                    egui::pos2(seg[0], seg[1]),
                                    local_to_world_offset,
                                ),
                                viewport,
                            );
                            let b = camera.world_to_screen(
                                self.local_to_world_point(
                                    egui::pos2(seg[2], seg[3]),
                                    local_to_world_offset,
                                ),
                                viewport,
                            );
                            ui.painter().line_segment([a, b], stroke);
                        }
                    }
                }
            }
        }

        if use_selected_only_fill_mesh && !selection_fill_composited_by_tiles {
            self.ensure_cpu_selection_fill_mesh();
        }

        if !use_fast_proxy_points
            && !selection_fill_composited_by_tiles
            && self.show_selection_overlay
            && self.selected_fill_opacity > 0.0
            && let Some(fill_mesh) = self.selected_fill_mesh.as_ref()
            && fill_mesh.vertices_local.len() <= Self::MAX_DIRECT_FILL_VERTICES
            && fill_mesh.bounds_local.intersects(visible_local)
        {
            if gpu_available {
                let item = PolygonFillGlDrawItem {
                    data: PolygonFillGlDrawData {
                        cache_id: 0x5345474f424a30u64,
                        generation: self.selection_generation,
                        vertices_local: Arc::clone(&fill_mesh.vertices_local),
                    },
                    params: PolygonFillGlDrawParams {
                        center_world: camera.center_world_lvl0,
                        zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                        color: selected_fill,
                        visible: self.visible,
                        local_to_world_offset: display_offset,
                        local_to_world_scale: display_scale,
                    },
                    visible_world: fill_mesh.bounds_local,
                };
                let renderer = self.gl_fill.clone();
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint_many(info, painter, &[item.clone()]);
                });
                ui.painter().add(egui::PaintCallback {
                    rect: viewport,
                    callback: Arc::new(cb),
                });
            } else {
                for tri in fill_mesh.vertices_local.chunks_exact(3) {
                    let points = tri
                        .iter()
                        .map(|p| {
                            camera.world_to_screen(
                                self.local_to_world_point(
                                    egui::pos2(p[0], p[1]),
                                    local_to_world_offset,
                                ),
                                viewport,
                            )
                        })
                        .collect::<Vec<_>>();
                    ui.painter().add(egui::Shape::convex_polygon(
                        points,
                        selected_fill,
                        egui::Stroke::NONE,
                    ));
                }
            }
        } else if !use_fast_proxy_points
            && !selection_fill_composited_by_tiles
            && self.show_selection_overlay
            && gpu_available
            && self.selected_fill_opacity > 0.0
            && !self.selected_object_indices.is_empty()
            && let Some(fill_mesh) = self.object_fill_mesh.as_ref()
            && fill_mesh.bounds_local.intersects(visible_local)
        {
            self.draw_object_selection_fill_overlay_gpu(
                ui,
                camera,
                viewport,
                visible_local,
                display_offset,
                display_scale,
                selected_fill,
            );
        }

        if !gpu_available && self.show_selection_overlay && !self.selected_object_indices.is_empty()
        {
            let secondary_stroke = egui::Stroke::new(
                (self.width_screen_px + 1.0).max(1.25),
                egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
            );
            let primary_stroke = egui::Stroke::new(
                (self.width_screen_px + 2.0).max(2.0),
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
            );
            if let Some(objects) = self.objects.as_ref() {
                for idx in &self.selected_object_indices {
                    if !self.is_index_visible(*idx) {
                        continue;
                    }
                    let Some(obj) = objects.get(*idx) else {
                        continue;
                    };
                    let stroke = if Some(*idx) == self.selected_object_index {
                        primary_stroke
                    } else {
                        secondary_stroke
                    };
                    let bbox_screen = egui::Rect::from_two_pos(
                        camera.world_to_screen(
                            self.local_to_world_point(obj.bbox_world.min, local_to_world_offset),
                            viewport,
                        ),
                        camera.world_to_screen(
                            self.local_to_world_point(obj.bbox_world.max, local_to_world_offset),
                            viewport,
                        ),
                    );
                    let screen_span = bbox_screen.width().abs().max(bbox_screen.height().abs());
                    let total_vertices = object_vertex_count(obj);
                    if screen_span <= 64.0 && total_vertices > 128 {
                        ui.painter().rect_stroke(
                            bbox_screen,
                            0.0,
                            stroke,
                            egui::StrokeKind::Middle,
                        );
                        continue;
                    }
                    let max_points = if screen_span <= 160.0 {
                        256
                    } else if screen_span <= 512.0 {
                        1024
                    } else {
                        usize::MAX
                    };
                    for poly in &obj.polygons_world {
                        let pts = simplified_polyline_screen_points(
                            poly,
                            max_points,
                            camera,
                            local_to_world_offset,
                            self.display_transform,
                            viewport,
                        );
                        if pts.len() >= 2 {
                            ui.painter().add(egui::Shape::line(pts, stroke));
                        }
                    }
                }
            }
        }
    }

    fn draw_object_selection_fill_overlay_gpu(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        display_offset: egui::Vec2,
        display_scale: egui::Vec2,
        selected_fill: egui::Color32,
    ) {
        let Some(fill_mesh) = self.object_fill_mesh.as_ref() else {
            return;
        };
        if fill_mesh.vertices_local.is_empty() {
            return;
        }

        let params = ObjectFillGlDrawParams {
            center_world: camera.center_world_lvl0,
            zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
            selected_color: selected_fill,
            primary_color: selected_fill,
            visible: self.visible,
            local_to_world_offset: display_offset,
            local_to_world_scale: display_scale,
            object_color_opacity: 1.0,
        };

        let mut items = Vec::new();
        let state_cache_id = object_render_cache_id(0x4a31, 0);
        if fill_mesh.vertices_local.len() < 500_000 {
            items.push(ObjectFillGlDrawItem {
                data: ObjectFillGlDrawData {
                    resource_cache_id: self.render_resource_cache_id,
                    style_cache_id: self.render_style_cache_id,
                    cache_id: object_render_cache_id(0x4a32, 0),
                    state_cache_id,
                    generation: self.geometry_generation,
                    vertices_local: Arc::clone(&fill_mesh.vertices_local),
                    object_count: fill_mesh.object_count,
                    selection_generation: self.selection_generation,
                    selection_state: Arc::clone(&self.selection_fill_state),
                    color_cache_id: 0,
                    color_generation: 0,
                    object_colors_rgba: None,
                },
                params,
                visible_world: fill_mesh.bounds_local,
            });
        } else {
            for slice in fill_mesh.spatial_slices_for_local_rect(visible_local) {
                items.push(ObjectFillGlDrawItem {
                    data: ObjectFillGlDrawData {
                        resource_cache_id: self.render_resource_cache_id,
                        style_cache_id: self.render_style_cache_id,
                        cache_id: object_render_cache_id_usize(0x4a80, slice.bin_index),
                        state_cache_id,
                        generation: self.geometry_generation,
                        vertices_local: slice.vertices_local,
                        object_count: fill_mesh.object_count,
                        selection_generation: self.selection_generation,
                        selection_state: Arc::clone(&self.selection_fill_state),
                        color_cache_id: 0,
                        color_generation: 0,
                        object_colors_rgba: None,
                    },
                    params: params.clone(),
                    visible_world: slice.bounds_local,
                });
            }
        }

        if items
            .iter()
            .map(|item| item.data.vertices_local.len())
            .sum::<usize>()
            > Self::MAX_DIRECT_FILL_VERTICES
        {
            return;
        }

        if items.is_empty() {
            return;
        }
        let renderer = self.gl_object_fill.clone();
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            renderer.paint_many(info, painter, &items);
        });
        ui.painter().add(egui::PaintCallback {
            rect: viewport,
            callback: Arc::new(cb),
        });
    }

    fn draw_selected_outline_overlay_gpu(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        dataset_long_side_screen_px: f32,
        display_offset: egui::Vec2,
        display_scale: egui::Vec2,
        allow_selection_state_fallback: bool,
    ) -> bool {
        if !self.show_selection_overlay || self.selected_object_indices.is_empty() {
            return false;
        }

        if let Some(selected_lods) = self.selected_render_lods.as_ref()
            && let Some(selected_lod) =
                selected_lods.get(choose_lod_index(selected_lods, dataset_long_side_screen_px))
        {
            let mut items = vec![LineBinsGlDrawItem {
                data: LineBinsGlDrawData {
                    cache_id: object_render_cache_id(0x4a50, selected_lod.lod as u64),
                    generation: self.selection_generation,
                    bins: Arc::clone(&selected_lod.bins),
                },
                params: LineBinsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    width_points: (self.width_screen_px + 1.0).max(1.25),
                    color: egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
                    visible: self.visible,
                    local_to_world_offset: display_offset,
                    local_to_world_scale: display_scale,
                },
                visible_world: visible_local,
            }];

            if let Some(primary_lods) = self.primary_selected_render_lods.as_ref()
                && let Some(primary_lod) =
                    primary_lods.get(choose_lod_index(primary_lods, dataset_long_side_screen_px))
            {
                items.push(LineBinsGlDrawItem {
                    data: LineBinsGlDrawData {
                        cache_id: object_render_cache_id(0x4a60, primary_lod.lod as u64),
                        generation: self.selection_generation,
                        bins: Arc::clone(&primary_lod.bins),
                    },
                    params: LineBinsGlDrawParams {
                        center_world: camera.center_world_lvl0,
                        zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                        width_points: (self.width_screen_px + 2.0).max(2.0),
                        color: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
                        visible: self.visible,
                        local_to_world_offset: display_offset,
                        local_to_world_scale: display_scale,
                    },
                    visible_world: visible_local,
                });
            }

            let renderer = self.gl.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint_many(info, painter, &items);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
            return true;
        }

        if allow_selection_state_fallback
            && let Some(selection_lods) = self.object_selection_lods.as_ref()
            && let Some(selection_lod) = selection_lods.get(choose_object_selection_lod_index(
                selection_lods,
                dataset_long_side_screen_px,
            ))
            && let Some(object_count) = self.objects.as_ref().map(|objects| objects.len())
        {
            let sel_items = [ObjectLineBinsGlDrawItem {
                data: ObjectLineBinsGlDrawData {
                    cache_id: object_render_cache_id(0x4a40, selection_lod.lod as u64),
                    state_cache_id: object_render_cache_id(0x4a41, 0),
                    generation: self.generation,
                    bins: Arc::clone(&selection_lod.bins),
                    selection_generation: self.selection_generation,
                    selection_state: Arc::clone(&self.selection_fill_state),
                    object_count,
                    color_cache_id: 0,
                    color_generation: 0,
                    object_colors_rgba: None,
                },
                params: ObjectLineBinsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    base_width_points: self.width_screen_px.max(0.0),
                    selected_width_points: (self.width_screen_px + 1.0).max(1.25),
                    primary_width_points: (self.width_screen_px + 2.0).max(2.0),
                    base_color: egui::Color32::TRANSPARENT,
                    selected_color: egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
                    primary_color: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
                    draw_unselected: false,
                    visible: self.visible,
                    local_to_world_offset: display_offset,
                    local_to_world_scale: display_scale,
                    object_color_opacity: 1.0,
                },
                visible_world: visible_local,
            }];
            let renderer = self.gl_object_selection.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint_many(info, painter, &sel_items);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
            return true;
        }
        false
    }

    fn draw_visible_selected_outline_overlay(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        local_to_world_offset: egui::Vec2,
        dataset_long_side_screen_px: f32,
        gpu_available: bool,
    ) {
        if !self.show_selection_overlay
            || self.display_mode != ObjectDisplayMode::Polygons
            || self.selected_object_indices.len() <= Self::SELECTED_RENDER_LOD_LIMIT
        {
            return;
        }
        if gpu_available {
            let Some((lods, generation)) = self
                .visible_selected_render_lods(visible_local)
                .map(|cache| (cache.lods.clone(), cache.generation))
            else {
                return;
            };
            let Some(lod) = lods.get(choose_lod_index(&lods, dataset_long_side_screen_px)) else {
                return;
            };
            let item = LineBinsGlDrawItem {
                data: LineBinsGlDrawData {
                    cache_id: object_render_cache_id(0x4a70, lod.lod as u64),
                    generation,
                    bins: Arc::clone(&lod.bins),
                },
                params: LineBinsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    width_points: (self.width_screen_px + 1.0).max(1.25),
                    color: egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
                    visible: self.visible,
                    local_to_world_offset: self.display_offset(local_to_world_offset),
                    local_to_world_scale: self.display_scale(),
                },
                visible_world: visible_local,
            };
            let renderer = self.gl.clone();
            let items = vec![item];
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint_many(info, painter, &items);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
            return;
        }

        let visible_indices = self.visible_selected_object_indices(visible_local);
        if visible_indices.is_empty() {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            return;
        };

        let selected_stroke = egui::Stroke::new(
            (self.width_screen_px + 1.0).max(1.25),
            egui::Color32::from_rgba_unmultiplied(255, 245, 140, 210),
        );
        let primary_stroke = egui::Stroke::new(
            (self.width_screen_px + 2.0).max(2.0),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 235),
        );
        for idx in visible_indices {
            let Some(obj) = objects.get(idx) else {
                continue;
            };
            let stroke = if self.selected_object_index == Some(idx) {
                primary_stroke
            } else {
                selected_stroke
            };
            for poly in &obj.polygons_world {
                for seg in poly.windows(2) {
                    let p0 = self.local_to_world_point(seg[0], local_to_world_offset);
                    let p1 = self.local_to_world_point(seg[1], local_to_world_offset);
                    if !visible_local.contains(seg[0]) && !visible_local.contains(seg[1]) {
                        let seg_rect = egui::Rect::from_two_pos(seg[0], seg[1]);
                        if !seg_rect.intersects(visible_local) {
                            continue;
                        }
                    }
                    ui.painter().line_segment(
                        [
                            camera.world_to_screen(p0, viewport),
                            camera.world_to_screen(p1, viewport),
                        ],
                        stroke,
                    );
                }
            }
        }
    }

    fn visible_selected_render_lods(
        &mut self,
        visible_local: egui::Rect,
    ) -> Option<&VisibleSelectedRenderCache> {
        let visible_indices = self.visible_selected_object_indices(visible_local);
        if visible_indices.is_empty() {
            self.visible_selected_render_cache = None;
            return None;
        }
        if self
            .visible_selected_render_cache
            .as_ref()
            .is_some_and(|cache| {
                cache.selection_generation == self.selection_generation
                    && cache.visible_indices == visible_indices
            })
        {
            return self.visible_selected_render_cache.as_ref();
        }

        let objects = self.objects.as_ref()?;
        let selected = visible_indices
            .iter()
            .filter_map(|idx| objects.get(*idx).cloned())
            .collect::<Vec<_>>();
        let lods = build_render_lods(&selected).ok()?;
        let generation =
            visible_selected_cache_generation(self.selection_generation, &visible_indices);
        self.visible_selected_render_cache = Some(VisibleSelectedRenderCache {
            selection_generation: self.selection_generation,
            visible_indices,
            lods,
            generation,
        });
        self.visible_selected_render_cache.as_ref()
    }

    fn visible_selected_object_indices(&self, visible_local: egui::Rect) -> Vec<usize> {
        let (Some(objects), Some(bins)) = (self.objects.as_ref(), self.bins.as_ref()) else {
            return Vec::new();
        };
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(visible_local);
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u32 in bins.bin_slice(bi) {
                    let idx = idx_u32 as usize;
                    if !seen.insert(idx) {
                        continue;
                    }
                    if !self.selected_object_indices.contains(&idx) || !self.is_index_visible(idx) {
                        continue;
                    }
                    let Some(obj) = objects.get(idx) else {
                        continue;
                    };
                    if obj.bbox_world.intersects(visible_local) {
                        out.push(idx);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    fn point_radius_screen_px(&self) -> f32 {
        (self.width_screen_px + 2.75).clamp(2.5, 10.0)
    }

    fn fill_proxy_radius_screen_px(&self) -> f32 {
        (self.width_screen_px + 0.75).clamp(1.25, 3.0)
    }

    fn point_pick_radius_world(&self, camera: &crate::camera::Camera) -> f32 {
        (self.point_radius_screen_px() + 4.0) / camera.zoom_screen_per_lvl0_px.max(1e-6)
    }

    fn effective_color_group_rgb(
        &self,
        property_key: &str,
        group: &ObjectColorGroup,
    ) -> Option<[u8; 3]> {
        self.color_value_visible_for_label(property_key, &group.value_label)
            .then_some(())?;
        if self.color_level_overrides_property_key != property_key {
            return Some(group.color_rgb);
        }
        let override_style = self.color_level_overrides.get(&group.value_label).copied();
        if override_style.is_some_and(|style| !style.visible) {
            return None;
        }
        Some(
            override_style
                .and_then(|style| style.color_rgb)
                .unwrap_or(group.color_rgb),
        )
    }

    fn should_use_fill_proxy_points(&self, lod: &ObjectRenderLod) -> bool {
        self.fast_rendering && self.fill_cells && self.fill_opacity > 0.0 && lod.lod >= 2
    }

    fn should_use_fast_proxy_points(&self, dataset_long_side_screen_px: f32) -> bool {
        const FAST_RENDER_MIN_OBJECTS: usize = 50_000;
        const FAST_RENDER_OUTLINE_SCREEN_PX: f32 = 3_000.0;
        self.fast_rendering
            && self.display_mode == ObjectDisplayMode::Polygons
            && self.filtered_count() >= FAST_RENDER_MIN_OBJECTS
            && dataset_long_side_screen_px < FAST_RENDER_OUTLINE_SCREEN_PX
    }

    fn proxy_point_style(&self, color_rgb: [u8; 3], opacity: f32) -> PointsStyle {
        PointsStyle {
            radius_screen_px: self.fill_proxy_radius_screen_px(),
            fill_positive: egui::Color32::from_rgba_unmultiplied(
                color_rgb[0],
                color_rgb[1],
                color_rgb[2],
                (opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
            ),
            fill_negative: egui::Color32::TRANSPARENT,
            stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
        }
    }

    fn draw_proxy_points(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
        opacity: f32,
        generation_seed: u64,
    ) {
        let continuous_colors = self.ensure_continuous_color_payload().cloned();
        if self.color_mode == ObjectColorMode::Continuous && continuous_colors.is_none() {
            return;
        }
        let color_groups_binding = self.active_color_groups().cloned();
        let active_color_groups = match self.color_mode {
            ObjectColorMode::Single => None,
            ObjectColorMode::ByProperty => color_groups_binding
                .as_ref()
                .filter(|groups| groups.property_key == self.color_property_key),
            ObjectColorMode::Continuous => None,
        };

        if let Some(color_groups) = active_color_groups {
            if self.gl_proxy_group_points.len() < color_groups.groups.len() {
                self.gl_proxy_group_points
                    .resize_with(color_groups.groups.len(), PointsGlRenderer::default);
            }
            for (group_idx, group) in color_groups.groups.iter().enumerate() {
                let Some(color_rgb) =
                    self.effective_color_group_rgb(&color_groups.property_key, group)
                else {
                    continue;
                };
                let style = self.proxy_point_style(color_rgb, opacity);
                self.draw_proxy_point_batch(
                    ui,
                    camera,
                    viewport,
                    visible_world,
                    local_to_world_offset,
                    gpu_available,
                    &group.point_positions_world,
                    &group.point_values,
                    style,
                    group.fill_generation ^ generation_seed,
                    None,
                    1.0,
                    Some(&self.gl_proxy_group_points[group_idx]),
                );
            }
            return;
        }

        let base_positions = if self.has_active_filter() {
            self.filtered_point_positions_world.as_ref()
        } else {
            self.point_positions_world.as_ref()
        };
        let base_values = if self.has_active_filter() {
            self.filtered_point_values.as_ref()
        } else {
            self.point_values.as_ref()
        };
        let (Some(base_positions), Some(base_values)) = (base_positions, base_values) else {
            return;
        };
        let point_colors = continuous_colors.as_ref().map(|payload| {
            if let Some(indices) = self.filtered_ordered_indices.as_ref() {
                Arc::new(
                    indices
                        .iter()
                        .filter_map(|index| payload.colors_rgba.get(*index).copied())
                        .collect::<Vec<_>>(),
                )
            } else {
                Arc::clone(&payload.colors_rgba)
            }
        });
        self.draw_proxy_point_batch(
            ui,
            camera,
            viewport,
            visible_world,
            local_to_world_offset,
            gpu_available,
            base_positions,
            base_values,
            self.proxy_point_style(self.color_rgb, opacity),
            self.render_cache_generation() ^ generation_seed,
            point_colors,
            opacity,
            Some(&self.gl_points),
        );
    }

    fn draw_proxy_point_batch(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
        positions_world: &Arc<Vec<egui::Pos2>>,
        values: &Arc<Vec<f32>>,
        style: PointsStyle,
        generation: u64,
        colors_rgba: Option<Arc<Vec<[u8; 4]>>>,
        color_opacity: f32,
        renderer: Option<&PointsGlRenderer>,
    ) {
        if positions_world.is_empty() || values.is_empty() {
            return;
        }

        if gpu_available {
            let Some(renderer) = renderer else {
                return;
            };
            let data = crate::render::points_gl::PointsGlDrawData {
                generation,
                positions_world: Arc::clone(positions_world),
                values: Arc::clone(values),
                colors_rgba: colors_rgba.clone(),
            };
            let params = crate::render::points_gl::PointsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                threshold: 0.5,
                style,
                visible: self.visible,
                local_to_world_offset: self.display_offset(local_to_world_offset),
                local_to_world_scale: self.display_scale(),
                color_opacity,
            };
            let renderer = renderer.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint(info, painter, &data, &params);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
            return;
        }

        for (index, pos) in positions_world.iter().enumerate() {
            let world = self.local_to_world_point(*pos, local_to_world_offset);
            if !visible_world.contains(world) {
                continue;
            }
            let screen = camera.world_to_screen(world, viewport);
            let fill = colors_rgba
                .as_ref()
                .and_then(|colors| colors.get(index))
                .map(|rgba| {
                    egui::Color32::from_rgba_unmultiplied(
                        rgba[0],
                        rgba[1],
                        rgba[2],
                        ((rgba[3] as f32) * color_opacity.clamp(0.0, 1.0)).round() as u8,
                    )
                })
                .unwrap_or(style.fill_positive);
            if fill.a() == 0 {
                continue;
            }
            ui.painter()
                .circle_filled(screen, style.radius_screen_px, fill);
        }
    }

    fn draw_points(
        &mut self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
    ) {
        let continuous_colors = self.ensure_continuous_color_payload().cloned();
        if self.color_mode == ObjectColorMode::Continuous && continuous_colors.is_none() {
            return;
        }
        let base_positions = if self.has_active_filter() {
            self.filtered_point_positions_world.as_ref()
        } else {
            self.point_positions_world.as_ref()
        };
        let base_values = if self.has_active_filter() {
            self.filtered_point_values.as_ref()
        } else {
            self.point_values.as_ref()
        };
        let (Some(base_positions), Some(base_values)) = (base_positions, base_values) else {
            return;
        };
        if base_positions.is_empty() {
            return;
        }

        let color_groups_binding = self.active_color_groups().cloned();
        let active_color_groups = match self.color_mode {
            ObjectColorMode::Single => None,
            ObjectColorMode::ByProperty => color_groups_binding
                .as_ref()
                .filter(|groups| groups.property_key == self.color_property_key),
            ObjectColorMode::Continuous => None,
        };
        if let Some(color_groups) = active_color_groups {
            if self.gl_proxy_group_points.len() < color_groups.groups.len() {
                self.gl_proxy_group_points
                    .resize_with(color_groups.groups.len(), PointsGlRenderer::default);
            }
            for (group_idx, group) in color_groups.groups.iter().enumerate() {
                let Some(color_rgb) =
                    self.effective_color_group_rgb(&color_groups.property_key, group)
                else {
                    continue;
                };
                self.draw_proxy_point_batch(
                    ui,
                    camera,
                    viewport,
                    visible_world,
                    local_to_world_offset,
                    gpu_available,
                    &group.point_positions_world,
                    &group.point_values,
                    self.point_style_for_rgb(color_rgb),
                    group.fill_generation,
                    None,
                    1.0,
                    Some(&self.gl_proxy_group_points[group_idx]),
                );
            }
            self.draw_point_selection_overlay(
                ui,
                camera,
                viewport,
                local_to_world_offset,
                gpu_available,
            );
            return;
        }

        let point_colors = continuous_colors.as_ref().map(|payload| {
            if let Some(indices) = self.filtered_ordered_indices.as_ref() {
                Arc::new(
                    indices
                        .iter()
                        .filter_map(|index| payload.colors_rgba.get(*index).copied())
                        .collect::<Vec<_>>(),
                )
            } else {
                Arc::clone(&payload.colors_rgba)
            }
        });

        if gpu_available {
            let data = crate::render::points_gl::PointsGlDrawData {
                generation: self.render_cache_generation(),
                positions_world: Arc::clone(base_positions),
                values: Arc::clone(base_values),
                colors_rgba: point_colors.clone(),
            };
            let params = crate::render::points_gl::PointsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                threshold: 0.5,
                style: self.base_point_style(),
                visible: self.visible,
                local_to_world_offset: self.display_offset(local_to_world_offset),
                local_to_world_scale: self.display_scale(),
                color_opacity: self.opacity,
            };
            let renderer = self.gl_points.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                renderer.paint(info, painter, &data, &params);
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
        } else {
            let world_margin =
                (self.point_radius_screen_px() + 4.0) / camera.zoom_screen_per_lvl0_px.max(1e-6);
            let visible_world = visible_world.expand(world_margin);
            let radius = self.point_radius_screen_px();
            for (index, &p) in base_positions.iter().enumerate() {
                let world = self.local_to_world_point(p, local_to_world_offset);
                if !visible_world.contains(world) {
                    continue;
                }
                let s = camera.world_to_screen(world, viewport);
                let fill = point_colors
                    .as_ref()
                    .and_then(|colors| colors.get(index))
                    .map(|rgba| {
                        egui::Color32::from_rgba_unmultiplied(
                            rgba[0],
                            rgba[1],
                            rgba[2],
                            ((rgba[3] as f32) * self.opacity.clamp(0.0, 1.0)).round() as u8,
                        )
                    })
                    .unwrap_or(self.base_point_style().fill_positive);
                if fill.a() == 0 {
                    continue;
                }
                ui.painter().circle_filled(s, radius, fill);
            }
        }
        self.draw_point_selection_overlay(
            ui,
            camera,
            viewport,
            local_to_world_offset,
            gpu_available,
        );
    }

    fn draw_point_selection_overlay(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        local_to_world_offset: egui::Vec2,
        gpu_available: bool,
    ) {
        if !self.show_selection_overlay {
            return;
        }
        if gpu_available {
            if let (Some(sel_positions), Some(sel_values)) = (
                self.selected_point_positions_world.as_ref(),
                self.selected_point_values.as_ref(),
            ) {
                let data = crate::render::points_gl::PointsGlDrawData {
                    generation: self.selection_generation,
                    positions_world: Arc::clone(sel_positions),
                    values: Arc::clone(sel_values),
                    colors_rgba: None,
                };
                let params = crate::render::points_gl::PointsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    threshold: 0.5,
                    style: self.selected_point_style(),
                    visible: self.visible,
                    local_to_world_offset: self.display_offset(local_to_world_offset),
                    local_to_world_scale: self.display_scale(),
                    color_opacity: 1.0,
                };
                let renderer = self.gl_points.clone();
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect: viewport,
                    callback: Arc::new(cb),
                });
            }

            if let (Some(primary_positions), Some(primary_values)) = (
                self.primary_selected_point_positions_world.as_ref(),
                self.primary_selected_point_values.as_ref(),
            ) {
                let data = crate::render::points_gl::PointsGlDrawData {
                    generation: self.selection_generation.wrapping_mul(1021),
                    positions_world: Arc::clone(primary_positions),
                    values: Arc::clone(primary_values),
                    colors_rgba: None,
                };
                let params = crate::render::points_gl::PointsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    threshold: 0.5,
                    style: self.primary_selected_point_style(),
                    visible: self.visible,
                    local_to_world_offset: self.display_offset(local_to_world_offset),
                    local_to_world_scale: self.display_scale(),
                    color_opacity: 1.0,
                };
                let renderer = self.gl_points.clone();
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect: viewport,
                    callback: Arc::new(cb),
                });
            }
        } else if let Some(positions) = self.selected_point_positions_world.as_ref() {
            let radius = self.selected_point_style().radius_screen_px;
            let color = self.selected_point_style().fill_positive;
            for &point in positions.iter() {
                let world = self.local_to_world_point(point, local_to_world_offset);
                let screen = camera.world_to_screen(world, viewport);
                ui.painter().circle_filled(screen, radius, color);
            }
        }
    }

    fn base_point_style(&self) -> PointsStyle {
        self.point_style_for_rgb(self.color_rgb)
    }

    fn point_style_for_rgb(&self, color_rgb: [u8; 3]) -> PointsStyle {
        PointsStyle {
            radius_screen_px: self.point_radius_screen_px(),
            fill_positive: egui::Color32::from_rgba_unmultiplied(
                color_rgb[0],
                color_rgb[1],
                color_rgb[2],
                (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
            ),
            fill_negative: egui::Color32::TRANSPARENT,
            stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
        }
    }

    fn selected_point_style(&self) -> PointsStyle {
        PointsStyle {
            radius_screen_px: self.point_radius_screen_px() + 0.75,
            fill_positive: egui::Color32::from_rgba_unmultiplied(
                255,
                245,
                140,
                (self.selected_fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
            ),
            fill_negative: egui::Color32::TRANSPARENT,
            stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
        }
    }

    fn primary_selected_point_style(&self) -> PointsStyle {
        PointsStyle {
            radius_screen_px: self.point_radius_screen_px() + 1.25,
            fill_positive: egui::Color32::WHITE,
            fill_negative: egui::Color32::TRANSPARENT,
            stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
        }
    }
}
