use super::core::{build_object_point_payload, object_proxy_position_world, rect_bins};
use super::*;
use anyhow::Context;
use lyon_path::Path;
use lyon_path::math::point;
use lyon_tessellation::{BuffersBuilder, FillOptions, FillTessellator, FillVertex, VertexBuffers};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl ObjectsLayer {
    pub(super) const SELECTED_RENDER_LOD_LIMIT: usize = 200_000;

    fn display_scale(&self) -> egui::Vec2 {
        egui::vec2(
            self.display_transform.scale[0].max(1e-6),
            self.display_transform.scale[1].max(1e-6),
        )
    }

    fn display_offset(&self, local_to_world_offset: egui::Vec2) -> egui::Vec2 {
        egui::vec2(
            local_to_world_offset.x + self.display_transform.translation[0],
            local_to_world_offset.y + self.display_transform.translation[1],
        )
    }

    fn local_to_world_point(
        &self,
        local: egui::Pos2,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Pos2 {
        let scale = self.display_scale();
        let offset = self.display_offset(local_to_world_offset);
        egui::pos2(local.x * scale.x + offset.x, local.y * scale.y + offset.y)
    }

    fn world_to_local_point(
        &self,
        world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Pos2 {
        let scale = self.display_scale();
        let offset = self.display_offset(local_to_world_offset);
        egui::pos2(
            (world.x - offset.x) / scale.x,
            (world.y - offset.y) / scale.y,
        )
    }

    fn world_to_local_rect(
        &self,
        world: egui::Rect,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Rect {
        let min = self.world_to_local_point(world.min, local_to_world_offset);
        let max = self.world_to_local_point(world.max, local_to_world_offset);
        egui::Rect::from_min_max(
            egui::pos2(min.x.min(max.x), min.y.min(max.y)),
            egui::pos2(min.x.max(max.x), min.y.max(max.y)),
        )
    }

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
        if use_selected_only_fill_mesh {
            self.ensure_cpu_selection_fill_mesh();
        }

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

        let a = (self.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let c = self.color_rgb;
        let color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], a);
        let selected_fill = egui::Color32::from_rgba_unmultiplied(
            255,
            245,
            140,
            (self.selected_fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
        );

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
            if gpu_available {
                let _ = self.draw_selected_outline_overlay_gpu(
                    ui,
                    camera,
                    viewport,
                    visible_local,
                    dataset_long_side_screen_px,
                    display_offset,
                    display_scale,
                    false,
                );
            }
            if self.selected_point_positions_world.is_none() {
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
            };

            if self.fill_cells
                && self.fill_opacity > 0.0
                && let Some(fill_mesh) = self.object_fill_mesh.as_ref()
                && fill_mesh.bounds_local.intersects(visible_local)
            {
                let fill_alpha = (self.fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
                if gpu_available {
                    let mut items = Vec::new();
                    if let Some(color_groups) = active_color_groups {
                        items.reserve(color_groups.groups.len());
                        for (group_idx, group) in color_groups.groups.iter().enumerate() {
                            let Some(c) =
                                self.effective_color_group_rgb(&color_groups.property_key, group)
                            else {
                                continue;
                            };
                            let fill_color =
                                egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], fill_alpha);
                            items.push(ObjectFillGlDrawItem {
                                data: ObjectFillGlDrawData {
                                    cache_id: object_render_cache_id_usize(0x4a20, group_idx),
                                    state_cache_id: object_render_cache_id_usize(0x4a21, group_idx),
                                    generation: self.generation,
                                    vertices_local: Arc::clone(&fill_mesh.vertices_local),
                                    object_count: fill_mesh.object_count,
                                    selection_generation: group.fill_generation,
                                    selection_state: Arc::clone(&group.fill_state),
                                },
                                params: ObjectFillGlDrawParams {
                                    center_world: camera.center_world_lvl0,
                                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                                    selected_color: fill_color,
                                    primary_color: fill_color,
                                    visible: self.visible,
                                    local_to_world_offset: display_offset,
                                    local_to_world_scale: display_scale,
                                },
                                visible_world: fill_mesh.bounds_local,
                            });
                        }
                    } else {
                        let fill_color =
                            egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], fill_alpha);
                        let mut visible_fill_state = vec![0u8; fill_mesh.object_count];
                        for idx in 0..fill_mesh.object_count {
                            if self.is_index_visible(idx)
                                && let Some(slot) = visible_fill_state.get_mut(idx)
                            {
                                *slot = 255;
                            }
                        }
                        items.push(ObjectFillGlDrawItem {
                            data: ObjectFillGlDrawData {
                                cache_id: object_render_cache_id(0x4a22, 0),
                                state_cache_id: object_render_cache_id(0x4a23, 0),
                                generation: self.generation,
                                vertices_local: Arc::clone(&fill_mesh.vertices_local),
                                object_count: fill_mesh.object_count,
                                selection_generation: self.generation,
                                selection_state: Arc::new(visible_fill_state),
                            },
                            params: ObjectFillGlDrawParams {
                                center_world: camera.center_world_lvl0,
                                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                                selected_color: fill_color,
                                primary_color: fill_color,
                                visible: self.visible,
                                local_to_world_offset: display_offset,
                                local_to_world_scale: display_scale,
                            },
                            visible_world: fill_mesh.bounds_local,
                        });
                    }
                    let renderer = self.gl_object_fill.clone();
                    let cb = egui_glow::CallbackFn::new(move |info, painter| {
                        renderer.paint_many(info, painter, &items);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect: viewport,
                        callback: Arc::new(cb),
                    });
                } else if let Some(color_groups) = active_color_groups {
                    let mut object_fill_colors =
                        vec![egui::Color32::TRANSPARENT; fill_mesh.object_count];
                    for group in &color_groups.groups {
                        let Some(c) =
                            self.effective_color_group_rgb(&color_groups.property_key, group)
                        else {
                            continue;
                        };
                        let fill_color =
                            egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], fill_alpha);
                        for (object_index, state) in group.fill_state.iter().copied().enumerate() {
                            if state > 0 {
                                object_fill_colors[object_index] = fill_color;
                            }
                        }
                    }
                    for tri in fill_mesh.vertices_local.chunks_exact(3) {
                        let object_index = tri[0][2].round().max(0.0) as usize;
                        let fill_color = object_fill_colors
                            .get(object_index)
                            .copied()
                            .unwrap_or(egui::Color32::TRANSPARENT);
                        if fill_color == egui::Color32::TRANSPARENT {
                            continue;
                        }
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
                            fill_color,
                            egui::Stroke::NONE,
                        ));
                    }
                } else {
                    let fill_color =
                        egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], fill_alpha);
                    let filtered_mask = self.filtered_mask.as_ref();
                    for tri in fill_mesh.vertices_local.chunks_exact(3) {
                        let object_index = tri[0][2].round().max(0.0) as usize;
                        if filtered_mask
                            .is_some_and(|mask| !mask.get(object_index).copied().unwrap_or(false))
                        {
                            continue;
                        }
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
                            fill_color,
                            egui::Stroke::NONE,
                        ));
                    }
                }
            }

            if gpu_available {
                let lod = &render_lods[lod_idx];
                if active_color_groups.is_none()
                    && self.show_selection_overlay
                    && !self.selected_object_indices.is_empty()
                    && let Some(selection_lods) = self.object_selection_lods.as_ref()
                    && let Some(selection_lod) =
                        selection_lods.get(choose_object_selection_lod_index(
                            selection_lods,
                            dataset_long_side_screen_px,
                        ))
                    && let Some(object_count) = self.objects.as_ref().map(|objects| objects.len())
                {
                    let item = ObjectLineBinsGlDrawItem {
                        data: ObjectLineBinsGlDrawData {
                            cache_id: object_render_cache_id(0x4a90, selection_lod.lod as u64),
                            state_cache_id: object_render_cache_id(0x4a91, 0),
                            generation: self.generation,
                            bins: Arc::clone(&selection_lod.bins),
                            selection_generation: self.selection_generation,
                            selection_state: Arc::clone(&self.selection_fill_state),
                            object_count,
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
                            draw_unselected: true,
                            visible: self.visible,
                            local_to_world_offset: display_offset,
                            local_to_world_scale: display_scale,
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
                                    cache_id: object_render_cache_id(
                                        0x4a00 | u32::from(lod.lod),
                                        group_idx as u64,
                                    ),
                                    generation: self.generation,
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
                                generation: self.generation,
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

        if !use_fast_proxy_points
            && self.show_selection_overlay
            && self.selected_fill_opacity > 0.0
            && let Some(fill_mesh) = self.selected_fill_mesh.as_ref()
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
        };

        let mut items = Vec::new();
        let state_cache_id = object_render_cache_id(0x4a31, 0);
        if fill_mesh.object_count <= Self::SELECTED_RENDER_LOD_LIMIT {
            items.push(ObjectFillGlDrawItem {
                data: ObjectFillGlDrawData {
                    cache_id: object_render_cache_id(0x4a32, 0),
                    state_cache_id,
                    generation: self.generation,
                    vertices_local: Arc::clone(&fill_mesh.vertices_local),
                    object_count: fill_mesh.object_count,
                    selection_generation: self.selection_generation,
                    selection_state: Arc::clone(&self.selection_fill_state),
                },
                params,
                visible_world: fill_mesh.bounds_local,
            });
        } else {
            let (bx0, by0, bx1, by1) = fill_mesh.bin_range_for_local_rect(visible_local);
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let bin_index = by * fill_mesh.bins_w + bx;
                    let Some(vertices) = fill_mesh.bin_vertices.get(bin_index) else {
                        continue;
                    };
                    if vertices.is_empty() {
                        continue;
                    }
                    items.push(ObjectFillGlDrawItem {
                        data: ObjectFillGlDrawData {
                            cache_id: object_render_cache_id_usize(0x4a80, bin_index),
                            state_cache_id,
                            generation: self.generation,
                            vertices_local: Arc::clone(vertices),
                            object_count: fill_mesh.object_count,
                            selection_generation: self.selection_generation,
                            selection_state: Arc::clone(&self.selection_fill_state),
                        },
                        params: params.clone(),
                        visible_world: visible_local,
                    });
                }
            }
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

    pub fn hover_tooltip(
        &self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<Vec<String>> {
        let objects = self.objects.as_ref()?;
        let local = self.world_to_local_point(pointer_world, local_to_world_offset);
        let idx = self.hover_object_index(local, pointer_world, local_to_world_offset, camera)?;
        let obj = objects.get(idx)?;
        let centroid_world = self.local_to_world_point(obj.centroid_world, local_to_world_offset);

        let mut lines = Vec::new();
        lines.push(format!("id: {}", obj.id));
        lines.push(format!("area_px: {:.2}", obj.area_px));
        lines.push(format!("perimeter_px: {:.2}", obj.perimeter_px));
        lines.push(format!(
            "centroid: ({:.2}, {:.2})",
            centroid_world.x, centroid_world.y
        ));

        const MAX_TOOLTIP_PROPERTIES: usize = 11;
        for (key, text) in self
            .loaded_property_display_pairs(idx, obj)
            .into_iter()
            .take(MAX_TOOLTIP_PROPERTIES)
        {
            lines.push(format!("{key}: {text}"));
        }

        Some(lines)
    }

    pub fn select_at(
        &mut self,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
        additive: bool,
        toggle: bool,
    ) {
        let local = self.world_to_local_point(pointer_world, local_to_world_offset);
        let hit = self.hover_object_index(local, pointer_world, local_to_world_offset, camera);
        match (hit, additive, toggle) {
            (Some(idx), _, true) => {
                if !self.selected_object_indices.insert(idx) {
                    self.selected_object_indices.remove(&idx);
                    if self.selected_object_index == Some(idx) {
                        self.selected_object_index =
                            self.selected_object_indices.iter().next().copied();
                    }
                } else {
                    self.selected_object_index = Some(idx);
                }
                self.rebuild_selection_render_lods();
                self.clear_measurements();
                self.invalidate_table_cache();
            }
            (Some(idx), true, false) => {
                self.selected_object_indices.insert(idx);
                self.selected_object_index = Some(idx);
                self.rebuild_selection_render_lods();
                self.clear_measurements();
                self.invalidate_table_cache();
            }
            (Some(idx), false, false) => {
                self.selected_object_index = Some(idx);
                if !self.has_live_analysis_selection() {
                    self.selected_object_indices.clear();
                    self.selected_object_indices.insert(idx);
                }
                self.rebuild_selection_render_lods();
                self.clear_measurements();
                self.invalidate_table_cache();
            }
            (None, false, false) => {
                if self.has_live_analysis_selection() {
                    self.selected_object_index = None;
                    self.rebuild_selection_render_lods();
                    self.clear_measurements();
                    self.invalidate_table_cache();
                } else {
                    self.clear_selection();
                }
            }
            (None, _, _) => {}
        }
    }

    pub fn selection_count(&self) -> usize {
        self.selected_object_indices.len()
    }

    pub fn selection_snapshot_json(
        &self,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        let Some(objects) = self.objects.as_ref() else {
            return serde_json::json!({
                "object_count": 0,
                "selection_count": 0,
                "primary": null,
                "selected": [],
                "truncated": false,
            });
        };
        let mut indices = self
            .selected_object_indices
            .iter()
            .copied()
            .collect::<Vec<_>>();
        indices.sort_unstable();
        let selected = self.object_entries_json(objects, &indices, local_to_world_offset, limit);
        serde_json::json!({
            "object_count": objects.len(),
            "selection_count": indices.len(),
            "primary": self.selected_object_index.and_then(|idx| {
                objects
                    .get(idx)
                    .map(|obj| self.object_entry_json(idx, obj, local_to_world_offset))
            }),
            "selected": selected,
            "truncated": indices.len() > limit,
        })
    }

    pub fn filtered_count(&self) -> usize {
        self.filtered_ordered_indices
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or_else(|| self.object_count())
    }

    pub fn clear_selection(&mut self) {
        self.selected_object_indices.clear();
        self.selected_object_index = None;
        self.rebuild_selection_render_lods();
        self.clear_measurements();
        self.invalidate_table_cache();
    }

    fn rebuild_selection_fill_state(&mut self, object_count: usize) {
        if object_count == 0 {
            self.selection_fill_state = Arc::new(Vec::new());
            return;
        }

        let mut state = vec![0u8; object_count];
        for idx in &self.selected_object_indices {
            if let Some(slot) = state.get_mut(*idx) {
                *slot = 128;
            }
        }
        if let Some(primary_idx) = self.selected_object_index
            && let Some(slot) = state.get_mut(primary_idx)
        {
            *slot = 255;
        }
        self.selection_fill_state = Arc::new(state);
    }

    fn ensure_cpu_selection_fill_mesh(&mut self) {
        if !self.selection_cpu_overlay_dirty {
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            self.selected_fill_mesh = None;
            self.selection_cpu_overlay_dirty = false;
            return;
        };

        let mut selected = Vec::with_capacity(self.selected_object_indices.len());
        for idx in &self.selected_object_indices {
            if let Some(obj) = objects.get(*idx) {
                selected.push(obj.clone());
            }
        }
        self.selected_fill_mesh = if selected.is_empty() {
            None
        } else {
            build_selection_fill_mesh(&selected).ok()
        };
        self.selection_cpu_overlay_dirty = false;
    }

    pub fn select_filtered_objects(&mut self) {
        if let Some(filtered) = self.filtered_ordered_indices.as_ref() {
            self.selected_object_indices = filtered.iter().copied().collect();
            self.selected_object_index = filtered.first().copied();
        } else if let Some(objects) = self.objects.as_ref() {
            self.selected_object_indices = (0..objects.len()).collect();
            self.selected_object_index = if objects.is_empty() { None } else { Some(0) };
        }
        self.rebuild_selection_render_lods();
        self.invalidate_table_cache();
    }

    pub fn select_in_world_rect(
        &mut self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        additive: bool,
    ) -> usize {
        let indices = self.query_indices_in_world_rect(world_rect, local_to_world_offset);
        self.apply_selection_indices(&indices, additive);
        let id_preview = self.selection_id_preview(&indices);
        self.status = if id_preview.is_empty() {
            format!("Selected {} object(s) by rectangle.", indices.len())
        } else {
            format!(
                "Selected {} object(s) by rectangle: {}",
                indices.len(),
                id_preview
            )
        };
        indices.len()
    }

    pub fn query_world_rect_snapshot_json(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> serde_json::Value {
        let indices = self.query_indices_in_world_rect(world_rect, local_to_world_offset);
        self.rect_query_snapshot_json(world_rect, local_to_world_offset, &indices, limit)
    }

    pub fn select_in_world_rect_snapshot_json(
        &mut self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        additive: bool,
        limit: usize,
    ) -> serde_json::Value {
        let indices = self.query_indices_in_world_rect(world_rect, local_to_world_offset);
        self.apply_selection_indices(&indices, additive);
        let id_preview = self.selection_id_preview(&indices);
        self.status = if id_preview.is_empty() {
            format!("Selected {} object(s) by rectangle.", indices.len())
        } else {
            format!(
                "Selected {} object(s) by rectangle: {}",
                indices.len(),
                id_preview
            )
        };
        serde_json::json!({
            "query": self.rect_query_snapshot_json(
                world_rect,
                local_to_world_offset,
                &indices,
                limit,
            ),
            "selection": self.selection_snapshot_json(local_to_world_offset, limit),
        })
    }

    pub fn select_in_world_lasso(
        &mut self,
        world_points: &[egui::Pos2],
        local_to_world_offset: egui::Vec2,
        additive: bool,
    ) -> usize {
        let indices = self.query_indices_in_world_lasso(world_points, local_to_world_offset);
        self.apply_selection_indices(&indices, additive);
        self.status = format!("Selected {} object(s) by lasso.", indices.len());
        indices.len()
    }

    fn selection_id_preview(&self, indices: &[usize]) -> String {
        let Some(objects) = self.objects.as_ref() else {
            return String::new();
        };
        let labels = indices
            .iter()
            .take(8)
            .filter_map(|idx| objects.get(*idx).map(|obj| obj.id.as_str()))
            .collect::<Vec<_>>();
        if labels.is_empty() {
            return String::new();
        }
        let extra = indices.len().saturating_sub(labels.len());
        let mut out = labels.join(", ");
        if extra > 0 {
            out.push_str(&format!(" +{extra}"));
        }
        out
    }

    fn query_indices_in_world_rect(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
    ) -> Vec<usize> {
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let local_rect = self.world_to_local_rect(world_rect, local_to_world_offset);
        let mut out = Vec::new();
        for (idx, obj) in objects.iter().enumerate() {
            if !self.is_index_visible(idx) {
                continue;
            }
            if object_intersects_rect_for_selection(obj, local_rect) {
                out.push(idx);
            }
        }
        out
    }

    fn rect_query_snapshot_json(
        &self,
        world_rect: egui::Rect,
        local_to_world_offset: egui::Vec2,
        indices: &[usize],
        limit: usize,
    ) -> serde_json::Value {
        let local_rect = self.world_to_local_rect(world_rect, local_to_world_offset);
        let objects = self.objects.as_ref();
        let hits = objects
            .map(|objects| self.object_entries_json(objects, indices, local_to_world_offset, limit))
            .unwrap_or_default();
        serde_json::json!({
            "world_rect": rect_json(world_rect),
            "local_rect": rect_json(local_rect),
            "match_count": indices.len(),
            "matches": hits,
            "truncated": indices.len() > limit,
        })
    }

    fn object_entries_json(
        &self,
        objects: &[ObjectFeature],
        indices: &[usize],
        local_to_world_offset: egui::Vec2,
        limit: usize,
    ) -> Vec<serde_json::Value> {
        indices
            .iter()
            .take(limit)
            .filter_map(|idx| {
                objects
                    .get(*idx)
                    .map(|obj| self.object_entry_json(*idx, obj, local_to_world_offset))
            })
            .collect()
    }

    fn object_entry_json(
        &self,
        idx: usize,
        obj: &ObjectFeature,
        local_to_world_offset: egui::Vec2,
    ) -> serde_json::Value {
        let centroid_world = self.local_to_world_point(obj.centroid_world, local_to_world_offset);
        let bbox_min_world = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
        let bbox_max_world = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
        let bbox_world = egui::Rect::from_min_max(
            egui::pos2(
                bbox_min_world.x.min(bbox_max_world.x),
                bbox_min_world.y.min(bbox_max_world.y),
            ),
            egui::pos2(
                bbox_min_world.x.max(bbox_max_world.x),
                bbox_min_world.y.max(bbox_max_world.y),
            ),
        );
        serde_json::json!({
            "index": idx,
            "id": obj.id.as_str(),
            "centroid_world": [centroid_world.x, centroid_world.y],
            "centroid_local": [obj.centroid_world.x, obj.centroid_world.y],
            "bbox_world": rect_json(bbox_world),
            "bbox_local": rect_json(obj.bbox_world),
            "area_px": obj.area_px,
            "perimeter_px": obj.perimeter_px,
        })
    }

    fn query_indices_in_world_lasso(
        &self,
        world_points: &[egui::Pos2],
        local_to_world_offset: egui::Vec2,
    ) -> Vec<usize> {
        if world_points.len() < 3 {
            return Vec::new();
        }
        let Some(objects) = self.objects.as_ref() else {
            return Vec::new();
        };
        let Some(bins) = self.bins.as_ref() else {
            return Vec::new();
        };

        let local_points = world_points
            .iter()
            .copied()
            .map(|point| self.world_to_local_point(point, local_to_world_offset))
            .collect::<Vec<_>>();
        let mut min = local_points[0];
        let mut max = local_points[0];
        for point in local_points.iter().copied().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        let local_bounds = egui::Rect::from_min_max(min, max);

        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(local_bounds);
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u32 in bins.bin_slice(bi) {
                    let idx = idx_u32 as usize;
                    if !seen.insert(idx) || !self.is_index_visible(idx) {
                        continue;
                    }
                    let Some(obj) = objects.get(idx) else {
                        continue;
                    };
                    if !local_bounds.contains(obj.centroid_world) {
                        continue;
                    }
                    if point_in_polygon(obj.centroid_world, &local_points) {
                        out.push(idx);
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }

    pub fn fit_bounds_world(&self, local_to_world_offset: egui::Vec2) -> Option<egui::Rect> {
        let objects = self.objects.as_ref()?;
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for idx in &self.selected_object_indices {
            let Some(obj) = objects.get(*idx) else {
                continue;
            };
            any = true;
            let min = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
            let max = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
            min_x = min_x.min(min.x);
            min_y = min_y.min(min.y);
            max_x = max_x.max(max.x);
            max_y = max_y.max(max.y);
        }

        any.then(|| {
            let rect = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
            let pad = rect.size().max_elem().max(32.0) * 0.08;
            rect.expand(pad)
        })
    }

    pub fn fit_object_bounds_world(
        &self,
        object_index: usize,
        local_to_world_offset: egui::Vec2,
    ) -> Option<egui::Rect> {
        let obj = self.objects.as_ref()?.get(object_index)?;
        let min = self.local_to_world_point(obj.bbox_world.min, local_to_world_offset);
        let max = self.local_to_world_point(obj.bbox_world.max, local_to_world_offset);
        let rect = egui::Rect::from_min_max(min, max);
        let pad = rect.size().max_elem().max(32.0) * 0.08;
        Some(rect.expand(pad))
    }

    pub(super) fn rebuild_selection_render_lods(&mut self) {
        let selected_count = self.selected_object_indices.len();
        if self.display_mode == ObjectDisplayMode::Points {
            self.selected_render_lods = None;
            self.primary_selected_render_lods = None;
            self.selected_fill_mesh = None;
            self.selection_fill_state = Arc::new(Vec::new());
            self.selection_cpu_overlay_dirty = false;
            let Some(objects) = self.objects.as_ref() else {
                self.selected_point_positions_world = None;
                self.selected_point_values = None;
                self.selected_point_lods = None;
                self.primary_selected_point_positions_world = None;
                self.primary_selected_point_values = None;
                self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
                return;
            };
            if selected_count == 0 || selected_count > Self::SELECTED_RENDER_LOD_LIMIT {
                self.selected_point_positions_world = None;
                self.selected_point_values = None;
                self.selected_point_lods = None;
            } else {
                let selected = self
                    .selected_object_indices
                    .iter()
                    .filter_map(|idx| objects.get(*idx).cloned())
                    .collect::<Vec<_>>();
                let (positions, values, lods) =
                    build_object_point_payload(&selected, self.display_transform);
                self.selected_point_positions_world = Some(positions);
                self.selected_point_values = Some(values);
                self.selected_point_lods = Some(lods);
            }
            if let Some(primary) = self.selected_object_index.and_then(|idx| objects.get(idx)) {
                let (positions, values, _) = build_object_point_payload(
                    std::slice::from_ref(primary),
                    self.display_transform,
                );
                self.primary_selected_point_positions_world = Some(positions);
                self.primary_selected_point_values = Some(values);
            } else {
                self.primary_selected_point_positions_world = None;
                self.primary_selected_point_values = None;
            }
            self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
            return;
        }
        let Some(objects) = self.objects.as_ref() else {
            self.selected_render_lods = None;
            self.primary_selected_render_lods = None;
            self.selected_fill_mesh = None;
            self.selected_point_positions_world = None;
            self.selected_point_values = None;
            self.selected_point_lods = None;
            self.selection_fill_state = Arc::new(Vec::new());
            self.selection_cpu_overlay_dirty = false;
            self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
            return;
        };
        self.selected_point_positions_world = None;
        self.selected_point_values = None;
        self.selected_point_lods = None;
        let object_count = objects.len();
        self.selected_render_lods =
            if selected_count == 0 || selected_count > Self::SELECTED_RENDER_LOD_LIMIT {
                None
            } else {
                let selected = self
                    .selected_object_indices
                    .iter()
                    .filter_map(|idx| objects.get(*idx).cloned())
                    .collect::<Vec<_>>();
                build_render_lods(&selected).ok()
            };
        self.primary_selected_render_lods = self
            .selected_object_index
            .and_then(|idx| objects.get(idx))
            .and_then(|object| build_render_lods(std::slice::from_ref(object)).ok());
        self.selected_fill_mesh = None;
        self.rebuild_selection_fill_state(object_count);
        self.selection_cpu_overlay_dirty = true;

        self.selection_generation = self.selection_generation.wrapping_add(1).max(1);
    }

    fn hover_object_index(
        &self,
        pointer_local: egui::Pos2,
        pointer_world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
        camera: &crate::camera::Camera,
    ) -> Option<usize> {
        let objects = self.objects.as_ref()?;
        let bins = self.bins.as_ref()?;
        if self.display_mode == ObjectDisplayMode::Points {
            let radius_world = self.point_pick_radius_world(camera);
            let scale = self.display_scale();
            let rect = egui::Rect::from_min_max(
                egui::pos2(
                    pointer_local.x - radius_world / scale.x.max(1e-6),
                    pointer_local.y - radius_world / scale.y.max(1e-6),
                ),
                egui::pos2(
                    pointer_local.x + radius_world / scale.x.max(1e-6),
                    pointer_local.y + radius_world / scale.y.max(1e-6),
                ),
            );
            let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(rect);
            let mut best_idx: Option<usize> = None;
            let mut best_dist_sq = f32::INFINITY;
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let bi = by * bins.bins_w + bx;
                    for &idx_u32 in bins.bin_slice(bi) {
                        let idx = idx_u32 as usize;
                        let Some(obj) = objects.get(idx) else {
                            continue;
                        };
                        if !self.is_index_visible(idx) {
                            continue;
                        }
                        let centroid_world =
                            self.local_to_world_point(obj.centroid_world, local_to_world_offset);
                        let dist_sq = centroid_world.distance_sq(pointer_world);
                        if dist_sq <= radius_world * radius_world && dist_sq < best_dist_sq {
                            best_dist_sq = dist_sq;
                            best_idx = Some(idx);
                        }
                    }
                }
            }
            return best_idx;
        }

        let rect = egui::Rect::from_center_size(pointer_local, egui::vec2(1.0, 1.0));
        let (bx0, by0, bx1, by1) = bins.bin_range_for_world_rect(rect);
        let mut best_idx: Option<usize> = None;
        let mut best_area = f32::INFINITY;

        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u32 in bins.bin_slice(bi) {
                    let idx = idx_u32 as usize;
                    let Some(obj) = objects.get(idx) else {
                        continue;
                    };
                    if !self.is_index_visible(idx) {
                        continue;
                    }
                    if !obj.bbox_world.contains(pointer_local) {
                        continue;
                    }
                    if !point_in_any_polygon(pointer_local, &obj.polygons_world) {
                        continue;
                    }
                    if obj.area_px < best_area {
                        best_area = obj.area_px;
                        best_idx = Some(idx);
                    }
                }
            }
        }

        best_idx
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
        self.fill_cells && self.fill_opacity > 0.0 && lod.lod >= 2
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
        let color_groups_binding = self.active_color_groups().cloned();
        let active_color_groups = match self.color_mode {
            ObjectColorMode::Single => None,
            ObjectColorMode::ByProperty => color_groups_binding
                .as_ref()
                .filter(|groups| groups.property_key == self.color_property_key),
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
            self.generation ^ generation_seed,
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
            };
            let params = crate::render::points_gl::PointsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                threshold: 0.5,
                style,
                visible: self.visible,
                local_to_world_offset: self.display_offset(local_to_world_offset),
                local_to_world_scale: self.display_scale(),
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

        for pos in positions_world.iter() {
            let world = self.local_to_world_point(*pos, local_to_world_offset);
            if !visible_world.contains(world) {
                continue;
            }
            let screen = camera.world_to_screen(world, viewport);
            ui.painter()
                .circle_filled(screen, style.radius_screen_px, style.fill_positive);
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

        if gpu_available {
            let data = crate::render::points_gl::PointsGlDrawData {
                generation: self.generation,
                positions_world: Arc::clone(base_positions),
                values: Arc::clone(base_values),
            };
            let params = crate::render::points_gl::PointsGlDrawParams {
                center_world: camera.center_world_lvl0,
                zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                threshold: 0.5,
                style: self.base_point_style(),
                visible: self.visible,
                local_to_world_offset: self.display_offset(local_to_world_offset),
                local_to_world_scale: self.display_scale(),
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
            for &p in base_positions.iter() {
                let world = self.local_to_world_point(p, local_to_world_offset);
                if !visible_world.contains(world) {
                    continue;
                }
                let s = camera.world_to_screen(world, viewport);
                ui.painter()
                    .circle_filled(s, radius, self.base_point_style().fill_positive);
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
                };
                let params = crate::render::points_gl::PointsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    threshold: 0.5,
                    style: self.selected_point_style(),
                    visible: self.visible,
                    local_to_world_offset: self.display_offset(local_to_world_offset),
                    local_to_world_scale: self.display_scale(),
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
                };
                let params = crate::render::points_gl::PointsGlDrawParams {
                    center_world: camera.center_world_lvl0,
                    zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                    threshold: 0.5,
                    style: self.primary_selected_point_style(),
                    visible: self.visible,
                    local_to_world_offset: self.display_offset(local_to_world_offset),
                    local_to_world_scale: self.display_scale(),
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

fn visible_selected_cache_generation(selection_generation: u64, indices: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    selection_generation.hash(&mut hasher);
    indices.hash(&mut hasher);
    hasher.finish()
}

fn object_render_cache_id(namespace: u32, index: u64) -> u64 {
    ((namespace as u64) << 32) | (index & 0xffff_ffff)
}

fn object_render_cache_id_usize(namespace: u32, index: usize) -> u64 {
    object_render_cache_id(namespace, index.min(u32::MAX as usize) as u64)
}

impl ObjectFillMesh {
    fn bin_range_for_local_rect(&self, rect: egui::Rect) -> (usize, usize, usize, usize) {
        rect_bins(rect, self.origin, self.bin_size, self.bins_w, self.bins_h)
    }
}

fn build_selection_fill_mesh(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<SelectionFillMesh> {
    let mut tess = FillTessellator::new();
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut bounds: Option<egui::Rect> = None;

    for obj in objects {
        bounds = Some(match bounds {
            Some(acc) => acc.union(obj.bbox_world),
            None => obj.bbox_world,
        });

        for poly in &obj.polygons_world {
            let Some(clean) = cleaned_fill_polygon(poly) else {
                continue;
            };
            let mut builder = Path::builder();
            let first = clean[0];
            builder.begin(point(first.x, first.y));
            for p in &clean[1..] {
                builder.line_to(point(p.x, p.y));
            }
            builder.close();
            let path = builder.build();
            tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| {
                    let pos = vertex.position();
                    [pos.x, pos.y]
                }),
            )?;
        }
    }

    let bounds_local = bounds.context("no selection fill bounds")?;
    if geometry.indices.is_empty() {
        anyhow::bail!("no valid triangles for selection fill");
    }

    let mut triangles = Vec::with_capacity(geometry.indices.len());
    for idx in geometry.indices {
        let vertex = geometry
            .vertices
            .get(idx as usize)
            .copied()
            .context("selection fill index out of range")?;
        triangles.push(vertex);
    }

    Ok(SelectionFillMesh {
        vertices_local: Arc::new(triangles),
        bounds_local,
    })
}

pub(super) fn build_object_fill_mesh(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<ObjectFillMesh> {
    let mut tess = FillTessellator::new();
    let mut triangles = Vec::new();
    let mut bounds: Option<egui::Rect> = None;

    for (object_index, obj) in objects.iter().enumerate() {
        bounds = Some(match bounds {
            Some(acc) => acc.union(obj.bbox_world),
            None => obj.bbox_world,
        });

        let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        for poly in &obj.polygons_world {
            let Some(clean) = cleaned_fill_polygon(poly) else {
                continue;
            };
            let mut builder = Path::builder();
            let first = clean[0];
            builder.begin(point(first.x, first.y));
            for p in &clean[1..] {
                builder.line_to(point(p.x, p.y));
            }
            builder.close();
            let path = builder.build();
            tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| {
                    let pos = vertex.position();
                    [pos.x, pos.y]
                }),
            )?;
        }

        for idx in geometry.indices {
            let vertex = geometry
                .vertices
                .get(idx as usize)
                .copied()
                .context("object fill index out of range")?;
            triangles.push([vertex[0], vertex[1], object_index as f32]);
        }
    }

    let bounds_local = bounds.context("no object fill bounds")?;
    if triangles.is_empty() {
        anyhow::bail!("no valid triangles for object fill rendering");
    }

    const OBJECT_FILL_BIN_SIZE: f32 = 2048.0;
    let w = bounds_local.width().max(1.0);
    let h = bounds_local.height().max(1.0);
    let bins_w = ((w / OBJECT_FILL_BIN_SIZE).ceil() as usize).max(1);
    let bins_h = ((h / OBJECT_FILL_BIN_SIZE).ceil() as usize).max(1);
    let origin = bounds_local.min;
    let bins_len = bins_w.saturating_mul(bins_h);
    let mut tmp_bins: Vec<Vec<[f32; 3]>> = vec![Vec::new(); bins_len];
    for tri in triangles.chunks_exact(3) {
        let min_x = tri.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
        let min_y = tri.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min);
        let max_x = tri.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
        let max_y = tri.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            continue;
        }
        let tri_rect = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
        let (bx0, by0, bx1, by1) =
            rect_bins(tri_rect, origin, OBJECT_FILL_BIN_SIZE, bins_w, bins_h);
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bin_index = by * bins_w + bx;
                if let Some(bin) = tmp_bins.get_mut(bin_index) {
                    bin.extend_from_slice(tri);
                }
            }
        }
    }

    Ok(ObjectFillMesh {
        vertices_local: Arc::new(triangles),
        bounds_local,
        object_count: objects.len(),
        origin,
        bin_size: OBJECT_FILL_BIN_SIZE,
        bins_w,
        bins_h,
        bin_vertices: tmp_bins.into_iter().map(Arc::new).collect(),
    })
}

fn cleaned_fill_polygon(poly: &[egui::Pos2]) -> Option<Vec<egui::Pos2>> {
    if poly.len() < 3 {
        return None;
    }
    let mut out = Vec::with_capacity(poly.len());
    for &p in poly {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        if out.last().copied() == Some(p) {
            continue;
        }
        out.push(p);
    }
    if out.len() >= 2 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        return None;
    }
    let area = polygon_signed_area_local(&out).abs();
    (area > 1e-3).then_some(out)
}

fn polygon_signed_area_local(points: &[egui::Pos2]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

fn object_vertex_count(object: &GeoJsonObjectFeature) -> usize {
    object.polygons_world.iter().map(|poly| poly.len()).sum()
}

fn simplified_polyline_screen_points(
    poly: &[egui::Pos2],
    max_points: usize,
    camera: &crate::camera::Camera,
    local_to_world_offset: egui::Vec2,
    display_transform: SpatialDataTransform2,
    viewport: egui::Rect,
) -> Vec<egui::Pos2> {
    if poly.len() < 2 {
        return Vec::new();
    }

    let step = if max_points == usize::MAX || poly.len() <= max_points {
        1
    } else {
        poly.len().div_ceil(max_points)
    };

    let mut pts = Vec::with_capacity(poly.len().div_ceil(step).saturating_add(1));
    for point in poly.iter().step_by(step).copied() {
        let world = egui::pos2(
            point.x * display_transform.scale[0].max(1e-6)
                + display_transform.translation[0]
                + local_to_world_offset.x,
            point.y * display_transform.scale[1].max(1e-6)
                + display_transform.translation[1]
                + local_to_world_offset.y,
        );
        pts.push(camera.world_to_screen(world, viewport));
    }
    if let Some(last) = poly.last().copied() {
        let last_world = egui::pos2(
            last.x * display_transform.scale[0].max(1e-6)
                + display_transform.translation[0]
                + local_to_world_offset.x,
            last.y * display_transform.scale[1].max(1e-6)
                + display_transform.translation[1]
                + local_to_world_offset.y,
        );
        let last_screen = camera.world_to_screen(last_world, viewport);
        if pts.last().copied() != Some(last_screen) {
            pts.push(last_screen);
        }
    }
    pts
}

pub(super) fn summarize_geometry(
    polygons: &[Vec<egui::Pos2>],
) -> Option<(egui::Rect, f32, f32, egui::Pos2)> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut area_sum = 0.0f32;
    let mut perimeter_sum = 0.0f32;
    let mut centroid_num = egui::Vec2::ZERO;
    let mut any = false;

    for poly in polygons {
        if poly.len() < 4 {
            continue;
        }
        for p in poly {
            if p.x.is_finite() && p.y.is_finite() {
                any = true;
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
        for win in poly.windows(2) {
            let a = win[0];
            let b = win[1];
            perimeter_sum += (b - a).length();
        }
        if let Some((area, centroid)) = polygon_area_and_centroid(poly) {
            area_sum += area;
            centroid_num += centroid.to_vec2() * area;
        }
    }

    if !any {
        return None;
    }
    let bbox = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
    let centroid = if area_sum > 1e-6 {
        (centroid_num / area_sum).to_pos2()
    } else {
        bbox.center()
    };
    Some((bbox, area_sum.max(0.0), perimeter_sum.max(0.0), centroid))
}

fn polygon_area_and_centroid(poly: &[egui::Pos2]) -> Option<(f32, egui::Pos2)> {
    if poly.len() < 4 {
        return None;
    }
    let mut cross_sum = 0.0f32;
    let mut cx_sum = 0.0f32;
    let mut cy_sum = 0.0f32;

    for win in poly.windows(2) {
        let a = win[0];
        let b = win[1];
        let cross = a.x * b.y - b.x * a.y;
        cross_sum += cross;
        cx_sum += (a.x + b.x) * cross;
        cy_sum += (a.y + b.y) * cross;
    }
    let area_signed = cross_sum * 0.5;
    let area = area_signed.abs();
    if area <= 1e-6 {
        return None;
    }
    let denom = 3.0 * cross_sum;
    if denom.abs() <= 1e-6 {
        return None;
    }
    Some((area, egui::pos2(cx_sum / denom, cy_sum / denom)))
}

fn point_in_any_polygon(p: egui::Pos2, polygons: &[Vec<egui::Pos2>]) -> bool {
    polygons.iter().any(|poly| point_in_polygon(p, poly))
}

fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let dy = pj.y - pi.y;
        let intersects = ((pi.y > p.y) != (pj.y > p.y))
            && dy.abs() > 1e-12
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / dy + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn object_intersects_rect_for_selection(object: &ObjectFeature, rect: egui::Rect) -> bool {
    if !object.polygons_world.is_empty() {
        if !rects_intersect_inclusive(object.bbox_world, rect) {
            return false;
        }
        let mut tested_polygon = false;
        for poly in &object.polygons_world {
            if poly.len() < 3 {
                continue;
            }
            tested_polygon = true;
            if polygon_intersects_rect(poly, rect) {
                return true;
            }
        }
        if tested_polygon {
            return false;
        }
    }

    let point = object.point_position_world.unwrap_or(object.centroid_world);
    rect_contains_point_inclusive(rect, point)
}

fn polygon_intersects_rect(poly: &[egui::Pos2], rect: egui::Rect) -> bool {
    if poly
        .iter()
        .any(|point| rect_contains_point_inclusive(rect, *point))
    {
        return true;
    }

    if rect_corners(rect)
        .iter()
        .any(|corner| point_in_polygon_or_on_edge(*corner, poly))
    {
        return true;
    }

    let edges = rect_edges(rect);
    for (a, b) in polygon_edges(poly) {
        for (c, d) in edges {
            if segments_intersect_inclusive(a, b, c, d) {
                return true;
            }
        }
    }

    false
}

fn polygon_edges(poly: &[egui::Pos2]) -> Vec<(egui::Pos2, egui::Pos2)> {
    if poly.len() < 2 {
        return Vec::new();
    }
    let mut edges = poly
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    if poly.len() >= 3 && poly.first() != poly.last() {
        edges.push((*poly.last().unwrap(), poly[0]));
    }
    edges
}

fn point_in_polygon_or_on_edge(point: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    polygon_edges(poly)
        .iter()
        .any(|(a, b)| point_on_segment_inclusive(point, *a, *b))
        || point_in_polygon(point, poly)
}

fn rect_corners(rect: egui::Rect) -> [egui::Pos2; 4] {
    [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ]
}

fn rect_edges(rect: egui::Rect) -> [(egui::Pos2, egui::Pos2); 4] {
    let [lt, rt, rb, lb] = rect_corners(rect);
    [(lt, rt), (rt, rb), (rb, lb), (lb, lt)]
}

fn rects_intersect_inclusive(a: egui::Rect, b: egui::Rect) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

fn segments_intersect_inclusive(
    a: egui::Pos2,
    b: egui::Pos2,
    c: egui::Pos2,
    d: egui::Pos2,
) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);

    if o1.abs() <= 1e-5 && point_on_segment_inclusive(c, a, b) {
        return true;
    }
    if o2.abs() <= 1e-5 && point_on_segment_inclusive(d, a, b) {
        return true;
    }
    if o3.abs() <= 1e-5 && point_on_segment_inclusive(a, c, d) {
        return true;
    }
    if o4.abs() <= 1e-5 && point_on_segment_inclusive(b, c, d) {
        return true;
    }

    ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0))
        && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))
}

fn point_on_segment_inclusive(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> bool {
    orient(a, b, point).abs() <= 1e-5
        && point.x >= a.x.min(b.x) - 1e-5
        && point.x <= a.x.max(b.x) + 1e-5
        && point.y >= a.y.min(b.y) - 1e-5
        && point.y <= a.y.max(b.y) + 1e-5
}

fn orient(a: egui::Pos2, b: egui::Pos2, c: egui::Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn rect_contains_point_inclusive(rect: egui::Rect, point: egui::Pos2) -> bool {
    point.x >= rect.min.x && point.x <= rect.max.x && point.y >= rect.min.y && point.y <= rect.max.y
}

#[cfg(test)]
mod rectangle_selection_tests {
    use super::*;

    fn object_with_polygons(polygons_world: Vec<Vec<egui::Pos2>>) -> ObjectFeature {
        let bbox_world = polygons_world
            .iter()
            .flat_map(|poly| poly.iter().copied())
            .fold(None, |acc: Option<egui::Rect>, point| {
                let rect = egui::Rect::from_min_max(point, point);
                Some(acc.map_or(rect, |acc| acc.union(rect)))
            })
            .unwrap_or_else(|| {
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(0.0, 0.0))
            });
        ObjectFeature {
            id: "test".to_string(),
            polygons_world,
            point_position_world: None,
            bbox_world,
            area_px: 0.0,
            perimeter_px: 0.0,
            centroid_world: egui::pos2(1000.0, 1000.0),
            inline_properties: serde_json::Map::new(),
            source_row_index: None,
        }
    }

    fn point_object(point: egui::Pos2, centroid: egui::Pos2) -> ObjectFeature {
        ObjectFeature {
            id: "point".to_string(),
            polygons_world: Vec::new(),
            point_position_world: Some(point),
            bbox_world: egui::Rect::from_min_max(point, point),
            area_px: 0.0,
            perimeter_px: 0.0,
            centroid_world: centroid,
            inline_properties: serde_json::Map::new(),
            source_row_index: None,
        }
    }

    #[test]
    fn rect_contains_point_inclusive_accepts_edges() {
        let rect = egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(30.0, 40.0));

        assert!(rect_contains_point_inclusive(rect, egui::pos2(10.0, 20.0)));
        assert!(rect_contains_point_inclusive(rect, egui::pos2(30.0, 40.0)));
        assert!(rect_contains_point_inclusive(rect, egui::pos2(20.0, 30.0)));
        assert!(!rect_contains_point_inclusive(rect, egui::pos2(30.1, 30.0)));
        assert!(!rect_contains_point_inclusive(rect, egui::pos2(20.0, 40.1)));
    }

    #[test]
    fn object_render_cache_ids_keep_namespace_and_index_separate() {
        let old_bin_98 = 0x5345474f424a80u64 | 98;
        let old_bin_226 = 0x5345474f424a80u64 | 226;
        assert_eq!(old_bin_98, old_bin_226);

        let new_bin_98 = object_render_cache_id(0x4a80, 98);
        let new_bin_226 = object_render_cache_id(0x4a80, 226);
        assert_ne!(new_bin_98, new_bin_226);
    }

    #[test]
    fn polygon_selection_uses_geometry_not_centroid() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(2.0, 2.0),
            egui::pos2(8.0, 2.0),
            egui::pos2(8.0, 8.0),
            egui::pos2(2.0, 8.0),
            egui::pos2(2.0, 2.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_rejects_bbox_outside_even_when_centroid_inside() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let mut object = object_with_polygons(vec![vec![
            egui::pos2(20.0, 20.0),
            egui::pos2(30.0, 20.0),
            egui::pos2(30.0, 30.0),
            egui::pos2(20.0, 30.0),
            egui::pos2(20.0, 20.0),
        ]]);
        object.centroid_world = egui::pos2(5.0, 5.0);

        assert!(!object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_detects_rect_inside_polygon() {
        let rect = egui::Rect::from_min_max(egui::pos2(4.0, 4.0), egui::pos2(6.0, 6.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(0.0, 0.0),
            egui::pos2(10.0, 0.0),
            egui::pos2(10.0, 10.0),
            egui::pos2(0.0, 10.0),
            egui::pos2(0.0, 0.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_detects_edge_crossing() {
        let rect = egui::Rect::from_min_max(egui::pos2(4.0, 4.0), egui::pos2(6.0, 6.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(2.0, 5.0),
            egui::pos2(8.0, 5.0),
            egui::pos2(8.0, 8.0),
            egui::pos2(2.0, 8.0),
            egui::pos2(2.0, 5.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn point_only_selection_uses_point_position_before_centroid() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let object = point_object(egui::pos2(20.0, 20.0), egui::pos2(5.0, 5.0));

        assert!(!object_intersects_rect_for_selection(&object, rect));
    }
}

pub(super) fn build_render_lods(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<Vec<ObjectRenderLod>> {
    let polylines_world = flatten_object_polylines(objects);
    build_render_lods_from_polylines(&polylines_world)
}

pub(super) fn build_object_selection_render_lods(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<Vec<ObjectSelectionRenderLod>> {
    let polylines_world = flatten_indexed_object_polylines(objects);
    build_object_selection_render_lods_from_polylines(&polylines_world)
}

fn build_render_lods_from_polylines(
    polylines_world: &[Vec<egui::Pos2>],
) -> anyhow::Result<Vec<ObjectRenderLod>> {
    if polylines_world.is_empty() {
        anyhow::bail!("no valid object outlines available for rendering");
    }

    let lod_specs: &[(u8, f32, f32)] = &[
        (0, 1.0, 2048.0),
        (1, 4.0, 8192.0),
        (2, 16.0, 32768.0),
        (3, 64.0, 1_000_000.0),
    ];

    let mut out = Vec::new();
    for (lod, step, bin_size) in lod_specs {
        let step = step.max(1.0);
        let bin_size = bin_size.max(256.0);
        let lines = if *lod == 0 {
            polylines_world.to_vec()
        } else if *lod >= 3 {
            let sampled = sample_polylines(polylines_world, 4_000);
            quantize_polylines(&sampled, step)
        } else {
            quantize_polylines(polylines_world, step)
        };
        let Some(bins) = LineSegmentsBins::build_from_polylines(&lines, bin_size) else {
            continue;
        };
        out.push(ObjectRenderLod {
            lod: *lod,
            bins: Arc::new(bins),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no valid renderable object outlines after parsing");
    }
    Ok(out)
}

fn build_object_selection_render_lods_from_polylines(
    polylines_world: &[(usize, Vec<egui::Pos2>)],
) -> anyhow::Result<Vec<ObjectSelectionRenderLod>> {
    if polylines_world.is_empty() {
        anyhow::bail!("no valid object outlines available for selection rendering");
    }

    let lod_specs: &[(u8, f32, f32)] = &[
        (0, 1.0, 2048.0),
        (1, 4.0, 8192.0),
        (2, 16.0, 32768.0),
        (3, 64.0, 1_000_000.0),
    ];

    let mut out = Vec::new();
    for (lod, step, bin_size) in lod_specs {
        let step = step.max(1.0);
        let bin_size = bin_size.max(1024.0);
        let lines = if *lod == 0 {
            polylines_world.to_vec()
        } else {
            quantize_indexed_polylines(polylines_world, step)
        };
        let Some(bins) = ObjectLineSegmentsBins::build_from_indexed_polylines(&lines, bin_size)
        else {
            continue;
        };
        out.push(ObjectSelectionRenderLod {
            lod: *lod,
            bins: Arc::new(bins),
        });
    }

    if out.is_empty() {
        anyhow::bail!("no valid renderable object selection outlines after parsing");
    }
    Ok(out)
}

pub(super) fn discover_categorical_color_keys(objects: &[GeoJsonObjectFeature]) -> Vec<String> {
    let mut distinct: HashMap<String, HashSet<String>> = HashMap::new();
    let mut overflow = HashSet::new();

    for obj in objects {
        for (key, value) in &obj.inline_properties {
            let Some(text) = property_scalar_value(value) else {
                continue;
            };
            if overflow.contains(key) {
                continue;
            }
            let set = distinct.entry(key.clone()).or_default();
            set.insert(text);
            if set.len() > 24 {
                distinct.remove(key);
                overflow.insert(key.clone());
            }
        }
    }

    let mut keys = distinct.into_keys().collect::<Vec<_>>();
    keys.sort();
    keys
}

pub(super) fn discover_property_keys(objects: &[GeoJsonObjectFeature]) -> Vec<String> {
    let mut keys = HashSet::new();
    for obj in objects {
        keys.extend(obj.inline_properties.keys().cloned());
    }
    let mut out = keys.into_iter().collect::<Vec<_>>();
    out.sort();
    out
}

pub(super) fn discover_scalar_property_keys(objects: &[GeoJsonObjectFeature]) -> Vec<String> {
    let mut keys = HashSet::new();
    for obj in objects {
        for (key, value) in &obj.inline_properties {
            if property_scalar_value(value).is_some() {
                keys.insert(key.clone());
            }
        }
    }
    let mut out = keys.into_iter().collect::<Vec<_>>();
    out.sort();
    out
}

pub(super) fn build_color_groups_for_property_labels<'a, I>(
    objects: I,
    property_key: &str,
) -> anyhow::Result<ObjectColorGroups>
where
    I: IntoIterator<Item = (usize, &'a GeoJsonObjectFeature, String)>,
{
    use std::collections::BTreeMap;
    use std::hash::{Hash, Hasher};

    let mut grouped: BTreeMap<String, Vec<Vec<egui::Pos2>>> = BTreeMap::new();
    let mut grouped_points: BTreeMap<String, Vec<egui::Pos2>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut grouped_indices: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut object_count = 0usize;

    for (object_index, obj, value) in objects {
        object_count = object_count.max(object_index.saturating_add(1));
        counts
            .entry(value.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        grouped
            .entry(value.clone())
            .or_default()
            .extend(obj.polygons_world.iter().cloned());
        grouped_points
            .entry(value.clone())
            .or_default()
            .push(object_proxy_position_world(obj));
        grouped_indices.entry(value).or_default().push(object_index);
    }

    if counts.is_empty() {
        anyhow::bail!("no scalar values found for property '{property_key}'");
    }

    let mut groups = Vec::new();
    for (value_label, _) in counts {
        let polylines = grouped.remove(&value_label).unwrap_or_default();
        let lods = if polylines.is_empty() {
            Vec::new()
        } else {
            build_render_lods_from_polylines(&polylines)?
        };
        let point_positions = grouped_points.remove(&value_label).unwrap_or_default();
        let mut object_indices = grouped_indices.remove(&value_label).unwrap_or_default();
        object_indices.sort_unstable();
        object_indices.dedup();

        let mut fill_state = vec![0u8; object_count];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        property_key.hash(&mut hasher);
        value_label.hash(&mut hasher);
        for object_index in object_indices {
            if let Some(slot) = fill_state.get_mut(object_index) {
                *slot = 255;
                object_index.hash(&mut hasher);
            }
        }

        groups.push(ObjectColorGroup {
            color_rgb: hashed_color_rgb(property_key, &value_label),
            value_label,
            lods,
            point_values: Arc::new(vec![1.0; point_positions.len()]),
            point_positions_world: Arc::new(point_positions),
            fill_state: Arc::new(fill_state),
            fill_generation: hasher.finish(),
        });
    }

    Ok(ObjectColorGroups {
        property_key: property_key.to_string(),
        groups,
    })
}

pub(super) fn property_scalar_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(v) => Some(v.to_string()),
        serde_json::Value::Number(v) => Some(v.to_string()),
        serde_json::Value::String(v) => {
            let trimmed = v.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        _ => None,
    }
}

pub(super) fn hashed_color_rgb(property_key: &str, value_label: &str) -> [u8; 3] {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    property_key.hash(&mut hasher);
    value_label.hash(&mut hasher);
    let hash = hasher.finish();
    let hue = (hash % 360) as f32;
    hsv_to_rgb(hue, 0.6, 0.95)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let c = v * s;
    let hh = (h / 60.0) % 6.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hh {
        h if (0.0..1.0).contains(&h) => (c, x, 0.0),
        h if (1.0..2.0).contains(&h) => (x, c, 0.0),
        h if (2.0..3.0).contains(&h) => (0.0, c, x),
        h if (3.0..4.0).contains(&h) => (0.0, x, c),
        h if (4.0..5.0).contains(&h) => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    ]
}

fn flatten_object_polylines(objects: &[GeoJsonObjectFeature]) -> Vec<Vec<egui::Pos2>> {
    let mut out = Vec::new();
    for obj in objects {
        for poly in &obj.polygons_world {
            if poly.len() >= 2 {
                out.push(poly.clone());
            }
        }
    }
    out
}

fn flatten_indexed_object_polylines(
    objects: &[GeoJsonObjectFeature],
) -> Vec<(usize, Vec<egui::Pos2>)> {
    let mut out = Vec::new();
    for (object_index, obj) in objects.iter().enumerate() {
        for poly in &obj.polygons_world {
            if poly.len() >= 2 {
                out.push((object_index, poly.clone()));
            }
        }
    }
    out
}

fn sample_polylines(polys: &[Vec<egui::Pos2>], max_polylines: usize) -> Vec<Vec<egui::Pos2>> {
    if polys.len() <= max_polylines {
        return polys.to_vec();
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for pts in polys {
        for p in pts {
            if p.x.is_finite() && p.y.is_finite() {
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
    }
    let w = (max_x - min_x).max(1.0);
    let h = (max_y - min_y).max(1.0);

    let grid_w = 32usize;
    let grid_h = 32usize;
    let cell_w = (w / grid_w as f32).max(1.0);
    let cell_h = (h / grid_h as f32).max(1.0);
    let cells = grid_w * grid_h;
    let per_cell_cap = max_polylines.div_ceil(cells).max(1);

    let mut chosen = vec![false; polys.len()];
    let mut buckets = Vec::with_capacity(max_polylines.min(polys.len()));
    let mut bucket_counts = vec![0usize; cells];

    for (i, pts) in polys.iter().enumerate() {
        if buckets.len() >= max_polylines {
            break;
        }
        if pts.len() < 2 {
            continue;
        }
        let Some(bounds) = polyline_bounds(pts) else {
            continue;
        };
        let cx = 0.5 * (bounds.min.x + bounds.max.x);
        let cy = 0.5 * (bounds.min.y + bounds.max.y);
        let gx = ((cx - min_x) / cell_w)
            .floor()
            .clamp(0.0, (grid_w - 1) as f32) as usize;
        let gy = ((cy - min_y) / cell_h)
            .floor()
            .clamp(0.0, (grid_h - 1) as f32) as usize;
        let bi = gy * grid_w + gx;
        if bucket_counts[bi] >= per_cell_cap {
            continue;
        }
        bucket_counts[bi] += 1;
        chosen[i] = true;
        buckets.push(i);
    }

    if buckets.len() < max_polylines {
        let remaining = max_polylines - buckets.len();
        let step = (polys.len() / remaining.max(1)).max(1);
        for i in (0..polys.len()).step_by(step) {
            if buckets.len() >= max_polylines {
                break;
            }
            if chosen[i] {
                continue;
            }
            chosen[i] = true;
            buckets.push(i);
        }
    }

    buckets
        .into_iter()
        .take(max_polylines)
        .map(|i| polys[i].clone())
        .collect()
}

fn polyline_bounds(poly: &[egui::Pos2]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;
    for p in poly {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        any = true;
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    any.then(|| egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y)))
}

fn quantize_polylines(polys: &[Vec<egui::Pos2>], step_world: f32) -> Vec<Vec<egui::Pos2>> {
    let s = step_world.max(1e-6);
    let inv = 1.0 / s;
    let mut out = Vec::with_capacity(polys.len());

    for pts in polys {
        if pts.len() < 2 {
            continue;
        }
        let is_closed = pts.first() == pts.last();
        let mut q = Vec::with_capacity(pts.len());
        for p in pts {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let qp = egui::pos2((p.x * inv).round() * s, (p.y * inv).round() * s);
            if q.last().copied() == Some(qp) {
                continue;
            }
            q.push(qp);
        }
        if q.len() < 2 {
            continue;
        }
        if is_closed && q.first() != q.last() {
            if let Some(first) = q.first().copied() {
                q.push(first);
            }
        }
        if q.len() >= 2 {
            out.push(q);
        }
    }

    out
}

fn quantize_indexed_polylines(
    polys: &[(usize, Vec<egui::Pos2>)],
    step_world: f32,
) -> Vec<(usize, Vec<egui::Pos2>)> {
    let s = step_world.max(1e-6);
    let inv = 1.0 / s;
    let mut out = Vec::with_capacity(polys.len());

    for (object_index, pts) in polys {
        if pts.len() < 2 {
            continue;
        }
        let is_closed = pts.first() == pts.last();
        let mut q = Vec::with_capacity(pts.len());
        for p in pts {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let qp = egui::pos2((p.x * inv).round() * s, (p.y * inv).round() * s);
            if q.last().copied() == Some(qp) {
                continue;
            }
            q.push(qp);
        }
        if q.len() < 2 {
            continue;
        }
        if is_closed && q.first() != q.last() {
            if let Some(first) = q.first().copied() {
                q.push(first);
            }
        }
        if q.len() >= 2 {
            out.push((*object_index, q));
        }
    }

    out
}

fn choose_lod_index(lods: &[ObjectRenderLod], dataset_long_side_screen_px: f32) -> usize {
    if lods.len() <= 1 {
        return 0;
    }
    let s = dataset_long_side_screen_px.max(1e-3);
    let desired_lod = if s < 160.0 {
        3u8
    } else if s < 420.0 {
        2u8
    } else if s < 1000.0 {
        1u8
    } else {
        0u8
    };

    let mut best_i = 0usize;
    let mut best_err = i32::MAX;
    for (i, lod) in lods.iter().enumerate() {
        let err = (lod.lod as i32 - desired_lod as i32).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        }
    }
    best_i
}

fn rect_json(rect: egui::Rect) -> serde_json::Value {
    serde_json::json!({
        "min_x": rect.min.x,
        "min_y": rect.min.y,
        "max_x": rect.max.x,
        "max_y": rect.max.y,
        "width": rect.width(),
        "height": rect.height(),
    })
}

fn choose_object_selection_lod_index(
    lods: &[ObjectSelectionRenderLod],
    dataset_long_side_screen_px: f32,
) -> usize {
    if lods.len() <= 1 {
        return 0;
    }
    let s = dataset_long_side_screen_px.max(1e-3);
    let desired_lod = if s < 160.0 {
        3u8
    } else if s < 420.0 {
        2u8
    } else if s < 1000.0 {
        1u8
    } else {
        0u8
    };

    let mut best_i = 0usize;
    let mut best_err = i32::MAX;
    for (i, lod) in lods.iter().enumerate() {
        let err = (lod.lod as i32 - desired_lod as i32).abs();
        if err < best_err {
            best_err = err;
            best_i = i;
        }
    }
    best_i
}
