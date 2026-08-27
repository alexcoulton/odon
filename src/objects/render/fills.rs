//! Hybrid vector and multiresolution object-fill orchestration.

use super::*;

impl ObjectsLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::objects) fn draw_object_fills(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        local_to_world_offset: egui::Vec2,
        display_offset: egui::Vec2,
        display_scale: egui::Vec2,
        gpu_available: bool,
        active_color_groups: Option<&ObjectColorGroups>,
        continuous_colors: Option<&ObjectContinuousColorPayload>,
        render_generation: u64,
    ) -> bool {
        if !self.fill_cells || self.fill_opacity <= 0.0 {
            return false;
        }
        let Some(fill_mesh) = self.object_fill_mesh.as_ref() else {
            return false;
        };
        if !fill_mesh.bounds_local.intersects(visible_local) {
            return false;
        }

        let tile_frame = gpu_available
            .then(|| {
                self.plan_object_fill_tile_frame(
                    fill_mesh,
                    visible_local,
                    camera,
                    display_offset,
                    display_scale,
                    active_color_groups,
                    continuous_colors,
                    render_generation,
                )
            })
            .flatten();
        let tile_coverage = tile_frame
            .as_ref()
            .is_some_and(|frame| self.gl_object_fill.id_tiles_have_coverage(&frame.keys()));
        let any_tile_coverage = tile_frame.as_ref().is_some_and(|frame| {
            self.gl_object_fill
                .id_tiles_have_any_coverage(&frame.keys())
        });
        let direct_fallback_is_bounded = tile_frame.as_ref().is_none_or(|frame| {
            frame
                .draw_items
                .iter()
                .flat_map(|item| item.geometry.iter())
                .map(|geometry| geometry.vertices_local.len())
                .sum::<usize>()
                <= Self::MAX_DIRECT_FILL_VERTICES
        });
        let tile_compose = tile_coverage || (!direct_fallback_is_bounded && any_tile_coverage);
        let selection_overlay_composited = tile_compose
            && tile_frame
                .as_ref()
                .is_some_and(|frame| frame.selection_overlay);

        if gpu_available {
            if !tile_coverage && direct_fallback_is_bounded {
                self.draw_direct_object_fill_gpu(
                    ui,
                    camera,
                    viewport,
                    visible_local,
                    display_offset,
                    display_scale,
                    fill_mesh,
                    active_color_groups,
                    continuous_colors,
                    render_generation,
                );
            }
        } else {
            self.draw_object_fill_cpu(
                ui,
                camera,
                viewport,
                visible_local,
                local_to_world_offset,
                fill_mesh,
                active_color_groups,
                continuous_colors,
            );
        }

        if let Some(frame) = tile_frame {
            let renderer = self.gl_object_fill.clone();
            let repaint = ui.ctx().clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                let result = renderer.paint_id_tiles(
                    info,
                    painter,
                    &frame.request_items,
                    &frame.draw_items,
                    &frame.styles,
                    &frame.params,
                    tile_compose,
                );
                if result.pending > 0 {
                    repaint.request_repaint();
                }
            });
            ui.painter().add(egui::PaintCallback {
                rect: viewport,
                callback: Arc::new(cb),
            });
        }
        selection_overlay_composited
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_direct_object_fill_gpu(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        display_offset: egui::Vec2,
        display_scale: egui::Vec2,
        fill_mesh: &ObjectFillMesh,
        active_color_groups: Option<&ObjectColorGroups>,
        continuous_colors: Option<&ObjectContinuousColorPayload>,
        render_generation: u64,
    ) {
        const SPATIAL_FILL_VERTEX_THRESHOLD: usize = 500_000;
        let fill_alpha = (self.fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let spatial_slices = (fill_mesh.vertices_local.len() >= SPATIAL_FILL_VERTEX_THRESHOLD)
            .then(|| fill_mesh.spatial_slices_for_local_rect(visible_local));
        let fill_geometry = spatial_slices
            .as_ref()
            .filter(|slices| !slices.is_empty())
            .map(|slices| {
                slices
                    .iter()
                    .map(|slice| {
                        (
                            object_render_cache_id_usize(0x4ab0, slice.bin_index),
                            Arc::clone(&slice.vertices_local),
                            slice.bounds_local,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| {
                vec![(
                    object_render_cache_id(0x4a22, 0),
                    Arc::clone(&fill_mesh.vertices_local),
                    fill_mesh.bounds_local,
                )]
            });
        let mut items = Vec::new();
        if let Some(color_groups) = active_color_groups {
            items.reserve(
                color_groups
                    .groups
                    .len()
                    .saturating_mul(fill_geometry.len()),
            );
            for (group_index, group) in color_groups.groups.iter().enumerate() {
                let Some(rgb) = self.effective_color_group_rgb(&color_groups.property_key, group)
                else {
                    continue;
                };
                let color =
                    egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
                for (cache_id, vertices_local, bounds_local) in &fill_geometry {
                    items.push(ObjectFillGlDrawItem {
                        data: ObjectFillGlDrawData {
                            cache_id: *cache_id,
                            state_cache_id: object_property_render_cache_id(
                                0x4a21,
                                &color_groups.property_key,
                                group_index,
                            ),
                            generation: self.geometry_generation,
                            vertices_local: Arc::clone(vertices_local),
                            object_count: fill_mesh.object_count,
                            selection_generation: group.fill_generation,
                            selection_state: Arc::clone(&group.fill_state),
                            color_cache_id: 0,
                            color_generation: 0,
                            object_colors_rgba: None,
                        },
                        params: ObjectFillGlDrawParams {
                            center_world: camera.center_world_lvl0,
                            zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                            selected_color: color,
                            primary_color: color,
                            visible: self.visible,
                            local_to_world_offset: display_offset,
                            local_to_world_scale: display_scale,
                            object_color_opacity: 1.0,
                        },
                        visible_world: *bounds_local,
                    });
                }
            }
        } else {
            let rgb = self.color_rgb;
            let color = egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
            let mut visible_state = vec![0u8; fill_mesh.object_count];
            for (index, state) in visible_state.iter_mut().enumerate() {
                if self.is_index_visible(index) {
                    *state = 255;
                }
            }
            let visible_state = Arc::new(visible_state);
            for (cache_id, vertices_local, bounds_local) in &fill_geometry {
                items.push(ObjectFillGlDrawItem {
                    data: ObjectFillGlDrawData {
                        cache_id: *cache_id,
                        state_cache_id: object_render_cache_id(0x4a23, 0),
                        generation: self.geometry_generation,
                        vertices_local: Arc::clone(vertices_local),
                        object_count: fill_mesh.object_count,
                        selection_generation: render_generation,
                        selection_state: Arc::clone(&visible_state),
                        color_cache_id: object_render_cache_id(0x4a24, 0),
                        color_generation: continuous_colors.map_or(0, |payload| payload.generation),
                        object_colors_rgba: continuous_colors
                            .map(|payload| Arc::clone(&payload.colors_rgba)),
                    },
                    params: ObjectFillGlDrawParams {
                        center_world: camera.center_world_lvl0,
                        zoom_screen_per_world: camera.zoom_screen_per_lvl0_px,
                        selected_color: color,
                        primary_color: color,
                        visible: self.visible,
                        local_to_world_offset: display_offset,
                        local_to_world_scale: display_scale,
                        object_color_opacity: self.fill_opacity,
                    },
                    visible_world: *bounds_local,
                });
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

    #[allow(clippy::too_many_arguments)]
    fn draw_object_fill_cpu(
        &self,
        ui: &mut egui::Ui,
        camera: &crate::camera::Camera,
        viewport: egui::Rect,
        visible_local: egui::Rect,
        local_to_world_offset: egui::Vec2,
        fill_mesh: &ObjectFillMesh,
        active_color_groups: Option<&ObjectColorGroups>,
        continuous_colors: Option<&ObjectContinuousColorPayload>,
    ) {
        let fill_alpha = (self.fill_opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        let base_rgb = self.color_rgb;
        let base_color = egui::Color32::from_rgba_unmultiplied(
            base_rgb[0],
            base_rgb[1],
            base_rgb[2],
            fill_alpha,
        );
        let categorical_colors = active_color_groups.map(|color_groups| {
            let mut colors = vec![egui::Color32::TRANSPARENT; fill_mesh.object_count];
            for group in &color_groups.groups {
                let Some(rgb) = self.effective_color_group_rgb(&color_groups.property_key, group)
                else {
                    continue;
                };
                let color =
                    egui::Color32::from_rgba_unmultiplied(rgb[0], rgb[1], rgb[2], fill_alpha);
                for (object_index, state) in group.fill_state.iter().copied().enumerate() {
                    if state > 0 {
                        colors[object_index] = color;
                    }
                }
            }
            colors
        });

        let geometry = if fill_mesh.vertices_local.len() >= 500_000 {
            fill_mesh
                .spatial_slices_for_local_rect(visible_local)
                .into_iter()
                .map(|slice| (slice.vertices_local, slice.bounds_local))
                .collect::<Vec<_>>()
        } else {
            vec![(
                Arc::clone(&fill_mesh.vertices_local),
                fill_mesh.bounds_local,
            )]
        };
        if geometry
            .iter()
            .map(|(vertices, _)| vertices.len())
            .sum::<usize>()
            > Self::MAX_DIRECT_FILL_VERTICES
        {
            return;
        }

        for (vertices, bounds_local) in geometry {
            let clip = egui::Rect::from_two_pos(
                camera.world_to_screen(
                    self.local_to_world_point(bounds_local.min, local_to_world_offset),
                    viewport,
                ),
                camera.world_to_screen(
                    self.local_to_world_point(bounds_local.max, local_to_world_offset),
                    viewport,
                ),
            )
            .intersect(viewport);
            if !clip.is_positive() {
                continue;
            }
            let painter = ui.painter().with_clip_rect(clip);
            for triangle in vertices.chunks_exact(3) {
                let object_index = triangle[0][2].round().max(0.0) as usize;
                if self
                    .filtered_mask
                    .as_ref()
                    .is_some_and(|mask| !mask.get(object_index).copied().unwrap_or(false))
                {
                    continue;
                }
                let color = categorical_colors
                    .as_ref()
                    .and_then(|colors| colors.get(object_index).copied())
                    .or_else(|| {
                        continuous_colors
                            .and_then(|payload| payload.colors_rgba.get(object_index))
                            .map(|rgba| {
                                egui::Color32::from_rgba_unmultiplied(
                                    rgba[0],
                                    rgba[1],
                                    rgba[2],
                                    ((rgba[3] as f32) * self.fill_opacity.clamp(0.0, 1.0)).round()
                                        as u8,
                                )
                            })
                    })
                    .unwrap_or(base_color);
                if color.a() == 0 {
                    continue;
                }
                let points = triangle
                    .iter()
                    .map(|point| {
                        camera.world_to_screen(
                            self.local_to_world_point(
                                egui::pos2(point[0], point[1]),
                                local_to_world_offset,
                            ),
                            viewport,
                        )
                    })
                    .collect::<Vec<_>>();
                painter.add(egui::Shape::convex_polygon(
                    points,
                    color,
                    egui::Stroke::NONE,
                ));
            }
        }
    }
}
