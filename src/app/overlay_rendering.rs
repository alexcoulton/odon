use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_active_layer_tooltip(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        rect: egui::Rect,
        response: &egui::Response,
        camera_changed: bool,
    ) {
        const HOVER_TOOLTIP_DWELL: Duration = Duration::from_millis(120);
        const HOVER_TOOLTIP_GRACE: Duration = Duration::from_millis(180);

        if !response.hovered() {
            self.hover_tooltip_state = None;
            return;
        }
        // Don't show tooltips while dragging/panning/transforming.
        if ui.input(|i| i.pointer.any_down()) {
            self.hover_tooltip_state = None;
            return;
        }
        if ctx.wants_keyboard_input() {
            self.hover_tooltip_state = None;
            return;
        }
        let Some(pointer_screen) = ui.input(|i| i.pointer.hover_pos()) else {
            self.hover_tooltip_state = None;
            return;
        };
        if !rect.contains(pointer_screen) {
            self.hover_tooltip_state = None;
            return;
        }
        let now = Instant::now();

        if camera_changed {
            self.hover_tooltip_state = None;
            return;
        }

        let pointer_world = self.camera.screen_to_world(pointer_screen, rect);
        let lines: Option<Vec<String>> = match self.active_layer {
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .and_then(|s| s.hover_tooltip(pointer_world, &self.camera)),
            LayerId::SpatialPoints => {
                let off = self.layer_offset_world(LayerId::SpatialPoints);
                self.spatial_layers
                    .points
                    .as_ref()
                    .and_then(|p| p.hover_tooltip(pointer_world, off, &self.camera))
            }
            LayerId::XeniumTranscripts => {
                let off = self.layer_offset_world(LayerId::XeniumTranscripts);
                self.xenium_layers
                    .transcripts
                    .as_ref()
                    .and_then(|t| t.hover_tooltip(pointer_world, off, &self.camera))
            }
            LayerId::SegmentationObjects => {
                let off = self.layer_offset_world(LayerId::SegmentationObjects);
                self.seg_objects
                    .hover_tooltip(pointer_world, off, &self.camera)
            }
            _ => None,
        };
        let has_lines = lines.is_some();

        if let Some(lines) = lines {
            let signature = lines.join("\n");
            match self.hover_tooltip_state.as_mut() {
                Some(state) if state.signature == signature => {
                    state.last_seen = now;
                    if !state.visible && now.duration_since(state.first_seen) >= HOVER_TOOLTIP_DWELL
                    {
                        state.visible = true;
                    }
                    state.lines = lines;
                }
                Some(state) => {
                    *state = HoverTooltipState {
                        signature,
                        lines,
                        first_seen: now,
                        last_seen: now,
                        visible: false,
                    };
                }
                None => {
                    self.hover_tooltip_state = Some(HoverTooltipState {
                        signature,
                        lines,
                        first_seen: now,
                        last_seen: now,
                        visible: false,
                    });
                }
            }
        } else if self
            .hover_tooltip_state
            .as_ref()
            .is_some_and(|state| now.duration_since(state.last_seen) > HOVER_TOOLTIP_GRACE)
        {
            self.hover_tooltip_state = None;
        }

        if let Some(state) = self.hover_tooltip_state.as_ref() {
            if !state.visible {
                let elapsed = now.duration_since(state.first_seen);
                if elapsed < HOVER_TOOLTIP_DWELL {
                    ctx.request_repaint_after(HOVER_TOOLTIP_DWELL - elapsed);
                } else {
                    ctx.request_repaint();
                }
            } else if !has_lines {
                let elapsed = now.duration_since(state.last_seen);
                if elapsed < HOVER_TOOLTIP_GRACE {
                    ctx.request_repaint_after(HOVER_TOOLTIP_GRACE - elapsed);
                }
            }
        }

        if let Some(state) = self.hover_tooltip_state.as_ref()
            && state.visible
            && now.duration_since(state.last_seen) <= HOVER_TOOLTIP_GRACE
        {
            let lines = state.lines.clone();
            crate::ui::tooltip::show_tooltip_at_pointer(
                ctx,
                ui.id().with("hover_layer_tooltip"),
                |ui| {
                    for l in lines {
                        ui.label(l);
                    }
                },
            );
        }
    }

    pub(super) fn draw_cells_segmentation_overlay(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        visible_world: egui::Rect,
        target_level: usize,
    ) {
        if !self.cells_outlines_visible {
            return;
        }

        let off = self.layer_offset_world(LayerId::SegmentationLabels);
        let visible_world = visible_world.translate(-off);
        let off_screen = off * self.camera.zoom_screen_per_lvl0_px;

        let (Some(lbl), Some(loader), Some(renderer)) = (
            self.label_cells.as_ref(),
            self.label_loader.as_ref(),
            self.labels_gl.clone(),
        ) else {
            return;
        };
        let Some(xforms) = self.label_cells_xform.as_ref() else {
            return;
        };

        // Keep segmentation level selection locked to the image level to avoid drift when the
        // label pyramid scale metadata or rounding differs.
        let target_label_level = target_level.min(lbl.levels.len().saturating_sub(1));
        let levels_to_draw = vec![target_label_level];

        let mut needed_per_level: Vec<(usize, crate::data::ome::LevelInfo, Vec<LabelTileKey>)> =
            Vec::with_capacity(levels_to_draw.len());
        for &level in &levels_to_draw {
            let level_info = lbl.levels[level].clone();
            let xform = xforms.get(level).copied().unwrap_or(LabelToWorld {
                scale_x: level_info.downsample,
                scale_y: level_info.downsample,
                offset_x: 0.0,
                offset_y: 0.0,
            });

            let inv_x = 1.0 / xform.scale_x.max(1e-6);
            let inv_y = 1.0 / xform.scale_y.max(1e-6);
            let visible_lvl = egui::Rect::from_min_max(
                egui::pos2(
                    (visible_world.min.x - xform.offset_x) * inv_x,
                    (visible_world.min.y - xform.offset_y) * inv_y,
                ),
                egui::pos2(
                    (visible_world.max.x - xform.offset_x) * inv_x,
                    (visible_world.max.y - xform.offset_y) * inv_y,
                ),
            );

            let y_dim = lbl.dims.y;
            let x_dim = lbl.dims.x;
            let shape_y = level_info.shape[y_dim] as f32;
            let shape_x = level_info.shape[x_dim] as f32;
            let image_rect_lvl =
                egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(shape_x, shape_y));
            let visible_lvl = visible_lvl.intersect(image_rect_lvl.expand(1.0));

            let chunk_y = level_info.chunks[y_dim] as f32;
            let chunk_x = level_info.chunks[x_dim] as f32;

            let tile_y0 = (visible_lvl.min.y / chunk_y).floor().max(0.0) as i64 - 1;
            let tile_x0 = (visible_lvl.min.x / chunk_x).floor().max(0.0) as i64 - 1;
            let tile_y1 = (visible_lvl.max.y / chunk_y).ceil().max(0.0) as i64 + 1;
            let tile_x1 = (visible_lvl.max.x / chunk_x).ceil().max(0.0) as i64 + 1;

            let needed = self.label_tiles_needed_with_xform(
                level,
                tile_y0,
                tile_y1,
                tile_x0,
                tile_x1,
                &level_info,
                &lbl.dims,
                xform,
            );
            needed_per_level.push((level, level_info, needed));
        }

        // Prune stale in-flight label tile requests so we don't keep repainting after a fast pan/zoom.
        if let Some(labels_gl_ref) = self.labels_gl.as_ref() {
            let mut keep: HashSet<LabelTileKey> = HashSet::new();
            for (_level, _level_info, needed) in needed_per_level.iter() {
                for k in needed {
                    keep.insert(*k);
                }
            }
            if let Some(active_keys) = self.viewport_label_active_keys.as_mut() {
                merge_viewport_active_keys(active_keys, keep);
            } else {
                labels_gl_ref.prune_in_flight(&keep);
            }
        }

        // Request fine -> coarse so zoom-in upgrades quickly.
        let mut requested_this_frame = 0usize;
        let max_requests_per_frame = 128usize;
        for (_level, _level_info, needed) in needed_per_level.iter().rev() {
            if requested_this_frame >= max_requests_per_frame {
                break;
            }
            for key in needed {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                if renderer.mark_in_flight(*key) {
                    let _ = loader.tx.send(LabelTileRequest { key: *key });
                    requested_this_frame += 1;
                }
            }
        }

        // Draw list coarse -> fine.
        let mut draws: Vec<LabelDraw> = Vec::new();
        draws.reserve(512);
        for (level, level_info, needed) in needed_per_level {
            let xform = xforms.get(level).copied().unwrap_or(LabelToWorld {
                scale_x: level_info.downsample,
                scale_y: level_info.downsample,
                offset_x: 0.0,
                offset_y: 0.0,
            });
            for key in needed {
                let (_world_rect, screen_rect) =
                    self.label_tile_rects(&key, rect, &level_info, &lbl.dims, xform);
                let screen_rect = screen_rect.translate(off_screen);
                if screen_rect.intersects(rect) {
                    draws.push(LabelDraw {
                        z_level0: key.z_level0,
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        screen_rect,
                    });
                }
            }
        }

        let c = self.cells_outlines_color_rgb;
        let params = OutlinesParams {
            visible: true,
            color_rgb: [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
            ],
            opacity: self.cells_outlines_opacity,
            width_screen_px: self.cells_outlines_width_px,
        };
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            renderer.paint(info, painter, &draws, params);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        });
    }

    pub(super) fn draw_mask_layer_overlay(&mut self, ui: &mut egui::Ui, rect: egui::Rect, id: u64) {
        let start = Instant::now();
        let mut stats = MaskDrawDebugStats::default();
        let Some(layer) = self.mask_layers.iter().find(|l| l.id == id) else {
            return;
        };
        if !layer.visible || layer.polygons_world.is_empty() {
            return;
        }
        stats.visible_layers = 1;

        let off = layer.offset_world;
        let c = layer.color_rgb;
        let stroke_alpha = mask_stroke_alpha(layer.opacity);
        let stroke_color = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], stroke_alpha);
        let stroke_width = match layer.display_mode {
            MaskDisplayMode::FilledPreview => layer.width_screen_px.min(1.0).max(0.5),
            _ => layer.width_screen_px,
        };
        let stroke = egui::Stroke::new(stroke_width, stroke_color);
        let fill_color = mask_fill_color(c, layer.opacity, layer.display_mode);
        let raster_drawn =
            if let (Some(fill_color), Some(raster)) = (fill_color, layer.raster_display.as_ref()) {
                self.draw_mask_raster_display(ui, rect, raster, off, fill_color)
            } else {
                false
            };

        if raster_drawn {
            if let Some(raster) = layer.raster_display.as_ref() {
                stats.raster_layers += 1;
                stats.raster_pixels += raster.width.saturating_mul(raster.height);
            }
        } else {
            for poly in &layer.polygons_world {
                if poly.len() < 2 {
                    continue;
                }
                stats.painted_polygons += 1;
                stats.painted_vertices += poly.len();
                let pts = poly
                    .iter()
                    .copied()
                    .map(|p| self.camera.world_to_screen(p + off, rect))
                    .collect::<Vec<_>>();
                let screen_bounds = bounds_for_points(&pts);
                if screen_bounds.is_some_and(|bounds| bounds.intersects(rect)) {
                    stats.screen_polygons += 1;
                    stats.screen_vertices += poly.len();
                }
                if let Some(fill_color) = fill_color {
                    stats.fill_polygons += 1;
                    stats.fill_vertices += poly.len();
                    paint_filled_polygon(ui, &pts, fill_color);
                }
                ui.painter().add(egui::Shape::line(pts, stroke));
            }
        }
        if let Some(selection) = self.selected_mask_polygon
            && selection.layer_id == id
            && let Some(poly) = layer.polygons_world.get(selection.polygon_idx)
        {
            let n = Self::mask_polygon_unique_vertex_count(poly);
            if n >= 3 {
                let mut selected_pts = poly
                    .iter()
                    .copied()
                    .take(n)
                    .map(|p| self.camera.world_to_screen(p + off, rect))
                    .collect::<Vec<_>>();
                selected_pts.push(selected_pts[0]);

                ui.painter().add(egui::Shape::line(
                    selected_pts.clone(),
                    egui::Stroke::new(layer.width_screen_px + 4.0, egui::Color32::WHITE),
                ));
                ui.painter().add(egui::Shape::line(
                    selected_pts.clone(),
                    egui::Stroke::new(layer.width_screen_px + 2.0, stroke_color),
                ));

                for (vertex_idx, p) in selected_pts.iter().copied().take(n).enumerate() {
                    let selected = self.selected_mask_vertex == Some(vertex_idx);
                    let fill = if selected {
                        egui::Color32::from_rgb(80, 220, 140)
                    } else {
                        egui::Color32::WHITE
                    };
                    let radius = if selected { 5.5 } else { 4.5 };
                    ui.painter().circle_filled(p, radius, fill);
                    ui.painter().circle_stroke(
                        p,
                        radius + 1.5,
                        egui::Stroke::new(1.5, egui::Color32::BLACK),
                    );
                }
            }
        }

        stats.draw_time += start.elapsed();
        self.mask_draw_debug_stats.visible_layers += stats.visible_layers;
        self.mask_draw_debug_stats.painted_polygons += stats.painted_polygons;
        self.mask_draw_debug_stats.painted_vertices += stats.painted_vertices;
        self.mask_draw_debug_stats.screen_polygons += stats.screen_polygons;
        self.mask_draw_debug_stats.screen_vertices += stats.screen_vertices;
        self.mask_draw_debug_stats.fill_polygons += stats.fill_polygons;
        self.mask_draw_debug_stats.fill_vertices += stats.fill_vertices;
        self.mask_draw_debug_stats.raster_layers += stats.raster_layers;
        self.mask_draw_debug_stats.raster_pixels += stats.raster_pixels;
        self.mask_draw_debug_stats.draw_time += stats.draw_time;
    }

    pub(super) fn draw_mask_raster_display(
        &self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        raster: &MaskRasterDisplayCache,
        layer_offset_world: egui::Vec2,
        tint: egui::Color32,
    ) -> bool {
        let Some(renderer) = self.threshold_preview_gl.clone() else {
            return false;
        };
        if raster.width == 0 || raster.height == 0 || raster.values.is_empty() {
            return false;
        }

        let corners_screen = raster.corners_world.map(|point| {
            self.camera
                .world_to_screen(point + layer_offset_world, rect)
        });
        let data = ThresholdPreviewGlDrawData {
            generation: raster.generation,
            width: raster.width,
            height: raster.height,
            values: raster.values.clone(),
        };
        let params = ThresholdPreviewGlDrawParams {
            visible: true,
            quad_screen: corners_screen,
            threshold_u16: 1,
            tint,
        };
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            renderer.paint(info, painter, &data, &params);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        });
        true
    }

    pub(super) fn draw_points_overlay(
        &mut self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        visible_world: egui::Rect,
    ) {
        let off = self.layer_offset_world(LayerId::Points);
        if let Some(renderer) = self.points_gl.clone() {
            if let Some((generation, positions_world, values)) =
                self.legacy_cell_threshold_points.gpu_points()
            {
                let data = PointsGlDrawData {
                    generation,
                    positions_world,
                    values,
                };
                let params = PointsGlDrawParams {
                    center_world: self.camera.center_world_lvl0,
                    zoom_screen_per_world: self.camera.zoom_screen_per_lvl0_px,
                    threshold: self.legacy_cell_threshold_points.threshold(),
                    style: self.cell_points.style.clone(),
                    visible: self.cell_points.visible,
                    local_to_world_offset: off,
                    local_to_world_scale: egui::vec2(1.0, 1.0),
                };
                let cb = egui_glow::CallbackFn::new(move |info, painter| {
                    renderer.paint(info, painter, &data, &params);
                });
                ui.painter().add(egui::PaintCallback {
                    rect,
                    callback: Arc::new(cb),
                });
            }
        } else {
            let world_to_screen = |p: egui::Pos2| self.camera.world_to_screen(p + off, rect);
            self.cell_points.draw(
                ui.painter(),
                rect,
                world_to_screen,
                visible_world.translate(-off),
                self.camera.zoom_screen_per_lvl0_px,
            );
        }
    }

    pub(super) fn label_tiles_needed_with_xform(
        &self,
        level: usize,
        tile_y0: i64,
        tile_y1: i64,
        tile_x0: i64,
        tile_x1: i64,
        level_info: &crate::data::ome::LevelInfo,
        dims: &crate::data::ome::Dims,
        xform: LabelToWorld,
    ) -> Vec<LabelTileKey> {
        let mut keys = Vec::new();

        let y_dim = dims.y;
        let x_dim = dims.x;
        let max_tiles_y = ((level_info.shape[y_dim] + level_info.chunks[y_dim] - 1)
            / level_info.chunks[y_dim]) as i64;
        let max_tiles_x = ((level_info.shape[x_dim] + level_info.chunks[x_dim] - 1)
            / level_info.chunks[x_dim]) as i64;

        let y0 = tile_y0.clamp(0, max_tiles_y);
        let y1 = tile_y1.clamp(0, max_tiles_y);
        let x0 = tile_x0.clamp(0, max_tiles_x);
        let x1 = tile_x1.clamp(0, max_tiles_x);

        for ty in y0..y1 {
            for tx in x0..x1 {
                keys.push(LabelTileKey {
                    z_level0: self.active_z_level0(),
                    level,
                    tile_y: ty as u64,
                    tile_x: tx as u64,
                });
            }
        }

        // Near-to-center priority for request ordering.
        let center_world = self.camera.center_world_lvl0;
        let inv_x = 1.0 / xform.scale_x.max(1e-6);
        let inv_y = 1.0 / xform.scale_y.max(1e-6);
        let center_lvl = egui::pos2(
            (center_world.x - xform.offset_x) * inv_x,
            (center_world.y - xform.offset_y) * inv_y,
        );
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;
        keys.sort_by(|a, b| {
            let ay = (a.tile_y as f32 + 0.5) * chunk_y;
            let ax = (a.tile_x as f32 + 0.5) * chunk_x;
            let by = (b.tile_y as f32 + 0.5) * chunk_y;
            let bx = (b.tile_x as f32 + 0.5) * chunk_x;
            let da = (ax - center_lvl.x).powi(2) + (ay - center_lvl.y).powi(2);
            let db = (bx - center_lvl.x).powi(2) + (by - center_lvl.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        keys
    }

    pub(super) fn label_tile_rects(
        &self,
        key: &LabelTileKey,
        viewport: egui::Rect,
        level_info: &crate::data::ome::LevelInfo,
        dims: &crate::data::ome::Dims,
        xform: LabelToWorld,
    ) -> (egui::Rect, egui::Rect) {
        let y_dim = dims.y;
        let x_dim = dims.x;
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;

        let y0 = key.tile_y as f32 * chunk_y;
        let x0 = key.tile_x as f32 * chunk_x;
        let y1 = (y0 + chunk_y).min(level_info.shape[y_dim] as f32);
        let x1 = (x0 + chunk_x).min(level_info.shape[x_dim] as f32);

        let world_min = egui::pos2(
            x0 * xform.scale_x + xform.offset_x,
            y0 * xform.scale_y + xform.offset_y,
        );
        let world_max = egui::pos2(
            x1 * xform.scale_x + xform.offset_x,
            y1 * xform.scale_y + xform.offset_y,
        );
        let world_rect = egui::Rect::from_min_max(world_min, world_max);

        let screen_min = self.camera.world_to_screen(world_min, viewport);
        let screen_max = self.camera.world_to_screen(world_max, viewport);
        let screen_rect = egui::Rect::from_min_max(screen_min, screen_max);

        (world_rect, screen_rect)
    }

    pub(super) fn visible_world_rect(&self, viewport: egui::Rect) -> egui::Rect {
        let world_min = self.camera.screen_to_world(viewport.left_top(), viewport);
        let world_max = self
            .camera
            .screen_to_world(viewport.right_bottom(), viewport);
        egui::Rect::from_min_max(world_min, world_max)
    }

    pub(super) fn image_local_rect_lvl0(&self) -> egui::Rect {
        let shape0 = &self.dataset.levels[0].shape;
        let axes = self
            .display_axes()
            .unwrap_or(crate::imaging::view_plane::DisplayAxes {
                vertical: self.dataset.dims.y,
                horizontal: self.dataset.dims.x,
            });
        let y = shape0[axes.vertical] as f32;
        let x = shape0[axes.horizontal] as f32;
        egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(x, y))
    }

    pub(super) fn primary_image_local_to_world(&self, p: egui::Pos2) -> egui::Pos2 {
        let Some(level0) = self.dataset.levels.first() else {
            return p;
        };
        let (sx, sy) = local_to_world_scale(&self.dataset.dims, level0, self.view_plane_mode);
        egui::pos2(p.x * sx, p.y * sy)
    }

    pub(super) fn primary_image_world_to_local(&self, p: egui::Pos2) -> egui::Pos2 {
        let Some(level0) = self.dataset.levels.first() else {
            return p;
        };
        let (sx, sy) = local_to_world_scale(&self.dataset.dims, level0, self.view_plane_mode);
        egui::pos2(p.x / sx.max(1e-6), p.y / sy.max(1e-6))
    }

    pub(super) fn primary_image_world_rect_to_local(&self, rect: egui::Rect) -> egui::Rect {
        egui::Rect::from_min_max(
            self.primary_image_world_to_local(rect.min),
            self.primary_image_world_to_local(rect.max),
        )
    }

    pub(super) fn image_world_rect_lvl0(&self) -> egui::Rect {
        let local = self.image_local_rect_lvl0();
        egui::Rect::from_min_max(
            self.primary_image_local_to_world(local.min),
            self.primary_image_local_to_world(local.max),
        )
    }

    pub(super) fn channel_transform_gizmo_screen(
        &self,
        viewport: egui::Rect,
        ch_idx: usize,
    ) -> (egui::Pos2, [egui::Pos2; 4], egui::Pos2) {
        let img_world = self.image_world_rect_lvl0();
        let pivot_world = img_world.center();
        let pivot_screen = self.camera.world_to_screen(pivot_world, viewport);

        let zoom = self.camera.zoom_screen_per_lvl0_px;
        let off_world = self
            .channel_offsets_world
            .get(ch_idx)
            .copied()
            .unwrap_or_default();
        let trans_screen = off_world * zoom;
        let pivot_screen_effective = pivot_screen + trans_screen;
        let scale = self
            .channel_scales
            .get(ch_idx)
            .copied()
            .unwrap_or(egui::Vec2::splat(1.0));
        let rot = self
            .channel_rotations_rad
            .get(ch_idx)
            .copied()
            .unwrap_or(0.0);

        // Base image corners in screen space (untransformed).
        let tl = self.camera.world_to_screen(img_world.left_top(), viewport);
        let tr = self
            .camera
            .world_to_screen(egui::pos2(img_world.right(), img_world.top()), viewport);
        let br = self
            .camera
            .world_to_screen(img_world.right_bottom(), viewport);
        let bl = self
            .camera
            .world_to_screen(egui::pos2(img_world.left(), img_world.bottom()), viewport);

        let corners = [
            xform_screen_point(tl, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(tr, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(br, pivot_screen, trans_screen, scale, rot),
            xform_screen_point(bl, pivot_screen, trans_screen, scale, rot),
        ];

        let center = quad_center(&corners);
        let top_mid = (corners[0].to_vec2() + corners[1].to_vec2()) * 0.5;
        let outward = {
            let v = top_mid - center.to_vec2();
            if v.length() > 1e-6 {
                v / v.length()
            } else {
                egui::vec2(0.0, -1.0)
            }
        };
        let rotate_handle_v = top_mid + outward * 26.0;
        let rotate_handle = egui::pos2(rotate_handle_v.x, rotate_handle_v.y);

        (pivot_screen_effective, corners, rotate_handle)
    }

    pub(super) fn draw_channel_transform_gizmo(
        &self,
        ui: &mut egui::Ui,
        viewport: egui::Rect,
        ch_idx: usize,
    ) {
        if ch_idx >= self.channels.len() {
            return;
        }

        let (_pivot, corners, rotate_handle) =
            self.channel_transform_gizmo_screen(viewport, ch_idx);
        let base_color = egui::Color32::from_rgb(120, 200, 255);
        let hover_color = egui::Color32::from_rgb(255, 180, 60);
        let stroke = egui::Stroke::new(1.6, base_color);
        let handle_r = 4.5;

        let mut hover_corner: Option<usize> = None;
        let mut hover_rotate = false;
        let hit_r = 10.0;
        if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
            if viewport.contains(pointer) {
                for (i, &c) in corners.iter().enumerate() {
                    if c.distance(pointer) <= hit_r {
                        hover_corner = Some(i);
                        break;
                    }
                }
                hover_rotate = rotate_handle.distance(pointer) <= hit_r;
            }
        }

        // Outline.
        let mut pts = Vec::with_capacity(5);
        pts.push(corners[0]);
        pts.push(corners[1]);
        pts.push(corners[2]);
        pts.push(corners[3]);
        pts.push(corners[0]);
        ui.painter().add(egui::Shape::line(pts, stroke));

        // Corner handles.
        for (i, &c) in corners.iter().enumerate() {
            let fill = if hover_corner == Some(i) {
                hover_color
            } else {
                base_color
            };
            ui.painter().circle_filled(c, handle_r, fill);
        }

        // Rotate handle + connector.
        let top_mid_v = (corners[0].to_vec2() + corners[1].to_vec2()) * 0.5;
        let top_mid = egui::pos2(top_mid_v.x, top_mid_v.y);
        let rotate_stroke = if hover_rotate {
            egui::Stroke::new(stroke.width, hover_color)
        } else {
            stroke
        };
        ui.painter()
            .line_segment([top_mid, rotate_handle], rotate_stroke);
        ui.painter()
            .circle_stroke(rotate_handle, handle_r + 1.0, rotate_stroke);
    }
}
