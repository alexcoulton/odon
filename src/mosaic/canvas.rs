//! Canvas drawing, tile scheduling, layout application, and mosaic labels.

use super::*;

impl MosaicViewerApp {
    pub(super) fn ui_canvas(&mut self, ui: &mut egui::Ui, _ctx: &egui::Context) {
        let camera_before = self.control_camera_snapshot();
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::drag());
        self.last_canvas_rect = Some(rect);
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(10));

        // Zoom + pan
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let pinch = ui.input(|i| i.zoom_delta());
            if scroll != 0.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let factor = (scroll * 0.0015).exp();
                    self.camera.zoom_about_screen_point(rect, pos, factor);
                }
            }
            if pinch != 1.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    self.camera.zoom_about_screen_point(rect, pos, pinch);
                }
            }
        }
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera.pan_by_screen_delta(delta);
        }

        if response.double_clicked() {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let world = self.camera.screen_to_world(pos, rect);
                if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world)) {
                    if self.control_actor_owned {
                        let roi_id = it.sample_id.clone();
                        self.submit_native_control_intent(
                            "mosaic.focus.set",
                            serde_json::json!({"roi_id":roi_id,"fit":true}),
                        );
                    } else {
                        self.focused_core_id = Some(it.id);
                        self.camera.fit_to_world_rect(rect, item_rect(it));
                    }
                }
            }
        }

        self.submit_camera_preview_if_changed(&camera_before);

        if self.active_layer == MosaicLayerId::SegmentationGeoJson
            && self.seg_geojson.visible
            && response.clicked_by(egui::PointerButton::Primary)
            && let Some(pos) = ui.input(|i| i.pointer.hover_pos())
            && rect.contains(pos)
        {
            let world = self.camera.screen_to_world(pos, rect);
            let mods = ui.input(|i| i.modifiers);
            if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world)) {
                self.seg_geojson
                    .select_at(it.id, world, &self.camera, mods.shift, mods.command);
            } else if !mods.shift && !mods.command {
                self.seg_geojson.clear_selection();
            }
        }

        // Draw
        //
        // The mosaic renderer works in two passes each frame:
        // 1. ensure every visible ROI has at least coarse coverage, so the whole mosaic appears
        // 2. spend the remaining request budget refining ROIs near the viewport center/current focus
        //
        // This prevents a zoomed-out mosaic from showing "holes" while still biasing bandwidth
        // toward the area the user is inspecting.
        let visible_world = visible_world_rect(&self.camera, rect);
        let prev_visible_world = self.last_visible_world.unwrap_or(visible_world);
        let channels_draw = self.visible_channel_draws();
        self.sync_tile_request_generation(visible_world, rect, &channels_draw);
        self.drain_raw_tiles();
        let request_generation = self.tile_request_generation;
        let mut draws: Vec<MosaicTileDraw> = Vec::new();

        let mut sent = 0usize;
        let max_requests_per_frame = 2048usize;
        let max_coarse_tiles_per_item_per_frame = 2usize;

        // Phase A: ensure coarsest level tiles for all items (so everything appears when zoomed out).
        for it in &self.items {
            let _ = self.collect_draws_and_requests_for_item(
                it,
                visible_world,
                Some(prev_visible_world),
                rect,
                Phase::CoarseOnly,
                request_generation,
                None,
                None,
                None,
                None,
                None,
                &mut draws,
                &mut sent,
                max_requests_per_frame,
                max_coarse_tiles_per_item_per_frame,
            );
        }
        // Phase B: refine near the current zoom.
        let refine_order = self.refine_item_order(visible_world);
        for idx in refine_order {
            if sent >= max_requests_per_frame {
                break;
            }
            let (id, target, ceiling) = {
                let Some(it) = self.items.get(idx) else {
                    continue;
                };
                let prev = self
                    .last_target_level_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_ceiling = self
                    .fallback_ceiling_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor = self
                    .zoom_out_floor_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor_until = self
                    .zoom_out_floor_until_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let prev_floor_world = self
                    .zoom_out_floor_world_by_dataset_id
                    .get(it.id)
                    .copied()
                    .flatten();
                let (target, ceiling, floor, floor_until, floor_world) = self
                    .collect_draws_and_requests_for_item(
                        it,
                        visible_world,
                        Some(prev_visible_world),
                        rect,
                        Phase::Refine,
                        request_generation,
                        prev,
                        prev_ceiling,
                        prev_floor,
                        prev_floor_until,
                        prev_floor_world,
                        &mut draws,
                        &mut sent,
                        max_requests_per_frame,
                        max_coarse_tiles_per_item_per_frame,
                    );
                if let Some(dst) = self.zoom_out_floor_by_dataset_id.get_mut(it.id) {
                    *dst = floor;
                }
                if let Some(dst) = self.zoom_out_floor_until_by_dataset_id.get_mut(it.id) {
                    *dst = floor_until;
                }
                if let Some(dst) = self.zoom_out_floor_world_by_dataset_id.get_mut(it.id) {
                    *dst = floor_world;
                }
                (it.id, target, ceiling)
            };
            if let Some(t) = target {
                if let Some(dst) = self.last_target_level_by_dataset_id.get_mut(id) {
                    *dst = Some(t);
                }
            }
            if let Some(c) = ceiling {
                if let Some(dst) = self.fallback_ceiling_by_dataset_id.get_mut(id) {
                    *dst = Some(c);
                }
            }
        }
        self.last_visible_world = Some(visible_world);

        // If the user navigated quickly, we may have in-flight requests for tiles that are no longer
        // relevant to the current view. Prune them so we can actually go idle.
        let mut keep: HashSet<MosaicRawTileKey> =
            HashSet::with_capacity(draws.len() * channels_draw.len());
        for td in &draws {
            for ch in &channels_draw {
                keep.insert(MosaicRawTileKey {
                    dataset_id: td.dataset_id,
                    level: td.level,
                    tile_y: td.tile_y,
                    tile_x: td.tile_x,
                    channel: ch.index,
                });
            }
        }
        self.tiles_gl.prune_in_flight(&keep);

        let tiles_gl = self.tiles_gl.clone();
        let sources = Arc::clone(&self.sources);
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            tiles_gl.paint(info, painter, &sources, &draws, &channels_draw);
        });
        ui.painter().add(egui::PaintCallback {
            rect,
            callback: Arc::new(cb),
        });

        let screenshot = self.screenshot_pending.take();
        let screenshot_active = screenshot.is_some();

        // Overlay layers (in the user-controlled order).
        self.seg_geojson_pending_visible = false;
        let overlay_order = self.overlay_layer_order.clone();

        let mut visible_rois: Vec<(String, egui::Vec2, f32)> = Vec::new();
        visible_rois.reserve(self.items.len().min(256));
        for it in &self.items {
            let r = item_rect(it);
            if r.intersects(visible_world) {
                visible_rois.push((it.sample_id.clone(), it.offset, it.scale));
            }
        }

        for layer in overlay_order.into_iter().rev() {
            match layer {
                MosaicLayerId::Channel(_) => {}
                MosaicLayerId::TextLabels => {
                    if self.show_text_labels {
                        self.draw_text_labels(ui, rect);
                    }
                }
                MosaicLayerId::SegmentationGeoJson => {
                    if self.seg_geojson.visible && self.seg_geojson.has_any_segpaths() {
                        let mut visible_items: Vec<(usize, egui::Rect, egui::Vec2, f32)> =
                            Vec::new();
                        visible_items.reserve(self.items.len().min(128));
                        for it in &self.items {
                            let r = item_rect(it);
                            if r.intersects(visible_world) {
                                visible_items.push((it.id, r, it.offset, it.scale));
                            }
                        }
                        let mut load_items = visible_items.clone();
                        for item in &self.items {
                            if self.pending_object_load_ids.contains(&item.id)
                                && !load_items.iter().any(|entry| entry.0 == item.id)
                            {
                                load_items.push((
                                    item.id,
                                    item_rect(item),
                                    item.offset,
                                    item.scale,
                                ));
                            }
                        }
                        let load_world = if self.pending_object_load_ids.is_empty() {
                            visible_world
                        } else {
                            egui::Rect::EVERYTHING
                        };
                        self.seg_geojson_pending_visible = self
                            .seg_geojson
                            .ensure_visible_items_loading(&load_items, load_world);
                        let pending_gpu = self.seg_geojson.paint(
                            ui,
                            &self.camera,
                            rect,
                            visible_world,
                            &visible_items,
                        );
                        self.seg_geojson_pending_visible |= pending_gpu;
                    }
                }
                MosaicLayerId::Annotation(id) => {
                    if let Some(layer) = self.annotation_layers.iter_mut().find(|l| l.id == id) {
                        let group_tint =
                            layer_groups::effective_annotation_tint(&self.layer_groups, id);
                        layer.draw_mosaic(
                            ui,
                            rect,
                            self.camera.center_world_lvl0,
                            self.camera.zoom_screen_per_lvl0_px,
                            &visible_rois,
                            group_tint,
                            true,
                        );
                        if self.active_layer == MosaicLayerId::Annotation(id) {
                            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                                if rect.contains(pointer) {
                                    let world = self.camera.screen_to_world(pointer, rect);
                                    if let Some(it) =
                                        self.items.iter().find(|it| item_rect(it).contains(world))
                                    {
                                        layer.maybe_hover_tooltip(
                                            ui.ctx(),
                                            rect,
                                            world,
                                            self.camera.zoom_screen_per_lvl0_px,
                                            it.sample_id.as_str(),
                                            it.offset,
                                            it.scale,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !screenshot_active
            && self.active_layer == MosaicLayerId::SegmentationGeoJson
            && self.seg_geojson.visible
            && let Some(pointer) = ui.input(|i| i.pointer.hover_pos())
            && rect.contains(pointer)
        {
            let world = self.camera.screen_to_world(pointer, rect);
            if let Some(it) = self.items.iter().find(|it| item_rect(it).contains(world))
                && let Some(lines) = self.seg_geojson.hover_tooltip(it.id, world, &self.camera)
            {
                crate::ui::tooltip::show_tooltip_at_pointer(
                    ui.ctx(),
                    egui::Id::new(("mosaic-segmentation-object-tooltip", it.id)),
                    |ui| {
                        for line in lines {
                            ui.label(line);
                        }
                    },
                );
            }
        }

        if self.show_group_labels && !self.group_by.is_empty() {
            self.draw_group_labels(ui, rect);
        }

        let tile_loading_count = self.tiles_gl.loading_tile_count_for(&keep);

        if !screenshot_active {
            let hud = format!(
                "zoom {:.5} center ({:.0}, {:.0})",
                self.camera.zoom_screen_per_lvl0_px,
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y
            );
            canvas_overlays::paint_hud(ui, rect, hud);
        }

        if screenshot
            .as_ref()
            .is_some_and(|spec| spec.settings.include_legend)
        {
            let mut entries: Vec<(egui::Color32, String)> = Vec::new();
            for idx in self.channel_layer_order.iter().copied() {
                let Some(ch) = self.channels.get(idx) else {
                    continue;
                };
                if !ch.visible {
                    continue;
                }
                let rgb = layer_groups::effective_channel_color_rgb(
                    &self.layer_groups,
                    ch.name.as_str(),
                    ch.color_rgb,
                );
                entries.push((
                    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                    ch.name.clone(),
                ));
            }
            canvas_overlays::paint_marker_legend(
                ui,
                rect,
                &entries,
                screenshot
                    .as_ref()
                    .map(|spec| spec.settings.legend_scale)
                    .unwrap_or(1.0),
            );
        }

        if !screenshot_active {
            let spinner_text = if self.show_tile_debug && tile_loading_count > 0 {
                Some(format!("{tile_loading_count} tiles"))
            } else {
                None
            };
            canvas_overlays::paint_spinner(
                ui,
                rect,
                tile_loading_count > 0
                    || self.seg_geojson.is_busy()
                    || self.seg_geojson_pending_visible,
                spinner_text.as_deref(),
            );
        }

        if let Some(spec) = screenshot {
            let tx = self.screenshot_worker.tx.clone();
            let id = spec.id;
            let path = spec.path.clone();
            let presentation = spec.presentation.clone();
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                let viewport = info.viewport_in_pixels();
                let x_px = viewport.left_px;
                let y_px = viewport.from_bottom_px;
                let w_px = viewport.width_px;
                let h_px = viewport.height_px;

                if w_px <= 0 || h_px <= 0 {
                    if let Some(reply) = presentation.as_ref() {
                        let _ =
                            reply
                                .tx
                                .send(odon::control::actor::PresentationCaptureCompletion {
                                    capture_id: reply.capture_id,
                                    result: Err("mosaic capture rectangle is empty".to_string()),
                                });
                    }
                    return;
                }

                let gl = painter.gl();
                let mut rgba = vec![0u8; (w_px as usize) * (h_px as usize) * 4];
                unsafe {
                    let gl_ref = gl.as_ref();
                    gl_ref.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gl_ref.read_pixels(
                        x_px,
                        y_px,
                        w_px,
                        h_px,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(rgba.as_mut_slice())),
                    );
                }
                if let Some(reply) = presentation.as_ref() {
                    let _ = reply
                        .tx
                        .send(odon::control::actor::PresentationCaptureCompletion {
                            capture_id: reply.capture_id,
                            result: Ok(odon::control::actor::PresentationPixels {
                                width: w_px as usize,
                                height: h_px as usize,
                                rgba,
                                bottom_up: true,
                            }),
                        });
                } else {
                    let _ = tx.send(ScreenshotWorkerMsg::SavePng {
                        id,
                        path: path.clone(),
                        width: w_px as usize,
                        height: h_px as usize,
                        rgba_bottom_up: rgba,
                    });
                }
            });
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(cb),
            });
        }
    }

    pub(super) fn visible_channel_draws(&self) -> Vec<ChannelDraw> {
        let mut out = Vec::new();
        for gid in self.channel_layer_order.iter().copied() {
            let Some(gch) = self.channels.get(gid) else {
                continue;
            };
            if !gch.visible {
                continue;
            }
            let rgb = layer_groups::effective_channel_color_rgb(
                &self.layer_groups,
                gch.name.as_str(),
                gch.color_rgb,
            );
            out.push(ChannelDraw {
                index: gid as u64,
                color_rgb: [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                ],
                window: gch.window.unwrap_or((0.0, self.abs_max)),
            });
        }
        out
    }

    pub(super) fn sort_tile_coords_near_center(
        &self,
        item: &MosaicItem,
        level_info: &crate::data::ome::LevelInfo,
        keys: &mut [TileCoord],
    ) {
        let y_dim = item.dataset.dims.y;
        let x_dim = item.dataset.dims.x;
        let center_world = self.camera.center_world_lvl0;
        let center_local = (center_world - item.offset) / item.scale;
        let downsample = level_info.downsample.max(1e-6);
        let center_lvl = egui::pos2(center_local.x / downsample, center_local.y / downsample);
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;
        let _ = x_dim;

        keys.sort_by(|a, b| {
            let ay = (a.tile_y as f32 + 0.5) * chunk_y;
            let ax = (a.tile_x as f32 + 0.5) * chunk_x;
            let by = (b.tile_y as f32 + 0.5) * chunk_y;
            let bx = (b.tile_x as f32 + 0.5) * chunk_x;
            let da = (ax - center_lvl.x).powi(2) + (ay - center_lvl.y).powi(2);
            let db = (bx - center_lvl.x).powi(2) + (by - center_lvl.y).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub(super) fn collect_draws_and_requests_for_item(
        &self,
        it: &MosaicItem,
        visible_world: egui::Rect,
        prev_visible_world: Option<egui::Rect>,
        viewport: egui::Rect,
        phase: Phase,
        request_generation: u64,
        prev_target_level: Option<usize>,
        prev_ceiling_level: Option<usize>,
        prev_floor_level: Option<usize>,
        prev_floor_until: Option<Instant>,
        prev_floor_world: Option<egui::Rect>,
        draws_out: &mut Vec<MosaicTileDraw>,
        sent: &mut usize,
        max_per_frame: usize,
        max_coarse_tiles_per_item_per_frame: usize,
    ) -> (
        Option<usize>,
        Option<usize>,
        Option<usize>,
        Option<Instant>,
        Option<egui::Rect>,
    ) {
        let item_rect = item_rect(it);
        if !item_rect.intersects(visible_world) {
            return (None, None, None, None, None);
        }

        // Translate the current mosaic viewport into this item's local level-0 pixel space.
        // All level choice and tile-key generation below happens in per-item local coordinates,
        // then the resulting draws are mapped back into shared mosaic world coordinates.
        //
        // The refinement path also tracks two short-lived pieces of history:
        // - a "ceiling" when zooming in, so intermediate levels remain eligible instead of
        //   jumping directly from very coarse to very fine
        // - a "floor" when zooming out, so the previous finer level can linger briefly and avoid
        //   a sudden blur jump while the new coarser level is still loading
        //
        // The return value carries that per-item refinement state back to the caller for reuse on
        // the next frame.
        // visible local (lvl0 px), intersection with ROI bounds
        let visible_in_item = visible_world.intersect(item_rect);
        let local_min = (visible_in_item.min.to_vec2() - it.offset) / it.scale;
        let local_max = (visible_in_item.max.to_vec2() - it.offset) / it.scale;
        let visible_local = egui::Rect::from_min_max(local_min.to_pos2(), local_max.to_pos2());

        let mut target_out: Option<usize> = None;
        let mut ceiling_out: Option<usize> = None;
        let mut zoom_out_floor_level_out: Option<usize> = prev_floor_level;
        let mut zoom_out_floor_until_out: Option<Instant> = prev_floor_until;
        let mut zoom_out_floor_world_out: Option<egui::Rect> = prev_floor_world;
        let levels = match phase {
            Phase::CoarseOnly => vec![it.dataset.levels.len().saturating_sub(1)],
            Phase::Refine => {
                let target_level = choose_level_auto(
                    &it.dataset.levels,
                    self.camera.zoom_screen_per_lvl0_px,
                    it.scale,
                );
                target_out = Some(target_level);
                let coarsest = it.dataset.levels.len().saturating_sub(1);
                let mut ceiling = prev_ceiling_level
                    .or(prev_target_level)
                    .unwrap_or(target_level);
                if let Some(prev_target) = prev_target_level {
                    if target_level < prev_target {
                        ceiling = ceiling.max(prev_target);
                    } else if target_level > prev_target {
                        ceiling = target_level;
                    }
                } else {
                    ceiling = target_level;
                }
                ceiling = ceiling.min(coarsest);
                ceiling_out = Some(ceiling);

                // We already have Phase::CoarseOnly ensuring coarsest coverage for all items; here
                // we focus on progressively refining between target and the sticky ceiling.
                let mut levels = Vec::new();
                for l in target_level..=ceiling {
                    levels.push(l);
                }
                levels.sort_unstable_by(|a, b| b.cmp(a)); // coarse -> fine
                levels.dedup();

                // Short-lived zoom-out floor: keep drawing the previous finer target level over the
                // previously-visible region for a moment to avoid sudden blur jumps.
                const ZOOM_OUT_FLOOR_MS: u64 = 400;
                let now = Instant::now();
                let prev_vis_world = prev_visible_world.unwrap_or(visible_world);
                if let Some(prev_target) = prev_target_level {
                    if target_level > prev_target {
                        zoom_out_floor_level_out = Some(prev_target);
                        zoom_out_floor_until_out =
                            Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_MS));
                        zoom_out_floor_world_out = Some(prev_vis_world);
                    } else if target_level < prev_target {
                        zoom_out_floor_level_out = None;
                        zoom_out_floor_until_out = None;
                        zoom_out_floor_world_out = None;
                    }
                }
                levels
            }
        };

        let Some(src) = self.sources.get(it.id) else {
            return (
                target_out,
                ceiling_out,
                zoom_out_floor_level_out,
                zoom_out_floor_until_out,
                zoom_out_floor_world_out,
            );
        };

        // Keep the zoom-out floor until the new (coarser) target has enough tiles, so we don't get
        // a sudden blur jump if IO is slower than expected.
        const ZOOM_OUT_FLOOR_EXTEND_MS: u64 = 200;
        if let (Some(target_level), Some(floor_level), Some(floor_world)) = (
            target_out,
            zoom_out_floor_level_out,
            zoom_out_floor_world_out,
        ) {
            if floor_level >= it.dataset.levels.len() || floor_level >= target_level {
                zoom_out_floor_level_out = None;
                zoom_out_floor_until_out = None;
                zoom_out_floor_world_out = None;
            } else {
                let probe_gid = self.channel_layer_order.iter().copied().find(|&gid| {
                    let visible = self.channels.get(gid).is_some_and(|ch| ch.visible);
                    visible && src.channel_map.get(gid).copied().flatten().is_some()
                });
                let probe_channel = probe_gid.map(|gid| gid as u64);

                let visible_floor_in_item = floor_world.intersect(item_rect);
                let mut ready_enough = true;
                if let (Some(probe_channel), Some(level_info_tgt)) =
                    (probe_channel, it.dataset.levels.get(target_level))
                {
                    if visible_floor_in_item.width() > 0.0 && visible_floor_in_item.height() > 0.0 {
                        let local_min =
                            (visible_floor_in_item.min.to_vec2() - it.offset) / it.scale;
                        let local_max =
                            (visible_floor_in_item.max.to_vec2() - it.offset) / it.scale;
                        let visible_local_floor =
                            egui::Rect::from_min_max(local_min.to_pos2(), local_max.to_pos2());
                        let coords_tgt = tiles_needed_lvl0_rect(
                            visible_local_floor,
                            level_info_tgt,
                            &it.dataset.dims,
                            0,
                        );
                        let sample_max = 8usize;
                        let stride = (coords_tgt.len() / sample_max).max(1);
                        let mut total = 0usize;
                        let mut ready = 0usize;
                        for c in coords_tgt.iter().step_by(stride).take(sample_max) {
                            total += 1;
                            let k = MosaicRawTileKey {
                                dataset_id: it.id,
                                level: target_level,
                                tile_y: c.tile_y,
                                tile_x: c.tile_x,
                                channel: probe_channel,
                            };
                            if self.tiles_gl.contains(&k) {
                                ready += 1;
                            }
                        }
                        ready_enough = total == 0 || ready * 10 >= total * 8; // >=80%
                    }
                }

                let now = Instant::now();
                if ready_enough {
                    zoom_out_floor_level_out = None;
                    zoom_out_floor_until_out = None;
                    zoom_out_floor_world_out = None;
                } else if zoom_out_floor_until_out.map(|u| now > u).unwrap_or(true) {
                    zoom_out_floor_until_out =
                        Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_EXTEND_MS));
                }
            }
        }

        for &level in &levels {
            let Some(level_info) = it.dataset.levels.get(level) else {
                continue;
            };
            let mut needed_tiles =
                tiles_needed_lvl0_rect(visible_local, level_info, &it.dataset.dims, 1);
            self.sort_tile_coords_near_center(it, level_info, &mut needed_tiles);

            let tile_limit = if matches!(phase, Phase::CoarseOnly) {
                max_coarse_tiles_per_item_per_frame.max(1)
            } else {
                usize::MAX
            };

            for (ti, key) in needed_tiles.iter().enumerate() {
                if ti >= tile_limit {
                    break;
                }
                if *sent >= max_per_frame {
                    break;
                }
                for gid in self.channel_layer_order.iter().copied() {
                    if *sent >= max_per_frame {
                        break;
                    }
                    let Some(gch) = self.channels.get(gid) else {
                        continue;
                    };
                    if !gch.visible {
                        continue;
                    }
                    if src.channel_map.get(gid).copied().flatten().is_none() {
                        continue;
                    }
                    let raw_key = MosaicRawTileKey {
                        dataset_id: it.id,
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        channel: gid as u64,
                    };
                    if self.tiles_gl.mark_in_flight(raw_key) {
                        if self
                            .loader
                            .tx
                            .try_send(MosaicRawTileRequest {
                                key: raw_key,
                                generation: request_generation,
                            })
                            .is_ok()
                        {
                            *sent += 1;
                        } else {
                            self.tiles_gl.cancel_in_flight(&raw_key);
                            break;
                        }
                    }
                }
            }

            // Draw tiles (coarse -> fine).
            for key in needed_tiles {
                let screen_rect =
                    tile_screen_rect_mosaic(&self.camera, it, level_info, &key, viewport);
                if screen_rect.intersects(viewport) {
                    draws_out.push(MosaicTileDraw {
                        dataset_id: it.id,
                        level,
                        tile_y: key.tile_y,
                        tile_x: key.tile_x,
                        screen_rect,
                    });
                }
            }
        }

        // Draw-only zoom-out floor overlay last (finer than the current target).
        if matches!(phase, Phase::Refine) {
            if let (Some(target_level), Some(floor_level)) = (target_out, zoom_out_floor_level_out)
            {
                if floor_level < target_level {
                    let now = Instant::now();
                    if zoom_out_floor_until_out.map(|u| now <= u).unwrap_or(false) {
                        if let Some(floor_world) = zoom_out_floor_world_out.or(prev_visible_world) {
                            let visible_floor_in_item =
                                floor_world.intersect(item_rect).intersect(visible_world);
                            if visible_floor_in_item.width() > 0.0
                                && visible_floor_in_item.height() > 0.0
                            {
                                let local_min =
                                    (visible_floor_in_item.min.to_vec2() - it.offset) / it.scale;
                                let local_max =
                                    (visible_floor_in_item.max.to_vec2() - it.offset) / it.scale;
                                let visible_local_floor = egui::Rect::from_min_max(
                                    local_min.to_pos2(),
                                    local_max.to_pos2(),
                                );
                                if let Some(level_info) = it.dataset.levels.get(floor_level) {
                                    let needed_tiles = tiles_needed_lvl0_rect(
                                        visible_local_floor,
                                        level_info,
                                        &it.dataset.dims,
                                        1,
                                    );
                                    for key in needed_tiles.into_iter().take(512) {
                                        let screen_rect = tile_screen_rect_mosaic(
                                            &self.camera,
                                            it,
                                            level_info,
                                            &key,
                                            viewport,
                                        );
                                        if screen_rect.intersects(viewport) {
                                            draws_out.push(MosaicTileDraw {
                                                dataset_id: it.id,
                                                level: floor_level,
                                                tile_y: key.tile_y,
                                                tile_x: key.tile_x,
                                                screen_rect,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        (
            target_out,
            ceiling_out,
            zoom_out_floor_level_out,
            zoom_out_floor_until_out,
            zoom_out_floor_world_out,
        )
    }

    pub(super) fn apply_sort_and_layout(&mut self) {
        // Sorting/grouping is allowed to reorder items freely, but we preserve user context by:
        // keeping the focused ROI selected if it still exists, and remapping the camera center as
        // a fraction of the old mosaic bounds into the new bounds after layout.
        let keep_focused = self.focused_core_id;
        let sort_by = self.sort_by.clone();
        let secondary = if self.sort_secondary_enabled {
            Some(self.sort_by_secondary.clone())
        } else {
            None
        };
        let group_by = self.group_by.clone();
        self.items.sort_by(|a, b| {
            if !group_by.is_empty() {
                let ga = group_label_for_item(a, &group_by);
                let gb = group_label_for_item(b, &group_by);
                let cg = cmp_sort_key(&ga, &gb);
                if cg != std::cmp::Ordering::Equal {
                    return cg;
                }
            }
            let c0 = cmp_sort_key(
                &sort_value_for_item(a, &sort_by),
                &sort_value_for_item(b, &sort_by),
            );
            if c0 != std::cmp::Ordering::Equal {
                return c0;
            }
            if let Some(sec) = secondary.as_deref() {
                let c1 = cmp_sort_key(&sort_value_for_item(a, sec), &sort_value_for_item(b, sec));
                if c1 != std::cmp::Ordering::Equal {
                    return c1;
                }
            }
            a.sample_id.cmp(&b.sample_id)
        });

        self.focused_core_id = keep_focused
            .filter(|id| self.items.iter().any(|it| it.id == *id))
            .or_else(|| self.items.first().map(|it| it.id));

        // Preserve camera center fraction within mosaic bounds.
        let old = self.mosaic_bounds;
        let fx = if old.width() > 0.0 {
            ((self.camera.center_world_lvl0.x - old.min.x) / old.width()).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let fy = if old.height() > 0.0 {
            ((self.camera.center_world_lvl0.y - old.min.y) / old.height()).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let (bounds, blocks) = layout_items_grouped(
            &mut self.items,
            self.grid_cols,
            self.grid_cell_w,
            self.grid_cell_h,
            self.grid_pad,
            (!self.group_by.is_empty()).then_some(self.group_by.as_str()),
            self.group_gap.max(0.0),
            self.layout_mode,
        );
        self.mosaic_bounds = bounds;
        self.group_blocks = blocks;
        let newb = self.mosaic_bounds;
        self.camera.center_world_lvl0 = egui::pos2(
            newb.min.x + newb.width() * fx,
            newb.min.y + newb.height() * fy,
        );
    }

    pub(super) fn draw_text_labels(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        let visible_world = visible_world_rect(&self.camera, viewport);
        let painter = ui.painter();
        let font = egui::FontId::proportional(13.0);
        let fg = egui::Color32::from_gray(240);
        let bg = egui::Color32::from_black_alpha(160);
        let line_gap = 1.0;

        for it in &self.items {
            let world_rect = item_rect(it);
            if !world_rect.intersects(visible_world) {
                continue;
            }
            let screen_min = self.camera.world_to_screen(world_rect.left_top(), viewport);
            let pos = screen_min + egui::vec2(6.0, 6.0);

            let lines = label_values_for_item(it, &self.label_columns);
            if lines.is_empty() {
                continue;
            }

            let galleys = lines
                .into_iter()
                .map(|line| painter.layout_no_wrap(line, font.clone(), fg))
                .collect::<Vec<_>>();
            let width = galleys
                .iter()
                .map(|galley| galley.size().x)
                .fold(0.0, f32::max);
            let height = galleys.iter().map(|galley| galley.size().y).sum::<f32>()
                + line_gap * galleys.len().saturating_sub(1) as f32;
            let rect = egui::Rect::from_min_size(pos, egui::vec2(width, height)).expand(2.0);
            painter.rect_filled(rect, 3.0, bg);

            let mut y = pos.y;
            for galley in galleys {
                painter.galley(egui::pos2(pos.x, y), galley.clone(), fg);
                y += galley.size().y + line_gap;
            }
        }
    }

    pub(super) fn draw_group_labels(&self, ui: &mut egui::Ui, viewport: egui::Rect) {
        if self.group_blocks.is_empty() {
            return;
        }
        let visible_world = visible_world_rect(&self.camera, viewport);
        let painter = ui.painter();
        let font = egui::FontId::proportional(15.0);
        let fg = egui::Color32::from_gray(245);
        let bg = egui::Color32::from_black_alpha(200);

        for g in &self.group_blocks {
            if !g.world_rect.intersects(visible_world) {
                continue;
            }
            let screen_min = self
                .camera
                .world_to_screen(g.world_rect.left_top(), viewport);
            let pos = screen_min + egui::vec2(8.0, 8.0);
            let galley = painter.layout_no_wrap(g.name.clone(), font.clone(), fg);
            let rect = egui::Rect::from_min_size(pos, galley.size()).expand2(egui::vec2(4.0, 3.0));
            painter.rect_filled(rect, 4.0, bg);
            painter.galley(pos, galley, fg);
        }
    }
}
