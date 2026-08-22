use super::*;

impl OmeZarrViewerApp {
    pub(super) fn schedule_repaint(&self, ctx: &egui::Context) {
        if self.is_loading_scene() {
            repaint_control::request_repaint_busy(ctx);
            return;
        }

        if self.has_pending_async_ui_work() {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
    }

    pub(super) fn has_pending_async_ui_work(&self) -> bool {
        let properties_hist_active = self.show_right_panel
            && self.right_tab == RightTab::Properties
            && matches!(self.active_layer, LayerId::Channel(_));
        (properties_hist_active
            && (self.hist_dirty
                || self.hist_request_pending
                || self.hist_navigation_dirty_since.is_some()))
            || self.chanmax_pending.iter().any(|pending| *pending)
            || !self.screenshot_in_flight.is_empty()
            || self
                .annotation_layers
                .iter()
                .any(|layer| layer.has_pending_work())
    }

    pub(super) fn request_tiles_with_budget(
        &mut self,
        level: usize,
        needed: &[TileKey],
        sent: &mut usize,
        max_per_frame: usize,
    ) {
        if *sent >= max_per_frame {
            return;
        }

        if self.viewport_cpu_active_keys.is_some() {
            self.loader.activate_render_id(self.active_render_id);
        } else {
            self.loader.set_latest_render_id(self.active_render_id);
        }
        let channels = self.render_channels_for_request(level);
        for key in needed {
            if *sent >= max_per_frame {
                break;
            }
            if !self.cache.mark_in_flight(*key) {
                continue;
            }
            if self.view_plane_is_xy()
                && let Some(level_info) = self.dataset.levels.get(level)
            {
                if let Some(tile) = self.pinned_levels.try_get_composited_tile(
                    *key,
                    &channels,
                    &self.dataset.dims,
                    level_info,
                ) {
                    self.cache.cancel_in_flight(key);
                    self.pending.push(tile);
                    *sent += 1;
                    continue;
                }
                if let Some(tile) =
                    self.try_get_composited_tile_from_pinned_finer(*key, &channels, level_info)
                {
                    self.cache.cancel_in_flight(key);
                    self.pending.push(tile);
                    *sent += 1;
                    continue;
                }
            }
            let _ = self.loader.tx.send(TileRequest {
                key: *key,
                channels: channels.clone(),
            });
            *sent += 1;
        }
    }

    pub(super) fn request_raw_tiles_with_budget(
        &mut self,
        tiles_gl: &TilesGl,
        raw_tx: &crossbeam_channel::Sender<RawTileRequest>,
        level: usize,
        needed: &[TileKey],
        render_channels: &[RenderChannel],
        sent: &mut usize,
        max_per_frame: usize,
    ) {
        if *sent >= max_per_frame {
            return;
        }

        for key in needed {
            if *sent >= max_per_frame {
                break;
            }
            for ch in render_channels {
                if *sent >= max_per_frame {
                    break;
                }
                let raw_key = RawTileKey {
                    view: key.view,
                    level,
                    tile_y: key.tile_y,
                    tile_x: key.tile_x,
                    channel: ch.index,
                };
                if !tiles_gl.mark_in_flight(raw_key) {
                    continue;
                }
                if self.view_plane_is_xy()
                    && let Some(level_info) = self.dataset.levels.get(level)
                {
                    if let Some(resp) =
                        self.pinned_levels
                            .try_get_raw_tile(raw_key, &self.dataset.dims, level_info)
                    {
                        tiles_gl.insert_pending(resp);
                        *sent += 1;
                        continue;
                    }
                    if let Some(resp) = self.try_get_raw_tile_from_pinned_finer(raw_key, level_info)
                    {
                        tiles_gl.insert_pending(resp);
                        *sent += 1;
                        continue;
                    }
                }
                let _ = raw_tx.send(RawTileRequest { key: raw_key });
                *sent += 1;
            }
        }
    }

    pub(super) fn prefetch_spec(&self, visible_count: usize) -> Option<(i64, usize)> {
        match self.tile_prefetch_mode {
            TilePrefetchMode::Off => None,
            TilePrefetchMode::TargetHalo | TilePrefetchMode::TargetAndFinerHalo => {
                let (small_pad, small_budget, medium_pad, medium_budget) =
                    match self.tile_prefetch_aggressiveness {
                        TilePrefetchAggressiveness::Conservative => (1, 16usize, 1, 8usize),
                        TilePrefetchAggressiveness::Balanced => (2, 48usize, 1, 24usize),
                        TilePrefetchAggressiveness::Aggressive => (2, 96usize, 2, 48usize),
                    };
                if visible_count <= 16 {
                    Some((small_pad, small_budget))
                } else if visible_count <= 48 {
                    Some((medium_pad, medium_budget))
                } else {
                    None
                }
            }
        }
    }

    pub(super) fn prefetch_keys_for_level(
        &self,
        level: usize,
        level_info: &crate::data::ome::LevelInfo,
        visible_world_tiles: egui::Rect,
        visible_needed: &[TileKey],
    ) -> Vec<TileKey> {
        let visible_count = visible_needed.len();
        let Some((pad_tiles, prefetch_budget)) = self.prefetch_spec(visible_count) else {
            return Vec::new();
        };

        // Prefetch only a halo around the already-visible set. This keeps ahead of short pans
        // without letting speculative IO dominate the queue when the viewport is large.
        let visible_set: HashSet<(u64, u64)> = visible_needed
            .iter()
            .map(|key| (key.tile_y, key.tile_x))
            .collect();
        let Some(level0) = self.dataset.levels.first() else {
            return Vec::new();
        };
        let Some(axes) = self.display_axes() else {
            return Vec::new();
        };
        let mut prefetch: Vec<TileKey> = tiles_needed_lvl0_rect_for_axes(
            visible_world_tiles,
            level0,
            level_info,
            axes,
            pad_tiles,
        )
        .into_iter()
        .filter_map(|coord| {
            (!visible_set.contains(&(coord.tile_y, coord.tile_x))).then_some(TileKey {
                render_id: self.active_render_id,
                view: self.displayed_view_selection(),
                level,
                tile_y: coord.tile_y,
                tile_x: coord.tile_x,
            })
        })
        .collect();
        self.sort_tile_keys_near_center(level_info, &mut prefetch);
        prefetch.truncate(prefetch_budget);
        prefetch
    }

    pub(super) fn render_channels_for_request(&self, _level: usize) -> Vec<RenderChannel> {
        let mut out = Vec::new();
        let groups = self.current_layer_groups();
        let order = if self.channel_layer_order.len() == self.channels.len() {
            self.channel_layer_order.clone()
        } else {
            (0..self.channels.len()).collect()
        };

        for idx in order {
            let Some(ch) = self.channels.get(idx) else {
                continue;
            };
            if !ch.visible {
                continue;
            }
            let rgb =
                layer_groups::effective_channel_color_rgb(&groups, ch.name.as_str(), ch.color_rgb);
            out.push(RenderChannel {
                index: ch.index as u64,
                color_rgb: [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                ],
                window: ch.window.unwrap_or((0.0, 65535.0)),
            });
        }
        out
    }

    pub(super) fn tile_rects(
        &self,
        key: &TileKey,
        viewport: egui::Rect,
        level_info: &crate::data::ome::LevelInfo,
    ) -> (egui::Rect, egui::Rect) {
        let level0 = self
            .dataset
            .levels
            .first()
            .expect("dataset should have a level 0");
        let axes = display_axes_for_mode(&self.dataset.dims, key.view.mode)
            .expect("view mode should be supported");
        let y_dim = axes.vertical;
        let x_dim = axes.horizontal;
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;

        let y0 = key.tile_y as f32 * chunk_y;
        let x0 = key.tile_x as f32 * chunk_x;
        let y1 = (y0 + chunk_y).min(level_info.shape[y_dim] as f32);
        let x1 = (x0 + chunk_x).min(level_info.shape[x_dim] as f32);

        let (downsample_y, downsample_x) =
            display_downsample(&self.dataset.dims, level0, level_info, key.view.mode).unwrap_or((
                level_info.downsample.max(1e-6),
                level_info.downsample.max(1e-6),
            ));
        let world_min =
            self.primary_image_local_to_world(egui::pos2(x0 * downsample_x, y0 * downsample_y));
        let world_max =
            self.primary_image_local_to_world(egui::pos2(x1 * downsample_x, y1 * downsample_y));
        let world_rect = egui::Rect::from_min_max(world_min, world_max);

        let screen_min = self.camera.world_to_screen(world_min, viewport);
        let screen_max = self.camera.world_to_screen(world_max, viewport);
        let screen_rect = egui::Rect::from_min_max(screen_min, screen_max);

        (world_rect, screen_rect)
    }

    pub(super) fn get_tile_texture(&mut self, key: &TileKey) -> Option<egui::TextureHandle> {
        if let Some(tex) = self.cache.get(key).cloned() {
            return Some(tex);
        }

        if let Some(view) = self
            .fallback_view_selection()
            .filter(|view| *view != key.view)
        {
            let fallback_key = TileKey {
                render_id: self.active_render_id,
                view,
                level: key.level,
                tile_y: key.tile_y,
                tile_x: key.tile_x,
            };
            if let Some(tex) = self.cache.get(&fallback_key).cloned() {
                return Some(tex);
            }
        }

        if let Some(prev) = self.previous_render_id {
            let prev_key = TileKey {
                render_id: prev,
                view: key.view,
                level: key.level,
                tile_y: key.tile_y,
                tile_x: key.tile_x,
            };
            if let Some(tex) = self.cache.get(&prev_key).cloned() {
                return Some(tex);
            }
            if let Some(view) = self
                .previous_view_selection
                .filter(|view| *view != key.view)
            {
                let prev_key = TileKey {
                    render_id: prev,
                    view,
                    level: key.level,
                    tile_y: key.tile_y,
                    tile_x: key.tile_x,
                };
                if let Some(tex) = self.cache.get(&prev_key).cloned() {
                    return Some(tex);
                }
            }
        }

        None
    }

    pub(super) fn bump_render_id(&mut self) {
        let new_id = self.compute_render_id();
        if new_id != self.active_render_id {
            self.previous_render_id = Some(self.active_render_id);
            self.previous_render_smooth_pixels = Some(self.active_render_smooth_pixels);
            self.previous_view_selection = Some(self.last_render_view_selection);
            self.active_render_id = new_id;
            self.active_render_smooth_pixels = self.smooth_pixels;
            self.last_render_view_selection = self.committed_view_selection();
        }
    }

    pub(super) fn compute_render_id(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.dataset.source.hash(&mut hasher);
        self.active_view_selection().hash(&mut hasher);
        self.channel_layer_order.hash(&mut hasher);
        self.smooth_pixels.hash(&mut hasher);
        let groups = self.current_layer_groups();
        for &idx in &self.channel_layer_order {
            if let Some(ch) = self.channels.get(idx) {
                ch.index.hash(&mut hasher);
                ch.visible.hash(&mut hasher);
                let rgb = layer_groups::effective_channel_color_rgb(
                    &groups,
                    ch.name.as_str(),
                    ch.color_rgb,
                );
                rgb.hash(&mut hasher);
                let (w0, w1) = ch.window.unwrap_or((0.0, 65535.0));
                w0.to_bits().hash(&mut hasher);
                w1.to_bits().hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    pub(super) fn drain_histogram(&mut self) {
        while let Ok(msg) = self.hist_loader.rx.try_recv() {
            if msg.request_id != self.hist_request_id {
                continue;
            }
            self.hist_request_pending = false;
            self.hist = Some(msg);
        }
    }

    pub(super) fn maybe_request_histogram(&mut self, ctx: &egui::Context) {
        if !self.hist_dirty {
            return;
        }
        if self.channels.is_empty() {
            self.hist_dirty = false;
            self.hist_request_pending = false;
            self.hist_navigation_dirty_since = None;
            self.hist = None;
            return;
        }
        let Some(viewport) = self.last_canvas_rect else {
            return;
        };
        let now = Instant::now();
        if let Some(dirty_since) = self.hist_navigation_dirty_since {
            let settled_for = now.duration_since(dirty_since);
            if settled_for < HISTOGRAM_NAVIGATION_DEBOUNCE {
                ctx.request_repaint_after(HISTOGRAM_NAVIGATION_DEBOUNCE - settled_for);
                return;
            }
            if self.hist_request_pending {
                ctx.request_repaint_after(Duration::from_millis(50));
                return;
            }
        }
        let elapsed = now.duration_since(self.hist_last_sent);
        if elapsed < HISTOGRAM_REQUEST_THROTTLE {
            ctx.request_repaint_after(HISTOGRAM_REQUEST_THROTTLE - elapsed);
            return;
        }

        let level = self.dataset.levels.len().saturating_sub(1);
        let level_info = &self.dataset.levels[level];
        let Some(level0) = self.dataset.levels.first() else {
            return;
        };
        let (downsample_y, downsample_x) =
            display_downsample(&self.dataset.dims, level0, level_info, self.view_plane_mode)
                .unwrap_or((
                    level_info.downsample.max(1.0),
                    level_info.downsample.max(1.0),
                ));

        let visible_local =
            self.primary_image_world_rect_to_local(self.visible_world_rect(viewport));
        let mut y0 = (visible_local.min.y / downsample_y.max(1e-6))
            .floor()
            .max(0.0) as u64;
        let mut y1 = (visible_local.max.y / downsample_y.max(1e-6))
            .ceil()
            .max(0.0) as u64;
        let mut x0 = (visible_local.min.x / downsample_x.max(1e-6))
            .floor()
            .max(0.0) as u64;
        let mut x1 = (visible_local.max.x / downsample_x.max(1e-6))
            .ceil()
            .max(0.0) as u64;

        let Some(axes) = self.display_axes() else {
            return;
        };
        let y_dim = axes.vertical;
        let x_dim = axes.horizontal;
        let shape_y = *level_info.shape.get(y_dim).unwrap_or(&0);
        let shape_x = *level_info.shape.get(x_dim).unwrap_or(&0);
        y0 = y0.min(shape_y);
        y1 = y1.min(shape_y).max(y0);
        x0 = x0.min(shape_x);
        x1 = x1.min(shape_x).max(x0);

        // Hard cap the sampled area to keep it responsive.
        let max_dim = 1024u64;
        if y1.saturating_sub(y0) > max_dim {
            let cy = (y0 + y1) / 2;
            y0 = cy.saturating_sub(max_dim / 2);
            y1 = (y0 + max_dim).min(shape_y);
        }
        if x1.saturating_sub(x0) > max_dim {
            let cx = (x0 + x1) / 2;
            x0 = cx.saturating_sub(max_dim / 2);
            x1 = (x0 + max_dim).min(shape_x);
        }

        self.hist_request_id = self.hist_request_id.wrapping_add(1);
        let req = crate::imaging::histogram::HistogramRequest {
            request_id: self.hist_request_id,
            view: self.active_view_selection(),
            level,
            channel: self.selected_channel as u64,
            y0,
            y1,
            x0,
            x1,
            bins: 256,
            abs_max: self.dataset.abs_max.max(1.0),
        };
        let _ = self.hist_loader.tx.send(req);
        self.hist_last_sent = Instant::now();
        self.hist_request_pending = true;
        self.hist_dirty = false;
        self.hist_navigation_dirty_since = None;
    }

    pub(super) fn drain_tiles(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.loader.rx.try_recv() {
            match msg {
                TileWorkerResponse::Tile(msg) => {
                    self.cache.cancel_in_flight(&msg.key);

                    // Loader responses can outlive the frame that requested them. Accept the
                    // current render epoch and, briefly, the immediately previous one so the draw
                    // path can finish a coarse->fine transition without showing obviously stale
                    // tiles from unrelated dataset/tool states.
                    if !self.cpu_render_id_is_current(msg.key.render_id) {
                        continue;
                    }
                    self.pending.push(msg);
                }
                TileWorkerResponse::Failed { key, error } => {
                    self.cache.cancel_in_flight(&key);
                    crate::log_warn!("tile load failed for {:?}: {}", key, error);
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
            }
        }

        if self.pending.is_empty() {
            return;
        }

        let pending = self.pending.drain(..).collect::<Vec<_>>();
        for TileResponse {
            key,
            width,
            height,
            rgba,
        } in pending
        {
            let image = egui::ColorImage::from_rgba_unmultiplied([width, height], &rgba);
            let options = if self.smooth_pixels_for_render_id(key.render_id) {
                egui::TextureOptions::LINEAR
            } else {
                egui::TextureOptions::NEAREST
            };
            let tex = ctx.load_texture(
                format!(
                    "tile-{}-{:?}-{}-{}-{}-{}",
                    key.render_id,
                    key.view.mode,
                    key.view.slice_level0,
                    key.level,
                    key.tile_y,
                    key.tile_x
                ),
                image,
                options,
            );
            self.cache.put(key, tex);
        }
    }

    pub(super) fn drain_raw_tiles(&mut self) {
        let (Some(loader), Some(tiles_gl)) = (self.raw_loader.as_ref(), self.tiles_gl.as_ref())
        else {
            return;
        };
        while let Ok(msg) = loader.rx.try_recv() {
            match msg {
                RawTileWorkerResponse::Tile(msg) => tiles_gl.insert_pending(msg),
                RawTileWorkerResponse::Failed { key, error } => {
                    tiles_gl.cancel_in_flight(&key);
                    crate::log_warn!("raw tile load failed for {:?}: {}", key, error);
                }
            }
        }
    }

    pub(super) fn drain_label_tiles(&mut self) {
        let (Some(loader), Some(labels_gl)) = (self.label_loader.as_ref(), self.labels_gl.as_ref())
        else {
            return;
        };
        while let Ok(msg) = loader.rx.try_recv() {
            labels_gl.insert_pending(msg);
        }
    }

    pub(super) fn drain_screenshots(&mut self) {
        while let Ok(resp) = self.screenshot_worker.rx.try_recv() {
            match resp {
                crate::app_support::screenshot::ScreenshotWorkerResp::Saved {
                    id,
                    path,
                    result,
                } => {
                    self.screenshot_in_flight.remove(&id);
                    match result {
                        Ok(()) => {
                            self.set_status(format!(
                                "Saved screenshot -> {}",
                                path.to_string_lossy()
                            ));
                        }
                        Err(err) => {
                            self.set_status(format!("Save screenshot failed: {err}"));
                        }
                    }
                }
            }
        }
    }
}
