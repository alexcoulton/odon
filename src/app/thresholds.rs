use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ensure_threshold_region_full_level_default(&mut self) {
        if self.threshold_region_full_level < self.dataset.levels.len()
            && self
                .threshold_region_level_pixel_count(self.threshold_region_full_level)
                .is_some_and(|pixels| pixels <= THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS)
        {
            return;
        }
        if let Some(level) = default_threshold_full_level(
            &self.dataset.levels,
            self.dataset.dims.y,
            self.dataset.dims.x,
            THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS,
        ) {
            self.threshold_region_full_level = level;
        } else {
            self.threshold_region_full_level = self.dataset.levels.len().saturating_sub(1);
        }
    }

    pub(super) fn threshold_region_level_pixel_count(&self, level_index: usize) -> Option<u64> {
        let level = self.dataset.levels.get(level_index)?;
        let (width, height) =
            threshold_level_size(level, self.dataset.dims.y, self.dataset.dims.x)?;
        threshold_region_pixel_count(width, height)
    }

    pub(super) fn threshold_region_full_level_summary(
        &self,
        level_index: usize,
    ) -> Option<(u64, u64, u64)> {
        let level = self.dataset.levels.get(level_index)?;
        let (width, height) =
            threshold_level_size(level, self.dataset.dims.y, self.dataset.dims.x)?;
        let pixels = threshold_region_pixel_count(width, height)?;
        Some((width, height, pixels))
    }

    pub(super) fn threshold_region_extent(
        &self,
        viewport: Option<egui::Rect>,
        ch_idx: usize,
    ) -> anyhow::Result<ThresholdRegionExtent> {
        match self.threshold_region_scope {
            ThresholdRegionScope::VisibleRegion => {
                let viewport =
                    viewport.ok_or_else(|| anyhow::anyhow!("canvas viewport unavailable"))?;
                let level_index = self.choose_level();
                let Some(level_info) = self.dataset.levels.get(level_index) else {
                    anyhow::bail!("invalid image level");
                };
                let visible_rect_lvl0 =
                    self.selected_channel_visible_data_rect_lvl0(viewport, ch_idx);
                if visible_rect_lvl0.width() <= 0.0 || visible_rect_lvl0.height() <= 0.0 {
                    anyhow::bail!("no visible region intersects the active channel");
                }
                let downsample = level_info.downsample.max(1e-6);
                let y_dim = self.dataset.dims.y;
                let x_dim = self.dataset.dims.x;
                let x0 = (visible_rect_lvl0.left() / downsample).floor().max(0.0) as u64;
                let y0 = (visible_rect_lvl0.top() / downsample).floor().max(0.0) as u64;
                let x1 = (visible_rect_lvl0.right() / downsample)
                    .ceil()
                    .min(level_info.shape[x_dim] as f32) as u64;
                let y1 = (visible_rect_lvl0.bottom() / downsample)
                    .ceil()
                    .min(level_info.shape[y_dim] as f32) as u64;
                if x1 <= x0 || y1 <= y0 {
                    anyhow::bail!("visible region is empty at this level");
                }
                let _ = threshold_region_pixel_count(x1 - x0, y1 - y0)
                    .ok_or_else(|| anyhow::anyhow!("threshold region is too large"))?;
                Ok(ThresholdRegionExtent {
                    scope: ThresholdRegionScope::VisibleRegion,
                    level_index,
                    x0,
                    y0,
                    x1,
                    y1,
                })
            }
            ThresholdRegionScope::EntireImage => {
                let level_index = self.threshold_region_full_level;
                let Some(level_info) = self.dataset.levels.get(level_index) else {
                    anyhow::bail!("invalid image level");
                };
                let (width, height) =
                    threshold_level_size(level_info, self.dataset.dims.y, self.dataset.dims.x)
                        .ok_or_else(|| anyhow::anyhow!("invalid image level shape"))?;
                let pixel_count = threshold_region_pixel_count(width, height)
                    .ok_or_else(|| anyhow::anyhow!("whole-image threshold size is too large"))?;
                if pixel_count > THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS {
                    anyhow::bail!(
                        "Whole-image thresholding at this level would read {} pixels; choose a coarser level.",
                        pixel_count
                    );
                }
                Ok(ThresholdRegionExtent {
                    scope: ThresholdRegionScope::EntireImage,
                    level_index,
                    x0: 0,
                    y0: 0,
                    x1: width,
                    y1: height,
                })
            }
        }
    }

    pub(super) fn uses_gpu_threshold_region_preview(
        &self,
        preview: &ThresholdRegionPreview,
    ) -> bool {
        self.threshold_preview_gl.is_some() && preview.min_component_pixels <= 1
    }

    pub(super) fn start_threshold_region_preview(
        &mut self,
        ctx: &egui::Context,
    ) -> anyhow::Result<()> {
        let ch_idx = self
            .selected_channel
            .min(self.channels.len().saturating_sub(1));
        let extent = self.threshold_region_extent(self.last_canvas_rect, ch_idx)?;
        let Some(level_info) = self.dataset.levels.get(extent.level_index) else {
            anyhow::bail!("invalid image level");
        };

        let downsample = level_info.downsample.max(1e-6);
        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let level0 = self
            .dataset
            .levels
            .first()
            .context("dataset has no image levels")?;
        let channel_index = self
            .channels
            .get(ch_idx)
            .map(|ch| ch.index as u64)
            .unwrap_or(0);
        let plane = plane_selection_for_z(&self.dataset.dims, level0, self.active_z_level0());
        let ranges = image_subset_ranges(
            &self.dataset.dims,
            level0,
            level_info,
            Some(channel_index),
            extent.y0..extent.y1,
            extent.x0..extent.x1,
            plane,
        );

        let zarr_path = format!("/{}", level_info.path.trim_start_matches('/'));
        let array = Array::open(self.store.clone(), &zarr_path).with_context(|| {
            format!("failed to open image array for level {}", level_info.index)
        })?;
        let subset = ArraySubset::new_with_ranges(&ranges);
        let data = retrieve_image_subset_u16(&array, &subset, &level_info.dtype)
            .context("failed to read active-channel viewport subset")?;
        let plane = crate::render::array_dims::squeeze_to_yx(data, y_dim, x_dim)
            .context("unexpected dimensionality for threshold region subset")?;

        let threshold = self
            .channels
            .get(ch_idx)
            .and_then(|channel| channel.window)
            .map(|(lo, _)| lo.round().clamp(0.0, u16::MAX as f32) as u16)
            .unwrap_or(0);
        let channel_name = self
            .channels
            .get(ch_idx)
            .map(|channel| channel.name.clone())
            .unwrap_or_else(|| format!("Channel {ch_idx}"));
        let raw_values = Arc::new(plane.iter().copied().collect::<Vec<_>>());
        let generation = self.threshold_region_preview_generation;
        self.threshold_region_preview_generation =
            self.threshold_region_preview_generation.wrapping_add(1);
        let mut preview = ThresholdRegionPreview {
            generation,
            channel_index: ch_idx,
            channel_name,
            scope: extent.scope,
            level_index: level_info.index,
            downsample,
            x0: extent.x0,
            y0: extent.y0,
            plane,
            raw_values,
            threshold,
            min_component_pixels: self.threshold_region_min_pixels.max(1),
            mask: ThresholdRegionMask {
                width: 0,
                height: 0,
                included: Vec::new(),
            },
            texture: None,
        };
        if !self.uses_gpu_threshold_region_preview(&preview) {
            Self::recompute_threshold_region_preview_cpu_data(ctx, &mut preview);
        }
        self.threshold_region_status = Self::threshold_region_preview_status_message(
            &preview,
            self.uses_gpu_threshold_region_preview(&preview),
        );
        self.threshold_region_preview = Some(preview);
        Ok(())
    }

    pub(super) fn recompute_threshold_region_preview(&mut self, ctx: &egui::Context) {
        let gpu_available = self.threshold_preview_gl.is_some();
        if let Some(preview) = self.threshold_region_preview.as_mut() {
            preview.min_component_pixels = self.threshold_region_min_pixels.max(1);
            let uses_gpu = gpu_available && preview.min_component_pixels <= 1;
            if uses_gpu {
                preview.mask = ThresholdRegionMask {
                    width: 0,
                    height: 0,
                    included: Vec::new(),
                };
                preview.texture = None;
            } else {
                Self::recompute_threshold_region_preview_cpu_data(ctx, preview);
            }
            self.threshold_region_status =
                Self::threshold_region_preview_status_message(preview, uses_gpu);
        }
    }

    pub(super) fn threshold_region_preview_status_message(
        preview: &ThresholdRegionPreview,
        uses_gpu: bool,
    ) -> String {
        if uses_gpu {
            format!(
                "Previewing {} {} at level {} on the GPU (threshold only; min component filtering is applied on Apply).",
                preview.channel_name,
                preview.scope.label(),
                preview.level_index
            )
        } else {
            let included = preview
                .mask
                .included
                .iter()
                .filter(|included| **included)
                .count();
            format!(
                "Preview: {} pixels selected in {} {} at level {}.",
                included,
                preview.channel_name,
                preview.scope.label(),
                preview.level_index
            )
        }
    }

    pub(super) fn recompute_threshold_region_preview_cpu_data(
        ctx: &egui::Context,
        preview: &mut ThresholdRegionPreview,
    ) {
        preview.mask = extract_threshold_region_mask(
            &preview.plane,
            preview.threshold,
            preview.min_component_pixels,
        );
        let mut rgba = vec![0u8; preview.mask.width * preview.mask.height * 4];
        for (idx, included) in preview.mask.included.iter().copied().enumerate() {
            if !included {
                continue;
            }
            let base = idx * 4;
            rgba[base] = 255;
            rgba[base + 1] = 210;
            rgba[base + 2] = 80;
            rgba[base + 3] = 120;
        }
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [preview.mask.width, preview.mask.height],
            &rgba,
        );
        let options = egui::TextureOptions::NEAREST;
        if let Some(texture) = preview.texture.as_mut() {
            texture.set(image, options);
        } else {
            preview.texture = Some(ctx.load_texture(
                format!(
                    "threshold-preview-{}-{}-{}-{}",
                    preview.channel_index, preview.level_index, preview.x0, preview.y0
                ),
                image,
                options,
            ));
        }
    }

    pub(super) fn threshold_mask_raster_display_cache(
        &self,
        mask: &ThresholdRegionMask,
        generation: u64,
        channel_index: usize,
        x0: u64,
        y0: u64,
        downsample: f32,
    ) -> MaskRasterDisplayCache {
        let width = mask.width;
        let height = mask.height;
        let x0 = x0 as f32 * downsample;
        let y0 = y0 as f32 * downsample;
        let x1 = x0 + width as f32 * downsample;
        let y1 = y0 + height as f32 * downsample;
        let corners_world = [
            self.selected_channel_local_to_world(channel_index, egui::pos2(x0, y0)),
            self.selected_channel_local_to_world(channel_index, egui::pos2(x1, y0)),
            self.selected_channel_local_to_world(channel_index, egui::pos2(x1, y1)),
            self.selected_channel_local_to_world(channel_index, egui::pos2(x0, y1)),
        ];
        let values = mask
            .included
            .iter()
            .copied()
            .map(|included| if included { u16::MAX } else { 0 })
            .collect::<Vec<_>>();
        MaskRasterDisplayCache {
            generation,
            width,
            height,
            values: Arc::new(values),
            corners_world,
        }
    }

    pub(super) fn create_threshold_mask_from_preview(&mut self) -> anyhow::Result<usize> {
        let (mask, polygons, channel_index, channel_name, scope, level_index, x0, y0, downsample) = {
            let Some(preview) = self.threshold_region_preview.as_ref() else {
                anyhow::bail!("no threshold preview is active");
            };
            let mask = extract_threshold_region_mask(
                &preview.plane,
                preview.threshold,
                preview.min_component_pixels,
            );
            let polygons = threshold_region_mask_to_polygons(&mask);
            if polygons.is_empty() {
                anyhow::bail!("no visible regions found above the current threshold");
            }
            (
                mask,
                polygons,
                preview.channel_index,
                preview.channel_name.clone(),
                preview.scope,
                preview.level_index,
                preview.x0,
                preview.y0,
                preview.downsample,
            )
        };
        let raster_generation = self.threshold_region_preview_generation;
        self.threshold_region_preview_generation =
            self.threshold_region_preview_generation.wrapping_add(1);
        let raster_display = self.threshold_mask_raster_display_cache(
            &mask,
            raster_generation,
            channel_index,
            x0,
            y0,
            downsample,
        );
        self.push_mask_undo_snapshot();
        let layer_id = self.create_editable_mask_layer(Some(format!(
            "Threshold {channel_name} {} level {level_index}",
            scope.layer_label()
        )));
        let mut created = 0usize;
        let world_polygons = polygons
            .into_iter()
            .map(|polygon| {
                polygon
                    .into_iter()
                    .map(|point| {
                        let local_lvl0 = egui::pos2(
                            (x0 as f32 + point.x) * downsample,
                            (y0 as f32 + point.y) * downsample,
                        );
                        self.selected_channel_local_to_world(channel_index, local_lvl0)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if let Some(layer) = self
            .mask_layers
            .iter_mut()
            .find(|layer| layer.id == layer_id)
        {
            layer.offset_world = egui::Vec2::ZERO;
            layer.polygons_world.clear();
            for polygon in world_polygons {
                layer.add_closed_polygon(polygon);
            }
            layer.display_mode = MaskDisplayMode::FilledPreview;
            layer.raster_display = Some(raster_display);
            layer.visible = true;
            created = layer.polygons_world.len();
            self.mark_mask_layers_project_dirty();
        }

        self.threshold_region_preview = None;
        self.active_layer = LayerId::Mask(layer_id);
        self.threshold_region_status = format!(
            "Created {created} threshold region(s) from {channel_name} {} at level {level_index}.",
            scope.label()
        );
        self.rebuild_layer_orders();
        self.bump_render_id();
        Ok(created)
    }

    pub(super) fn draw_threshold_region_preview(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let Some(preview) = self.threshold_region_preview.as_ref() else {
            return;
        };
        let width = preview.plane.dim().1;
        let height = preview.plane.dim().0;
        if width == 0 || height == 0 {
            return;
        }

        let x0 = preview.x0 as f32 * preview.downsample;
        let y0 = preview.y0 as f32 * preview.downsample;
        let x1 = (preview.x0 as f32 + width as f32) * preview.downsample;
        let y1 = (preview.y0 as f32 + height as f32) * preview.downsample;
        let corners_world = [
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x1, y0)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x1, y1)),
            self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y1)),
        ];
        let corners_screen = corners_world.map(|point| self.camera.world_to_screen(point, rect));
        let uses_gpu = self.uses_gpu_threshold_region_preview(preview);
        if uses_gpu {
            if let Some(renderer) = self.threshold_preview_gl.clone() {
                let data = ThresholdPreviewGlDrawData {
                    generation: preview.generation,
                    width,
                    height,
                    values: preview.raw_values.clone(),
                };
                let params = ThresholdPreviewGlDrawParams {
                    visible: true,
                    quad_screen: corners_screen,
                    threshold_u16: preview.threshold,
                    tint: egui::Color32::from_rgba_unmultiplied(255, 210, 80, 120),
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
            let Some(texture) = preview.texture.as_ref() else {
                return;
            };
            let mut mesh = egui::Mesh::with_texture(texture.id());
            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[0],
                uv: egui::pos2(0.0, 0.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[1],
                uv: egui::pos2(1.0, 0.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[2],
                uv: egui::pos2(1.0, 1.0),
                color: egui::Color32::WHITE,
            });
            mesh.vertices.push(egui::epaint::Vertex {
                pos: corners_screen[3],
                uv: egui::pos2(0.0, 1.0),
                color: egui::Color32::WHITE,
            });
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            ui.painter().add(egui::Shape::mesh(mesh));
        }

        ui.painter().add(egui::Shape::closed_line(
            vec![
                corners_screen[0],
                corners_screen[1],
                corners_screen[2],
                corners_screen[3],
            ],
            egui::Stroke::new(
                1.5,
                egui::Color32::from_rgba_unmultiplied(255, 210, 80, 220),
            ),
        ));

        let pixel_step_x_screen = {
            let p0 =
                self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0));
            let p1 = self.selected_channel_local_to_world(
                preview.channel_index,
                egui::pos2(x0 + preview.downsample, y0),
            );
            self.camera
                .world_to_screen(p0, rect)
                .distance(self.camera.world_to_screen(p1, rect))
        };
        let pixel_step_y_screen = {
            let p0 =
                self.selected_channel_local_to_world(preview.channel_index, egui::pos2(x0, y0));
            let p1 = self.selected_channel_local_to_world(
                preview.channel_index,
                egui::pos2(x0, y0 + preview.downsample),
            );
            self.camera
                .world_to_screen(p0, rect)
                .distance(self.camera.world_to_screen(p1, rect))
        };
        let show_grid = pixel_step_x_screen.max(pixel_step_y_screen) >= 12.0
            && width.saturating_add(height) <= 2048;
        if show_grid {
            let grid_stroke = egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 72),
            );
            for x in 0..=width {
                let local_x = x0 + x as f32 * preview.downsample;
                let p0 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(local_x, y0),
                );
                let p1 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(local_x, y1),
                );
                ui.painter().line_segment(
                    [
                        self.camera.world_to_screen(p0, rect),
                        self.camera.world_to_screen(p1, rect),
                    ],
                    grid_stroke,
                );
            }
            for y in 0..=height {
                let local_y = y0 + y as f32 * preview.downsample;
                let p0 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(x0, local_y),
                );
                let p1 = self.selected_channel_local_to_world(
                    preview.channel_index,
                    egui::pos2(x1, local_y),
                );
                ui.painter().line_segment(
                    [
                        self.camera.world_to_screen(p0, rect),
                        self.camera.world_to_screen(p1, rect),
                    ],
                    grid_stroke,
                );
            }
        }
    }
}
