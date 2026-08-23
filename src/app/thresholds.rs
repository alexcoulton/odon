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

    pub(super) fn uses_gpu_threshold_region_preview(
        &self,
        preview: &ThresholdRegionPreview,
    ) -> bool {
        self.threshold_preview_gl.is_some() && preview.min_component_pixels <= 1
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
