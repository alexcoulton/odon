use super::*;

impl OmeZarrViewerApp {
    pub fn apply_control_actor_screenshot_preferences(
        &mut self,
        preferences: &odon::model::ScreenshotPreferences,
    ) {
        self.screenshot_output_dir = preferences.output_dir().map(Path::to_path_buf);
        self.screenshot_settings = ScreenshotSettings {
            include_scale_bar: preferences.include_scale_bar(),
            include_legend: preferences.include_legend(),
            scale_bar_scale: preferences.scale_bar_scale(),
            legend_scale: preferences.legend_scale(),
        };
    }

    pub fn open_screenshot_settings(&mut self) {
        self.screenshot_settings_open = true;
    }

    pub fn set_auto_contrast_settings(&mut self, settings: AutoContrastSettings) {
        self.auto_contrast_settings = settings.normalized();
    }

    pub fn set_fast_object_rendering(&mut self, enabled: bool) {
        self.fast_object_rendering = enabled;
        let mut changed = self.seg_objects.fast_rendering != enabled;
        self.seg_objects.fast_rendering = enabled;
        for layer in &mut self.spatial_layers.shapes {
            if let Some(objects) = layer.object_layer_mut() {
                changed |= objects.fast_rendering != enabled;
                objects.fast_rendering = enabled;
            }
        }
        if changed {
            self.bump_render_id();
        }
    }

    pub fn apply_auto_contrast_now(&mut self) {
        self.request_auto_contrast(true);
        self.set_status(format!(
            "Applying auto contrast ({}) to all channels...",
            self.auto_contrast_settings.method.label()
        ));
    }

    pub fn open_roi_info_window(&mut self) {
        self.roi_info_open = true;
    }

    pub fn screenshot_output_dir(&self) -> Option<&Path> {
        self.screenshot_output_dir.as_deref()
    }

    pub fn request_screenshot_png(&mut self, path: PathBuf) {
        let Some(viewport_id) = self
            .viewport_workspace
            .as_ref()
            .map(|workspace| workspace.active_id().clone())
        else {
            self.set_status("Cannot capture screenshot before the viewer workspace is ready.");
            return;
        };
        self.request_screenshot_png_for_viewport(path, viewport_id);
    }

    pub(super) fn request_screenshot_png_for_viewport(
        &mut self,
        path: PathBuf,
        viewport_id: ViewportId,
    ) {
        let id = self.screenshot_next_id;
        self.screenshot_next_id = self.screenshot_next_id.wrapping_add(1).max(1);
        self.screenshot_pending
            .push_back(PendingViewportScreenshot {
                viewport_id,
                request: ScreenshotRequest {
                    id,
                    path,
                    settings: self.screenshot_settings,
                    presentation: None,
                },
            });
        // Avoid capturing floating dialogs over the canvas.
        self.screenshot_settings_open = false;
        self.set_status("Capturing screenshot...".to_string());
    }

    pub fn request_actor_screenshot(
        &mut self,
        capture_id: u64,
        viewport_id: Option<&str>,
        preferences: &odon::model::ScreenshotPreferences,
        tx: crossbeam_channel::Sender<odon::control::actor::PresentationCaptureCompletion>,
    ) -> anyhow::Result<()> {
        let workspace = self
            .viewport_workspace
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("viewer workspace is not initialized"))?;
        let viewport_id = match viewport_id {
            Some(id) => ViewportId::new(id)?,
            None => workspace.active_id().clone(),
        };
        anyhow::ensure!(
            workspace.get(&viewport_id).is_some(),
            "viewport '{viewport_id}' was not found"
        );
        let id = self.screenshot_next_id;
        self.screenshot_next_id = self.screenshot_next_id.wrapping_add(1).max(1);
        self.screenshot_pending
            .push_back(PendingViewportScreenshot {
                viewport_id,
                request: ScreenshotRequest {
                    id,
                    path: PathBuf::new(),
                    settings: ScreenshotSettings {
                        include_scale_bar: preferences.include_scale_bar(),
                        include_legend: preferences.include_legend(),
                        scale_bar_scale: preferences.scale_bar_scale(),
                        legend_scale: preferences.legend_scale(),
                    },
                    presentation: Some(
                        crate::app_support::screenshot::PresentationScreenshotReply {
                            capture_id,
                            tx,
                        },
                    ),
                },
            });
        self.screenshot_settings_open = false;
        self.set_status("Capturing actor-requested screenshot...");
        Ok(())
    }

    pub(super) fn request_quick_screenshot_png_for_viewport(
        &mut self,
        viewport_id: ViewportId,
    ) -> anyhow::Result<PathBuf> {
        let Some(dir) = self.screenshot_output_dir.as_deref() else {
            anyhow::bail!("No screenshot folder configured");
        };
        let path = next_numbered_screenshot_path(dir, &self.default_screenshot_filename())?;
        self.request_screenshot_png_for_viewport(path.clone(), viewport_id);
        Ok(path)
    }

    pub fn request_quick_screenshot_png(&mut self) -> anyhow::Result<PathBuf> {
        let viewport_id = self
            .viewport_workspace
            .as_ref()
            .map(|workspace| workspace.active_id().clone())
            .ok_or_else(|| anyhow::anyhow!("Viewer workspace is not initialized"))?;
        self.request_quick_screenshot_png_for_viewport(viewport_id)
    }

    pub fn default_screenshot_filename(&self) -> String {
        let base = self.dataset.source.display_name();
        let stem = base
            .strip_suffix(".ome.zarr")
            .or_else(|| base.strip_suffix(".zarr"))
            .unwrap_or(base.as_str());
        let sanitized = stem
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        if sanitized.is_empty() {
            "odon.screenshot.png".to_string()
        } else {
            format!("{sanitized}.screenshot.png")
        }
    }

    pub(super) fn default_mask_layer_export_filename(&self, layer_id: u64) -> String {
        let stem = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == layer_id)
            .map(|layer| layer.name.as_str())
            .unwrap_or("mask-layer");
        let sanitized = stem
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        if sanitized.is_empty() {
            "mask-layer.geojson".to_string()
        } else {
            format!("{sanitized}.geojson")
        }
    }

    pub fn export_masks_geojson(&self, path: &Path) -> anyhow::Result<()> {
        save_mask_layers_geojson(path, &self.mask_layers)
    }

    pub fn export_mask_layer_geojson(&self, layer_id: u64, path: &Path) -> anyhow::Result<()> {
        let layer = self
            .mask_layers
            .iter()
            .find(|layer| layer.id == layer_id)
            .ok_or_else(|| anyhow::anyhow!("mask layer not found"))?;
        save_mask_layers_geojson(path, std::slice::from_ref(layer))
    }
}
