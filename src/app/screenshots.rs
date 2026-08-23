use super::*;

impl OmeZarrViewerApp {
    pub fn apply_control_actor_screenshot_preferences(
        &mut self,
        preferences: &odon::model::ScreenshotPreferences,
    ) {
        self.screenshot_dialog.output_dir = preferences.output_dir().map(Path::to_path_buf);
        self.screenshot_dialog.settings = ScreenshotSettings {
            include_scale_bar: preferences.include_scale_bar(),
            include_legend: preferences.include_legend(),
            scale_bar_scale: preferences.scale_bar_scale(),
            legend_scale: preferences.legend_scale(),
        };
    }

    pub fn open_screenshot_settings(&mut self) {
        self.screenshot_dialog.open = true;
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

    pub fn open_roi_info_window(&mut self) {
        self.roi_info_open = true;
    }

    pub fn screenshot_output_dir(&self) -> Option<&Path> {
        self.screenshot_dialog.output_dir()
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
        anyhow::ensure!(
            !self.screenshot_capture.pending.iter().any(|pending| pending
                .request
                .presentation
                .capture_id
                == capture_id),
            "capture {capture_id} is already installed in the viewer renderer"
        );
        self.screenshot_capture
            .pending
            .push_back(PendingViewportScreenshot {
                viewport_id,
                request: RendererScreenshotRequest {
                    settings: ScreenshotSettings {
                        include_scale_bar: preferences.include_scale_bar(),
                        include_legend: preferences.include_legend(),
                        scale_bar_scale: preferences.scale_bar_scale(),
                        legend_scale: preferences.legend_scale(),
                    },
                    presentation: crate::app_support::screenshot::PresentationScreenshotReply {
                        capture_id,
                        tx,
                    },
                },
            });
        self.screenshot_dialog.open = false;
        self.set_status("Capturing actor-requested screenshot...");
        Ok(())
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
}
