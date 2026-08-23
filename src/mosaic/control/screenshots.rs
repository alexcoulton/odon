use super::super::*;

impl MosaicViewerApp {
    pub fn open_screenshot_settings(&mut self) {
        self.screenshot_settings_open = true;
    }

    pub fn apply_control_actor_screenshot_preferences(
        &mut self,
        preferences: &odon::model::ScreenshotPreferences,
    ) {
        self.screenshot_output_dir = preferences.output_dir().map(Path::to_path_buf);
        self.screenshot_settings = ScreenshotSettings {
            include_scale_bar: false,
            include_legend: preferences.include_legend(),
            scale_bar_scale: preferences.scale_bar_scale(),
            legend_scale: preferences.legend_scale(),
        };
    }

    pub fn screenshot_output_dir(&self) -> Option<&Path> {
        self.screenshot_output_dir.as_deref()
    }

    pub fn request_screenshot_png(&mut self, path: PathBuf) {
        let id = self.screenshot_next_id;
        self.screenshot_next_id = self.screenshot_next_id.wrapping_add(1).max(1);
        self.screenshot_pending = Some(ScreenshotRequest {
            id,
            path,
            settings: self.screenshot_settings,
            presentation: None,
        });
        self.screenshot_in_flight = Some(id);
        self.screenshot_settings_open = false;
        self.renderer_status = "Capturing screenshot...".to_string();
    }

    pub fn request_actor_screenshot(
        &mut self,
        capture_id: u64,
        preferences: &odon::model::ScreenshotPreferences,
        tx: crossbeam_channel::Sender<odon::control::actor::PresentationCaptureCompletion>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.screenshot_pending.is_none(),
            "the mosaic renderer already has a pending screenshot"
        );
        let id = self.screenshot_next_id;
        self.screenshot_next_id = self.screenshot_next_id.wrapping_add(1).max(1);
        self.screenshot_pending = Some(ScreenshotRequest {
            id,
            path: PathBuf::new(),
            settings: ScreenshotSettings {
                include_scale_bar: false,
                include_legend: preferences.include_legend(),
                scale_bar_scale: preferences.scale_bar_scale(),
                legend_scale: preferences.legend_scale(),
            },
            presentation: Some(
                crate::app_support::screenshot::PresentationScreenshotReply { capture_id, tx },
            ),
        });
        self.screenshot_settings_open = false;
        self.renderer_status = "Capturing actor-requested screenshot...".to_string();
        Ok(())
    }

    pub fn request_quick_screenshot_png(&mut self) -> anyhow::Result<PathBuf> {
        let Some(dir) = self.screenshot_output_dir.as_deref() else {
            anyhow::bail!("No screenshot folder configured");
        };
        let path = next_numbered_screenshot_path(dir, &self.default_screenshot_filename())?;
        self.request_screenshot_png(path.clone());
        Ok(path)
    }

    pub fn default_screenshot_filename(&self) -> String {
        let base = self
            .focused_item()
            .or_else(|| self.items.first())
            .map(|it| it.sample_id.clone())
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| "mosaic".to_string());
        let sanitized = base
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        if sanitized.is_empty() {
            "odon.mosaic.screenshot.png".to_string()
        } else {
            format!("{sanitized}.mosaic.screenshot.png")
        }
    }
}
