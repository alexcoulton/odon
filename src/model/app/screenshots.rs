//! Screenshot preferences and generation-safe persistence state.

use super::*;

impl AppModel {
    pub fn screenshot_settings_snapshot(&self) -> Result<Value, ControlError> {
        let default_filename = if self.mode == ModelMode::Mosaic {
            self.mosaic.default_screenshot_filename()?
        } else {
            let dataset = self.dataset()?;
            default_screenshot_filename(&dataset.descriptor.source.display_name())
        };
        Ok(self.screenshot_preferences.snapshot(
            &default_filename,
            self.screenshot_settings_generation,
            self.screenshot_settings_pending,
        ))
    }

    pub fn begin_screenshot_settings_update(
        &mut self,
        params: &Value,
        normalized_output_dir: Option<Option<PathBuf>>,
    ) -> Result<(u64, ScreenshotPreferences), ControlError> {
        if self.mode == ModelMode::Mosaic {
            self.mosaic.require_ready()?;
        } else {
            self.dataset()?;
        }
        let candidate = self
            .screenshot_preferences
            .updated(params, normalized_output_dir)?;
        self.screenshot_settings_generation =
            self.screenshot_settings_generation.wrapping_add(1).max(1);
        self.screenshot_settings_pending = true;
        self.readiness.begin(
            OperationKind::ScreenshotSettings,
            self.screenshot_settings_generation,
            "Validating screenshot settings",
        );
        Ok((self.screenshot_settings_generation, candidate))
    }

    pub fn install_screenshot_settings_for_generation(
        &mut self,
        generation: u64,
        preferences: ScreenshotPreferences,
    ) -> Option<Value> {
        if generation != self.screenshot_settings_generation || !self.screenshot_settings_pending {
            return None;
        }
        self.screenshot_preferences = preferences;
        self.screenshot_settings_pending = false;
        self.readiness.finish(
            OperationKind::ScreenshotSettings,
            generation,
            "Screenshot settings ready",
        );
        self.screenshot_settings_snapshot().ok()
    }

    pub fn fail_screenshot_settings_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if generation != self.screenshot_settings_generation || !self.screenshot_settings_pending {
            return false;
        }
        self.screenshot_settings_pending = false;
        self.readiness.fail(
            OperationKind::ScreenshotSettings,
            generation,
            message.into(),
        );
        true
    }

    pub fn cancel_screenshot_settings_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if generation != self.screenshot_settings_generation || !self.screenshot_settings_pending {
            return false;
        }
        self.screenshot_settings_pending = false;
        self.readiness.cancel(
            OperationKind::ScreenshotSettings,
            generation,
            message.into(),
        );
        true
    }
}
