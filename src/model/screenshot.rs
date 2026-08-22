use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

#[derive(Debug, Clone, PartialEq)]
pub struct ScreenshotPreferences {
    output_dir: Option<PathBuf>,
    include_scale_bar: bool,
    include_legend: bool,
    scale_bar_scale: f32,
    legend_scale: f32,
}

impl Default for ScreenshotPreferences {
    fn default() -> Self {
        Self {
            output_dir: None,
            include_scale_bar: true,
            include_legend: true,
            scale_bar_scale: 1.0,
            legend_scale: 1.0,
        }
    }
}

impl ScreenshotPreferences {
    pub fn output_dir(&self) -> Option<&Path> {
        self.output_dir.as_deref()
    }

    pub fn include_scale_bar(&self) -> bool {
        self.include_scale_bar
    }

    pub fn include_legend(&self) -> bool {
        self.include_legend
    }

    pub fn scale_bar_scale(&self) -> f32 {
        self.scale_bar_scale
    }

    pub fn legend_scale(&self) -> f32 {
        self.legend_scale
    }

    pub(crate) fn updated(
        &self,
        params: &Value,
        normalized_output_dir: Option<Option<PathBuf>>,
    ) -> Result<Self, ControlError> {
        let mut candidate = self.clone();
        if let Some(output_dir) = normalized_output_dir {
            candidate.output_dir = output_dir;
        }
        if let Some(value) = params.get("include_scale_bar") {
            candidate.include_scale_bar = value.as_bool().ok_or_else(|| {
                ControlError::invalid_params(
                    "viewer.screenshot.settings.set",
                    "include_scale_bar must be a boolean",
                )
            })?;
        }
        if let Some(value) = params.get("include_legend") {
            candidate.include_legend = value.as_bool().ok_or_else(|| {
                ControlError::invalid_params(
                    "viewer.screenshot.settings.set",
                    "include_legend must be a boolean",
                )
            })?;
        }
        for (key, target) in [
            ("scale_bar_scale", &mut candidate.scale_bar_scale),
            ("legend_scale", &mut candidate.legend_scale),
        ] {
            if let Some(value) = params.get(key) {
                let value = value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| {
                        ControlError::invalid_params(
                            "viewer.screenshot.settings.set",
                            format!("{key} must be a finite number"),
                        )
                    })?;
                if !(0.5..=3.0).contains(&value) {
                    return Err(ControlError::invalid_params(
                        "viewer.screenshot.settings.set",
                        format!("{key} must be between 0.5 and 3.0"),
                    ));
                }
                *target = value as f32;
            }
        }
        Ok(candidate)
    }

    pub(crate) fn snapshot(
        &self,
        default_filename: &str,
        settings_generation: u64,
        settings_pending: bool,
    ) -> Value {
        json!({
            "output_dir":self.output_dir.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "include_scale_bar":self.include_scale_bar,
            "include_legend":self.include_legend,
            "scale_bar_scale":self.scale_bar_scale,
            "legend_scale":self.legend_scale,
            "pending":false,
            "pending_count":0,
            "in_flight":false,
            "in_flight_count":0,
            "default_filename":default_filename,
            "settings_generation":settings_generation,
            "settings_pending":settings_pending,
        })
    }

    pub(crate) fn validate_output_dir(&self) -> Result<(), ControlError> {
        if let Some(path) = self.output_dir.as_deref()
            && !path.is_dir()
        {
            return Err(ControlError::new(
                ControlErrorKind::Application,
                format!(
                    "screenshot output directory does not exist: {}",
                    path.to_string_lossy()
                ),
            ));
        }
        Ok(())
    }
}

pub(crate) fn default_screenshot_filename(source_name: &str) -> String {
    let stem = source_name
        .strip_suffix(".ome.zarr")
        .or_else(|| source_name.strip_suffix(".zarr"))
        .unwrap_or(source_name);
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
