use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AutoContrastMethod {
    #[default]
    ZeroToP97,
    P1ToP99,
    ZeroToMax,
}

impl AutoContrastMethod {
    pub const ALL: [Self; 3] = [Self::ZeroToP97, Self::P1ToP99, Self::ZeroToMax];

    pub fn label(self) -> &'static str {
        match self {
            Self::ZeroToP97 => "Zero to P97",
            Self::P1ToP99 => "P1 to P99",
            Self::ZeroToMax => "Zero to Max",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::ZeroToP97 => {
                "Fast default. Keeps the lower bound at zero and clips bright outliers at the 97th percentile."
            }
            Self::P1ToP99 => {
                "Robust range. Ignores both dark and bright outliers by using the 1st and 99th percentiles."
            }
            Self::ZeroToMax => {
                "Full data range. Keeps the lower bound at zero and uses the brightest observed value."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoContrastSettings {
    pub enabled_on_open: bool,
    pub method: AutoContrastMethod,
    pub lower_percentile: u8,
    pub upper_percentile: u8,
}

impl Default for AutoContrastSettings {
    fn default() -> Self {
        Self {
            enabled_on_open: true,
            method: AutoContrastMethod::ZeroToP97,
            lower_percentile: 1,
            upper_percentile: 97,
        }
    }
}

impl AutoContrastSettings {
    pub fn normalized(mut self) -> Self {
        self.lower_percentile = self.lower_percentile.min(99);
        self.upper_percentile = self.upper_percentile.clamp(1, 100);
        if self.lower_percentile >= self.upper_percentile {
            self.lower_percentile = self.upper_percentile.saturating_sub(1);
        }
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppSettings {
    pub auto_contrast: AutoContrastSettings,
}

impl AppSettings {
    pub fn load() -> anyhow::Result<Self> {
        let path = settings_file_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let settings: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse settings file {}", path.display()))?;
        Ok(Self {
            auto_contrast: settings.auto_contrast.normalized(),
        })
    }

    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let path = settings_file_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create settings directory {}", parent.display())
            })?;
        }
        let normalized = Self {
            auto_contrast: self.auto_contrast.normalized(),
        };
        let text =
            serde_json::to_string_pretty(&normalized).context("failed to serialize settings")?;
        fs::write(&path, text)
            .with_context(|| format!("failed to write settings file {}", path.display()))?;
        Ok(path)
    }
}

pub fn settings_file_path() -> anyhow::Result<PathBuf> {
    let base = dirs::config_dir().context("system config directory is not available")?;
    Ok(base.join("odon").join("settings.json"))
}
