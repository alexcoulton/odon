use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_RECENT_PROJECTS: usize = 20;

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

pub fn auto_contrast_window_from_histogram(
    settings: AutoContrastSettings,
    histogram: &[u64],
    sample_count: u64,
    observed_max: u16,
) -> (u16, u16) {
    let settings = settings.normalized();
    match settings.method {
        AutoContrastMethod::ZeroToP97 => (
            0,
            percentile_from_histogram(histogram, sample_count, settings.upper_percentile as u64),
        ),
        AutoContrastMethod::P1ToP99 => (
            percentile_from_histogram(histogram, sample_count, settings.lower_percentile as u64),
            percentile_from_histogram(histogram, sample_count, settings.upper_percentile as u64),
        ),
        AutoContrastMethod::ZeroToMax => (0, observed_max),
    }
}

fn percentile_from_histogram(histogram: &[u64], sample_count: u64, percentile: u64) -> u16 {
    if sample_count == 0 || histogram.is_empty() {
        return 0;
    }
    let target = (sample_count.saturating_mul(percentile).saturating_add(99)) / 100;
    let mut accumulated = 0_u64;
    for (index, count) in histogram.iter().enumerate() {
        accumulated = accumulated.saturating_add(*count);
        if accumulated >= target {
            return index.min(u16::MAX as usize) as u16;
        }
    }
    histogram.len().saturating_sub(1).min(u16::MAX as usize) as u16
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub auto_contrast: AutoContrastSettings,
    #[serde(default = "default_true")]
    pub fast_object_rendering: bool,
    pub recent_projects: Vec<RecentProject>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            auto_contrast: AutoContrastSettings::default(),
            fast_object_rendering: true,
            recent_projects: Vec::new(),
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RecentProject {
    pub path: PathBuf,
    pub last_opened_unix_ms: u64,
}

impl Default for RecentProject {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            last_opened_unix_ms: 0,
        }
    }
}

impl RecentProject {
    pub fn display_name(&self) -> String {
        self.path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.path.to_string_lossy().to_string())
    }
}

impl AppSettings {
    pub fn normalized(mut self) -> Self {
        self.auto_contrast = self.auto_contrast.normalized();
        self.recent_projects
            .retain(|project| !project.path.as_os_str().is_empty());
        self.recent_projects
            .sort_by_key(|project| std::cmp::Reverse(project.last_opened_unix_ms));
        let mut seen = Vec::<PathBuf>::new();
        self.recent_projects.retain(|project| {
            let normalized = normalize_recent_project_path(&project.path);
            if seen.iter().any(|path| paths_match(path, &normalized)) {
                false
            } else {
                seen.push(normalized);
                true
            }
        });
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
        self
    }

    pub fn load() -> anyhow::Result<Self> {
        Self::load_from(&settings_file_path()?)
    }

    pub fn load_from(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let settings: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse settings file {}", path.display()))?;
        Ok(settings.normalized())
    }

    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let path = settings_file_path()?;
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create settings directory {}", parent.display())
            })?;
        }
        let normalized = self.clone().normalized();
        let text =
            serde_json::to_string_pretty(&normalized).context("failed to serialize settings")?;
        let temporary = temporary_settings_path(path);
        fs::write(&temporary, text).with_context(|| {
            format!(
                "failed to write temporary settings file {}",
                temporary.display()
            )
        })?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error)
                .with_context(|| format!("failed to replace settings file {}", path.display()));
        }
        Ok(())
    }

    pub fn patched(&self, params: &Value) -> Result<Self, String> {
        let mut candidate = self.clone();
        if let Some(value) = params.get("fast_object_rendering") {
            candidate.fast_object_rendering = value
                .as_bool()
                .ok_or_else(|| "fast_object_rendering must be a boolean".to_string())?;
        }
        if let Some(value) = params.get("auto_contrast") {
            let settings = value
                .as_object()
                .ok_or_else(|| "auto_contrast must be an object".to_string())?;
            if let Some(value) = settings.get("enabled_on_open") {
                candidate.auto_contrast.enabled_on_open = value
                    .as_bool()
                    .ok_or_else(|| "auto_contrast.enabled_on_open must be a boolean".to_string())?;
            }
            if let Some(value) = settings.get("method") {
                candidate.auto_contrast.method =
                    match value.as_str() {
                        Some("zero_to_p97") => AutoContrastMethod::ZeroToP97,
                        Some("p1_to_p99") => AutoContrastMethod::P1ToP99,
                        Some("zero_to_max") => AutoContrastMethod::ZeroToMax,
                        _ => return Err(
                            "auto_contrast.method must be zero_to_p97, p1_to_p99, or zero_to_max"
                                .to_string(),
                        ),
                    };
            }
            for (key, target) in [
                (
                    "lower_percentile",
                    &mut candidate.auto_contrast.lower_percentile,
                ),
                (
                    "upper_percentile",
                    &mut candidate.auto_contrast.upper_percentile,
                ),
            ] {
                if let Some(value) = settings.get(key) {
                    let value = value
                        .as_u64()
                        .filter(|value| *value <= 100)
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| {
                            format!("auto_contrast.{key} must be an integer from 0 to 100")
                        })?;
                    *target = value;
                }
            }
            if candidate.auto_contrast.lower_percentile >= candidate.auto_contrast.upper_percentile
            {
                return Err(
                    "auto_contrast.lower_percentile must be less than upper_percentile".to_string(),
                );
            }
        }
        Ok(candidate.normalized())
    }

    pub fn record_recent_project(&mut self, path: &Path) -> bool {
        if path.as_os_str().is_empty() {
            return false;
        }
        let normalized = normalize_recent_project_path(path);
        let before = self.recent_projects.clone();
        self.recent_projects
            .retain(|project| !paths_match(&project.path, &normalized));
        self.recent_projects.insert(
            0,
            RecentProject {
                path: normalized,
                last_opened_unix_ms: current_unix_ms(),
            },
        );
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
        self.recent_projects != before
    }

    pub fn forget_recent_project(&mut self, path: &Path) -> bool {
        let normalized = normalize_recent_project_path(path);
        let before = self.recent_projects.len();
        self.recent_projects
            .retain(|project| !paths_match(&project.path, &normalized));
        self.recent_projects.len() != before
    }

    pub fn clear_recent_projects(&mut self) -> bool {
        if self.recent_projects.is_empty() {
            false
        } else {
            self.recent_projects.clear();
            true
        }
    }
}

pub fn settings_file_path() -> anyhow::Result<PathBuf> {
    let base = dirs::config_dir().context("system config directory is not available")?;
    Ok(base.join("odon").join("settings.json"))
}

fn temporary_settings_path(path: &Path) -> PathBuf {
    let nonce = current_unix_ms();
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("settings.json");
    path.with_file_name(format!(".{name}.{}.{nonce}.tmp", std::process::id()))
}

fn normalize_recent_project_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

fn paths_match(a: &Path, b: &Path) -> bool {
    normalize_recent_project_path(a) == normalize_recent_project_path(b)
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_object_rendering_defaults_on_for_old_settings() {
        let settings: AppSettings = serde_json::from_str(r#"{"recent_projects":[]}"#).unwrap();
        assert!(settings.fast_object_rendering);
        assert!(AppSettings::default().fast_object_rendering);
    }

    #[test]
    fn settings_patch_validation_is_typed() {
        let settings = AppSettings::default()
            .patched(&serde_json::json!({
                "fast_object_rendering":false,
                "auto_contrast":{"method":"p1_to_p99","lower_percentile":2,"upper_percentile":98}
            }))
            .unwrap();
        assert!(!settings.fast_object_rendering);
        assert_eq!(settings.auto_contrast.method, AutoContrastMethod::P1ToP99);
        assert!(
            AppSettings::default()
                .patched(&serde_json::json!({
                    "auto_contrast":{"lower_percentile":99,"upper_percentile":2}
                }))
                .is_err()
        );
    }

    #[test]
    fn recent_projects_are_deduped_and_most_recent_first() {
        let mut settings = AppSettings::default();
        let a = PathBuf::from("/tmp/a.project.json");
        let b = PathBuf::from("/tmp/b.project.json");
        assert!(settings.record_recent_project(&a));
        assert!(settings.record_recent_project(&b));
        assert!(settings.record_recent_project(&a));
        assert_eq!(settings.recent_projects.len(), 2);
        assert_eq!(settings.recent_projects[0].path, a);
        assert_eq!(settings.recent_projects[1].path, b);
    }
}
