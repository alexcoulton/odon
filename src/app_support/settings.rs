use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppSettings {
    pub auto_contrast: AutoContrastSettings,
    pub recent_projects: Vec<RecentProject>,
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
    fn normalized(mut self) -> Self {
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
        let path = settings_file_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path)
            .with_context(|| format!("failed to read settings file {}", path.display()))?;
        let settings: Self = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse settings file {}", path.display()))?;
        Ok(settings.normalized())
    }

    pub fn save(&self) -> anyhow::Result<PathBuf> {
        let path = settings_file_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create settings directory {}", parent.display())
            })?;
        }
        let normalized = self.clone().normalized();
        let text =
            serde_json::to_string_pretty(&normalized).context("failed to serialize settings")?;
        fs::write(&path, text)
            .with_context(|| format!("failed to write settings file {}", path.display()))?;
        Ok(path)
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

fn normalize_recent_project_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
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

    #[test]
    fn recent_projects_can_be_forgotten_and_cleared() {
        let mut settings = AppSettings::default();
        let a = PathBuf::from("/tmp/a.project.json");
        let b = PathBuf::from("/tmp/b.project.json");
        settings.record_recent_project(&a);
        settings.record_recent_project(&b);

        assert!(settings.forget_recent_project(&a));
        assert_eq!(settings.recent_projects.len(), 1);
        assert_eq!(settings.recent_projects[0].path, b);
        assert!(settings.clear_recent_projects());
        assert!(settings.recent_projects.is_empty());
    }
}
