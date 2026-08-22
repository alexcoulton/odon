use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};

use super::{ControlObjectResource, ProjectModelSnapshot};
use crate::control::{ControlError, ControlErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProjectObjectPreloadMode {
    FullGeometry,
    CentroidPoints,
}

impl ProjectObjectPreloadMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullGeometry => "full_geometry",
            Self::CentroidPoints => "centroid_points",
        }
    }

    pub fn parse(value: &str) -> Result<Self, ControlError> {
        match value {
            "full_geometry" => Ok(Self::FullGeometry),
            "centroid_points" => Ok(Self::CentroidPoints),
            _ => Err(ControlError::new(
                ControlErrorKind::InvalidParams,
                format!("unknown object preload mode '{value}'"),
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProjectObjectPreloadSettings {
    pub mode: ProjectObjectPreloadMode,
    pub lazy_properties: bool,
}

impl Default for ProjectObjectPreloadSettings {
    fn default() -> Self {
        Self {
            mode: ProjectObjectPreloadMode::FullGeometry,
            lazy_properties: true,
        }
    }
}

impl ProjectObjectPreloadSettings {
    pub fn worker_options(self) -> Value {
        json!({
            "project_preload": {
                "mode": self.mode.as_str(),
                "lazy_properties": self.lazy_properties,
            }
        })
    }

    fn snapshot(self) -> Value {
        json!({
            "mode": self.mode.as_str(),
            "lazy_properties": self.lazy_properties,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectObjectPreloadScope {
    pub saved_path: Option<PathBuf>,
    pub load_generation: u64,
    pub resource_generation: u64,
}

impl ProjectObjectPreloadScope {
    pub(crate) fn from_project(project: &ProjectModelSnapshot) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for roi in &project.rois {
            roi.id.hash(&mut hasher);
            roi.source_key().hash(&mut hasher);
            roi.segpath.hash(&mut hasher);
        }
        Self {
            saved_path: project.saved_path.clone(),
            load_generation: project.load_generation,
            resource_generation: hasher.finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProjectObjectPreloadSource {
    pub path: PathBuf,
    pub bytes: u64,
}

/// Immutable renderer projection of the actor-owned preload catalog. The JSON state is suitable
/// for UI counters and public API responses; resources retain native renderer payloads behind
/// shared handles and are never serialized.
#[derive(Debug, Clone)]
pub struct ProjectObjectPreloadProjection {
    pub state: Value,
    pub settings: ProjectObjectPreloadSettings,
    pub resources: Arc<Vec<(PathBuf, Arc<ControlObjectResource>)>>,
}

#[derive(Debug, Clone)]
pub(crate) struct ProjectObjectPreloadCatalog {
    scope: ProjectObjectPreloadScope,
    sources: Vec<ProjectObjectPreloadSource>,
    resources: HashMap<(PathBuf, ProjectObjectPreloadSettings), Arc<ControlObjectResource>>,
    settings: ProjectObjectPreloadSettings,
    total: usize,
    done: usize,
    failed: usize,
    loading: bool,
    operation_generation: u64,
}

impl Default for ProjectObjectPreloadCatalog {
    fn default() -> Self {
        Self {
            scope: ProjectObjectPreloadScope {
                saved_path: None,
                load_generation: 0,
                resource_generation: 0,
            },
            sources: Vec::new(),
            resources: HashMap::new(),
            settings: ProjectObjectPreloadSettings::default(),
            total: 0,
            done: 0,
            failed: 0,
            loading: false,
            operation_generation: 0,
        }
    }
}

impl ProjectObjectPreloadCatalog {
    pub(crate) fn sync_scope(&mut self, project: &ProjectModelSnapshot) {
        let scope = ProjectObjectPreloadScope::from_project(project);
        if self.scope == scope {
            return;
        }
        self.scope = scope;
        self.invalidate();
        self.settings = ProjectObjectPreloadSettings::default();
    }

    pub(crate) fn scope(&self) -> ProjectObjectPreloadScope {
        self.scope.clone()
    }

    pub(crate) fn begin(
        &mut self,
        settings: ProjectObjectPreloadSettings,
        candidate_count: usize,
    ) -> u64 {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.sources.clear();
        self.resources.clear();
        self.settings = settings;
        self.total = candidate_count;
        self.done = 0;
        self.failed = 0;
        self.loading = true;
        self.operation_generation
    }

    pub(crate) fn is_loading(&self) -> bool {
        self.loading
    }

    pub(crate) fn is_current(&self, scope: &ProjectObjectPreloadScope, generation: u64) -> bool {
        self.scope == *scope && self.loading && self.operation_generation == generation
    }

    pub(crate) fn install_sources(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        sources: Vec<ProjectObjectPreloadSource>,
    ) -> bool {
        if self.scope != *scope {
            return false;
        }
        self.sources = sources;
        true
    }

    pub(crate) fn finish(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        generation: u64,
        sources: Vec<ProjectObjectPreloadSource>,
        resources: Vec<(PathBuf, ControlObjectResource)>,
        failed: usize,
    ) -> bool {
        if !self.is_current(scope, generation) {
            return false;
        }
        self.sources = sources;
        self.total = self.sources.len();
        self.done = self.total;
        self.failed = failed;
        self.resources = resources
            .into_iter()
            .map(|(path, resource)| ((path, self.settings), Arc::new(resource)))
            .collect();
        self.loading = false;
        true
    }

    pub(crate) fn fail(&mut self, scope: &ProjectObjectPreloadScope, generation: u64) -> bool {
        if !self.is_current(scope, generation) {
            return false;
        }
        self.loading = false;
        true
    }

    pub(crate) fn clear(&mut self) -> (usize, bool, u64) {
        let removed = self.resources.len();
        let cancelled = self.loading;
        let generation = self.operation_generation;
        self.invalidate();
        (removed, cancelled, generation)
    }

    fn invalidate(&mut self) {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.sources.clear();
        self.resources.clear();
        self.total = 0;
        self.done = 0;
        self.failed = 0;
        self.loading = false;
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "available_count": self.sources.len(),
            "on_disk_bytes": self.sources.iter().map(|source| source.bytes).sum::<u64>(),
            "cached": self.resources.len(),
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "loading": self.loading,
            "settings": self.settings.snapshot(),
            "project_path": self.scope.saved_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
        })
    }

    pub(crate) fn list_sources(&self, offset: usize, limit: usize) -> Value {
        let total = self.sources.len();
        let sources = self
            .sources
            .iter()
            .skip(offset)
            .take(limit)
            .map(|source| {
                json!({
                    "path": source.path.to_string_lossy(),
                    "bytes": source.bytes,
                    "cached": self.resources.contains_key(&(source.path.clone(), self.settings)),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(sources.len()) < total,
            "sources": sources,
        })
    }

    pub(crate) fn projection(&self) -> ProjectObjectPreloadProjection {
        let mut resources = self
            .resources
            .iter()
            .filter_map(|((path, settings), resource)| {
                (*settings == self.settings).then_some((path.clone(), Arc::clone(resource)))
            })
            .collect::<Vec<_>>();
        resources.sort_by(|left, right| left.0.cmp(&right.0));
        ProjectObjectPreloadProjection {
            state: self.snapshot(),
            settings: self.settings,
            resources: Arc::new(resources),
        }
    }

    pub(crate) fn cached_resource(&self, path: &PathBuf) -> Option<Arc<ControlObjectResource>> {
        self.resources.get(&(path.clone(), self.settings)).cloned()
    }

    pub(crate) fn remember_resource(
        &mut self,
        scope: &ProjectObjectPreloadScope,
        path: PathBuf,
        resource: Arc<ControlObjectResource>,
    ) -> bool {
        if self.scope != *scope {
            return false;
        }
        self.resources.insert((path, self.settings), resource);
        true
    }
}

pub(crate) fn project_object_preload_candidates(project: &ProjectModelSnapshot) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    project
        .rois
        .iter()
        .filter_map(|roi| project_roi_segmentation_path(project, roi))
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "parquet" | "geoparquet"
                    )
                })
        })
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

pub(crate) fn project_roi_segmentation_path(
    project: &ProjectModelSnapshot,
    roi: &crate::data::project_config::ProjectRoi,
) -> Option<PathBuf> {
    let path = roi.segpath.as_ref()?;
    if path.is_absolute() {
        Some(path.clone())
    } else {
        Some(
            project
                .saved_path
                .as_deref()
                .and_then(std::path::Path::parent)
                .map_or_else(|| path.clone(), |directory| directory.join(path)),
        )
    }
}
