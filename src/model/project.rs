mod commands;
mod state;
mod validation;

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::data::project_config::{ProjectConfig, ProjectRoi};

pub(crate) use state::normalized_loaded_project_snapshot;

#[derive(Debug, Clone)]
pub struct ProjectModelSnapshot {
    /// Complete persisted project configuration. The explicit fields below are indexed copies
    /// used by actor commands; `ProjectModel` keeps both representations synchronized.
    pub config: ProjectConfig,
    /// Complete version-6 project state. This remains JSON in the renderer-independent crate so
    /// UI-specific view-state DTOs do not become actor dependencies.
    pub state: Value,
    /// Changes only when a complete project is replaced (open/create), allowing the renderer to
    /// distinguish a full materialization from an ordinary semantic projection.
    pub load_generation: u64,
    pub rois: Vec<ProjectRoi>,
    pub default_dataset: Option<String>,
    pub secondary_dataset: Option<String>,
    pub default_threshold_marker: Option<String>,
    pub mosaic_segmentation_search_roots: Vec<PathBuf>,
    pub dataset_keys: Vec<String>,
    pub selected_source_keys: Vec<String>,
    pub focused_source_key: Option<String>,
    pub saved_path: Option<PathBuf>,
    pub config_generation: u64,
    pub view_presets: Vec<Value>,
    pub view_count: usize,
    pub dirty: bool,
}

impl Default for ProjectModelSnapshot {
    fn default() -> Self {
        Self {
            config: ProjectConfig::default(),
            state: json!({}),
            load_generation: 0,
            rois: Vec::new(),
            default_dataset: None,
            secondary_dataset: None,
            default_threshold_marker: None,
            mosaic_segmentation_search_roots: Vec::new(),
            dataset_keys: Vec::new(),
            selected_source_keys: Vec::new(),
            focused_source_key: None,
            saved_path: None,
            config_generation: 0,
            view_presets: Vec::new(),
            view_count: 0,
            dirty: false,
        }
    }
}

/// The project snapshot, shared with every projection that was published from it.
///
/// Publishing a projection used to deep-copy the whole project (every ROI, its mask polygons,
/// the saved views) for each state change, including each camera step. Handing out the same
/// allocation costs nothing, and the copy is deferred to the next real project mutation: any
/// mutable access detaches from the shared allocation first.
#[derive(Debug, Clone, Default)]
struct SharedProjectSnapshot(std::sync::Arc<ProjectModelSnapshot>);

impl std::ops::Deref for SharedProjectSnapshot {
    type Target = ProjectModelSnapshot;

    fn deref(&self) -> &ProjectModelSnapshot {
        &self.0
    }
}

impl std::ops::DerefMut for SharedProjectSnapshot {
    fn deref_mut(&mut self) -> &mut ProjectModelSnapshot {
        std::sync::Arc::make_mut(&mut self.0)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ProjectModel {
    snapshot: SharedProjectSnapshot,
}
