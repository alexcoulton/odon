//! Canonical, renderer-independent application state.

mod app;
mod labels;
mod layers;
mod masks;
mod objects;
mod project;
mod project_preload;
mod readiness;
mod selection;

pub use app::{
    AppModel, ChannelIntensitySpec, ModelDispatch, ModelMode, SettingsMutationOutcome,
    SettingsSaveOperation,
};
pub use labels::{ControlLabelResource, LabelZarrDataset, discover_label_names_local};
pub(crate) use masks::{MaskModel, load_geojson_mask_polylines};
pub use objects::{
    ControlObjectFeature, ControlObjectFilterResult, ControlObjectResource, ObjectResourceLoader,
};
pub use project::ProjectModelSnapshot;
pub(crate) use project::normalized_loaded_project_snapshot;
pub(crate) use project_preload::{
    ProjectObjectPreloadCatalog, ProjectObjectPreloadScope, ProjectObjectPreloadSource,
    project_object_preload_candidates, project_roi_segmentation_path,
};
pub use project_preload::{
    ProjectObjectPreloadMode, ProjectObjectPreloadProjection, ProjectObjectPreloadSettings,
};
pub(crate) use readiness::{OperationKind, ReadinessModel};
pub(crate) use selection::{ObjectSelectionModel, parse_world_points, parse_world_rect};
