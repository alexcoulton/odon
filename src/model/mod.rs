//! Canonical, renderer-independent application state.

mod analysis;
mod app;
mod labels;
mod layers;
mod masks;
mod measurement;
mod memory;
mod mosaic;
mod object_export;
mod objects;
mod project;
mod project_preload;
mod readiness;
mod screenshot;
mod selection;
mod threshold;

pub(crate) use analysis::AnalysisModel;
pub(crate) use app::{
    AnalysisResourceSpec, DeepLinkApplyGuard, DeepLinkCurrentResources, MeasurementSpec,
    MemoryPinSpec, ObjectTarget, ProjectViewApplySpec, ThresholdPreviewApplySpec,
    ThresholdPreviewLoadSpec, ThresholdPreviewRecomputeSpec,
};
pub use app::{
    AppModel, ChannelIntensitySpec, ControlSecondaryObjectProjection, ModelDispatch, ModelMode,
    SettingsMutationOutcome, SettingsSaveOperation,
};
pub use labels::{ControlLabelResource, LabelZarrDataset, discover_label_names_local};
pub(crate) use masks::{MaskModel, load_geojson_mask_polylines};
pub(crate) use measurement::{MeasurementMetric, MeasurementModel};
pub use memory::{
    ControlPinnedLevelResource, TileLoadingPolicy, TilePrefetchAggressiveness, TilePrefetchMode,
};
pub(crate) use memory::{PinnedMemoryModel, SystemMemorySnapshot, TileLoadingModel};
pub use mosaic::{ControlMosaicItemResource, ControlMosaicResource};
pub(crate) use mosaic::{
    MosaicMemoryPinItemSpec, MosaicMemoryPinResult, MosaicMemoryPinSpec, MosaicModel,
    MosaicObjectLoadResult, MosaicObjectLoadSpec,
};
pub(crate) use object_export::{
    ObjectExportFormat, ObjectExportModel, ObjectExportResult, ObjectExportSpec,
    object_export_columns, write_object_export,
};
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
pub use screenshot::ScreenshotPreferences;
pub(crate) use screenshot::default_screenshot_filename;
pub(crate) use selection::{ObjectSelectionModel, parse_world_points, parse_world_rect};
pub(crate) use threshold::ThresholdPreviewModel;
pub use threshold::{ControlThresholdPreviewResource, ThresholdScope};
pub(crate) use threshold::{ThresholdMask, extract_threshold_mask, threshold_mask_polygons};
