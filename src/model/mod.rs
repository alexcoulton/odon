//! Canonical, renderer-independent application state.

mod analysis;
mod annotations;
mod app;
mod command_surface;
mod labels;
mod layers;
mod masks;
mod measurement;
mod memory;
mod mosaic;
mod object_color;
mod object_export;
mod objects;
mod project;
mod project_preload;
mod readiness;
mod screenshot;
mod segmentation_geojson;
mod selection;
mod shell;
mod threshold;

pub(crate) use analysis::AnalysisModel;
pub(crate) use annotations::{AnnotationLoadResult, AnnotationLoadSpec, AnnotationModel};
pub use annotations::{ControlAnnotationLayerProjection, ControlAnnotationResource};
pub(crate) use app::{
    AnalysisResourceSpec, AutoContrastChannelResult, AutoContrastSpec, DeepLinkApplyGuard,
    DeepLinkCurrentResources, MeasurementSpec, MemoryPinSpec, ObjectTarget, ProjectViewApplySpec,
    ThresholdPreviewApplySpec, ThresholdPreviewLoadSpec, ThresholdPreviewRecomputeSpec,
};
pub use app::{
    AppModel, ChannelIntensitySpec, ControlSecondaryObjectProjection, ModelDispatch, ModelMode,
    SettingsMutationOutcome, SettingsSaveOperation,
};
#[doc(hidden)]
pub use command_surface::command_surface_native_actions;
pub(crate) use command_surface::{
    CommandEvaluationContext, CommandInvocation, CommandSurfaceModel,
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
pub use object_color::{
    ContinuousColorConfig, ContinuousColorStop, ContinuousDomain, ContinuousPalette,
    ContinuousScale, ObjectColorMapping, OutOfRangeMode,
};
pub(crate) use object_export::{
    ObjectExportFormat, ObjectExportModel, ObjectExportResult, ObjectExportSpec,
    object_export_columns, write_object_export,
};
pub use objects::{
    ControlObjectFeature, ControlObjectFilterResult, ControlObjectNumericSummary,
    ControlObjectResource, ObjectResourceLoader,
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
pub use segmentation_geojson::ControlSegmentationGeoJsonResource;
pub(crate) use segmentation_geojson::{SegmentationGeoJsonLoadSpec, SegmentationGeoJsonModel};
pub(crate) use selection::{ObjectSelectionModel, parse_world_points, parse_world_rect};
pub(crate) use shell::ShellModel;

#[doc(hidden)]
pub use shell::shell_component_catalog;

#[doc(hidden)]
pub fn shell_component_minimum_size(id: &str) -> Option<[f32; 2]> {
    shell::shell_component_minimum_size(id)
}

#[doc(hidden)]
pub fn validate_shell_layout_document(
    document: &serde_json::Value,
) -> Result<(), crate::control::ControlError> {
    shell::validate_layout_document(document)
}

#[doc(hidden)]
pub fn normalize_shell_layout_document(
    document: &serde_json::Value,
) -> Result<serde_json::Value, crate::control::ControlError> {
    shell::normalize_layout_document(document)
}
pub(crate) use threshold::ThresholdPreviewModel;
pub use threshold::{ControlThresholdPreviewResource, ThresholdScope};
pub(crate) use threshold::{ThresholdMask, extract_threshold_mask, threshold_mask_polygons};
