//! UI-independent control actor and renderer projection queue.
//!
//! The runtime remains one serial ordering authority, but its implementation is split by
//! responsibility: [`runtime`] owns the mailbox loop, [`dispatch`] routes requests, [`worker`]
//! executes bounded jobs, the `completion_*` modules commit results by domain, and the remaining
//! domain modules prepare their own work. Renderer projections and diagnostics have independent
//! modules, while actor-capability metadata is owned by the control registry.

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use serde_json::{Value, json};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::control::registry::ExecutionOwner;
use crate::control::{
    ControlError, ControlErrorKind, ResourceRegistry, TaskRegistry, TaskServiceHandle, TaskState,
};
use crate::data::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
use crate::data::document::{
    AlternateDatasetBackend, AlternateDocumentResource, ControlOpenedDocument,
    CoreDatasetInspector, DatasetInspection, DatasetInspector, OmeZarrDocumentResource,
    OpenedDocument, UnavailableAlternateDatasetBackend, open_local_ome_zarr,
};
use crate::data::ome::{OmeZarrDataset, retrieve_image_subset_u16};
use crate::data::project_config::{ProjectConfig, ProjectMaskLayer, ProjectRoi};
use crate::data::remote_store::{CoreRemoteDatasetBackend, RemoteDatasetBackend};
use crate::data::samplesheet::{
    SampleRow, SampleSheet, load_samplesheet_csv, write_samplesheet_csv,
};
use crate::deep_link::{
    DeepLinkRequest, DeepLinkResolution, apply_example_defaults, object_filter_model,
    requested_bundled_label, resolve_example_project_path, resolve_roi_target,
};
use crate::mcp::OdonControlRequest;
use crate::model::{
    AnalysisResourceSpec, AppModel, ChannelIntensitySpec, ControlLabelResource,
    ControlMosaicItemResource, ControlMosaicResource, ControlObjectFilterResult,
    ControlObjectResource, ControlPinnedLevelResource, ControlThresholdPreviewResource,
    DeepLinkApplyGuard, DeepLinkCurrentResources, LabelZarrDataset, MeasurementMetric,
    MeasurementSpec, MemoryPinSpec, ModelMode, MosaicObjectLoadResult, MosaicObjectLoadSpec,
    ObjectExportFormat, ObjectExportResult, ObjectExportSpec, ObjectResourceLoader,
    ProjectModelSnapshot, ProjectObjectPreloadScope, ProjectObjectPreloadSettings,
    ProjectObjectPreloadSource, ScreenshotPreferences, SettingsMutationOutcome,
    SystemMemorySnapshot, ThresholdMask, ThresholdPreviewApplySpec, ThresholdPreviewLoadSpec,
    ThresholdPreviewRecomputeSpec, TileLoadingPolicy, discover_label_names_local,
    extract_threshold_mask, project_roi_segmentation_path, threshold_mask_polygons,
    write_object_export,
};
use crate::settings::AppSettings;

mod analysis;
mod application;
mod channel_io;
mod channels;
mod completion;
mod completion_masks;
mod completion_mosaic;
mod completion_objects;
mod completion_opening;
mod completion_project;
mod completion_resources;
mod datasets;
mod deep_links;
mod diagnostics;
mod dispatch;
mod jobs;
mod mask_io;
mod masks;
mod measurements;
mod memory;
mod mosaics;
mod object_exports;
mod objects;
mod project_io;
mod project_preload;
mod project_roi;
mod projection;
mod projects;
mod remote;
mod request;
mod resources;
mod routing;
mod runtime;
mod screenshots;
mod thresholds;
mod worker;

pub use crate::control::registry::ACTOR_CAPABLE_METHODS as MIGRATED_METHODS;
use analysis::{
    begin_analysis_compute, begin_analysis_preset_export, begin_analysis_preset_import,
};
use application::{
    begin_lifecycle_request, begin_settings_mutation, enqueue_recent_project_persistence,
};
pub use channel_io::read_channel_intensity_stats;
use channels::begin_channel_intensity;
use completion::{finish_load, reject_actor_request};
use datasets::{
    begin_dataset_inspection, begin_ome_zarr_load, begin_spatialdata_load, begin_tiff_load,
    begin_xenium_load,
};
use deep_links::{
    apply_deep_link_on_worker, begin_deep_link_application, begin_deep_link_resolution,
    deep_link_resolution_response, resolve_deep_link_on_worker,
};
pub use diagnostics::ActorDiagnostics;
use dispatch::dispatch_request;
use jobs::{
    AnalysisComputeKind, CompletionDomain, DeepLinkApplySpec, DeepLinkApplyWorkerResult,
    DeepLinkResolveWorkerResult, LoadCompletion, LoadJob, MemoryPinWorkerOutcome,
    MemoryPinWorkerResult, MosaicOpenWorkerResult, ProjectObjectPreloadWorkerResult,
    ProjectRoiOpenSpec, ProjectRoiOpenWorkerResult,
};
use mask_io::export_mask_layers_geojson;
use masks::{begin_mask_export, begin_mask_import};
use measurements::begin_measurement;
use memory::begin_memory_pin;
use mosaics::{
    begin_mosaic_object_load, begin_mosaic_open, load_mosaic_objects_on_worker,
    open_mosaic_project_on_worker, open_mosaic_samplesheet_on_worker,
};
use object_exports::begin_object_export;
use objects::{begin_object_filter_evaluation, begin_object_selection_filter_evaluation};
use project_io::{
    discover_omezarr_roots_under, export_samplesheet_rois, import_samplesheet_rois,
    inspect_samplesheet, read_project_file,
};
use project_preload::{begin_project_object_preload, begin_project_object_source_scan};
use project_roi::begin_project_roi_open;
use projection::publish_projection;
pub use projection::{PlatformEffect, RenderDocument, RenderProjection};
use projects::{
    begin_project_discovery, begin_project_open, begin_project_save, begin_samplesheet_export,
    begin_samplesheet_import, begin_samplesheet_inspect,
};
use remote::{
    RemoteOpenIdentity, RemoteOpenSpec, RemoteSessionState, begin_remote_http_open,
    begin_remote_list, begin_remote_s3_open,
};
use request::{expand_path, finish_request, reject_worker_submission};
use resources::{begin_label_load, begin_object_resource_load};
pub use routing::execution_diagnostics;
pub use runtime::{
    ActorModelUpdate, ControlActorChannels, spawn_control_actor,
    spawn_control_actor_with_object_loader, spawn_control_actor_with_services,
};
use screenshots::begin_screenshot_settings_update;
use thresholds::{begin_threshold_apply, begin_threshold_configure, begin_threshold_load};
use worker::{open_project_roi_on_worker, spawn_resource_workers};

const ACTOR_QUEUE_CAPACITY: usize = 256;
const WORKER_COMPLETION_CAPACITY: usize = 64;
const LOAD_JOB_CAPACITY: usize = 32;
const LOAD_WORKERS: usize = 2;

pub type UiWake = Arc<dyn Fn() + Send + Sync>;

#[cfg(test)]
mod tests;
