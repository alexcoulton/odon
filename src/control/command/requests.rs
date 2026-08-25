//! Typed request DTOs and deserialization defaults for control commands.

use serde::Deserialize;
use serde_json::Value;

use crate::control::ControlError;
use crate::control::registry::{self, RequestShape};
use crate::data::project_config::ProjectRoi;

mod validation;

pub(super) use validation::validate_params;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellGetRequest {
    mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MenuReplaceRequest {
    if_command_revision: Option<u64>,
    transaction_id: Option<String>,
    menu: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolbarReplaceRequest {
    if_command_revision: Option<u64>,
    transaction_id: Option<String>,
    toolbar: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PaletteReplaceRequest {
    if_command_revision: Option<u64>,
    transaction_id: Option<String>,
    palette: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandRegisterRequest {
    extension_id: String,
    if_command_revision: Option<u64>,
    transaction_id: Option<String>,
    command: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandRemoveRequest {
    extension_id: String,
    command_id: String,
    if_command_revision: Option<u64>,
    transaction_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandExecuteRequest {
    command_id: String,
    checked: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandCleanupRequest {
    extensions: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandSyncRequest {
    context: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellImportLayoutRequest {
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
    document: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellPatchRequest {
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
    visibility: Option<std::collections::BTreeMap<String, bool>>,
    orders: Option<std::collections::BTreeMap<String, Vec<String>>>,
    selected: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellReplaceLayoutRequest {
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
    desired_tree: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellPatchLayoutRequest {
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
    visibility: Option<std::collections::BTreeMap<String, bool>>,
    selected: Option<std::collections::BTreeMap<String, String>>,
    sizes: Option<std::collections::BTreeMap<String, Value>>,
    splits: Option<std::collections::BTreeMap<String, Value>>,
    collapsed: Option<std::collections::BTreeMap<String, bool>>,
    configurations: Option<std::collections::BTreeMap<String, Value>>,
    active_region_id: Option<String>,
    focused_node_id: Option<String>,
    clear_focus: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellResetRequest {
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellProfileListRequest {
    scope: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellProfileSaveRequest {
    name: String,
    scope: Option<String>,
    mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellProfileLoadRequest {
    name: String,
    scope: Option<String>,
    mode: Option<String>,
    if_shell_revision: Option<u64>,
    transaction_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellProfileRemoveRequest {
    name: String,
    scope: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetSidePanelsRequest {
    left: Option<bool>,
    right: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetSmoothPixelsRequest {
    smooth: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChannelSelector {
    Name(String),
    Index(usize),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetVisibleChannelsRequest {
    channels: Vec<ChannelSelector>,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetCameraRequest {
    center_world_lvl0: Option<[f64; 2]>,
    center_x: Option<f64>,
    center_y: Option<f64>,
    zoom: Option<f64>,
    zoom_screen_per_lvl0_px: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureScreenshotRequest {
    path: Option<String>,
    viewport_id: Option<String>,
    #[serde(default)]
    overwrite: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AppSettingsRequest {
    auto_contrast: Option<AutoContrastRequest>,
    fast_object_rendering: Option<bool>,
    shell_layout_startup_profiles: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AutoContrastRequest {
    enabled_on_open: Option<bool>,
    method: Option<String>,
    lower_percentile: Option<u8>,
    upper_percentile: Option<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LifecycleRequest {
    save: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetScaleBarRequest {
    visible: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScreenshotSettingsRequest {
    output_dir: Option<Option<String>>,
    include_scale_bar: Option<bool>,
    include_legend: Option<bool>,
    scale_bar_scale: Option<f64>,
    legend_scale: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TileLoadingRequest {
    workers: Option<usize>,
    prefetch_mode: Option<String>,
    prefetch_aggressiveness: Option<String>,
    prefer_pinned_finer_levels: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryPinRequest {
    level: usize,
    channels: Option<Vec<ChannelSelector>>,
    scope: Option<String>,
    item: Option<Value>,
    #[serde(default)]
    force: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryUnpinRequest {
    level: usize,
    scope: Option<String>,
    item: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelLoadRequest {
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelVisibilityRequest {
    visible: bool,
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChannelPresentationRequest {
    search: Option<String>,
    sort: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodAvailabilityRequest {
    methods: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetPlaneRequest {
    mode: Option<String>,
    slice: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepPlaneRequest {
    #[serde(default = "default_plane_step")]
    step: u64,
    #[serde(default)]
    wrap: bool,
}

fn default_plane_step() -> u64 {
    1
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetChannelColorRequest {
    index: Option<usize>,
    channel_index: Option<usize>,
    name: Option<String>,
    channel: Option<ChannelSelector>,
    marker: Option<String>,
    color_rgb: [u8; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetChannelNoteRequest {
    index: Option<usize>,
    channel_index: Option<usize>,
    name: Option<String>,
    channel: Option<ChannelSelector>,
    marker: Option<String>,
    note: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetChannelTransformRequest {
    viewport_id: Option<String>,
    if_presentation_revision: Option<u64>,
    index: Option<usize>,
    channel_index: Option<usize>,
    name: Option<String>,
    channel: Option<ChannelSelector>,
    marker: Option<String>,
    offset_world: Option<[f64; 2]>,
    scale: Option<[f64; 2]>,
    rotation_rad: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLayerSelectorRequest {
    layer_id: Option<String>,
    id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLayerVisibilityRequest {
    layer_id: Option<String>,
    id: Option<String>,
    visible: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLayerOrderRequest {
    stack: String,
    layers: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLayerOffsetRequest {
    layer_id: Option<String>,
    id: Option<String>,
    offset_world: [f64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectViewSelectorRequest {
    index: Option<usize>,
    name: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectViewCreateRequest {
    name: String,
    spec: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectViewCaptureRequest {
    name: String,
    viewport_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectViewRenameRequest {
    index: Option<usize>,
    name: Option<String>,
    new_name: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectCreateRequest {
    default_dataset: Option<String>,
    config: Option<crate::data::project_config::ProjectConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathRequest {
    path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectMetadataRequest {
    default_dataset: Option<Value>,
    secondary_dataset: Option<Value>,
    default_threshold_marker: Option<Value>,
    mosaic_segmentation_search_roots: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplesheetInspectRequest {
    path: String,
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_samplesheet_limit")]
    limit: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplesheetExportRequest {
    path: String,
    #[serde(default)]
    overwrite: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpatialDataOpenRequest {
    path: String,
    image: String,
    #[serde(default)]
    extra_images: Vec<String>,
    labels: Option<String>,
    #[serde(default)]
    shapes: Vec<String>,
    points: Option<String>,
    #[serde(default = "default_points_max")]
    points_max: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct XeniumOpenRequest {
    path: String,
    #[serde(default = "default_xenium_imagery")]
    imagery: String,
    #[serde(default = "default_true")]
    load_cells: bool,
    #[serde(default = "default_true")]
    load_transcripts: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HttpOpenRequest {
    url: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct S3SessionRequest {
    endpoint: String,
    #[serde(default = "default_s3_region")]
    region: String,
    bucket: String,
    access_key: String,
    secret_key: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct S3PrefixRequest {
    #[serde(default)]
    prefix: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TiffOpenRequest {
    path: String,
    #[serde(default)]
    z: usize,
    #[serde(default)]
    t: usize,
}

fn default_s3_region() -> String {
    "auto".to_string()
}

fn default_points_max() -> usize {
    200_000
}

fn default_xenium_imagery() -> String {
    "auto".to_string()
}

fn default_samplesheet_limit() -> usize {
    200
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiIdRequest {
    id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiAddRequest {
    id: Option<String>,
    path: Option<String>,
    display_name: Option<String>,
    dataset: Option<String>,
    segmentation_path: Option<String>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
    replacement: Option<ProjectRoi>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiChanges {
    id: Option<String>,
    path: Option<String>,
    display_name: Option<Value>,
    dataset: Option<Value>,
    segmentation_path: Option<Value>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiUpdateRequest {
    target_id: String,
    changes: Option<ProjectRoiChanges>,
    replacement: Option<ProjectRoi>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiOrderRequest {
    ids: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectRoiSelectRequest {
    ids: Vec<String>,
    #[serde(default = "default_roi_selection_mode")]
    mode: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MosaicFocusRequest {
    index: Option<usize>,
    roi_id: Option<String>,
    id: Option<String>,
    #[serde(default = "default_true")]
    fit: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MosaicItemsRequest {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_samplesheet_limit")]
    limit: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MosaicSelectRequest {
    #[serde(default)]
    ids: Vec<String>,
    #[serde(default = "default_roi_selection_mode")]
    mode: String,
    start: Option<String>,
    end: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MosaicLayoutRequest {
    group_by: Option<String>,
    sort_by: Option<String>,
    sort_by_secondary: Option<String>,
    sort_secondary_enabled: Option<bool>,
    show_group_labels: Option<bool>,
    show_text_labels: Option<bool>,
    group_gap: Option<f64>,
    columns: Option<usize>,
    layout: Option<String>,
    layout_mode: Option<String>,
    label_columns: Option<Vec<String>>,
    fit: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObjectPreloadStartRequest {
    #[serde(default = "default_object_preload_mode")]
    mode: String,
    #[serde(default = "default_true")]
    lazy_properties: bool,
}

fn default_object_preload_mode() -> String {
    "full_geometry".to_string()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepLinkUriRequest {
    url: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepLinkGenerateRequest {
    request: Option<crate::deep_link::DeepLinkRequest>,
    #[serde(default = "default_true")]
    include_project: bool,
    roi: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeepLinkApplyRequest {
    url: Option<String>,
    request: Option<crate::deep_link::DeepLinkRequest>,
}

fn default_true() -> bool {
    true
}

fn default_roi_selection_mode() -> String {
    "replace".to_string()
}
