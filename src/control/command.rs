use serde::Deserialize;
use serde_json::{Value, json};

use super::ControlError;
use super::registry::{self, MethodDescriptor, RequestShape};

#[derive(Debug, Clone)]
pub struct ControlCommand {
    descriptor: &'static MethodDescriptor,
    params: Value,
    if_revision: Option<u64>,
}

impl ControlCommand {
    pub fn decode(method: &str, params: Value) -> Result<Self, ControlError> {
        let descriptor = registry::method(method).ok_or_else(|| {
            ControlError::new(
                super::ControlErrorKind::MethodNotFound,
                format!("unknown Odon control method '{method}'"),
            )
            .with_data(json!({"method": method}))
        })?;
        let mut params = if params.is_null() { json!({}) } else { params };
        if !params.is_object() {
            return Err(ControlError::invalid_params(
                method,
                "params must be an object",
            ));
        }
        let if_revision = params
            .as_object_mut()
            .and_then(|object| object.remove("if_revision"))
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    ControlError::invalid_params(method, "if_revision must be an unsigned integer")
                })
            })
            .transpose()?;
        if if_revision.is_some() && !descriptor.mutates {
            return Err(ControlError::invalid_params(
                method,
                "if_revision is only valid for mutating methods",
            ));
        }
        validate_params(method, descriptor.request_shape, &params)?;
        Ok(Self {
            descriptor,
            params,
            if_revision,
        })
    }

    pub fn method(&self) -> &'static str {
        self.descriptor.name
    }

    pub fn params(&self) -> &Value {
        &self.params
    }

    pub fn mutates(&self) -> bool {
        self.descriptor.mutates
    }

    pub fn starts_task(&self) -> bool {
        self.descriptor.starts_task
    }

    pub fn event_name(&self) -> Option<&'static str> {
        self.descriptor.event
    }

    pub fn available_in(&self) -> &'static [&'static str] {
        self.descriptor.available_in
    }

    pub fn if_revision(&self) -> Option<u64> {
        self.if_revision
    }
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
    #[serde(default)]
    overwrite: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AppSettingsRequest {
    auto_contrast: Option<AutoContrastRequest>,
    fast_object_rendering: Option<bool>,
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
    id: String,
    path: String,
    display_name: Option<String>,
    dataset: Option<String>,
    segmentation_path: Option<String>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
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
    changes: ProjectRoiChanges,
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

fn validate_optional_nullable_string(
    method: &str,
    name: &str,
    value: Option<&Value>,
) -> Result<(), ControlError> {
    if value.is_some_and(|value| !(value.is_null() || value.is_string())) {
        return Err(ControlError::invalid_params(
            method,
            format!("{name} must be a string or null"),
        ));
    }
    Ok(())
}

fn validate_project_view_selector(
    method: &str,
    index: Option<usize>,
    name: Option<&str>,
) -> Result<(), ControlError> {
    if index.is_some() == name.is_some() {
        return Err(ControlError::invalid_params(
            method,
            "provide exactly one of index or name",
        ));
    }
    if name.is_some_and(|name| name.trim().is_empty()) {
        return Err(ControlError::invalid_params(
            method,
            "view preset name must not be empty",
        ));
    }
    Ok(())
}

fn validate_native_layer_selector(
    method: &str,
    layer_id: Option<&str>,
    id: Option<&str>,
) -> Result<(), ControlError> {
    if layer_id.or(id).is_none_or(|value| value.trim().is_empty()) {
        return Err(ControlError::invalid_params(
            method,
            "a non-empty layer_id is required",
        ));
    }
    if layer_id.is_some() && id.is_some() {
        return Err(ControlError::invalid_params(
            method,
            "provide layer_id or id, not both",
        ));
    }
    Ok(())
}

fn has_channel_selector(
    index: Option<usize>,
    channel_index: Option<usize>,
    name: Option<&str>,
    channel: Option<&ChannelSelector>,
    marker: Option<&str>,
) -> bool {
    index.is_some()
        || channel_index.is_some()
        || name.is_some_and(|value| !value.trim().is_empty())
        || channel.is_some()
        || marker.is_some_and(|value| !value.trim().is_empty())
}

fn validate_params(method: &str, shape: RequestShape, params: &Value) -> Result<(), ControlError> {
    let invalid = |error: serde_json::Error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    };
    match shape {
        RequestShape::Empty => {
            if params.as_object().is_some_and(|object| !object.is_empty()) {
                return Err(ControlError::invalid_params(
                    method,
                    "this method does not accept parameters",
                ));
            }
        }
        RequestShape::SetSidePanels => {
            let request: SetSidePanelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.left.is_none() && request.right.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "left and/or right is required",
                ));
            }
        }
        RequestShape::SetSmoothPixels => {
            let request: SetSmoothPixelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let _ = request.smooth;
        }
        RequestShape::SetVisibleChannels => {
            let request: SetVisibleChannelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.channels.is_empty() && request.mode.as_deref() != Some("only") {
                return Err(ControlError::invalid_params(
                    method,
                    "channels must not be empty unless mode is 'only'",
                ));
            }
            if let Some(mode) = request.mode.as_deref()
                && !matches!(mode, "only" | "show" | "hide")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be 'only', 'show', or 'hide'",
                ));
            }
            for channel in request.channels {
                match channel {
                    ChannelSelector::Name(name) if name.trim().is_empty() => {
                        return Err(ControlError::invalid_params(
                            method,
                            "channel names must not be empty",
                        ));
                    }
                    ChannelSelector::Name(_) => {}
                    ChannelSelector::Index(index) => {
                        let _ = index;
                    }
                }
            }
        }
        RequestShape::SetCamera => {
            let request: SetCameraRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let values = request
                .center_world_lvl0
                .into_iter()
                .flatten()
                .chain(request.center_x)
                .chain(request.center_y);
            if values.into_iter().any(|value| !value.is_finite()) {
                return Err(ControlError::invalid_params(
                    method,
                    "camera coordinates must be finite",
                ));
            }
            for zoom in [request.zoom, request.zoom_screen_per_lvl0_px]
                .into_iter()
                .flatten()
            {
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err(ControlError::invalid_params(
                        method,
                        "zoom must be finite and greater than zero",
                    ));
                }
            }
        }
        RequestShape::CaptureScreenshot => {
            let request: CaptureScreenshotRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .path
                .as_deref()
                .is_some_and(|path| path.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = request.overwrite;
        }
        RequestShape::AppSettings => {
            let request: AppSettingsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if let Some(auto) = request.auto_contrast {
                if let Some(method_name) = auto.method
                    && !matches!(
                        method_name.as_str(),
                        "zero_to_p97" | "p1_to_p99" | "zero_to_max"
                    )
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "unknown auto-contrast method",
                    ));
                }
                if auto.lower_percentile.is_some_and(|value| value > 99)
                    || auto
                        .upper_percentile
                        .is_some_and(|value| value == 0 || value > 100)
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "auto-contrast percentiles must be in range",
                    ));
                }
                if let (Some(lower), Some(upper)) = (auto.lower_percentile, auto.upper_percentile)
                    && lower >= upper
                {
                    return Err(ControlError::invalid_params(
                        method,
                        "lower_percentile must be less than upper_percentile",
                    ));
                }
                let _ = auto.enabled_on_open;
            }
            let _ = request.fast_object_rendering;
        }
        RequestShape::LifecycleRequest => {
            let request: LifecycleRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .save
                .as_deref()
                .is_some_and(|value| !matches!(value, "prompt" | "save" | "discard"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "save must be prompt, save, or discard",
                ));
            }
        }
        RequestShape::SetScaleBar => {
            let request: SetScaleBarRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let _ = request.visible;
        }
        RequestShape::ScreenshotSettings => {
            let request: ScreenshotSettingsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .output_dir
                .as_ref()
                .and_then(|value| value.as_deref())
                .is_some_and(|path| path.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "output_dir must not be empty",
                ));
            }
            for value in [request.scale_bar_scale, request.legend_scale]
                .into_iter()
                .flatten()
            {
                if !value.is_finite() || !(0.5..=3.0).contains(&value) {
                    return Err(ControlError::invalid_params(
                        method,
                        "screenshot scales must be between 0.5 and 3.0",
                    ));
                }
            }
            let _ = (request.include_scale_bar, request.include_legend);
        }
        RequestShape::TileLoading => {
            let request: TileLoadingRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .workers
                .is_some_and(|value| !(1..=12).contains(&value))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "workers must be from 1 to 12",
                ));
            }
            if request.prefetch_mode.as_deref().is_some_and(|value| {
                !matches!(value, "off" | "target_halo" | "target_and_finer_halo")
            }) {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown prefetch_mode",
                ));
            }
            if request
                .prefetch_aggressiveness
                .as_deref()
                .is_some_and(|value| !matches!(value, "conservative" | "balanced" | "aggressive"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown prefetch_aggressiveness",
                ));
            }
            let _ = request.prefer_pinned_finer_levels;
        }
        RequestShape::MemoryPin => {
            let request: MemoryPinRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .scope
                .as_deref()
                .is_some_and(|value| !matches!(value, "focused" | "item" | "all"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scope must be focused, item, or all",
                ));
            }
            if request.scope.as_deref() == Some("item") && request.item.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "item is required when scope is item",
                ));
            }
            if request
                .item
                .as_ref()
                .is_some_and(|value| !(value.is_string() || value.is_u64()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "item must be a string or non-negative integer",
                ));
            }
            let _ = (request.level, request.channels, request.force);
        }
        RequestShape::MemoryUnpin => {
            let request: MemoryUnpinRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .scope
                .as_deref()
                .is_some_and(|value| !matches!(value, "focused" | "item" | "all"))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scope must be focused, item, or all",
                ));
            }
            if request.scope.as_deref() == Some("item") && request.item.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "item is required when scope is item",
                ));
            }
            if request
                .item
                .as_ref()
                .is_some_and(|value| !(value.is_string() || value.is_u64()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "item must be a string or non-negative integer",
                ));
            }
            let _ = request.level;
        }
        RequestShape::LabelLoad => {
            let request: LabelLoadRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "label name must not be empty",
                ));
            }
        }
        RequestShape::LabelVisibility => {
            let request: LabelVisibilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "label name must not be empty",
                ));
            }
            let _ = request.visible;
        }
        RequestShape::ChannelPresentation => {
            let request: ChannelPresentationRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.search.is_none() && request.sort.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "search and/or sort is required",
                ));
            }
            if request
                .search
                .as_ref()
                .is_some_and(|search| search.len() > 4096)
            {
                return Err(ControlError::invalid_params(method, "search is too long"));
            }
            if request.sort.as_deref().is_some_and(|sort| {
                !matches!(
                    sort,
                    "manual" | "name_asc" | "name_desc" | "visible_first" | "hidden_first"
                )
            }) {
                return Err(ControlError::invalid_params(
                    method,
                    "unknown channel sort mode",
                ));
            }
        }
        RequestShape::MethodAvailability => {
            let request: MethodAvailabilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if let Some(methods) = request.methods {
                if methods.len() > 256 {
                    return Err(ControlError::invalid_params(
                        method,
                        "methods must contain at most 256 entries",
                    ));
                }
                let mut seen = std::collections::BTreeSet::new();
                for candidate in methods {
                    if candidate.trim().is_empty() {
                        return Err(ControlError::invalid_params(
                            method,
                            "method names must not be empty",
                        ));
                    }
                    let Some(descriptor) = registry::method(&candidate) else {
                        return Err(ControlError::invalid_params(
                            method,
                            format!("unknown method '{candidate}'"),
                        ));
                    };
                    if !seen.insert(descriptor.name) {
                        return Err(ControlError::invalid_params(
                            method,
                            format!("method '{}' is duplicated", descriptor.name),
                        ));
                    }
                }
            }
        }
        RequestShape::SetPlane => {
            let request: SetPlaneRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.mode.is_none() && request.slice.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "mode and/or slice is required",
                ));
            }
            if let Some(mode) = request.mode.as_deref()
                && !matches!(mode.to_ascii_lowercase().as_str(), "xy" | "xz" | "yz")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be 'xy', 'xz', or 'yz'",
                ));
            }
        }
        RequestShape::StepPlane => {
            let request: StepPlaneRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.step == 0 {
                return Err(ControlError::invalid_params(
                    method,
                    "step must be greater than zero",
                ));
            }
            let _ = request.wrap;
        }
        RequestShape::SetChannelColor => {
            let request: SetChannelColorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            let _ = request.color_rgb;
        }
        RequestShape::SetChannelNote => {
            let request: SetChannelNoteRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            if request.note.len() > 16_384 {
                return Err(ControlError::invalid_params(
                    method,
                    "note must be at most 16384 bytes",
                ));
            }
        }
        RequestShape::SetChannelTransform => {
            let request: SetChannelTransformRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !has_channel_selector(
                request.index,
                request.channel_index,
                request.name.as_deref(),
                request.channel.as_ref(),
                request.marker.as_deref(),
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "a channel selector is required",
                ));
            }
            if request.offset_world.is_none()
                && request.scale.is_none()
                && request.rotation_rad.is_none()
            {
                return Err(ControlError::invalid_params(
                    method,
                    "offset_world, scale, and/or rotation_rad is required",
                ));
            }
            if request
                .offset_world
                .into_iter()
                .flatten()
                .chain(request.rotation_rad)
                .any(|value| !value.is_finite())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "transform values must be finite",
                ));
            }
            if request
                .scale
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || !(0.01..=100.0).contains(&value))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "scale values must be finite and between 0.01 and 100",
                ));
            }
        }
        RequestShape::NativeLayerSelector => {
            let request: NativeLayerSelectorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
        }
        RequestShape::NativeLayerVisibility => {
            let request: NativeLayerVisibilityRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
            let _ = request.visible;
        }
        RequestShape::NativeLayerOrder => {
            let request: NativeLayerOrderRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(request.stack.as_str(), "channels" | "overlays") {
                return Err(ControlError::invalid_params(
                    method,
                    "stack must be 'channels' or 'overlays'",
                ));
            }
            if request.layers.len() > 4096 {
                return Err(ControlError::invalid_params(
                    method,
                    "layers must contain at most 4096 entries",
                ));
            }
            let mut unique = std::collections::BTreeSet::new();
            if request
                .layers
                .iter()
                .any(|layer| layer.trim().is_empty() || !unique.insert(layer))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "layers must contain unique, non-empty layer IDs",
                ));
            }
        }
        RequestShape::NativeLayerOffset => {
            let request: NativeLayerOffsetRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_native_layer_selector(
                method,
                request.layer_id.as_deref(),
                request.id.as_deref(),
            )?;
            if request
                .offset_world
                .into_iter()
                .any(|value| !value.is_finite())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "offset_world values must be finite",
                ));
            }
        }
        RequestShape::ProjectViewSelector => {
            let request: ProjectViewSelectorRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_project_view_selector(method, request.index, request.name.as_deref())?;
        }
        RequestShape::ProjectViewCreate => {
            let request: ProjectViewCreateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "name must not be empty",
                ));
            }
            if request.spec.as_ref().is_some_and(|spec| !spec.is_object()) {
                return Err(ControlError::invalid_params(
                    method,
                    "spec must be an object",
                ));
            }
        }
        RequestShape::ProjectViewCapture => {
            let request: ProjectViewCaptureRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "name must not be empty",
                ));
            }
        }
        RequestShape::ProjectViewRename => {
            let request: ProjectViewRenameRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_project_view_selector(method, request.index, request.name.as_deref())?;
            if request.new_name.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "new_name must not be empty",
                ));
            }
        }
        RequestShape::ProjectCreate => {
            let request: ProjectCreateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .default_dataset
                .as_deref()
                .is_some_and(|value| value.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "default_dataset must not be empty",
                ));
            }
        }
        RequestShape::Path => {
            let request: PathRequest = serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
        }
        RequestShape::ProjectMetadata => {
            let request: ProjectMetadataRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            validate_optional_nullable_string(
                method,
                "default_dataset",
                request.default_dataset.as_ref(),
            )?;
            validate_optional_nullable_string(
                method,
                "secondary_dataset",
                request.secondary_dataset.as_ref(),
            )?;
            validate_optional_nullable_string(
                method,
                "default_threshold_marker",
                request.default_threshold_marker.as_ref(),
            )?;
            if request
                .mosaic_segmentation_search_roots
                .as_ref()
                .is_some_and(|roots| {
                    roots.len() > 4096 || roots.iter().any(|root| root.trim().is_empty())
                })
            {
                return Err(ControlError::invalid_params(
                    method,
                    "search roots must contain at most 4096 non-empty paths",
                ));
            }
        }
        RequestShape::SamplesheetInspect => {
            let request: SamplesheetInspectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() || request.limit == 0 || request.limit > 10_000 {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty and limit must be between 1 and 10000",
                ));
            }
            let _ = request.offset;
        }
        RequestShape::SamplesheetExport => {
            let request: SamplesheetExportRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = request.overwrite;
        }
        RequestShape::SpatialDataOpen => {
            let request: SpatialDataOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty()
                || request.image.trim().is_empty()
                || request.points_max > 200_000_000
                || request
                    .extra_images
                    .iter()
                    .chain(&request.shapes)
                    .any(|name| name.trim().is_empty())
                || request
                    .labels
                    .as_deref()
                    .into_iter()
                    .chain(request.points.as_deref())
                    .any(|name| name.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "SpatialData paths and element names must be non-empty and points_max must not exceed 200000000",
                ));
            }
        }
        RequestShape::XeniumOpen => {
            let request: XeniumOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty()
                || !matches!(request.imagery.as_str(), "auto" | "ome_zarr" | "tiff")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty and imagery must be auto, ome_zarr, or tiff",
                ));
            }
            let _ = (request.load_cells, request.load_transcripts);
        }
        RequestShape::HttpOpen => {
            let request: HttpOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let url = request.url.trim();
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(ControlError::invalid_params(
                    method,
                    "url must begin with http:// or https://",
                ));
            }
        }
        RequestShape::S3Session => {
            let request: S3SessionRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if [
                request.endpoint.as_str(),
                request.region.as_str(),
                request.bucket.as_str(),
                request.access_key.as_str(),
                request.secret_key.as_str(),
            ]
            .iter()
            .any(|value| value.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "endpoint, region, bucket, access_key, and secret_key must not be empty",
                ));
            }
        }
        RequestShape::S3Prefix => {
            let request: S3PrefixRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.prefix.contains('\0') {
                return Err(ControlError::invalid_params(
                    method,
                    "prefix must not contain NUL characters",
                ));
            }
        }
        RequestShape::TiffOpen => {
            let request: TiffOpenRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
            let _ = (request.z, request.t);
        }
        RequestShape::ProjectRoiId => {
            let request: ProjectRoiIdRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.id.trim().is_empty() {
                return Err(ControlError::invalid_params(method, "id must not be empty"));
            }
        }
        RequestShape::ProjectRoiAdd => {
            let request: ProjectRoiAddRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.id.trim().is_empty() || request.path.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "id and path must not be empty",
                ));
            }
            let _ = (
                request.display_name,
                request.dataset,
                request.segmentation_path,
                request.metadata,
            );
        }
        RequestShape::ProjectRoiUpdate => {
            let request: ProjectRoiUpdateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.target_id.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "target_id must not be empty",
                ));
            }
            let changes = request.changes;
            let changes_object = params
                .get("changes")
                .and_then(Value::as_object)
                .expect("deserialized project ROI changes must be an object");
            if changes_object.is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "changes must not be empty",
                ));
            }
            for name in ["display_name", "dataset", "segmentation_path"] {
                let value = changes_object.get(name);
                validate_optional_nullable_string(method, name, value)?;
            }
            let _ = (
                changes.id,
                changes.path,
                changes.display_name,
                changes.dataset,
                changes.segmentation_path,
                changes.metadata,
            );
        }
        RequestShape::ProjectRoiOrder => {
            let request: ProjectRoiOrderRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let mut unique = std::collections::BTreeSet::new();
            if request.ids.len() > 100_000
                || request
                    .ids
                    .iter()
                    .any(|id| id.trim().is_empty() || !unique.insert(id))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must contain at most 100000 unique, non-empty ROI IDs",
                ));
            }
        }
        RequestShape::ProjectRoiSelect => {
            let request: ProjectRoiSelectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(
                request.mode.as_str(),
                "replace" | "add" | "remove" | "toggle"
            ) {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be replace, add, remove, or toggle",
                ));
            }
            let mut unique = std::collections::BTreeSet::new();
            if request.ids.len() > 100_000
                || request
                    .ids
                    .iter()
                    .any(|id| id.trim().is_empty() || !unique.insert(id))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must contain at most 100000 unique, non-empty ROI IDs",
                ));
            }
        }
        RequestShape::MosaicFocus => {
            let request: MosaicFocusRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let selector_count = usize::from(request.index.is_some())
                + usize::from(request.roi_id.is_some())
                + usize::from(request.id.is_some());
            if selector_count != 1 {
                return Err(ControlError::invalid_params(
                    method,
                    "provide exactly one of index, roi_id, or id",
                ));
            }
            if request
                .roi_id
                .as_deref()
                .or(request.id.as_deref())
                .is_some_and(|id| id.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mosaic ROI ID must not be empty",
                ));
            }
            let _ = request.fit;
        }
        RequestShape::MosaicItems => {
            let request: MosaicItemsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.limit == 0 || request.limit > 10_000 {
                return Err(ControlError::invalid_params(
                    method,
                    "limit must be between 1 and 10000",
                ));
            }
            let _ = request.offset;
        }
        RequestShape::MosaicSelect => {
            let request: MosaicSelectRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let mut unique = std::collections::BTreeSet::new();
            if request
                .ids
                .iter()
                .any(|id| id.trim().is_empty() || !unique.insert(id.as_str()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "ids must be unique and non-empty",
                ));
            }
            match request.mode.as_str() {
                "all"
                    if request.ids.is_empty()
                        && request.start.is_none()
                        && request.end.is_none() => {}
                "range"
                    if request.ids.is_empty()
                        && request
                            .start
                            .as_deref()
                            .is_some_and(|id| !id.trim().is_empty())
                        && request
                            .end
                            .as_deref()
                            .is_some_and(|id| !id.trim().is_empty()) => {}
                "replace" | "add" | "remove" | "toggle"
                    if !request.ids.is_empty()
                        && request.start.is_none()
                        && request.end.is_none() => {}
                _ => {
                    return Err(ControlError::invalid_params(
                        method,
                        "all takes no IDs, range requires start/end, and other modes require ids",
                    ));
                }
            }
        }
        RequestShape::MosaicLayout => {
            let request: MosaicLayoutRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if params.as_object().is_none_or(|params| params.is_empty()) {
                return Err(ControlError::invalid_params(
                    method,
                    "at least one layout property is required",
                ));
            }
            if request.columns == Some(0)
                || request
                    .group_gap
                    .is_some_and(|gap| !gap.is_finite() || gap < 0.0)
                || request.layout.is_some() && request.layout_mode.is_some()
                || request
                    .label_columns
                    .as_ref()
                    .is_some_and(|columns| columns.iter().any(|column| column.trim().is_empty()))
            {
                return Err(ControlError::invalid_params(
                    method,
                    "layout fields are invalid or conflicting",
                ));
            }
            let _ = (
                request.group_by,
                request.sort_by,
                request.sort_by_secondary,
                request.sort_secondary_enabled,
                request.show_group_labels,
                request.show_text_labels,
                request.fit,
            );
        }
        RequestShape::ObjectPreloadStart => {
            let request: ObjectPreloadStartRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if !matches!(request.mode.as_str(), "full_geometry" | "centroid_points") {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be full_geometry or centroid_points",
                ));
            }
            let _ = request.lazy_properties;
        }
        RequestShape::DeepLinkUri => {
            let request: DeepLinkUriRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.url.trim().is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    "url must not be empty",
                ));
            }
            crate::deep_link::DeepLinkRequest::parse_arg(&request.url)
                .map_err(|error| ControlError::invalid_params(method, error.to_string()))?
                .ok_or_else(|| {
                    ControlError::invalid_params(method, "url must use the odon: scheme")
                })?;
        }
        RequestShape::DeepLinkGenerate => {
            let request: DeepLinkGenerateRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .roi
                .as_deref()
                .is_some_and(|roi| roi.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "roi must not be empty",
                ));
            }
            let _ = (request.request, request.include_project);
        }
        RequestShape::DeepLinkApply => {
            let request: DeepLinkApplyRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.url.is_some() == request.request.is_some() {
                return Err(ControlError::invalid_params(
                    method,
                    "provide exactly one of url or request",
                ));
            }
            if let Some(url) = request.url {
                crate::deep_link::DeepLinkRequest::parse_arg(&url)
                    .map_err(|error| ControlError::invalid_params(method, error.to_string()))?
                    .ok_or_else(|| {
                        ControlError::invalid_params(method, "url must use the odon: scheme")
                    })?;
            }
        }
        RequestShape::Object => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_commands_validate_representative_parameters() {
        assert!(ControlCommand::decode("get_camera", json!({})).is_ok());
        assert!(ControlCommand::decode("get_camera", json!({"extra": true})).is_err());
        assert!(ControlCommand::decode("set_side_panels", json!({})).is_err());
        assert!(ControlCommand::decode("set_side_panels", json!({"left": false})).is_ok());
        assert!(
            ControlCommand::decode(
                "set_visible_channels",
                json!({"channels": ["DAPI", 2], "mode": "only"})
            )
            .is_ok()
        );
        assert!(ControlCommand::decode("set_camera", json!({"zoom": 0.0})).is_err());
        let command = ControlCommand::decode("set_camera", json!({"zoom": 2.0, "if_revision": 4}))
            .expect("revision precondition");
        assert_eq!(command.if_revision(), Some(4));
        assert_eq!(command.params(), &json!({"zoom": 2.0}));
        assert_eq!(command.method(), "viewer.camera.set");
        assert_eq!(command.event_name(), Some("viewer.camera.changed"));
        assert!(command.available_in().contains(&"single"));
        assert!(ControlCommand::decode("get_camera", json!({"if_revision": 4})).is_err());
    }

    #[test]
    fn phase_g_commands_have_typed_validation() {
        assert!(ControlCommand::decode("app.settings.set", json!({
            "auto_contrast": {"method": "p1_to_p99", "lower_percentile": 1, "upper_percentile": 99},
            "fast_object_rendering": false
        })).is_ok());
        assert!(
            ControlCommand::decode(
                "app.settings.set",
                json!({
                    "auto_contrast": {"lower_percentile": 99, "upper_percentile": 20}
                })
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode("app.lifecycle.request_close", json!({"save": "prompt"}))
                .is_ok()
        );
        assert!(
            ControlCommand::decode("app.lifecycle.request_close", json!({"save": "maybe"}))
                .is_err()
        );
        assert!(ControlCommand::decode("viewer.scale_bar.set", json!({"visible": true})).is_ok());
        assert!(
            ControlCommand::decode(
                "viewer.screenshot.settings.set",
                json!({"legend_scale": 4.0})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "memory.tiles.set",
                json!({"workers": 6, "prefetch_mode": "target_halo"})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode("memory.pin", json!({"level": 1, "scope": "item"})).is_err()
        );
        assert!(
            ControlCommand::decode(
                "memory.pin",
                json!({"level": 1, "scope": "all", "force": true})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode("memory.unpin", json!({"level": 1, "unknown": true})).is_err()
        );
    }

    #[test]
    fn plane_commands_have_typed_validation() {
        assert!(ControlCommand::decode("viewer.planes.get", json!({})).is_ok());
        assert!(ControlCommand::decode("viewer.planes.set", json!({})).is_err());
        assert!(
            ControlCommand::decode("viewer.planes.set", json!({"mode": "XZ", "slice": 12})).is_ok()
        );
        assert!(ControlCommand::decode("viewer.planes.set", json!({"mode": "time"})).is_err());
        assert!(ControlCommand::decode("viewer.planes.next", json!({"step": 0})).is_err());
        let command = ControlCommand::decode(
            "viewer.planes.previous",
            json!({"step": 3, "wrap": true, "if_revision": 9}),
        )
        .expect("valid plane step");
        assert_eq!(command.if_revision(), Some(9));
        assert_eq!(command.event_name(), Some("viewer.planes.changed"));
        assert_eq!(command.available_in(), &["single"]);
    }

    #[test]
    fn channel_property_commands_have_typed_validation() {
        assert!(
            ControlCommand::decode(
                "viewer.channels.set_color",
                json!({"name": "DAPI", "color_rgb": [1, 2, 255]})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode(
                "viewer.channels.set_color",
                json!({"name": "DAPI", "color_rgb": [1, 2, 256]})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "viewer.channels.set_note",
                json!({"note": "missing selector"})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "viewer.channels.set_transform",
                json!({"index": 0, "scale": [0.01, 100.0], "rotation_rad": 0.5})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode(
                "viewer.channels.set_transform",
                json!({"index": 0, "scale": [0.0, 1.0]})
            )
            .is_err()
        );
    }

    #[test]
    fn native_layer_commands_have_typed_validation() {
        assert!(
            ControlCommand::decode(
                "viewer.native_layers.set_visibility",
                json!({"layer_id": "channel:0", "visible": false})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode(
                "viewer.native_layers.set_visibility",
                json!({"layer_id": "channel:0"})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "viewer.native_layers.set_order",
                json!({"stack": "channels", "layers": ["channel:1", "channel:1"]})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "viewer.native_layers.set_offset",
                json!({"layer_id": "mask:2", "offset_world": [1.0, 2.0]})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode(
                "viewer.native_layers.get",
                json!({"layer_id": "channel:0", "id": "channel:0"})
            )
            .is_err()
        );
    }

    #[test]
    fn project_view_commands_have_typed_validation() {
        assert!(ControlCommand::decode("project.views.get", json!({"name": "Review"})).is_ok());
        assert!(
            ControlCommand::decode("project.views.get", json!({"index": 0, "name": "Review"}))
                .is_err()
        );
        assert!(
            ControlCommand::decode(
                "project.views.create",
                json!({"name": "Review", "spec": {"visible_channels": ["DAPI"]}})
            )
            .is_ok()
        );
        assert!(
            ControlCommand::decode("project.views.create", json!({"name": "Review", "spec": 4}))
                .is_err()
        );
        assert!(
            ControlCommand::decode(
                "project.views.rename",
                json!({"index": 0, "new_name": "Overview"})
            )
            .is_ok()
        );
    }

    #[test]
    fn project_roi_mosaic_and_deep_link_commands_are_typed() {
        assert!(
            ControlCommand::decode(
                "project.rois.add",
                json!({"id": "ROI-1", "path": "/tmp/roi.zarr"})
            )
            .is_ok()
        );
        assert!(ControlCommand::decode("project.rois.add", json!({"id": "ROI-1"})).is_err());
        assert!(
            ControlCommand::decode(
                "project.rois.update",
                json!({"target_id": "ROI-1", "changes": {}})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode(
                "project.rois.select",
                json!({"ids": ["ROI-1"], "mode": "unexpected"})
            )
            .is_err()
        );
        assert!(
            ControlCommand::decode("mosaic.focus.set", json!({"index": 0, "roi_id": "ROI-1"}))
                .is_err()
        );
        assert!(
            ControlCommand::decode("deep_links.parse", json!({"url": "odon://open?roi=ROI-1"}))
                .is_ok()
        );
        assert!(
            ControlCommand::decode("deep_links.parse", json!({"url": "https://example.com"}))
                .is_err()
        );
        assert!(
            ControlCommand::decode(
                "deep_links.apply",
                json!({"url": "odon://open", "request": {}})
            )
            .is_err()
        );
        assert!(ControlCommand::decode("deep_links.apply", json!({"request": {}})).is_ok());
    }
}
