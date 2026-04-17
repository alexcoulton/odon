pub(crate) mod geojson;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, anyhow};
use crossbeam_channel::Receiver;
use eframe::egui;
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use zarrs::storage::ReadableStorageTraits;

use crate::data::ome::{ChannelInfo, OmeZarrDataset};
use crate::features::points::FeaturePointLod;
use crate::render::line_bins::{LineSegmentsBins, ObjectLineSegmentsBins};
use crate::render::line_bins_gl::{
    LineBinsGlDrawData, LineBinsGlDrawItem, LineBinsGlDrawParams, LineBinsGlRenderer,
    ObjectLineBinsGlDrawData, ObjectLineBinsGlDrawItem, ObjectLineBinsGlDrawParams,
    ObjectLineBinsGlRenderer,
};
use crate::render::points::PointsStyle;
use crate::render::points_gl::PointsGlRenderer;
use crate::render::polygon_fill_gl::{
    ObjectFillGlDrawData, ObjectFillGlDrawItem, ObjectFillGlDrawParams, ObjectFillGlRenderer,
    PolygonFillGlDrawData, PolygonFillGlDrawItem, PolygonFillGlDrawParams, PolygonFillGlRenderer,
};
use crate::spatialdata::{
    ShapesLoadOptions, ShapesObjectSchema, SpatialDataElement, SpatialDataTransform2,
    inspect_shapes_object_schema, load_shapes_objects, load_shapes_xy_point_objects,
};

mod analysis;
mod core;
mod measurements;
mod render;

use self::analysis::SimpleHistogram;
pub(crate) use self::geojson::GeoJsonSegmentationLayer;

#[derive(Debug, Clone)]
pub struct ObjectFeature {
    pub id: String,
    pub polygons_world: Vec<Vec<egui::Pos2>>,
    pub point_position_world: Option<egui::Pos2>,
    pub bbox_world: egui::Rect,
    pub area_px: f32,
    pub perimeter_px: f32,
    pub centroid_world: egui::Pos2,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub source_row_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SelectedObjectDetails {
    pub id: String,
    pub area_px: f32,
    pub perimeter_px: f32,
    pub centroid_world: egui::Pos2,
    pub properties: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct ObjectColorLegendEntry {
    pub value_label: String,
    pub count: usize,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectColorLevelOverride {
    pub visible: bool,
    pub color_rgb: Option<[u8; 3]>,
}

impl Default for ObjectColorLevelOverride {
    fn default() -> Self {
        Self {
            visible: true,
            color_rgb: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObjectsLayer {
    pub visible: bool,
    pub opacity: f32,
    pub width_screen_px: f32,
    pub color_rgb: [u8; 3],
    pub fill_cells: bool,
    pub fill_opacity: f32,
    pub selected_fill_opacity: f32,
    pub show_selection_overlay: bool,

    pub loaded_geojson: Option<PathBuf>,
    pub downsample_factor: f32,
    display_transform: SpatialDataTransform2,
    display_mode: ObjectDisplayMode,

    objects: Option<Arc<Vec<ObjectFeature>>>,
    bins: Option<Arc<ObjectIndexBins>>,
    render_lods: Option<Vec<ObjectRenderLod>>,
    object_fill_mesh: Option<ObjectFillMesh>,
    object_selection_lods: Option<Vec<ObjectSelectionRenderLod>>,
    point_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    point_values: Option<Arc<Vec<f32>>>,
    point_lods: Option<Arc<Vec<FeaturePointLod>>>,
    object_property_keys: Vec<String>,
    scalar_property_keys: Vec<String>,
    color_property_keys: Vec<String>,
    lazy_parquet_source: Option<LazyParquetSource>,
    color_legend_cache: Option<ObjectColorLegendCache>,
    color_groups: Option<ObjectColorGroups>,
    color_mode: ObjectColorMode,
    color_property_key: String,
    color_level_overrides_property_key: String,
    color_level_overrides: BTreeMap<String, ObjectColorLevelOverride>,
    filter_property_key: String,
    filter_query: String,
    filtered_indices: Option<HashSet<usize>>,
    filtered_render_lods: Option<Vec<ObjectRenderLod>>,
    filtered_point_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    filtered_point_values: Option<Arc<Vec<f32>>>,
    filtered_point_lods: Option<Arc<Vec<FeaturePointLod>>>,
    filtered_color_groups: Option<ObjectColorGroups>,
    selected_object_indices: HashSet<usize>,
    selected_object_index: Option<usize>,
    selection_elements: Vec<SelectionElement>,
    selection_element_selected: Option<usize>,
    selection_element_name_draft: String,
    selected_render_lods: Option<Vec<ObjectRenderLod>>,
    primary_selected_render_lods: Option<Vec<ObjectRenderLod>>,
    selected_fill_mesh: Option<SelectionFillMesh>,
    selection_fill_state: Arc<Vec<u8>>,
    selection_cpu_overlay_dirty: bool,
    selected_point_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    selected_point_values: Option<Arc<Vec<f32>>>,
    selected_point_lods: Option<Arc<Vec<FeaturePointLod>>>,
    primary_selected_point_positions_world: Option<Arc<Vec<egui::Pos2>>>,
    primary_selected_point_values: Option<Arc<Vec<f32>>>,
    selection_generation: u64,
    bulk_measurement_request_id: u64,
    bulk_measurement_rx: Option<Receiver<BulkMeasurementEvent>>,
    bulk_measurement_cancel: Option<Arc<AtomicBool>>,
    bulk_measurement_progress_completed: usize,
    bulk_measurement_progress_total: usize,
    bulk_measurement_status: String,
    bulk_measurement_metric: BulkMeasurementMetric,
    bulk_measurement_level: usize,
    bulk_measurement_concurrency: usize,
    bulk_measurement_filtered_only: bool,
    bulk_measurement_prefix: String,
    analysis_plot_mode: AnalysisPlotMode,
    analysis_hist_channel: usize,
    analysis_scatter_x_channel: usize,
    analysis_scatter_y_channel: usize,
    analysis_property_thresholds: Vec<ObjectPropertyThresholdRule>,
    analysis_threshold_set_name: String,
    analysis_threshold_elements: Vec<ThresholdSetElement>,
    analysis_threshold_selected_element: Option<usize>,
    analysis_follow_active_channel: bool,
    analysis_live_threshold_channel_name: Option<String>,
    analysis_channel_mapping_overrides: HashMap<String, String>,
    analysis_channel_mapping_popup_open: bool,
    analysis_channel_mapping_search: String,
    analysis_channel_mapping_suggestions_cache_key: u64,
    analysis_channel_mapping_suggestions_cache_channels_len: usize,
    analysis_channel_mapping_suggestions_cache_numeric_len: usize,
    analysis_channel_mapping_suggestions_cache: HashMap<String, Vec<String>>,
    analysis_hist_value_transform: HistogramValueTransform,
    analysis_hist_level_method: HistogramLevelMethod,
    analysis_hist_level_count: usize,
    analysis_hist_snapped_level: Option<HistogramLevelSelection>,
    analysis_hist_focus_object_index: Option<usize>,
    analysis_hist_drag_rule: Option<usize>,
    analysis_hist_brush: Option<(f32, f32)>,
    analysis_scatter_brush: Option<egui::Rect>,
    analysis_hist_drag_anchor: Option<f32>,
    analysis_scatter_drag_anchor: Option<egui::Pos2>,
    analysis_scatter_view_key: Option<String>,
    analysis_scatter_view_rect: Option<egui::Rect>,
    analysis_live_selection_generation: u64,
    analysis_live_selection_applied_generation: u64,
    object_property_numeric_keys_cache: Option<Vec<String>>,
    object_property_base_pairs_cache: HashMap<String, Arc<Vec<(usize, f32)>>>,
    object_property_base_sorted_pairs_cache: HashMap<String, Arc<Vec<(usize, f32)>>>,
    object_property_base_hist_cache: HashMap<(String, HistogramValueTransform), SimpleHistogram>,
    object_property_base_hist_levels_cache:
        HashMap<(String, HistogramValueTransform, HistogramLevelMethod, usize), Arc<Vec<f32>>>,
    object_property_pairs_cache: HashMap<String, Arc<Vec<(usize, f32)>>>,
    object_property_hist_cache: HashMap<(String, HistogramValueTransform), SimpleHistogram>,
    object_property_scatter_cache: HashMap<(String, String), Arc<Vec<(usize, f32, f32)>>>,
    object_property_hist_levels_cache:
        HashMap<(String, HistogramValueTransform, HistogramLevelMethod, usize), Arc<Vec<f32>>>,
    object_property_threshold_selection_cache_key: Option<String>,
    object_property_threshold_selection_cache: Arc<Vec<usize>>,
    object_property_threshold_order_cache_key: Option<String>,
    object_property_threshold_order_cache: Arc<Vec<usize>>,
    analysis_warm_started: bool,
    analysis_warm_request_id: u64,
    analysis_warm_rx: Option<Receiver<AnalysisWarmupEvent>>,
    analysis_warm_total_columns: usize,
    analysis_warm_completed_columns: usize,
    table_indices_cache: Vec<usize>,
    table_cache_dirty: bool,
    bounds_local: Option<egui::Rect>,
    generation: u64,
    gl: LineBinsGlRenderer,
    gl_object_selection: ObjectLineBinsGlRenderer,
    gl_fill: PolygonFillGlRenderer,
    gl_object_fill: ObjectFillGlRenderer,
    gl_points: PointsGlRenderer,
    gl_proxy_group_points: Vec<PointsGlRenderer>,
    object_load_dialog: Option<ObjectTableLoadDialog>,
    object_export_dialog: Option<ObjectExportDialog>,
    object_export_rx: Option<Receiver<ObjectExportEvent>>,
    object_export_request_id: u64,
    pending_zoom_object_index: Option<usize>,

    object_load_request_id: u64,
    object_load_cancel: Option<Arc<AtomicBool>>,
    load_rx: Option<Receiver<LoadResult>>,
    property_load_rx: Option<Receiver<PropertyLoadResult>>,
    property_load_key: Option<String>,
    status: String,
}

pub type GeoJsonObjectFeature = ObjectFeature;

#[derive(Debug)]
struct LoadResult {
    request_id: u64,
    path: PathBuf,
    downsample_factor: f32,
    display_transform: SpatialDataTransform2,
    display_mode: ObjectDisplayMode,
    objects: Arc<Vec<GeoJsonObjectFeature>>,
    bins: Arc<ObjectIndexBins>,
    render_lods: Vec<ObjectRenderLod>,
    object_fill_mesh: Option<ObjectFillMesh>,
    object_selection_lods: Option<Vec<ObjectSelectionRenderLod>>,
    point_positions_world: Arc<Vec<egui::Pos2>>,
    point_values: Arc<Vec<f32>>,
    point_lods: Arc<Vec<FeaturePointLod>>,
    object_property_keys: Vec<String>,
    scalar_property_keys: Vec<String>,
    color_property_keys: Vec<String>,
    lazy_parquet_source: Option<LazyParquetSource>,
    bounds_local: egui::Rect,
}

#[derive(Debug)]
struct PropertyLoadResult {
    property_key: String,
    values_by_row: HashMap<usize, serde_json::Value>,
}

#[derive(Debug, Clone)]
struct LazyParquetSource {
    geometry_column: String,
    available_property_columns: Vec<String>,
    loaded_property_columns: HashSet<String>,
}

#[derive(Debug, Clone)]
struct ObjectColorLegendCache {
    property_key: String,
    generation: u64,
    entries: Vec<ObjectColorLegendEntry>,
}

#[derive(Debug, Clone)]
struct ObjectTableLoadDialog {
    source_kind: ObjectTableSourceKind,
    path: PathBuf,
    display_mode: ObjectDisplayMode,
    point_source: GeoParquetPointSource,
    geometry_candidates: Vec<String>,
    geometry_column: String,
    geometry_search: String,
    numeric_columns: Vec<String>,
    x_column: String,
    y_column: String,
    x_search: String,
    y_search: String,
    property_columns: Vec<String>,
    property_search: String,
    selected_property_columns: HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectTableSourceKind {
    GeoParquet,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectDisplayMode {
    Polygons,
    Points,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeoParquetPointSource {
    Geometry,
    XYColumns,
}

#[derive(Debug, Clone)]
enum ObjectParquetSource {
    Geometry(ShapesLoadOptions),
    XYColumns {
        x_column: String,
        y_column: String,
        property_columns: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone)]
struct ObjectParquetLoadOptions {
    display_mode: ObjectDisplayMode,
    source: ObjectParquetSource,
}

#[derive(Debug, Clone)]
struct ObjectCsvLoadOptions {
    x_column: String,
    y_column: String,
    property_columns: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
enum ObjectLoadOptions {
    Parquet(ObjectParquetLoadOptions),
    Csv(ObjectCsvLoadOptions),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectExportFormat {
    GeoParquet,
    Csv,
}

#[derive(Debug, Clone)]
struct ObjectExportColumnSelection {
    name: String,
    selected: bool,
}

#[derive(Debug, Clone)]
struct ObjectExportDialog {
    format: ObjectExportFormat,
    columns: Vec<ObjectExportColumnSelection>,
}

#[derive(Debug, Clone)]
struct ObjectExportSnapshot {
    objects: Arc<Vec<ObjectFeature>>,
    property_keys: Vec<String>,
    selected_object_indices: HashSet<usize>,
    analysis_property_thresholds: Vec<ObjectPropertyThresholdRule>,
    analysis_live_threshold_channel_name: Option<String>,
    analysis_threshold_elements: Vec<ThresholdSetElement>,
    selection_elements: Vec<SelectionElement>,
}

#[derive(Debug)]
enum ObjectExportEvent {
    Finished {
        request_id: u64,
        path: PathBuf,
        object_count: usize,
        error: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BulkMeasurementMetric {
    Mean,
    Median,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BulkMeasurementPhase {
    RasterizingLabels,
    MeasuringChannels,
}

#[derive(Debug, Clone)]
struct BulkMeasurementResult {
    metric: BulkMeasurementMetric,
    scope_label: String,
    level_index: usize,
    level_downsample: f32,
    object_count: usize,
    measured_count: usize,
    failed_count: usize,
    column_values: Vec<(String, Vec<Option<f32>>)>,
}

#[derive(Debug)]
enum BulkMeasurementEvent {
    Progress {
        request_id: u64,
        phase: BulkMeasurementPhase,
        completed: usize,
        total: usize,
    },
    Finished {
        request_id: u64,
        result: Option<BulkMeasurementResult>,
        cancelled: bool,
        error: Option<String>,
    },
}

#[derive(Debug)]
enum AnalysisWarmupEvent {
    Started {
        request_id: u64,
        numeric_columns: Vec<String>,
        total: usize,
    },
    ColumnReady {
        request_id: u64,
        key: String,
        pairs: Arc<Vec<(usize, f32)>>,
        sorted_pairs: Arc<Vec<(usize, f32)>>,
        histograms: Vec<((String, HistogramValueTransform), SimpleHistogram)>,
        levels: Vec<(
            (String, HistogramValueTransform, HistogramLevelMethod, usize),
            Arc<Vec<f32>>,
        )>,
        completed: usize,
        total: usize,
    },
    Finished {
        request_id: u64,
    },
}

#[derive(Debug, Clone)]
struct SelectionFillMesh {
    vertices_local: Arc<Vec<[f32; 2]>>,
    bounds_local: egui::Rect,
}

#[derive(Debug, Clone)]
struct ObjectFillMesh {
    vertices_local: Arc<Vec<[f32; 3]>>,
    bounds_local: egui::Rect,
    object_count: usize,
}

#[derive(Debug, Clone)]
struct ObjectSelectionRenderLod {
    lod: u8,
    bins: Arc<ObjectLineSegmentsBins>,
}

#[derive(Debug, Clone)]
struct ObjectIndexBins {
    origin: egui::Pos2,
    bin_size: f32,
    bins_w: usize,
    bins_h: usize,
    indices: Vec<u32>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
}

#[derive(Debug, Clone)]
struct ObjectRenderLod {
    lod: u8,
    bins: Arc<LineSegmentsBins>,
}

#[derive(Debug, Clone)]
struct ObjectColorGroups {
    property_key: String,
    groups: Vec<ObjectColorGroup>,
}

#[derive(Debug, Clone)]
struct ObjectColorGroup {
    value_label: String,
    color_rgb: [u8; 3],
    lods: Vec<ObjectRenderLod>,
    point_positions_world: Arc<Vec<egui::Pos2>>,
    point_values: Arc<Vec<f32>>,
    fill_state: Arc<Vec<u8>>,
    fill_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObjectColorMode {
    Single,
    ByProperty,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnalysisPlotMode {
    Histogram,
    Scatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum AnalysisThresholdOp {
    GreaterEqual,
    LessEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum HistogramLevelMethod {
    Quantiles,
    KMeans,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum HistogramValueTransform {
    None,
    Arcsinh,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ObjectPropertyThresholdRule {
    column_key: String,
    #[serde(default)]
    channel_name: Option<String>,
    op: AnalysisThresholdOp,
    value: f32,
    value_transform: HistogramValueTransform,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ThresholdCallScope {
    Marker { channel_name: String },
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct ThresholdSetElement {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scope: Option<ThresholdCallScope>,
    #[serde(default, skip_serializing_if = "is_false")]
    mark_failed: bool,
    rules: Vec<ObjectPropertyThresholdRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct SelectionElement {
    name: String,
    object_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub(crate) struct ObjectProjectAnalysisState {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub threshold_set_name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub threshold_elements: Vec<ThresholdSetElement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_selected_element: Option<usize>,
    #[serde(default = "default_true")]
    pub follow_active_channel: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub live_threshold_channel_name: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub channel_mapping_overrides: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub selection_elements: Vec<SelectionElement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection_element_selected: Option<usize>,
    #[serde(default = "default_true")]
    pub show_selection_overlay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThresholdSetFile {
    name: String,
    elements: Vec<ThresholdSetElement>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn default_true() -> bool {
    true
}

fn sanitize_export_key(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut prev_underscore = false;
    for ch in name.chars() {
        let normalized = if ch.is_ascii_alphanumeric() {
            prev_underscore = false;
            ch.to_ascii_lowercase()
        } else {
            if prev_underscore {
                continue;
            }
            prev_underscore = true;
            '_'
        };
        out.push(normalized);
    }
    let out = out.trim_matches('_');
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out.to_string()
    }
}

fn threshold_call_display_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        "Unnamed call".to_string()
    } else {
        trimmed.to_string()
    }
}

fn infer_threshold_call_scope(rules: &[ObjectPropertyThresholdRule]) -> ThresholdCallScope {
    match threshold_call_single_channel_name(rules) {
        Some(channel_name) => ThresholdCallScope::Marker {
            channel_name: channel_name.to_string(),
        },
        None => ThresholdCallScope::Composite,
    }
}

fn threshold_call_scope(element: &ThresholdSetElement) -> ThresholdCallScope {
    element
        .scope
        .clone()
        .unwrap_or_else(|| infer_threshold_call_scope(&element.rules))
}

fn threshold_call_bound_channel_name(element: &ThresholdSetElement) -> Option<&str> {
    match element.scope.as_ref() {
        Some(ThresholdCallScope::Marker { channel_name }) => Some(channel_name.as_str()),
        Some(ThresholdCallScope::Composite) => None,
        None => threshold_call_single_channel_name(&element.rules),
    }
}

fn threshold_call_single_channel_name(rules: &[ObjectPropertyThresholdRule]) -> Option<&str> {
    let mut channel_name = None;
    for rule in rules {
        let Some(next) = rule.channel_name.as_deref() else {
            return None;
        };
        match channel_name {
            None => channel_name = Some(next),
            Some(current) if current == next => {}
            Some(_) => return None,
        }
    }
    channel_name
}

fn threshold_call_type_label(element: &ThresholdSetElement) -> &'static str {
    match threshold_call_scope(element) {
        ThresholdCallScope::Marker { .. } => "Marker call",
        ThresholdCallScope::Composite => "Composite call",
    }
}

fn threshold_call_scope_text(element: &ThresholdSetElement) -> String {
    match threshold_call_scope(element) {
        ThresholdCallScope::Marker { channel_name } => format!("Marker: {channel_name}"),
        ThresholdCallScope::Composite => threshold_call_composite_scope_text(&element.rules),
    }
}

fn threshold_call_composite_scope_text(rules: &[ObjectPropertyThresholdRule]) -> String {
    if rules.is_empty() {
        return "Composite".to_string();
    }
    let mut marker_names = BTreeSet::new();
    let mut has_global_rules = false;
    for rule in rules {
        match rule.channel_name.as_deref() {
            Some(name) => {
                marker_names.insert(name.to_string());
            }
            None => {
                has_global_rules = true;
            }
        }
    }
    if marker_names.is_empty() {
        return "Composite".to_string();
    }
    let marker_summary = threshold_call_marker_summary(&marker_names);
    if has_global_rules {
        format!("Composite: global + {marker_summary}")
    } else {
        format!("Composite: {marker_summary}")
    }
}

fn threshold_call_marker_summary(marker_names: &BTreeSet<String>) -> String {
    let labels = marker_names.iter().take(3).cloned().collect::<Vec<_>>();
    let extra = marker_names.len().saturating_sub(labels.len());
    if extra == 0 {
        labels.join(", ")
    } else {
        format!("{} +{extra}", labels.join(", "))
    }
}

fn threshold_call_export_column_name(element: &ThresholdSetElement) -> String {
    let label_token = sanitize_export_key(&threshold_call_display_name(&element.name));
    if let Some(channel_name) = threshold_call_bound_channel_name(element) {
        let channel_token = sanitize_export_key(channel_name);
        if label_token == "unnamed" {
            return format!("_odon_call_{channel_token}");
        }
        if label_token == channel_token || label_token.starts_with(&format!("{channel_token}_")) {
            return format!("_odon_call_{label_token}");
        }
        return format!("_odon_call_{channel_token}_{label_token}");
    }
    format!("_odon_call_{label_token}")
}

fn threshold_call_marks_failed(element: &ThresholdSetElement) -> bool {
    element.mark_failed
        && matches!(
            threshold_call_scope(element),
            ThresholdCallScope::Marker { .. }
        )
}

fn live_threshold_call_export_column_name(
    rules: &[ObjectPropertyThresholdRule],
    live_channel_name: Option<&str>,
) -> String {
    let channel_name = live_channel_name.or_else(|| threshold_call_single_channel_name(rules));
    match channel_name {
        Some(channel_name) => format!("_odon_live_call_{}", sanitize_export_key(channel_name)),
        None => "_odon_live_call".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistogramLevelSelection {
    column_key: String,
    value_transform: HistogramValueTransform,
    method: HistogramLevelMethod,
    level_count: usize,
    level_index: usize,
}

impl Default for ObjectsLayer {
    fn default() -> Self {
        Self {
            visible: false,
            opacity: 0.75,
            width_screen_px: 1.25,
            color_rgb: [255, 255, 255],
            fill_cells: false,
            fill_opacity: 0.30,
            selected_fill_opacity: 0.70,
            show_selection_overlay: true,
            loaded_geojson: None,
            downsample_factor: 1.0,
            display_transform: SpatialDataTransform2::default(),
            display_mode: ObjectDisplayMode::Polygons,
            objects: None,
            bins: None,
            render_lods: None,
            object_fill_mesh: None,
            object_selection_lods: None,
            point_positions_world: None,
            point_values: None,
            point_lods: None,
            object_property_keys: Vec::new(),
            scalar_property_keys: Vec::new(),
            color_property_keys: Vec::new(),
            lazy_parquet_source: None,
            color_legend_cache: None,
            color_groups: None,
            color_mode: ObjectColorMode::Single,
            color_property_key: String::new(),
            color_level_overrides_property_key: String::new(),
            color_level_overrides: BTreeMap::new(),
            filter_property_key: "id".to_string(),
            filter_query: String::new(),
            filtered_indices: None,
            filtered_render_lods: None,
            filtered_point_positions_world: None,
            filtered_point_values: None,
            filtered_point_lods: None,
            filtered_color_groups: None,
            selected_object_indices: HashSet::new(),
            selected_object_index: None,
            selection_elements: Vec::new(),
            selection_element_selected: None,
            selection_element_name_draft: "Selection Element 1".to_string(),
            selected_render_lods: None,
            primary_selected_render_lods: None,
            selected_fill_mesh: None,
            selection_fill_state: Arc::new(Vec::new()),
            selection_cpu_overlay_dirty: false,
            selected_point_positions_world: None,
            selected_point_values: None,
            selected_point_lods: None,
            primary_selected_point_positions_world: None,
            primary_selected_point_values: None,
            selection_generation: 1,
            bulk_measurement_request_id: 0,
            bulk_measurement_rx: None,
            bulk_measurement_cancel: None,
            bulk_measurement_progress_completed: 0,
            bulk_measurement_progress_total: 0,
            bulk_measurement_status: String::new(),
            bulk_measurement_metric: BulkMeasurementMetric::Mean,
            bulk_measurement_level: 0,
            bulk_measurement_concurrency: std::thread::available_parallelism()
                .map(|n| n.get().clamp(1, 16))
                .unwrap_or(4),
            bulk_measurement_filtered_only: false,
            bulk_measurement_prefix: "mean_intensity_".to_string(),
            analysis_plot_mode: AnalysisPlotMode::Histogram,
            analysis_hist_channel: 0,
            analysis_scatter_x_channel: 0,
            analysis_scatter_y_channel: 0,
            analysis_property_thresholds: Vec::new(),
            analysis_threshold_set_name: "Threshold Set".to_string(),
            analysis_threshold_elements: Vec::new(),
            analysis_threshold_selected_element: None,
            analysis_follow_active_channel: true,
            analysis_live_threshold_channel_name: None,
            analysis_channel_mapping_overrides: HashMap::new(),
            analysis_channel_mapping_popup_open: false,
            analysis_channel_mapping_search: String::new(),
            analysis_channel_mapping_suggestions_cache_key: 0,
            analysis_channel_mapping_suggestions_cache_channels_len: 0,
            analysis_channel_mapping_suggestions_cache_numeric_len: 0,
            analysis_channel_mapping_suggestions_cache: HashMap::new(),
            analysis_hist_value_transform: HistogramValueTransform::None,
            analysis_hist_level_method: HistogramLevelMethod::Quantiles,
            analysis_hist_level_count: 4,
            analysis_hist_snapped_level: None,
            analysis_hist_focus_object_index: None,
            analysis_hist_drag_rule: None,
            analysis_hist_brush: None,
            analysis_scatter_brush: None,
            analysis_hist_drag_anchor: None,
            analysis_scatter_drag_anchor: None,
            analysis_scatter_view_key: None,
            analysis_scatter_view_rect: None,
            analysis_live_selection_generation: 1,
            analysis_live_selection_applied_generation: 1,
            object_property_numeric_keys_cache: None,
            object_property_base_pairs_cache: HashMap::new(),
            object_property_base_sorted_pairs_cache: HashMap::new(),
            object_property_base_hist_cache: HashMap::new(),
            object_property_base_hist_levels_cache: HashMap::new(),
            object_property_pairs_cache: HashMap::new(),
            object_property_hist_cache: HashMap::new(),
            object_property_scatter_cache: HashMap::new(),
            object_property_hist_levels_cache: HashMap::new(),
            object_property_threshold_selection_cache_key: None,
            object_property_threshold_selection_cache: Arc::new(Vec::new()),
            object_property_threshold_order_cache_key: None,
            object_property_threshold_order_cache: Arc::new(Vec::new()),
            analysis_warm_started: false,
            analysis_warm_request_id: 0,
            analysis_warm_rx: None,
            analysis_warm_total_columns: 0,
            analysis_warm_completed_columns: 0,
            table_indices_cache: Vec::new(),
            table_cache_dirty: true,
            bounds_local: None,
            generation: 1,
            gl: LineBinsGlRenderer::new(2048),
            gl_object_selection: ObjectLineBinsGlRenderer::new(256, 4),
            gl_fill: PolygonFillGlRenderer::new(8),
            gl_object_fill: ObjectFillGlRenderer::new(4, 4),
            gl_points: PointsGlRenderer::default(),
            gl_proxy_group_points: Vec::new(),
            object_load_dialog: None,
            object_export_dialog: None,
            object_export_rx: None,
            object_export_request_id: 0,
            pending_zoom_object_index: None,
            object_load_request_id: 0,
            object_load_cancel: None,
            load_rx: None,
            property_load_rx: None,
            property_load_key: None,
            status: String::new(),
        }
    }
}
