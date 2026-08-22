use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::document::ControlOpenedDocument;
use crate::data::ome::ChannelInfo;
use crate::data::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectLayerGroups,
};

use super::layers::NativeLayersModel;
use super::{ControlPinnedLevelResource, SystemMemorySnapshot};

const DEFAULT_LOGICAL_CANVAS: [f32; 2] = [960.0, 720.0];
const DEFAULT_GRID_PAD: f32 = 64.0;
const DEFAULT_GROUP_GAP: f32 = 96.0;
const GROUP_HEADER_HEIGHT: f32 = 48.0;

/// Immutable, worker-prepared data for one mosaic item.
///
/// Dataset metadata and storage handles are opened before this resource reaches the actor. The
/// renderer may create tile loaders and GPU caches from the resource later, without reopening the
/// source or delaying semantic command completion.
#[derive(Clone)]
pub struct ControlMosaicItemResource {
    pub id: usize,
    pub roi_id: String,
    pub metadata: HashMap<String, String>,
    pub document: ControlOpenedDocument,
    pub segmentation_path: Option<PathBuf>,
}

impl fmt::Debug for ControlMosaicItemResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlMosaicItemResource")
            .field("id", &self.id)
            .field("roi_id", &self.roi_id)
            .field("metadata", &self.metadata)
            .field("source", &self.document.descriptor.source)
            .field("segmentation_path", &self.segmentation_path)
            .finish()
    }
}

/// Immutable set of opened mosaic datasets shared between the actor and renderer.
#[derive(Clone)]
pub struct ControlMosaicResource {
    pub generation: u64,
    pub source: String,
    pub base_dir: Option<PathBuf>,
    pub initial_columns: Option<usize>,
    pub metadata_columns: Arc<Vec<String>>,
    pub items: Arc<Vec<ControlMosaicItemResource>>,
}

impl fmt::Debug for ControlMosaicResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlMosaicResource")
            .field("generation", &self.generation)
            .field("source", &self.source)
            .field("base_dir", &self.base_dir)
            .field("initial_columns", &self.initial_columns)
            .field("metadata_columns", &self.metadata_columns)
            .field("item_count", &self.items.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MosaicLayoutMode {
    FitCells,
    NativePixels,
}

impl MosaicLayoutMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::FitCells => "fit_cells",
            Self::NativePixels => "native_pixels",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "fit_cells" => Some(Self::FitCells),
            "native_pixels" => Some(Self::NativePixels),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct MosaicChannelModel {
    index: usize,
    name: String,
    visible: bool,
    color_rgb: [u8; 3],
    window: Option<(f32, f32)>,
    note: String,
}

#[derive(Debug, Clone)]
struct MosaicItemModel {
    id: usize,
    roi_id: String,
    metadata: HashMap<String, String>,
    source: String,
    level0_size: [f32; 2],
    offset: [f32; 2],
    scale: f32,
    placed_size: [f32; 2],
    segmentation_path: Option<PathBuf>,
}

impl MosaicItemModel {
    fn bounds(&self) -> [[f32; 2]; 2] {
        [
            self.offset,
            [
                self.offset[0] + self.placed_size[0],
                self.offset[1] + self.placed_size[1],
            ],
        ]
    }

    fn sort_value(&self, column: &str) -> String {
        if column == "id" {
            self.roi_id.clone()
        } else {
            self.metadata.get(column).cloned().unwrap_or_default()
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MosaicObjectLoadSpec {
    pub resource_generation: u64,
    pub operation_generation: u64,
    pub downsample_factor: f32,
    pub items: Vec<(usize, PathBuf)>,
    pub cancel: Arc<AtomicBool>,
}

impl MosaicObjectLoadSpec {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.load(AtomicOrdering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MosaicObjectLoadResult {
    pub loaded: Vec<(usize, Arc<super::ControlObjectResource>)>,
    pub failures: Vec<(usize, String)>,
    pub cancelled: bool,
}

#[derive(Clone)]
pub(crate) struct MosaicMemoryPinItemSpec {
    pub item_id: usize,
    pub document: ControlOpenedDocument,
    /// Mapping from global mosaic channel index to this document's channel index.
    pub channel_map: Vec<Option<u64>>,
}

#[derive(Clone)]
pub(crate) struct MosaicMemoryPinSpec {
    pub resource_generation: u64,
    pub operation_generation: u64,
    pub level: usize,
    pub channel_ids: Vec<u64>,
    pub items: Vec<MosaicMemoryPinItemSpec>,
    pub estimated_bytes: u64,
    pub pinned_bytes: u64,
    pub force: bool,
}

#[derive(Debug)]
pub(crate) struct MosaicMemoryPinResult {
    pub loaded: Vec<(usize, ControlPinnedLevelResource)>,
    pub failures: Vec<(usize, String)>,
}

#[derive(Debug, Clone)]
enum MosaicPinnedLevelState {
    Loaded(Arc<ControlPinnedLevelResource>),
    Failed(String),
}

/// Canonical, renderer-independent semantic mosaic state.
#[derive(Debug, Clone)]
pub(crate) struct MosaicModel {
    resource: Option<Arc<ControlMosaicResource>>,
    items: Vec<MosaicItemModel>,
    channels: Vec<MosaicChannelModel>,
    active_channel: usize,
    channel_order: Vec<usize>,
    channel_search: String,
    channel_sort: String,
    layer_groups: ProjectLayerGroups,
    native_layers: NativeLayersModel,
    selected_ids: HashSet<usize>,
    focused_id: Option<usize>,
    right_tab: String,
    group_by: String,
    sort_by: String,
    sort_secondary_enabled: bool,
    sort_by_secondary: String,
    layout_mode: MosaicLayoutMode,
    columns: usize,
    group_gap: f32,
    show_group_labels: bool,
    show_text_labels: bool,
    label_columns: Vec<String>,
    grid_cell_size: [f32; 2],
    grid_pad: f32,
    bounds: [[f32; 2]; 2],
    logical_canvas: [f32; 2],
    camera_center: [f32; 2],
    camera_zoom: f32,
    show_left_panel: bool,
    show_right_panel: bool,
    smooth_pixels: bool,
    objects_visible: bool,
    fast_object_rendering: bool,
    object_resources: BTreeMap<usize, Arc<super::ControlObjectResource>>,
    object_operation_generation: u64,
    object_pending_ids: HashSet<usize>,
    object_failures: BTreeMap<usize, String>,
    object_cancel: Option<Arc<AtomicBool>>,
    object_status: String,
    pinned_levels: BTreeMap<(usize, usize), MosaicPinnedLevelState>,
    memory_selected_channels: Vec<usize>,
    memory_operation_generation: u64,
    memory_pending: HashMap<(usize, usize), u64>,
    memory_status: String,
    system_memory: Option<SystemMemorySnapshot>,
}

impl Default for MosaicModel {
    fn default() -> Self {
        Self {
            resource: None,
            items: Vec::new(),
            channels: Vec::new(),
            active_channel: 0,
            channel_order: Vec::new(),
            channel_search: String::new(),
            channel_sort: "manual".to_string(),
            layer_groups: ProjectLayerGroups::default(),
            native_layers: NativeLayersModel::channels(&[]),
            selected_ids: HashSet::new(),
            focused_id: None,
            right_tab: "properties".to_string(),
            group_by: String::new(),
            sort_by: "id".to_string(),
            sort_secondary_enabled: false,
            sort_by_secondary: "id".to_string(),
            layout_mode: MosaicLayoutMode::FitCells,
            columns: 1,
            group_gap: DEFAULT_GROUP_GAP,
            show_group_labels: true,
            show_text_labels: true,
            label_columns: vec!["id".to_string()],
            grid_cell_size: [1.0, 1.0],
            grid_pad: DEFAULT_GRID_PAD,
            bounds: [[0.0, 0.0], [1.0, 1.0]],
            logical_canvas: DEFAULT_LOGICAL_CANVAS,
            camera_center: [0.5, 0.5],
            camera_zoom: 1.0,
            show_left_panel: true,
            show_right_panel: true,
            smooth_pixels: true,
            objects_visible: false,
            fast_object_rendering: true,
            object_resources: BTreeMap::new(),
            object_operation_generation: 0,
            object_pending_ids: HashSet::new(),
            object_failures: BTreeMap::new(),
            object_cancel: None,
            object_status: String::new(),
            pinned_levels: BTreeMap::new(),
            memory_selected_channels: Vec::new(),
            memory_operation_generation: 0,
            memory_pending: HashMap::new(),
            memory_status: String::new(),
            system_memory: None,
        }
    }
}

impl MosaicModel {
    pub(crate) fn require_ready(&self) -> Result<(), ControlError> {
        self.require_resource().map(|_| ())
    }

    pub(crate) fn default_screenshot_filename(&self) -> Result<String, ControlError> {
        self.require_resource()?;
        let source_name = self
            .focused_id
            .and_then(|id| self.items.iter().find(|item| item.id == id))
            .or_else(|| self.items.first())
            .map(|item| item.roi_id.as_str())
            .unwrap_or("mosaic");
        let stem = source_name
            .strip_suffix(".ome.zarr")
            .or_else(|| source_name.strip_suffix(".zarr"))
            .unwrap_or(source_name);
        let sanitized = stem
            .chars()
            .map(|ch| match ch {
                '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
                _ => ch,
            })
            .collect::<String>();
        let sanitized = sanitized.trim().trim_matches('.').trim_matches('_');
        Ok(if sanitized.is_empty() {
            "odon.mosaic.screenshot.png".to_string()
        } else {
            format!("{sanitized}.mosaic.screenshot.png")
        })
    }

    pub(crate) fn resource(&self) -> Option<Arc<ControlMosaicResource>> {
        self.resource.clone()
    }

    pub(crate) fn resource_generation(&self) -> u64 {
        self.resource
            .as_ref()
            .map_or(0, |resource| resource.generation)
    }

    pub(crate) fn object_resources(&self) -> Vec<(usize, Arc<super::ControlObjectResource>)> {
        self.object_resources
            .iter()
            .map(|(id, resource)| (*id, Arc::clone(resource)))
            .collect()
    }

    pub(crate) fn object_operation_generation(&self) -> u64 {
        self.object_operation_generation
    }

    pub(crate) fn install_resource(&mut self, resource: Arc<ControlMosaicResource>) {
        let mut items = resource
            .items
            .iter()
            .map(|item| {
                let descriptor = &item.document.descriptor;
                let level0 = descriptor.levels.first();
                let width = level0
                    .and_then(|level| level.shape.get(descriptor.dims.x))
                    .copied()
                    .unwrap_or(1) as f32;
                let height = level0
                    .and_then(|level| level.shape.get(descriptor.dims.y))
                    .copied()
                    .unwrap_or(1) as f32;
                MosaicItemModel {
                    id: item.id,
                    roi_id: item.roi_id.clone(),
                    metadata: item.metadata.clone(),
                    source: descriptor.source.source_key(),
                    level0_size: [width.max(1.0), height.max(1.0)],
                    offset: [0.0, 0.0],
                    scale: 1.0,
                    placed_size: [1.0, 1.0],
                    segmentation_path: item.segmentation_path.clone(),
                }
            })
            .collect::<Vec<_>>();
        let mut channels = Vec::new();
        let mut seen = HashSet::new();
        for item in resource.items.iter() {
            for channel in &item.document.descriptor.channels {
                if seen.insert(channel.name.clone()) {
                    channels.push(channel.clone());
                }
            }
        }
        let channels = channels
            .iter()
            .enumerate()
            .map(|(index, channel)| mosaic_channel(index, channel))
            .collect::<Vec<_>>();
        let grid_cell_size = items.iter().fold([1.0_f32, 1.0_f32], |size, item| {
            [
                size[0].max(item.level0_size[0]),
                size[1].max(item.level0_size[1]),
            ]
        });
        let columns = resource
            .initial_columns
            .filter(|columns| *columns > 0)
            .unwrap_or_else(|| ((items.len() as f32).sqrt().ceil() as usize).max(1));
        let focused_id = items.first().map(|item| item.id);

        self.cancel_object_load("Mosaic resource was replaced");
        self.resource = Some(resource);
        self.items.clear();
        self.items.append(&mut items);
        self.channels = channels;
        self.active_channel = 0;
        self.channel_order = (0..self.channels.len()).collect();
        self.channel_search.clear();
        self.channel_sort = "manual".to_string();
        self.layer_groups = ProjectLayerGroups::default();
        self.native_layers = initial_native_layers(&self.channels, &self.items);
        self.focused_id = focused_id;
        self.selected_ids = focused_id.into_iter().collect();
        self.right_tab = "properties".to_string();
        self.group_by.clear();
        self.sort_by = "id".to_string();
        self.sort_secondary_enabled = false;
        self.sort_by_secondary = "id".to_string();
        self.layout_mode = MosaicLayoutMode::FitCells;
        self.columns = columns;
        self.group_gap = DEFAULT_GROUP_GAP;
        self.show_group_labels = true;
        self.show_text_labels = true;
        self.label_columns = vec!["id".to_string()];
        self.grid_cell_size = grid_cell_size;
        self.grid_pad = DEFAULT_GRID_PAD;
        self.show_left_panel = true;
        self.show_right_panel = true;
        self.smooth_pixels = true;
        self.objects_visible = false;
        self.fast_object_rendering = true;
        self.object_resources.clear();
        self.object_failures.clear();
        self.object_pending_ids.clear();
        self.object_status.clear();
        self.pinned_levels.clear();
        self.memory_selected_channels = (0..self.channels.len()).collect();
        self.memory_operation_generation = 0;
        self.memory_pending.clear();
        self.memory_status.clear();
        self.system_memory = None;
        self.apply_layout();
        self.fit_bounds(self.bounds);
    }

    pub(crate) fn restore_renderer_state(&mut self, state: &Value) -> Result<(), ControlError> {
        self.require_resource()?;
        if let Some(layout) = state.get("layout") {
            self.configure_layout(layout)?;
        }
        if let Some(tab) = state.get("right_tab").and_then(Value::as_str) {
            self.set_right_tab(&json!({"tab":tab}))?;
        }
        if let Some(items) = state.get("items").and_then(Value::as_array) {
            self.selected_ids = items
                .iter()
                .filter(|item| item["selected"].as_bool() == Some(true))
                .filter_map(|item| item["id"].as_u64())
                .map(|id| id as usize)
                .filter(|id| self.items.iter().any(|item| item.id == *id))
                .collect();
            self.focused_id = items
                .iter()
                .find(|item| item["focused"].as_bool() == Some(true))
                .and_then(|item| item["id"].as_u64())
                .map(|id| id as usize)
                .filter(|id| self.items.iter().any(|item| item.id == *id));
        }
        if let Some(camera) = state.get("camera") {
            self.set_camera(camera)?;
        }
        if let Some(panels) = state.get("panels") {
            self.set_panels(panels)?;
        }
        if let Some(smooth) = state.get("smooth_pixels").and_then(Value::as_bool) {
            self.smooth_pixels = smooth;
        }
        if let Some(visible) = state.get("objects_visible").and_then(Value::as_bool) {
            self.objects_visible = visible;
        }
        if let Some(fast) = state.get("fast_object_rendering").and_then(Value::as_bool) {
            self.fast_object_rendering = fast;
        }
        if let Some(channels) = state.get("channels").and_then(Value::as_array) {
            for projected in channels {
                let Some(index) = projected["index"].as_u64().map(|index| index as usize) else {
                    continue;
                };
                let Some(channel) = self.channels.get_mut(index) else {
                    continue;
                };
                if let Some(visible) = projected["visible"].as_bool() {
                    channel.visible = visible;
                }
                if let Some(values) = projected["color_rgb"]
                    .as_array()
                    .filter(|values| values.len() == 3)
                {
                    channel.color_rgb = [
                        json_u8(&values[0])?,
                        json_u8(&values[1])?,
                        json_u8(&values[2])?,
                    ];
                }
                if let Some(values) = projected["window"]
                    .as_array()
                    .filter(|values| values.len() == 2)
                {
                    channel.window = values[0]
                        .as_f64()
                        .zip(values[1].as_f64())
                        .map(|(minimum, maximum)| (minimum as f32, maximum as f32));
                }
                if let Some(note) = projected["note"].as_str() {
                    channel.note = note.to_string();
                }
                if projected["active"].as_bool() == Some(true) {
                    self.active_channel = index;
                }
            }
        }
        if let Some(order) = state.get("channel_order").and_then(Value::as_array) {
            let order = order
                .iter()
                .filter_map(Value::as_u64)
                .map(|index| index as usize)
                .filter(|index| *index < self.channels.len())
                .collect::<Vec<_>>();
            if order.len() == self.channels.len() {
                self.channel_order = order;
            }
        }
        if let Some(presentation) = state.get("channel_presentation") {
            if let Some(search) = presentation.get("search").and_then(Value::as_str) {
                self.channel_search = search.to_string();
            }
            if let Some(sort) = presentation
                .get("sort")
                .and_then(Value::as_str)
                .and_then(canonical_channel_sort)
            {
                self.channel_sort = sort.to_string();
            }
        }
        if let Some(groups) = state.get("layer_groups") {
            self.layer_groups = serde_json::from_value(groups.clone())
                .map_err(|error| invalid(format!("invalid mosaic channel groups: {error}")))?;
        }
        if let Some(layers) = state.get("native_layers") {
            self.native_layers = NativeLayersModel::restore(layers)?;
            self.sync_semantics_from_native_layers()?;
        } else {
            self.sync_native_layers_from_semantics();
        }
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let unresolved = [self.group_by.as_str(), self.sort_by.as_str()]
            .into_iter()
            .chain(
                self.sort_secondary_enabled
                    .then_some(self.sort_by_secondary.as_str()),
            )
            .chain(self.label_columns.iter().map(String::as_str))
            .filter(|field| {
                !field.is_empty()
                    && *field != "id"
                    && !self.metadata_columns().iter().any(|column| column == field)
            })
            .map(str::to_string)
            .collect::<HashSet<_>>();
        Ok(json!({
            "generation":self.resource_generation(),
            "roi_count":self.items.len(),
            "focused":self.focus_snapshot(),
            "selection":self.selection_snapshot(),
            "metadata_columns":self.metadata_columns(),
            "mosaic_bounds":{"min":self.bounds[0],"max":self.bounds[1]},
            "camera":self.camera_snapshot(),
            "right_tab":self.right_tab,
            "layout":{
                "group_by":self.group_by,
                "sort_by":self.sort_by,
                "sort_secondary_enabled":self.sort_secondary_enabled,
                "sort_by_secondary":self.sort_by_secondary,
                "layout":self.layout_mode.as_str(),
                "columns":self.columns,
                "group_gap":self.group_gap,
                "show_group_labels":self.show_group_labels,
                "show_text_labels":self.show_text_labels,
                "label_columns":self.label_columns,
                "unresolved_fields":unresolved,
            },
            "rois":self.items.iter().enumerate().map(|(index,item)| json!({
                "index":index,
                "id":item.id,
                "roi_id":item.roi_id,
                "metadata":item.metadata,
                "focused":self.focused_id == Some(item.id),
                "selected":self.selected_ids.contains(&item.id),
            })).collect::<Vec<_>>(),
        }))
    }

    pub(crate) fn projection_state(&self) -> Value {
        if self.resource.is_none() {
            return json!({});
        }
        json!({
            "generation":self.resource_generation(),
            "items":self.items.iter().map(|item| json!({
                "id":item.id,
                "roi_id":item.roi_id,
                "offset_world":item.offset,
                "scale":item.scale,
                "placed_size":item.placed_size,
                "selected":self.selected_ids.contains(&item.id),
                "focused":self.focused_id == Some(item.id),
            })).collect::<Vec<_>>(),
            "bounds":{"min":self.bounds[0],"max":self.bounds[1]},
            "camera":self.camera_snapshot(),
            "right_tab":self.right_tab,
            "layout":{
                "group_by":self.group_by,
                "sort_by":self.sort_by,
                "sort_secondary_enabled":self.sort_secondary_enabled,
                "sort_by_secondary":self.sort_by_secondary,
                "layout":self.layout_mode.as_str(),
                "columns":self.columns,
                "group_gap":self.group_gap,
                "show_group_labels":self.show_group_labels,
                "show_text_labels":self.show_text_labels,
                "label_columns":self.label_columns,
            },
            "panels":{"left":self.show_left_panel,"right":self.show_right_panel},
            "smooth_pixels":self.smooth_pixels,
            "objects_visible":self.objects_visible,
            "fast_object_rendering":self.fast_object_rendering,
            "channels":self.channels.iter().map(|channel| json!({
                "index":channel.index,
                "name":channel.name,
                "visible":channel.visible,
                "color_rgb":channel.color_rgb,
                "window":channel.window.map(|(min,max)| [min,max]),
                "note":channel.note,
                "active":channel.index == self.active_channel,
            })).collect::<Vec<_>>(),
            "channel_order":self.channel_order,
            "channel_presentation":self.channel_presentation_snapshot(),
            "layer_groups":self.layer_groups,
            "native_layers":self.native_layers.snapshots(),
            "objects":self.object_state(),
        })
    }

    pub(crate) fn dispatch(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Option<Result<Value, ControlError>> {
        let result = match method {
            "mosaic.ui.set_right_tab" => self.set_right_tab(params),
            "mosaic.layout.configure" => self.configure_layout(params),
            "mosaic.get_state" => self.snapshot(),
            "mosaic.items.list" => self.list_items(params),
            "mosaic.selection.get" => self.require_resource().map(|_| self.selection_snapshot()),
            "mosaic.selection.set" => self.set_selection(params),
            "mosaic.selection.clear" => self.clear_selection(),
            "mosaic.focus.get" => self.require_resource().map(|_| self.focus_snapshot()),
            "mosaic.focus.set" => self.set_focus(params),
            "mosaic.focus.next" => self.step_focus(params, true),
            "mosaic.focus.previous" => self.step_focus(params, false),
            "mosaic.focus.fit" => self.fit_focus(),
            "mosaic.focus.clear" => self.clear_focus(),
            "mosaic.fit_all" => self.fit_all(),
            "mosaic.objects.get_state" => self.require_resource().map(|_| self.object_state()),
            _ => return None,
        };
        Some(result)
    }

    pub(crate) fn dispatch_shared(
        &mut self,
        method: &str,
        params: &Value,
    ) -> Option<Result<(Value, bool), ControlError>> {
        let read_only = matches!(
            method,
            "viewer.channels.list"
                | "viewer.channels.list_visible"
                | "viewer.channels.get_active"
                | "viewer.channels.get_contrast"
                | "viewer.channels.presentation.get"
                | "viewer.channels.list_groups"
                | "viewer.native_layers.list"
                | "viewer.native_layers.get"
                | "viewer.camera.get"
                | "viewer.panels.get"
                | "viewer.rendering.get_smooth_pixels"
                | "viewer.rendering.get_state"
                | "viewer.objects.get_visibility"
                | "viewer.objects.rendering.get_fast"
                | "memory.get"
        );
        let result = match method {
            "viewer.channels.list" => self.channels_snapshot(),
            "viewer.channels.list_visible" => self.visible_channels_snapshot(),
            "viewer.channels.get_active" => self.active_channel_snapshot(),
            "viewer.channels.set_active" => self.set_active_channel(params),
            "viewer.channels.set_visible" => self.set_visible_channels(params),
            "viewer.channels.get_contrast" => self.get_channel_contrast(params),
            "viewer.channels.set_contrast" => self.set_channel_contrast(params),
            "viewer.channels.set_color" => self.set_channel_color(params),
            "viewer.channels.set_note" => self.set_channel_note(params),
            "viewer.channels.set_order" => self.set_channel_order(params),
            "viewer.channels.presentation.get" => self.channel_presentation(),
            "viewer.channels.presentation.set" => self.set_channel_presentation(params),
            "viewer.channels.list_groups" => self.channel_groups(),
            "viewer.channels.set_group" => self.set_channel_group(params),
            "viewer.native_layers.list" => self.native_layers_snapshot(),
            "viewer.native_layers.get" => self.native_layer_snapshot(params),
            "viewer.native_layers.set_active" => self.set_native_layer_active(params),
            "viewer.native_layers.set_visibility" => self.set_native_layer_visibility(params),
            "viewer.native_layers.set_order" => self.set_native_layer_order(params),
            "viewer.camera.get" => self
                .require_resource()
                .map(|_| json!({"mode":"mosaic","camera":self.camera_snapshot()})),
            "viewer.camera.set" => self.set_camera(params),
            "viewer.camera.zoom_in" => self.zoom_camera(params, true),
            "viewer.camera.zoom_out" => self.zoom_camera(params, false),
            "viewer.camera.fit" => self.fit_camera(),
            "viewer.panels.get" => self.panels_snapshot(),
            "viewer.panels.set" => self.set_panels(params),
            "viewer.rendering.get_smooth_pixels" => self.smooth_pixels_snapshot(),
            "viewer.rendering.set_smooth_pixels" => self.set_smooth_pixels(params),
            "viewer.rendering.get_state" => self.rendering_snapshot(),
            "viewer.objects.get_visibility" => self.object_visibility_snapshot(params),
            "viewer.objects.set_visibility" => self.set_object_visibility(params),
            "viewer.objects.rendering.get_fast" => self.fast_object_rendering_snapshot(),
            "viewer.objects.rendering.set_fast" => self.set_fast_object_rendering(params),
            "memory.get" => self.memory_snapshot(),
            "memory.unpin" => self.unpin_memory(params),
            "memory.unpin_all" => self.unpin_all_memory(),
            _ => return None,
        };
        Some(result.map(|response| (response, !read_only)))
    }

    fn channel_presentation_snapshot(&self) -> Value {
        json!({
            "search":self.channel_search,
            "sort":self.channel_sort,
            "order":self.channel_order.iter().map(|index| json!({
                "index":index,
                "name":self.channels[*index].name,
                "visible":self.channels[*index].visible,
            })).collect::<Vec<_>>(),
        })
    }

    fn channel_presentation(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","presentation":self.channel_presentation_snapshot()}))
    }

    fn set_channel_presentation(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(search) = params.get("search") {
            self.channel_search = search
                .as_str()
                .ok_or_else(|| invalid("search must be a string"))?
                .to_string();
        }
        if let Some(sort) = params.get("sort") {
            self.channel_sort = canonical_channel_sort(
                sort.as_str()
                    .ok_or_else(|| invalid("sort must be a string"))?,
            )
            .ok_or_else(|| invalid("unknown channel sort mode"))?
            .to_string();
        }
        Ok(json!({"mode":"mosaic","presentation":self.channel_presentation_snapshot()}))
    }

    fn channel_groups_snapshot(&self) -> Value {
        Value::Array(
            self.layer_groups
                .channel_groups
                .iter()
                .map(|group| {
                    let members = self
                        .channels
                        .iter()
                        .filter_map(|channel| {
                            let member = self.layer_groups.channel_members.get(&channel.name)?;
                            (member.group_id == group.id).then(|| {
                                json!({
                                    "index":channel.index,
                                    "name":channel.name,
                                    "inherit_color":member.inherit_color,
                                })
                            })
                        })
                        .collect::<Vec<_>>();
                    json!({
                        "id":group.id,
                        "name":group.name,
                        "expanded":group.expanded,
                        "color_rgb":group.color_rgb,
                        "members":members,
                    })
                })
                .collect(),
        )
    }

    fn channel_groups(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","groups":self.channel_groups_snapshot()}))
    }

    fn set_channel_group(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(state) = params.get("state") {
            let replacement: ProjectLayerGroups = serde_json::from_value(state.clone())
                .map_err(|error| invalid(format!("invalid channel-group state: {error}")))?;
            let changed = self.layer_groups != replacement;
            self.layer_groups = replacement;
            return Ok(json!({
                "mode":"mosaic",
                "result":{
                    "changed":changed,
                    "groups":self.channel_groups_snapshot(),
                },
            }));
        }
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("set_channel_group requires channels"))?;
        let mut indices = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<Vec<_>, _>>()?;
        indices.sort_unstable();
        indices.dedup();
        if indices.is_empty() {
            return Err(invalid("no channels resolved"));
        }
        let requested_id = params.get("group_id").and_then(Value::as_u64);
        let requested_name = params
            .get("group")
            .or_else(|| params.get("name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty());
        let color = color_from_params(params)?;
        let group_id =
            ensure_channel_group(&mut self.layer_groups, requested_id, requested_name, color);
        if params
            .get("replace_group_members")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            self.layer_groups
                .channel_members
                .retain(|_, member| member.group_id != group_id);
        }
        let inherit_color = params
            .get("inherit_color")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        for index in indices {
            self.layer_groups.channel_members.insert(
                self.channels[index].name.clone(),
                ProjectChannelGroupMember {
                    group_id,
                    inherit_color,
                },
            );
        }
        Ok(json!({
            "mode":"mosaic",
            "result":{
                "changed":true,
                "group_id":group_id,
                "groups":self.channel_groups_snapshot(),
            },
        }))
    }

    fn native_layers_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","layers":self.native_layers.snapshots()}))
    }

    fn native_layer_id<'a>(&self, params: &'a Value) -> Result<&'a str, ControlError> {
        params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("layer_id is required"))
    }

    fn native_layer_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?;
        let layer = self
            .native_layers
            .snapshots()
            .into_iter()
            .find(|layer| layer["layer_id"].as_str() == Some(layer_id))
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        Ok(json!({"mode":"mosaic","layer":layer}))
    }

    fn set_native_layer_active(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?.to_string();
        let changed = self.native_layers.set_active(&layer_id)?;
        self.sync_semantics_from_native_layers()?;
        let layer = self.native_layer_snapshot(&json!({"layer_id":layer_id}))?["layer"].clone();
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"layer":layer}}))
    }

    fn set_native_layer_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let layer_id = self.native_layer_id(params)?.to_string();
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("visible is required"))?;
        let changed = self.native_layers.set_visibility(&layer_id, visible)?;
        self.sync_semantics_from_native_layers()?;
        let layer = self.native_layer_snapshot(&json!({"layer_id":layer_id}))?["layer"].clone();
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"layer":layer}}))
    }

    fn set_native_layer_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let stack = params
            .get("stack")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("stack is required"))?;
        let layers = params
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("layers is required"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("layer IDs must be strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let changed = self.native_layers.set_order(stack, &layers)?;
        self.sync_semantics_from_native_layers()?;
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":changed,"layers":self.native_layers.snapshots()},
        }))
    }

    fn sync_semantics_from_native_layers(&mut self) -> Result<(), ControlError> {
        let snapshots = self.native_layers.snapshots();
        let mut channel_order = Vec::new();
        for layer in &snapshots {
            let Some(layer_id) = layer["layer_id"].as_str() else {
                continue;
            };
            let visible = layer["visible"].as_bool().unwrap_or(false);
            if let Some(index) = layer_id
                .strip_prefix("channel:")
                .and_then(|value| value.parse::<usize>().ok())
            {
                let channel = self
                    .channels
                    .get_mut(index)
                    .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
                channel.visible = visible;
                channel_order.push(index);
                if layer["active"].as_bool() == Some(true) {
                    self.active_channel = index;
                }
            } else if layer_id == "segmentation_geojson" {
                self.objects_visible = visible;
            } else if layer_id == "text_labels" {
                self.show_text_labels = visible;
            }
        }
        if channel_order.len() == self.channels.len() {
            self.channel_order = channel_order;
        }
        Ok(())
    }

    fn sync_native_layers_from_semantics(&mut self) {
        for channel in &self.channels {
            let _ = self
                .native_layers
                .set_visibility(&format!("channel:{}", channel.index), channel.visible);
        }
        let _ = self
            .native_layers
            .set_visibility("segmentation_geojson", self.objects_visible);
        let _ = self
            .native_layers
            .set_visibility("text_labels", self.show_text_labels);
        let _ = self
            .native_layers
            .set_active(&format!("channel:{}", self.active_channel));
        let order = self
            .channel_order
            .iter()
            .map(|index| format!("channel:{index}"))
            .collect::<Vec<_>>();
        let _ = self.native_layers.set_order("channels", &order);
    }

    pub(crate) fn pinned_level_resources(&self) -> Vec<(usize, Arc<ControlPinnedLevelResource>)> {
        self.pinned_levels
            .iter()
            .filter_map(|((item_id, _), state)| match state {
                MosaicPinnedLevelState::Loaded(resource) => Some((*item_id, Arc::clone(resource))),
                MosaicPinnedLevelState::Failed(_) => None,
            })
            .collect()
    }

    pub(crate) fn prepare_memory_pin(
        &mut self,
        params: &Value,
    ) -> Result<MosaicMemoryPinSpec, ControlError> {
        let resource = Arc::clone(self.require_resource()?);
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok())
            .ok_or_else(|| invalid("memory level is required"))?;
        let selected_channels = match params.get("channels") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| invalid("channels must be an array"))?
                .iter()
                .map(|selector| self.channel_index(selector))
                .collect::<Result<Vec<_>, _>>()?,
            None if self.memory_selected_channels.is_empty() => (0..self.channels.len()).collect(),
            None => self.memory_selected_channels.clone(),
        };
        let mut selected_channels = selected_channels;
        selected_channels.sort_unstable();
        selected_channels.dedup();
        if selected_channels.is_empty() {
            return Err(invalid("select at least one channel to pin"));
        }
        let item_ids = self.memory_item_ids(params)?;
        let mut items = Vec::new();
        let mut estimated_bytes = 0_u64;
        for item_id in item_ids {
            let Some(item) = resource.items.iter().find(|item| item.id == item_id) else {
                continue;
            };
            let z_extent = item
                .document
                .descriptor
                .dims
                .z
                .and_then(|dimension| {
                    item.document
                        .descriptor
                        .levels
                        .first()
                        .and_then(|level| level.shape.get(dimension))
                })
                .copied()
                .unwrap_or(1);
            if z_extent > 1 {
                return Err(invalid(format!(
                    "RAM pinning is unavailable for z-stack mosaic ROI '{}'",
                    item.roi_id
                )));
            }
            if level >= item.document.descriptor.levels.len() {
                continue;
            }
            let channel_map = self
                .channels
                .iter()
                .map(|global| {
                    item.document
                        .descriptor
                        .channels
                        .iter()
                        .find(|local| local.name == global.name)
                        .map(|local| local.index as u64)
                })
                .collect::<Vec<_>>();
            let present_channels = selected_channels
                .iter()
                .filter(|index| channel_map.get(**index).copied().flatten().is_some())
                .count();
            if present_channels == 0 {
                continue;
            }
            estimated_bytes = estimated_bytes.saturating_add(estimate_level_bytes(
                &item.document,
                level,
                present_channels,
            ));
            items.push(MosaicMemoryPinItemSpec {
                item_id,
                document: item.document.clone(),
                channel_map,
            });
        }
        if items.is_empty() {
            return Err(invalid(
                "the requested level and channels are unavailable for the selected mosaic scope",
            ));
        }
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        self.memory_selected_channels = selected_channels.clone();
        for item in &items {
            self.memory_pending
                .insert((item.item_id, level), self.memory_operation_generation);
        }
        self.memory_status = format!(
            "Loading {} channel(s) from level {level} into RAM for {} ROI(s)",
            selected_channels.len(),
            items.len()
        );
        Ok(MosaicMemoryPinSpec {
            resource_generation: self.resource_generation(),
            operation_generation: self.memory_operation_generation,
            level,
            channel_ids: selected_channels
                .iter()
                .map(|index| *index as u64)
                .collect(),
            items,
            estimated_bytes,
            pinned_bytes: self.pinned_bytes(),
            force: params
                .get("force")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    pub(crate) fn memory_pin_is_current(&self, spec: &MosaicMemoryPinSpec) -> bool {
        self.resource_generation() == spec.resource_generation
            && spec.items.iter().all(|item| {
                self.memory_pending
                    .get(&(item.item_id, spec.level))
                    .copied()
                    == Some(spec.operation_generation)
            })
    }

    pub(crate) fn install_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        result: MosaicMemoryPinResult,
        system: Option<SystemMemorySnapshot>,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec) {
            return None;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        for (item_id, resource) in result.loaded {
            self.pinned_levels.insert(
                (item_id, spec.level),
                MosaicPinnedLevelState::Loaded(Arc::new(resource)),
            );
        }
        for (item_id, error) in result.failures {
            self.pinned_levels
                .insert((item_id, spec.level), MosaicPinnedLevelState::Failed(error));
        }
        self.system_memory = system;
        self.memory_status = format!("Pinned mosaic level {} into RAM", spec.level);
        Some(json!({
            "started":true,
            "completed":true,
            "level":spec.level,
            "items":spec.items.len(),
            "estimated_bytes":spec.estimated_bytes,
            "memory":self.memory_snapshot().ok()?,
        }))
    }

    pub(crate) fn finish_memory_confirmation(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        system: Option<SystemMemorySnapshot>,
        risk: &str,
        projected_bytes: u64,
        available_bytes: u64,
    ) -> Option<Value> {
        if !self.memory_pin_is_current(spec) {
            return None;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        self.system_memory = system;
        self.memory_status = format!(
            "RAM pinning mosaic level {} requires confirmation",
            spec.level
        );
        Some(json!({
            "confirmation_required":true,
            "level":spec.level,
            "items":spec.items.len(),
            "requested_bytes":spec.estimated_bytes,
            "projected_bytes":projected_bytes,
            "available_bytes":available_bytes,
            "risk":risk,
        }))
    }

    pub(crate) fn fail_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        let message = message.into();
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
            self.pinned_levels.insert(
                (item.item_id, spec.level),
                MosaicPinnedLevelState::Failed(message.clone()),
            );
        }
        self.memory_status = message;
        true
    }

    pub(crate) fn cancel_memory_pin(
        &mut self,
        spec: &MosaicMemoryPinSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.memory_pin_is_current(spec) {
            return false;
        }
        for item in &spec.items {
            self.memory_pending.remove(&(item.item_id, spec.level));
        }
        self.memory_status = message.into();
        true
    }

    fn memory_item_ids(&self, params: &Value) -> Result<Vec<usize>, ControlError> {
        match params
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or("focused")
        {
            "all" => Ok(self.items.iter().map(|item| item.id).collect()),
            "focused" => self
                .focused_id
                .map(|id| vec![id])
                .ok_or_else(|| invalid("mosaic has no focused ROI")),
            "item" => {
                let selector = params
                    .get("item")
                    .ok_or_else(|| invalid("item is required when scope is item"))?;
                let id = if let Some(id) = selector.as_u64() {
                    usize::try_from(id)
                        .ok()
                        .filter(|id| self.items.iter().any(|item| item.id == *id))
                } else if let Some(roi_id) = selector.as_str() {
                    self.items
                        .iter()
                        .find(|item| item.roi_id == roi_id)
                        .map(|item| item.id)
                } else {
                    None
                };
                id.map(|id| vec![id])
                    .ok_or_else(|| invalid(format!("unknown mosaic item selector: {selector}")))
            }
            scope => Err(invalid(format!(
                "unknown memory scope '{scope}'; use focused, item, or all"
            ))),
        }
    }

    fn memory_snapshot(&self) -> Result<Value, ControlError> {
        let resource = self.require_resource()?;
        let selected = if self.memory_selected_channels.is_empty() {
            (0..self.channels.len()).collect::<Vec<_>>()
        } else {
            self.memory_selected_channels.clone()
        };
        let items = resource
            .items
            .iter()
            .map(|item| {
                let levels = item
                    .document
                    .descriptor
                    .levels
                    .iter()
                    .enumerate()
                    .map(|(level, descriptor)| {
                        let key = (item.id, level);
                        let (status, bytes, channels_loaded, error) =
                            if self.memory_pending.contains_key(&key) {
                                ("loading", None, None, None)
                            } else {
                                match self.pinned_levels.get(&key) {
                                    None => ("unloaded", None, None, None),
                                    Some(MosaicPinnedLevelState::Loaded(resource)) => (
                                        "loaded",
                                        Some(resource.bytes()),
                                        Some(resource.channels_loaded()),
                                        None,
                                    ),
                                    Some(MosaicPinnedLevelState::Failed(error)) => {
                                        ("failed", None, None, Some(error.as_str()))
                                    }
                                }
                            };
                        let present = selected
                            .iter()
                            .filter(|index| {
                                self.channels.get(**index).is_some_and(|global| {
                                    item.document
                                        .descriptor
                                        .channels
                                        .iter()
                                        .any(|local| local.name == global.name)
                                })
                            })
                            .count();
                        json!({
                            "level":level,
                            "shape":descriptor.shape,
                            "selected_channel_estimate_bytes":estimate_level_bytes(&item.document, level, present),
                            "status":status,
                            "loaded_bytes":bytes,
                            "channels_loaded":channels_loaded,
                            "error":error,
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "id":item.id,
                    "sample_id":item.roi_id,
                    "focused":self.focused_id == Some(item.id),
                    "levels":levels,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "mode":"mosaic",
            "running":!self.memory_pending.is_empty(),
            "status":self.memory_status,
            "pinned_bytes":self.pinned_bytes(),
            "system":self.system_memory.map(|memory| json!({
                "total_bytes":memory.total_bytes,
                "available_bytes":memory.available_bytes,
            })),
            "selected_channels":selected,
            "items":items,
        }))
    }

    fn unpin_memory(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok())
            .ok_or_else(|| invalid("memory level is required"))?;
        let item_ids = self.memory_item_ids(params)?;
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        let mut unloaded = 0;
        for item_id in item_ids {
            self.memory_pending.remove(&(item_id, level));
            if self.pinned_levels.remove(&(item_id, level)).is_some() {
                unloaded += 1;
            }
        }
        self.memory_status = format!("Unloaded level {level} from {unloaded} ROI(s)");
        Ok(json!({"unloaded_items":unloaded,"level":level}))
    }

    fn unpin_all_memory(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.memory_operation_generation = self.memory_operation_generation.wrapping_add(1).max(1);
        self.memory_pending.clear();
        let unloaded = self.pinned_levels.len();
        self.pinned_levels.clear();
        self.memory_status = format!("Unloaded {unloaded} pinned mosaic level(s) from RAM");
        Ok(json!({"unloaded_item_levels":unloaded}))
    }

    fn pinned_bytes(&self) -> u64 {
        self.pinned_levels
            .values()
            .filter_map(|state| match state {
                MosaicPinnedLevelState::Loaded(resource) => Some(resource.bytes()),
                MosaicPinnedLevelState::Failed(_) => None,
            })
            .sum()
    }

    fn channels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "channels":self.channels.iter().map(|channel| self.channel_json(channel)).collect::<Vec<_>>(),
        }))
    }

    fn visible_channels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "channels":self.channels.iter().filter(|channel| channel.visible).map(|channel| json!({
                "index":channel.index,"name":channel.name
            })).collect::<Vec<_>>(),
        }))
    }

    fn active_channel_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "active_channel":self.channels.get(self.active_channel).map(|channel| self.channel_json(channel)),
        }))
    }

    fn set_active_channel(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let changed = self.active_channel != index;
        self.active_channel = index;
        let _ = self.native_layers.set_active(&format!("channel:{index}"));
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":changed,"active_channel":self.channel_json(&self.channels[index])},
        }))
    }

    fn set_visible_channels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let selectors = params
            .get("channels")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("channels must be an array"))?;
        let indices = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<HashSet<_>, _>>()?;
        let mode = params.get("mode").and_then(Value::as_str).unwrap_or("only");
        if !matches!(mode, "only" | "show" | "hide" | "add" | "remove") {
            return Err(invalid(format!("unknown visibility mode '{mode}'")));
        }
        let before = self
            .channels
            .iter()
            .map(|channel| channel.visible)
            .collect::<Vec<_>>();
        for channel in &mut self.channels {
            channel.visible = match mode {
                "show" | "add" => channel.visible || indices.contains(&channel.index),
                "hide" | "remove" => channel.visible && !indices.contains(&channel.index),
                "only" => indices.contains(&channel.index),
                _ => unreachable!(),
            };
        }
        if let Some(first) = indices.iter().next() {
            self.active_channel = *first;
        }
        self.sync_native_layers_from_semantics();
        Ok(json!({
            "mode":"mosaic",
            "result":{
                "changed":before != self.channels.iter().map(|channel| channel.visible).collect::<Vec<_>>(),
                "mode":match mode { "show"=>"add", "hide"=>"remove", mode=>mode },
                "visible_channels":self.channels.iter().filter(|channel| channel.visible).map(|channel| json!({"index":channel.index,"name":channel.name})).collect::<Vec<_>>(),
            },
        }))
    }

    fn get_channel_contrast(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = if params.as_object().is_some_and(|object| !object.is_empty()) {
            self.channel_index_from_params(params)?
        } else {
            self.active_channel
        };
        let channel = &self.channels[index];
        let (minimum, maximum) = channel.window.unwrap_or((0.0, self.abs_max()));
        Ok(json!({
            "mode":"mosaic",
            "contrast":{"index":index,"name":channel.name,"min":minimum,"max":maximum,"abs_max":self.abs_max()},
        }))
    }

    fn set_channel_contrast(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let minimum = params
            .get("min")
            .or_else(|| params.get("lo"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("min is required"))? as f32;
        let maximum = params
            .get("max")
            .or_else(|| params.get("hi"))
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("max is required"))? as f32;
        if !minimum.is_finite() || !maximum.is_finite() || maximum <= minimum {
            return Err(invalid("contrast max must be greater than min"));
        }
        let changed = self.channels[index].window != Some((minimum, maximum));
        self.channels[index].window = Some((minimum, maximum));
        Ok(json!({
            "mode":"mosaic",
            "contrast":{"changed":changed,"index":index,"name":self.channels[index].name,"min":minimum,"max":maximum,"abs_max":self.abs_max()},
        }))
    }

    fn set_channel_color(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let values = params
            .get("color_rgb")
            .or_else(|| params.get("color"))
            .and_then(Value::as_array)
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers"))?;
        let color = [
            json_u8(&values[0])?,
            json_u8(&values[1])?,
            json_u8(&values[2])?,
        ];
        let changed = self.channels[index].color_rgb != color;
        self.channels[index].color_rgb = color;
        Ok(
            json!({"mode":"mosaic","result":{"changed":changed,"channel":self.channel_json(&self.channels[index])}}),
        )
    }

    fn set_channel_note(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = self.channel_index_from_params(params)?;
        let note = params
            .get("note")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("set_channel_note requires note"))?
            .to_string();
        let changed = self.channels[index].note != note;
        self.channels[index].note = note;
        Ok(json!({"changed":changed,"channel":self.channel_json(&self.channels[index])}))
    }

    fn set_channel_order(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let selectors = params
            .get("order")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("order must be an array"))?;
        let mut order = selectors
            .iter()
            .map(|selector| self.channel_index(selector))
            .collect::<Result<Vec<_>, _>>()?;
        let mut seen = HashSet::new();
        order.retain(|index| seen.insert(*index));
        if order.len() != self.channels.len() {
            return Err(invalid(
                "channel order must contain every channel exactly once",
            ));
        }
        let changed = self.channel_order != order;
        self.channel_order = order;
        self.channel_sort = "manual".to_string();
        self.sync_native_layers_from_semantics();
        Ok(json!({
            "changed":changed,
            "order":self.channel_order.iter().map(|index| json!({"index":index,"name":self.channels[*index].name})).collect::<Vec<_>>(),
        }))
    }

    fn set_camera(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let before = (self.camera_center, self.camera_zoom);
        if let Some(center) = params.get("center_world_lvl0").and_then(json_pair) {
            self.camera_center = center;
        }
        if let Some(x) = params.get("center_x").and_then(Value::as_f64) {
            if !x.is_finite() {
                return Err(invalid("center_x must be finite"));
            }
            self.camera_center[0] = x as f32;
        }
        if let Some(y) = params.get("center_y").and_then(Value::as_f64) {
            if !y.is_finite() {
                return Err(invalid("center_y must be finite"));
            }
            self.camera_center[1] = y as f32;
        }
        if let Some(zoom) = params
            .get("zoom_screen_per_lvl0_px")
            .or_else(|| params.get("zoom"))
            .and_then(Value::as_f64)
        {
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(invalid("zoom must be finite and greater than zero"));
            }
            self.camera_zoom = (zoom as f32).clamp(0.000_01, 5000.0);
        }
        Ok(json!({
            "mode":"mosaic",
            "camera":self.camera_snapshot(),
            "changed":before != (self.camera_center,self.camera_zoom),
        }))
    }

    fn zoom_camera(&mut self, params: &Value, zoom_in: bool) -> Result<Value, ControlError> {
        let factor = params.get("factor").and_then(Value::as_f64).unwrap_or(1.5) as f32;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(invalid("zoom factor must be finite and > 0"));
        }
        let factor = if zoom_in { factor } else { 1.0 / factor };
        self.set_camera(&json!({"zoom":self.camera_zoom * factor}))
    }

    fn fit_camera(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.fit_bounds(self.bounds);
        Ok(json!({"mode":"mosaic","camera":self.camera_snapshot()}))
    }

    fn panels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(
            json!({"mode":"mosaic","panels":{"left":self.show_left_panel,"right":self.show_right_panel}}),
        )
    }

    fn set_panels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let before = (self.show_left_panel, self.show_right_panel);
        if let Some(value) = params.get("left").and_then(Value::as_bool) {
            self.show_left_panel = value;
        }
        if let Some(value) = params.get("right").and_then(Value::as_bool) {
            self.show_right_panel = value;
        }
        Ok(json!({
            "mode":"mosaic",
            "result":{"changed":before != (self.show_left_panel,self.show_right_panel),"panels":{"left":self.show_left_panel,"right":self.show_right_panel}},
        }))
    }

    fn smooth_pixels_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"mode":"mosaic","smooth_pixels":{"smooth":self.smooth_pixels}}))
    }

    fn set_smooth_pixels(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let smooth = params
            .get("smooth")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_smooth_pixels requires smooth"))?;
        let changed = self.smooth_pixels != smooth;
        self.smooth_pixels = smooth;
        Ok(json!({"mode":"mosaic","result":{"changed":changed,"smooth_pixels":{"smooth":smooth}}}))
    }

    fn rendering_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({
            "mode":"mosaic",
            "gpu_available":true,
            "renderer":"opengl",
            "compositing":"additive",
            "smooth_pixels":self.smooth_pixels,
        }))
    }

    fn object_visibility_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let target = params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("objects");
        Ok(json!({
            "mode":"mosaic",
            "overlay":{"target":target,"segmentation_objects":self.objects_visible,"object_count":self.object_resources.values().map(|resource| resource.features.len()).sum::<usize>()},
        }))
    }

    fn set_object_visibility(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let visible = params
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("set_object_overlay_visibility requires visible"))?;
        let changed = self.objects_visible != visible;
        self.objects_visible = visible;
        let _ = self
            .native_layers
            .set_visibility("segmentation_geojson", visible);
        let mut response = self.object_visibility_snapshot(params)?;
        response
            .as_object_mut()
            .expect("object visibility response is an object")
            .insert("changed".to_string(), Value::Bool(changed));
        Ok(response)
    }

    fn fast_object_rendering_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"enabled":self.fast_object_rendering}))
    }

    fn set_fast_object_rendering(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let enabled = params
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid("enabled is required"))?;
        let changed = self.fast_object_rendering != enabled;
        self.fast_object_rendering = enabled;
        Ok(json!({"enabled":enabled,"changed":changed}))
    }

    fn channel_index_from_params(&self, params: &Value) -> Result<usize, ControlError> {
        let selector = params
            .get("index")
            .or_else(|| params.get("channel_index"))
            .or_else(|| params.get("name"))
            .or_else(|| params.get("channel"))
            .or_else(|| params.get("marker"))
            .ok_or_else(|| invalid("provide index, name, channel, or marker"))?;
        self.channel_index(selector)
    }

    fn channel_index(&self, selector: &Value) -> Result<usize, ControlError> {
        if let Some(index) = selector.as_u64() {
            return usize::try_from(index)
                .ok()
                .filter(|index| *index < self.channels.len())
                .ok_or_else(|| invalid(format!("channel index {index} is out of range")));
        }
        let name = selector
            .as_str()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid(format!("invalid channel selector: {selector}")))?;
        let needle = normalize_name(name);
        if let Some(index) = self
            .channels
            .iter()
            .position(|channel| normalize_name(&channel.name) == needle)
        {
            return Ok(index);
        }
        let matches = self
            .channels
            .iter()
            .enumerate()
            .filter(|(_, channel)| normalize_name(&channel.name).contains(&needle))
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("no channel matches '{name}'"),
            )),
            _ => Err(invalid(format!("channel selector '{name}' is ambiguous"))),
        }
    }

    fn channel_json(&self, channel: &MosaicChannelModel) -> Value {
        json!({
            "index":channel.index,
            "name":channel.name,
            "visible":channel.visible,
            "active":channel.index == self.active_channel,
            "color_rgb":channel.color_rgb,
            "window":channel.window.map(|(minimum,maximum)| [minimum,maximum]),
            "note":channel.note,
        })
    }

    fn abs_max(&self) -> f32 {
        self.resource
            .as_ref()
            .into_iter()
            .flat_map(|resource| resource.items.iter())
            .map(|item| item.document.descriptor.abs_max)
            .fold(1.0_f32, f32::max)
    }

    pub(crate) fn prepare_object_load(
        &mut self,
        downsample_factor: f32,
    ) -> Result<MosaicObjectLoadSpec, ControlError> {
        self.require_resource()?;
        if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
            return Err(invalid(
                "downsample_factor must be finite and greater than zero",
            ));
        }
        if self.selected_ids.is_empty() {
            return Err(invalid("Select at least one mosaic ROI first."));
        }
        let items = self
            .items
            .iter()
            .filter(|item| self.selected_ids.contains(&item.id))
            .filter_map(|item| {
                item.segmentation_path
                    .as_ref()
                    .map(|path| (item.id, path.clone()))
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "None of the selected mosaic ROIs has an object segmentation source.",
            ));
        }
        self.cancel_object_load("Superseded by a newer mosaic object load");
        self.object_operation_generation = self.object_operation_generation.wrapping_add(1).max(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.object_pending_ids = items.iter().map(|(id, _)| *id).collect();
        self.object_failures.clear();
        self.object_cancel = Some(Arc::clone(&cancel));
        self.object_status = format!("Loading objects for {} mosaic ROI(s)", items.len());
        Ok(MosaicObjectLoadSpec {
            resource_generation: self.resource_generation(),
            operation_generation: self.object_operation_generation,
            downsample_factor,
            items,
            cancel,
        })
    }

    pub(crate) fn finish_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        result: MosaicObjectLoadResult,
    ) -> Option<Value> {
        if !self.object_spec_is_current(spec) {
            return None;
        }
        let requested = spec.items.len();
        for (id, resource) in result.loaded {
            self.object_resources.insert(id, resource);
            self.object_pending_ids.remove(&id);
        }
        for (id, error) in result.failures {
            self.object_failures.insert(id, error);
            self.object_pending_ids.remove(&id);
        }
        let cancelled = result.cancelled || spec.is_cancelled();
        self.object_pending_ids.clear();
        self.object_cancel = None;
        self.object_status = if cancelled {
            "Mosaic object loading cancelled".to_string()
        } else if self.object_failures.is_empty() {
            format!("Loaded objects for {requested} mosaic ROI(s)")
        } else {
            format!(
                "Loaded objects for {} of {requested} mosaic ROI(s)",
                requested.saturating_sub(self.object_failures.len())
            )
        };
        Some(json!({
            "settled":true,
            "cancelled":cancelled,
            "requested":requested,
            "loaded":requested.saturating_sub(self.object_failures.len()),
            "failed":self.object_failures.len(),
            "state":self.object_state(),
        }))
    }

    pub(crate) fn fail_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.object_spec_is_current(spec) {
            return false;
        }
        self.object_pending_ids.clear();
        self.object_cancel = None;
        self.object_status = message.into();
        true
    }

    pub(crate) fn cancel_object_load(&mut self, message: impl Into<String>) -> usize {
        let cancelled = self.object_pending_ids.len();
        if let Some(cancel) = self.object_cancel.take() {
            cancel.store(true, AtomicOrdering::Relaxed);
        }
        self.object_pending_ids.clear();
        if cancelled > 0 {
            self.object_status = message.into();
        }
        cancelled
    }

    pub(crate) fn cancel_object_load_response(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let cancelled = self.cancel_object_load("Mosaic object loading cancelled");
        Ok(json!({
            "cancelled_requests":cancelled,
            "in_flight_cancelled":cancelled > 0,
            "state":self.object_state(),
        }))
    }

    fn object_spec_is_current(&self, spec: &MosaicObjectLoadSpec) -> bool {
        self.resource_generation() == spec.resource_generation
            && self.object_operation_generation == spec.operation_generation
    }

    fn object_state(&self) -> Value {
        let items = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let resource = self.object_resources.get(&item.id);
                json!({
                    "index":index,
                    "item_id":item.id,
                    "roi_id":item.roi_id,
                    "selected":self.selected_ids.contains(&item.id),
                    "available":item.segmentation_path.is_some(),
                    "path":item.segmentation_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
                    "requested":self.object_pending_ids.contains(&item.id),
                    "loaded":resource.is_some(),
                    "object_count":resource.map_or(0, |resource| resource.features.len()),
                    "error":self.object_failures.get(&item.id),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "generation":self.object_operation_generation,
            "requested_count":self.object_pending_ids.len(),
            "requested_loading":self.object_pending_ids.len(),
            "settled":self.object_pending_ids.is_empty(),
            "loaded_count":self.object_resources.len(),
            "failed_count":self.object_failures.len(),
            "status":self.object_status,
            "items":items,
        })
    }

    fn set_right_tab(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let tab = params
            .get("tab")
            .or_else(|| params.get("right_tab"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|tab| !tab.is_empty())
            .ok_or_else(|| invalid("set_mosaic_right_tab requires tab"))?;
        if !matches!(tab, "properties" | "views" | "layout" | "memory") {
            return Err(invalid(
                "unknown right tab; expected properties, views, layout, or memory",
            ));
        }
        self.right_tab = tab.to_string();
        Ok(json!({"right_tab":self.right_tab}))
    }

    fn configure_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if let Some(group_by) = params.get("group_by").and_then(Value::as_str) {
            self.group_by = group_by.trim().to_string();
        }
        if let Some(sort_by) = params
            .get("sort_by")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by = sort_by.to_string();
        }
        if let Some(sort_by) = params
            .get("sort_by_secondary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.sort_by_secondary = sort_by.to_string();
            self.sort_secondary_enabled = true;
        }
        if let Some(enabled) = params
            .get("sort_secondary_enabled")
            .and_then(Value::as_bool)
        {
            self.sort_secondary_enabled = enabled;
        }
        if let Some(show) = params.get("show_group_labels").and_then(Value::as_bool) {
            self.show_group_labels = show;
        }
        if let Some(show) = params.get("show_text_labels").and_then(Value::as_bool) {
            self.show_text_labels = show;
            let _ = self.native_layers.set_visibility("text_labels", show);
        }
        if let Some(gap) = params.get("group_gap").and_then(Value::as_f64) {
            if !gap.is_finite() {
                return Err(invalid("group_gap must be finite"));
            }
            self.group_gap = gap.max(0.0) as f32;
        }
        if let Some(columns) = params.get("columns").and_then(Value::as_u64) {
            self.columns = usize::try_from(columns).unwrap_or(usize::MAX).max(1);
        }
        if let Some(layout) = params
            .get("layout")
            .or_else(|| params.get("layout_mode"))
            .and_then(Value::as_str)
        {
            self.layout_mode = MosaicLayoutMode::parse(layout.trim())
                .ok_or_else(|| invalid("unknown layout; expected fit_cells or native_pixels"))?;
        }
        if let Some(columns) = params.get("label_columns").and_then(Value::as_array) {
            self.label_columns = columns
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect();
        }
        let preserve_center = self.camera_center;
        self.apply_layout();
        self.camera_center = clamp_point_to_bounds(preserve_center, self.bounds);
        if params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            self.fit_bounds(self.bounds);
        }
        Ok(self.layout_snapshot())
    }

    fn layout_snapshot(&self) -> Value {
        json!({
            "right_tab":self.right_tab,
            "group_by":self.group_by,
            "sort_by":self.sort_by,
            "sort_secondary_enabled":self.sort_secondary_enabled,
            "sort_by_secondary":self.sort_by_secondary,
            "layout":self.layout_mode.as_str(),
            "columns":self.columns,
            "group_gap":self.group_gap,
            "show_group_labels":self.show_group_labels,
            "show_text_labels":self.show_text_labels,
            "label_columns":self.label_columns,
        })
    }

    fn list_items(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let offset = params.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = params.get("limit").and_then(Value::as_u64).unwrap_or(200) as usize;
        let total = self.items.len();
        let items = self
            .items
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, item)| {
                json!({
                    "index":index,
                    "id":item.id,
                    "roi_id":item.roi_id,
                    "metadata":item.metadata,
                    "source":item.source,
                    "offset_world":item.offset,
                    "scale":item.scale,
                    "placed_size":item.placed_size,
                    "bounds_world":{"min":item.bounds()[0],"max":item.bounds()[1]},
                    "focused":self.focused_id == Some(item.id),
                    "selected":self.selected_ids.contains(&item.id),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "total":total,
            "offset":offset,
            "limit":limit,
            "has_more":offset.saturating_add(items.len()) < total,
            "items":items,
        }))
    }

    fn selection_snapshot(&self) -> Value {
        let selected = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.selected_ids.contains(&item.id))
            .map(|(index, item)| json!({"index":index,"id":item.id,"roi_id":item.roi_id}))
            .collect::<Vec<_>>();
        json!({"count":selected.len(),"selected":selected})
    }

    fn set_selection(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("replace");
        if mode == "all" {
            self.selected_ids = self.items.iter().map(|item| item.id).collect();
            return Ok(self.selection_snapshot());
        }
        if mode == "range" {
            let start = params
                .get("start")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("range selection requires start and end"))?;
            let end = params
                .get("end")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("range selection requires start and end"))?;
            let start = self.item_index_for_roi(start)?;
            let end = self.item_index_for_roi(end)?;
            let (lo, hi) = if start <= end {
                (start, end)
            } else {
                (end, start)
            };
            self.selected_ids = self.items[lo..=hi].iter().map(|item| item.id).collect();
            return Ok(self.selection_snapshot());
        }
        let ids = params
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("ids is required"))?
            .iter()
            .map(|id| {
                id.as_str()
                    .ok_or_else(|| invalid("mosaic ROI IDs must be strings"))
                    .and_then(|id| self.item_index_for_roi(id))
                    .map(|index| self.items[index].id)
            })
            .collect::<Result<HashSet<_>, _>>()?;
        match mode {
            "replace" => self.selected_ids = ids,
            "add" => self.selected_ids.extend(ids),
            "remove" => self.selected_ids.retain(|id| !ids.contains(id)),
            "toggle" => {
                for id in ids {
                    if !self.selected_ids.insert(id) {
                        self.selected_ids.remove(&id);
                    }
                }
            }
            _ => return Err(invalid("unknown mosaic selection mode")),
        }
        Ok(self.selection_snapshot())
    }

    fn clear_selection(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.selected_ids.clear();
        Ok(self.selection_snapshot())
    }

    fn focus_snapshot(&self) -> Value {
        self.focused_id
            .and_then(|id| {
                self.items
                    .iter()
                    .position(|item| item.id == id)
                    .map(|index| {
                        let item = &self.items[index];
                        json!({
                            "index":index,
                            "id":item.id,
                            "roi_id":item.roi_id,
                            "metadata":item.metadata,
                        })
                    })
            })
            .unwrap_or(Value::Null)
    }

    fn set_focus(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let index = if let Some(index) = params.get("index").and_then(Value::as_u64) {
            usize::try_from(index)
                .ok()
                .filter(|index| *index < self.items.len())
                .ok_or_else(|| invalid(format!("mosaic ROI index {index} is out of range")))?
        } else if let Some(roi_id) = params
            .get("roi_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_str)
        {
            self.item_index_for_roi(roi_id)?
        } else {
            return Err(invalid("provide index or roi_id"));
        };
        let before = self.focused_id;
        self.focused_id = Some(self.items[index].id);
        if params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            self.fit_bounds(self.items[index].bounds());
        }
        Ok(json!({
            "changed":before != self.focused_id,
            "focused":self.focus_snapshot(),
        }))
    }

    fn step_focus(&mut self, params: &Value, forward: bool) -> Result<Value, ControlError> {
        self.require_resource()?;
        let step = params.get("step").and_then(Value::as_u64).unwrap_or(1) as usize;
        let wrap = params.get("wrap").and_then(Value::as_bool).unwrap_or(true);
        let current = self
            .focused_id
            .and_then(|id| self.items.iter().position(|item| item.id == id))
            .unwrap_or(0);
        let index = if wrap {
            let offset = step % self.items.len();
            if forward {
                (current + offset) % self.items.len()
            } else {
                (current + self.items.len() - offset) % self.items.len()
            }
        } else if forward {
            current.saturating_add(step).min(self.items.len() - 1)
        } else {
            current.saturating_sub(step)
        };
        let mut next =
            json!({"index":index,"fit":params.get("fit").and_then(Value::as_bool).unwrap_or(true)});
        if let Some(object) = next.as_object_mut() {
            object.insert("wrap".to_string(), Value::Bool(wrap));
        }
        self.set_focus(&next)
    }

    fn fit_focus(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let id = self
            .focused_id
            .ok_or_else(|| invalid("mosaic has no focused ROI"))?;
        let bounds = self
            .items
            .iter()
            .find(|item| item.id == id)
            .map(MosaicItemModel::bounds)
            .ok_or_else(|| invalid("focused mosaic ROI is not loaded"))?;
        self.fit_bounds(bounds);
        Ok(json!({"focused":self.focus_snapshot(),"camera":self.camera_snapshot()}))
    }

    fn clear_focus(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let changed = self.focused_id.take().is_some();
        Ok(json!({"changed":changed,"focused":null}))
    }

    fn fit_all(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        self.fit_bounds(self.bounds);
        Ok(json!({"camera":self.camera_snapshot()}))
    }

    pub(crate) fn camera_snapshot(&self) -> Value {
        json!({
            "center_world_lvl0":self.camera_center,
            "zoom_screen_per_lvl0_px":self.camera_zoom,
        })
    }

    fn fit_bounds(&mut self, bounds: [[f32; 2]; 2]) {
        let width = (bounds[1][0] - bounds[0][0]).max(1.0);
        let height = (bounds[1][1] - bounds[0][1]).max(1.0);
        self.camera_center = [
            (bounds[0][0] + bounds[1][0]) * 0.5,
            (bounds[0][1] + bounds[1][1]) * 0.5,
        ];
        self.camera_zoom = (self.logical_canvas[0] / width)
            .min(self.logical_canvas[1] / height)
            .max(0.000_01);
    }

    fn apply_layout(&mut self) {
        let focused = self.focused_id;
        let group_by = self.group_by.clone();
        let sort_by = self.sort_by.clone();
        let secondary = self
            .sort_secondary_enabled
            .then(|| self.sort_by_secondary.clone());
        self.items.sort_by(|left, right| {
            if !group_by.is_empty() {
                let ordering = compare_sort_values(
                    &group_value(left, &group_by),
                    &group_value(right, &group_by),
                );
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            let ordering =
                compare_sort_values(&left.sort_value(&sort_by), &right.sort_value(&sort_by));
            if ordering != Ordering::Equal {
                return ordering;
            }
            if let Some(secondary) = secondary.as_deref() {
                let ordering =
                    compare_sort_values(&left.sort_value(secondary), &right.sort_value(secondary));
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.roi_id.cmp(&right.roi_id)
        });
        self.focused_id = focused
            .filter(|id| self.items.iter().any(|item| item.id == *id))
            .or_else(|| self.items.first().map(|item| item.id));

        let mut max_width = 1.0_f32;
        let mut y = 0.0_f32;
        if self.group_by.is_empty() {
            let [width, height] = layout_block(
                &mut self.items,
                0.0,
                self.columns,
                self.grid_cell_size,
                self.grid_pad,
                self.layout_mode,
            );
            self.bounds = [[0.0, 0.0], [width, height]];
            return;
        }

        let mut start = 0;
        while start < self.items.len() {
            let group = group_value(&self.items[start], &self.group_by).to_ascii_lowercase();
            let mut end = start + 1;
            while end < self.items.len()
                && group_value(&self.items[end], &self.group_by).to_ascii_lowercase() == group
            {
                end += 1;
            }
            let [width, height] = layout_block(
                &mut self.items[start..end],
                y + GROUP_HEADER_HEIGHT,
                self.columns,
                self.grid_cell_size,
                self.grid_pad,
                self.layout_mode,
            );
            max_width = max_width.max(width);
            y += GROUP_HEADER_HEIGHT + height;
            if end < self.items.len() {
                y += self.group_gap;
            }
            start = end;
        }
        self.bounds = [[0.0, 0.0], [max_width.max(1.0), y.max(1.0)]];
    }

    fn item_index_for_roi(&self, roi_id: &str) -> Result<usize, ControlError> {
        let matches = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.roi_id == roi_id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("mosaic ROI '{roi_id}' was not found"),
            )),
            _ => Err(invalid(format!("mosaic ROI '{roi_id}' is ambiguous"))),
        }
    }

    fn metadata_columns(&self) -> &[String] {
        self.resource
            .as_ref()
            .map_or(&[], |resource| resource.metadata_columns.as_slice())
    }

    fn require_resource(&self) -> Result<&Arc<ControlMosaicResource>, ControlError> {
        self.resource.as_ref().ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "No mosaic resource is currently open",
            )
        })
    }
}

fn initial_native_layers(
    channels: &[MosaicChannelModel],
    items: &[MosaicItemModel],
) -> NativeLayersModel {
    let mut layers = channels
        .iter()
        .map(|channel| {
            json!({
                "layer_id":format!("channel:{}", channel.index),
                "kind":"channel",
                "name":channel.name,
                "stack":"channels",
                "order":channel.index,
                "active":channel.index == 0,
                "visible":channel.visible,
                "available":true,
                "offset_world":[0.0,0.0],
                "presentation":{
                    "visible":channel.visible,
                    "color_rgb":channel.color_rgb,
                    "window":channel.window.map(|(minimum,maximum)| json!({"min":minimum,"max":maximum})),
                },
            })
        })
        .collect::<Vec<_>>();
    let has_segmentation = items.iter().any(|item| item.segmentation_path.is_some());
    layers.push(json!({
        "layer_id":"segmentation_geojson",
        "kind":"segmentation_geojson",
        "name":"Segmentation (GeoJSON)",
        "stack":"overlays",
        "order":0,
        "active":channels.is_empty(),
        "visible":false,
        "available":has_segmentation,
        "offset_world":[0.0,0.0],
        "presentation":{"visible":false},
    }));
    layers.push(json!({
        "layer_id":"text_labels",
        "kind":"text_labels",
        "name":"Text labels",
        "stack":"overlays",
        "order":1,
        "active":channels.is_empty() && !has_segmentation,
        "visible":true,
        "available":true,
        "offset_world":[0.0,0.0],
        "presentation":{"visible":true},
    }));
    NativeLayersModel::restore(&Value::Array(layers))
        .expect("canonical mosaic native-layer inventory is valid")
}

fn estimate_level_bytes(
    document: &ControlOpenedDocument,
    level: usize,
    selected_channel_count: usize,
) -> u64 {
    let descriptor = &document.descriptor;
    let Some(level) = descriptor.levels.get(level) else {
        return 0;
    };
    let Some(&height) = level.shape.get(descriptor.dims.y) else {
        return 0;
    };
    let Some(&width) = level.shape.get(descriptor.dims.x) else {
        return 0;
    };
    let channel_count = if descriptor.dims.c.is_some() {
        selected_channel_count as u64
    } else {
        u64::from(selected_channel_count > 0)
    };
    let bytes_per_sample = match level.dtype.as_str() {
        "|u1" | "|i1" => 1,
        "<u2" | ">u2" | "<i2" | ">i2" => 2,
        "<f4" | ">f4" | "<u4" | ">u4" | "<i4" | ">i4" => 4,
        _ => 2,
    };
    channel_count
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(bytes_per_sample))
        .unwrap_or(0)
}

fn canonical_channel_sort(value: &str) -> Option<&'static str> {
    match value {
        "manual" | "project_order" => Some("manual"),
        "name_asc" | "alphabetical_asc" => Some("name_asc"),
        "name_desc" | "alphabetical_desc" => Some("name_desc"),
        "visible_first" | "enabled_desc" => Some("visible_first"),
        "hidden_first" | "enabled_asc" => Some("hidden_first"),
        _ => None,
    }
}

fn ensure_channel_group(
    groups: &mut ProjectLayerGroups,
    requested_id: Option<u64>,
    requested_name: Option<&str>,
    color_rgb: Option<[u8; 3]>,
) -> u64 {
    if let Some(group_id) = requested_id
        && let Some(group) = groups
            .channel_groups
            .iter_mut()
            .find(|group| group.id == group_id)
    {
        if let Some(name) = requested_name {
            group.name = name.to_string();
        }
        if let Some(color) = color_rgb {
            group.color_rgb = color;
        }
        return group_id;
    }
    if let Some(name) = requested_name
        && let Some(group) = groups
            .channel_groups
            .iter_mut()
            .find(|group| group.name == name)
    {
        if let Some(color) = color_rgb {
            group.color_rgb = color;
        }
        return group.id;
    }
    let next_id = requested_id
        .filter(|id| !groups.channel_groups.iter().any(|group| group.id == *id))
        .unwrap_or_else(|| {
            groups
                .channel_groups
                .iter()
                .map(|group| group.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1)
                .max(1)
        });
    groups.channel_groups.push(ProjectChannelGroup {
        id: next_id,
        name: requested_name
            .map(str::to_string)
            .unwrap_or_else(|| format!("Group {next_id}")),
        expanded: true,
        color_rgb: color_rgb.unwrap_or([255, 255, 255]),
    });
    next_id
}

fn color_from_params(params: &Value) -> Result<Option<[u8; 3]>, ControlError> {
    if let Some(values) = params.get("color_rgb") {
        let values = values
            .as_array()
            .filter(|values| values.len() == 3)
            .ok_or_else(|| invalid("color_rgb must contain three integers from 0 to 255"))?;
        return Ok(Some([
            json_u8(&values[0])?,
            json_u8(&values[1])?,
            json_u8(&values[2])?,
        ]));
    }
    let Some(value) = params
        .get("color")
        .or_else(|| params.get("colour"))
        .and_then(Value::as_str)
    else {
        return Ok(None);
    };
    let color = match value.trim().to_ascii_lowercase().as_str() {
        "white" => Some([255, 255, 255]),
        "black" => Some([0, 0, 0]),
        "red" => Some([230, 57, 70]),
        "green" => Some([42, 157, 143]),
        "blue" => Some([69, 123, 157]),
        "cyan" => Some([0, 188, 212]),
        "magenta" => Some([216, 27, 96]),
        "yellow" => Some([255, 202, 40]),
        "orange" => Some([251, 133, 0]),
        "purple" => Some([126, 87, 194]),
        "pink" => Some([244, 143, 177]),
        "lime" => Some([139, 195, 74]),
        "teal" => Some([0, 150, 136]),
        "amber" => Some([255, 193, 7]),
        "gray" | "grey" => Some([158, 158, 158]),
        _ => parse_hex_color(value),
    };
    color
        .map(Some)
        .ok_or_else(|| invalid(format!("unknown color '{value}'")))
}

fn parse_hex_color(value: &str) -> Option<[u8; 3]> {
    let value = value.trim().strip_prefix('#').unwrap_or(value.trim());
    match value.len() {
        6 => Some([
            u8::from_str_radix(&value[0..2], 16).ok()?,
            u8::from_str_radix(&value[2..4], 16).ok()?,
            u8::from_str_radix(&value[4..6], 16).ok()?,
        ]),
        3 => Some([
            u8::from_str_radix(&value[0..1], 16)
                .ok()?
                .saturating_mul(17),
            u8::from_str_radix(&value[1..2], 16)
                .ok()?
                .saturating_mul(17),
            u8::from_str_radix(&value[2..3], 16)
                .ok()?
                .saturating_mul(17),
        ]),
        _ => None,
    }
}

fn mosaic_channel(index: usize, channel: &ChannelInfo) -> MosaicChannelModel {
    MosaicChannelModel {
        index,
        name: channel.name.clone(),
        visible: channel.visible,
        color_rgb: channel.color_rgb,
        window: channel.window,
        note: channel.note.clone(),
    }
}

fn layout_block(
    items: &mut [MosaicItemModel],
    y_offset: f32,
    columns: usize,
    cell_size: [f32; 2],
    padding: f32,
    mode: MosaicLayoutMode,
) -> [f32; 2] {
    if items.is_empty() {
        return [1.0, 1.0];
    }
    let columns = columns.max(1);
    match mode {
        MosaicLayoutMode::FitCells => {
            for (position, item) in items.iter_mut().enumerate() {
                let scale = (cell_size[0] / item.level0_size[0])
                    .min(cell_size[1] / item.level0_size[1])
                    .max(0.000_001);
                item.scale = scale;
                item.placed_size = [item.level0_size[0] * scale, item.level0_size[1] * scale];
                let column = (position % columns) as f32;
                let row = (position / columns) as f32;
                item.offset = [
                    column * (cell_size[0] + padding) + (cell_size[0] - item.placed_size[0]) * 0.5,
                    y_offset
                        + row * (cell_size[1] + padding)
                        + (cell_size[1] - item.placed_size[1]) * 0.5,
                ];
            }
            let rows = items.len().div_ceil(columns);
            [
                columns as f32 * (cell_size[0] + padding) - padding,
                rows as f32 * (cell_size[1] + padding) - padding,
            ]
        }
        MosaicLayoutMode::NativePixels => {
            let mut max_width = 1.0_f32;
            let mut y = y_offset;
            for row in items.chunks_mut(columns) {
                let row_height = row
                    .iter()
                    .map(|item| item.level0_size[1])
                    .fold(1.0_f32, f32::max);
                let mut x = 0.0_f32;
                for item in row {
                    item.scale = 1.0;
                    item.placed_size = item.level0_size;
                    item.offset = [x, y + (row_height - item.level0_size[1]) * 0.5];
                    x += item.level0_size[0] + padding;
                }
                max_width = max_width.max((x - padding).max(1.0));
                y += row_height + padding;
            }
            [max_width, (y - padding - y_offset).max(1.0)]
        }
    }
}

fn group_value(item: &MosaicItemModel, column: &str) -> String {
    let value = item.sort_value(column);
    if value.trim().is_empty() {
        "(missing)".to_string()
    } else {
        value
    }
}

fn compare_sort_values(left: &str, right: &str) -> Ordering {
    let left_empty = left.trim().is_empty();
    let right_empty = right.trim().is_empty();
    if left_empty != right_empty {
        return left_empty.cmp(&right_empty);
    }
    left.to_ascii_lowercase().cmp(&right.to_ascii_lowercase())
}

fn clamp_point_to_bounds(point: [f32; 2], bounds: [[f32; 2]; 2]) -> [f32; 2] {
    [
        point[0].clamp(bounds[0][0], bounds[1][0]),
        point[1].clamp(bounds[0][1], bounds[1][1]),
    ]
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn json_pair(value: &Value) -> Option<[f32; 2]> {
    let values = value.as_array()?;
    if values.len() != 2 {
        return None;
    }
    let x = values[0].as_f64()?;
    let y = values[1].as_f64()?;
    (x.is_finite() && y.is_finite()).then_some([x as f32, y as f32])
}

fn json_u8(value: &Value) -> Result<u8, ControlError> {
    value
        .as_u64()
        .and_then(|value| u8::try_from(value).ok())
        .ok_or_else(|| invalid("color components must be integers from 0 through 255"))
}

fn normalize_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_layout_accumulates_rows_without_overlapping() {
        let mut items = (0..3)
            .map(|id| MosaicItemModel {
                id,
                roi_id: format!("roi-{id}"),
                metadata: HashMap::new(),
                source: format!("source-{id}"),
                level0_size: [100.0 + id as f32 * 10.0, 50.0 + id as f32 * 5.0],
                offset: [0.0, 0.0],
                scale: 1.0,
                placed_size: [1.0, 1.0],
                segmentation_path: None,
            })
            .collect::<Vec<_>>();
        let size = layout_block(
            &mut items,
            48.0,
            2,
            [1.0, 1.0],
            10.0,
            MosaicLayoutMode::NativePixels,
        );
        assert!(items[2].offset[1] > items[0].bounds()[1][1]);
        assert!(size[0] >= 220.0);
        assert!(size[1] >= 115.0);
    }
}
