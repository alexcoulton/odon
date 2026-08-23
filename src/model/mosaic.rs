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

mod layout;
mod memory;
mod objects;
mod presentation;
#[cfg(test)]
mod tests;

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
