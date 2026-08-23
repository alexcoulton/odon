mod commands;
mod geojson;
mod state;
mod validation;

#[cfg(test)]
mod tests;

use crate::data::project_config::ProjectMaskLayer;

pub(crate) use geojson::load_geojson_mask_polylines;

const MAX_UNDO_STATES: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskSelection {
    layer_id: u64,
    polygon_index: usize,
    vertex_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct MaskUndoState {
    layers: Vec<ProjectMaskLayer>,
    next_id: u64,
    active_layer_id: Option<u64>,
    selection: Option<MaskSelection>,
    dirty: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct MaskModel {
    layers: Vec<ProjectMaskLayer>,
    next_id: u64,
    active_layer_id: Option<u64>,
    selection: Option<MaskSelection>,
    undo: Vec<MaskUndoState>,
    generation: u64,
    dirty: bool,
}

impl Default for MaskModel {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            next_id: 1,
            active_layer_id: None,
            selection: None,
            undo: Vec::new(),
            generation: 1,
            dirty: false,
        }
    }
}
