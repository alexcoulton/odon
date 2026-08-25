mod canonical;
mod parsing;
mod resolution;
mod semantics;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub use parsing::is_deep_link;
use parsing::parse_deep_link;
pub use resolution::{
    DeepLinkResolution, apply_example_defaults, resolve_example_project_path, resolve_roi_target,
};
pub(crate) use semantics::{
    object_filter_model, object_segmentation_requested, requested_bundled_label,
};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepLinkChannelContrast {
    pub channel: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkChannelColor {
    pub channel: String,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepLinkChannelOrder {
    Listed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkObjectLevelColor {
    pub value: String,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkObjectFilterClause {
    pub property_key: String,
    pub query: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepLinkObjectFilterLogic {
    All,
    Any,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeepLinkRequest {
    pub example: Option<String>,
    pub project_path: Option<PathBuf>,
    pub roi: Option<String>,
    pub sample: Option<String>,
    pub channel: Option<String>,
    pub channel_alternatives: Vec<String>,
    pub visible_channels: Vec<String>,
    pub visible_channel_alternatives: Vec<Vec<String>>,
    pub group_visible_channels: bool,
    pub visible_channel_group: Option<String>,
    pub visible_channel_group_color: Option<[u8; 3]>,
    pub channel_order: Option<DeepLinkChannelOrder>,
    pub hidden_channels: Vec<String>,
    pub hidden_channel_alternatives: Vec<Vec<String>>,
    pub contrast_min: Option<f32>,
    pub contrast_max: Option<f32>,
    pub channel_contrasts: Vec<DeepLinkChannelContrast>,
    pub channel_colors: Vec<DeepLinkChannelColor>,
    pub segmentation: Option<String>,
    pub segmentation_source: Option<String>,
    pub load_segmentation_labels: Option<bool>,
    pub cell_color_by: Option<String>,
    pub object_color_mapping: Option<crate::model::ObjectColorMapping>,
    pub fill_cells: Option<bool>,
    pub show_selection_overlay: Option<bool>,
    pub fast_object_rendering: Option<bool>,
    pub visible_cell_types: Vec<String>,
    pub hidden_cell_types: Vec<String>,
    pub object_level_colors: Vec<DeepLinkObjectLevelColor>,
    pub object_filters: Vec<DeepLinkObjectFilterClause>,
    pub object_filter_logic: Option<DeepLinkObjectFilterLogic>,
    pub object_query: Option<String>,
    pub center_world: Option<[f32; 2]>,
    pub zoom: Option<f32>,
}

impl DeepLinkRequest {
    pub fn parse_arg(arg: &str) -> anyhow::Result<Option<Self>> {
        if !is_deep_link(arg) {
            return Ok(None);
        }
        Ok(Some(parse_deep_link(arg)?))
    }
}

impl Default for DeepLinkRequest {
    fn default() -> Self {
        Self {
            example: None,
            project_path: None,
            roi: None,
            sample: None,
            channel: None,
            channel_alternatives: Vec::new(),
            visible_channels: Vec::new(),
            visible_channel_alternatives: Vec::new(),
            group_visible_channels: false,
            visible_channel_group: None,
            visible_channel_group_color: None,
            channel_order: None,
            hidden_channels: Vec::new(),
            hidden_channel_alternatives: Vec::new(),
            contrast_min: None,
            contrast_max: None,
            channel_contrasts: Vec::new(),
            channel_colors: Vec::new(),
            segmentation: None,
            segmentation_source: None,
            load_segmentation_labels: None,
            cell_color_by: None,
            object_color_mapping: None,
            fill_cells: None,
            show_selection_overlay: None,
            fast_object_rendering: None,
            visible_cell_types: Vec::new(),
            hidden_cell_types: Vec::new(),
            object_level_colors: Vec::new(),
            object_filters: Vec::new(),
            object_filter_logic: None,
            object_query: None,
            center_world: None,
            zoom: None,
        }
    }
}
