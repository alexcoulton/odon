mod parquet;

use std::collections::HashMap;
use std::sync::Arc;

use eframe::egui;
use serde::{Deserialize, Serialize};

use crate::data::point_bins::PointIndexBins;

pub use parquet::{load_annotations_parquet, read_parquet_columns};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotationValueMode {
    Categorical,
    Continuous,
}

#[derive(Debug, Clone)]
pub struct AnnotationRoiData {
    pub positions_local: Arc<Vec<egui::Pos2>>,
    pub values: Arc<Vec<f32>>,
    pub count: usize,
    pub bins_local: Option<Arc<PointIndexBins>>,
}

#[derive(Debug, Clone)]
pub struct AnnotationDataset {
    pub mode: AnnotationValueMode,
    pub categories: Vec<String>,
    pub roi: HashMap<String, AnnotationRoiData>,
    pub value_min: f32,
    pub value_max: f32,
    pub total_points: usize,
    pub total_rois: usize,
}

#[derive(Debug, Clone)]
pub struct AnnotationColumnInfo {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectAnnotationCategoryStyleState {
    pub name: String,
    pub visible: bool,
    pub color_rgb: [u8; 3],
    pub shape: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectAnnotationLayerState {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub radius_screen_px: f32,
    pub opacity: f32,
    pub stroke_width: f32,
    pub stroke_color_rgb: [u8; 3],
    pub stroke_color_alpha: u8,
    pub offset_world: [f32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parquet_path: Option<String>,
    pub roi_id_column: String,
    pub x_column: String,
    pub y_column: String,
    pub value_column: String,
    pub selected_value_column: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category_styles: Vec<ProjectAnnotationCategoryStyleState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_shape: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous_range: Option<[f32; 2]>,
}

pub fn default_category_style_states(
    categories: &[String],
) -> Vec<ProjectAnnotationCategoryStyleState> {
    const COLORS: &[[u8; 3]] = &[
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [174, 199, 232],
        [255, 187, 120],
        [152, 223, 138],
        [255, 152, 150],
        [197, 176, 213],
        [196, 156, 148],
        [247, 182, 210],
        [199, 199, 199],
        [219, 219, 141],
        [158, 218, 229],
    ];
    const SHAPES: &[&str] = &["circle", "square", "diamond", "cross"];
    categories
        .iter()
        .enumerate()
        .map(|(index, name)| ProjectAnnotationCategoryStyleState {
            name: name.clone(),
            visible: true,
            color_rgb: COLORS[index % COLORS.len()],
            shape: SHAPES[index % SHAPES.len()].to_string(),
        })
        .collect()
}
