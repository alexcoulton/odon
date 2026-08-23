mod parquet;

use std::collections::HashMap;
use std::sync::Arc;

use eframe::egui;

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
