use std::path::PathBuf;

use crate::render::polygon_fill_gl::ObjectFillGlRenderer;
use crate::spatialdata::{SpatialDataElement, SpatialDataTransform2};

mod points;
mod shapes;

pub use points::SpatialPointsLayer;
pub(crate) use points::{PreparedSpatialPointsLayer, prepare_spatial_points_layer};
pub use shapes::SpatialShapesLayer;
pub(crate) use shapes::{PreparedSpatialShape, prepare_spatial_shape_data};

// SpatialData elements are discovered from format-specific metadata, then adapted
// into the viewer's native overlay types. The rest of the app should not need to
// care whether a shape/point layer came from SpatialData or from another source.

#[derive(Debug)]
pub struct SpatialDataLayers {
    pub root: Option<PathBuf>,
    pub tables: Vec<SpatialDataElement>,
    pub shapes: Vec<SpatialShapesLayer>,
    pub points: Option<SpatialPointsLayer>,
    next_shape_layer_id: u64,
    object_fill_renderer: ObjectFillGlRenderer,
}

impl Default for SpatialDataLayers {
    fn default() -> Self {
        Self {
            root: None,
            tables: Vec::new(),
            shapes: Vec::new(),
            points: None,
            next_shape_layer_id: 0,
            object_fill_renderer: ObjectFillGlRenderer::application_pool(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositiveCellSelectionTarget {
    SegmentationObjects,
    AllObjectLayers,
    ShapeLayer(u64),
}

impl SpatialDataLayers {
    pub(crate) fn set_object_fill_renderer(&mut self, renderer: ObjectFillGlRenderer) {
        self.object_fill_renderer = renderer.clone();
        for shape in &mut self.shapes {
            if let Some(objects) = shape.object_layer_mut() {
                objects.set_object_fill_renderer(renderer.clone());
            }
        }
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.tables.clear();
        self.shapes.clear();
        self.points = None;
        self.next_shape_layer_id = 1;
    }

    pub fn set_root(&mut self, root: PathBuf) {
        self.root = Some(root);
    }

    pub fn set_tables(&mut self, tables: Vec<SpatialDataElement>) {
        self.tables = tables;
    }

    pub(crate) fn attach_prepared_shapes(&mut self, shapes: Vec<PreparedSpatialShape>) {
        self.next_shape_layer_id = shapes
            .iter()
            .map(|shape| shape.id)
            .max()
            .unwrap_or(0)
            .wrapping_add(1)
            .max(1);
        self.shapes = shapes
            .into_iter()
            .map(SpatialShapesLayer::from_prepared)
            .collect();
        self.set_object_fill_renderer(self.object_fill_renderer.clone());
    }

    pub fn load_external_shapes(
        &mut self,
        external_id: String,
        external_resource_id: String,
        name: String,
        parquet_path: PathBuf,
        transform: SpatialDataTransform2,
    ) -> u64 {
        let id = self.next_shape_layer_id.max(1);
        self.next_shape_layer_id = id.wrapping_add(1).max(1);
        let mut shape = SpatialShapesLayer::new(
            id,
            Some(external_id),
            Some(external_resource_id),
            name,
            parquet_path,
            transform,
        );
        if let Some(objects) = shape.object_layer_mut() {
            objects.set_object_fill_renderer(self.object_fill_renderer.clone());
        }
        self.shapes.push(shape);
        id
    }

    pub(crate) fn attach_prepared_points(&mut self, points: Option<PreparedSpatialPointsLayer>) {
        self.points = points.map(SpatialPointsLayer::from_prepared);
    }

    pub fn tick(&mut self) {
        for s in &mut self.shapes {
            s.tick();
        }
    }

    pub fn is_loading_shapes(&self) -> bool {
        self.shapes.iter().any(|s| s.is_loading())
    }

    pub fn is_loading_points(&self) -> bool {
        self.points.as_ref().is_some_and(|p| p.is_loading())
    }

    pub fn is_busy(&self) -> bool {
        self.is_loading_shapes()
            || self.is_loading_points()
            || self.shapes.iter().any(SpatialShapesLayer::is_busy)
    }

    pub fn positive_cell_selection_targets(&self) -> Vec<(PositiveCellSelectionTarget, String)> {
        self.shapes
            .iter()
            .filter(|layer| layer.has_object_layer())
            .map(|layer| {
                (
                    PositiveCellSelectionTarget::ShapeLayer(layer.id),
                    layer.name.clone(),
                )
            })
            .collect()
    }

    pub fn table_elements(&self) -> &[SpatialDataElement] {
        &self.tables
    }
}
