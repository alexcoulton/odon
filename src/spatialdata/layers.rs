use std::path::{Path, PathBuf};

use crate::spatialdata::{SpatialDataElement, SpatialDataTransform2};

mod points;
mod shapes;

pub use points::SpatialPointsLayer;
pub use shapes::SpatialShapesLayer;

// SpatialData elements are discovered from format-specific metadata, then adapted
// into the viewer's native overlay types. The rest of the app should not need to
// care whether a shape/point layer came from SpatialData or from another source.

#[derive(Debug, Default)]
pub struct SpatialDataLayers {
    pub root: Option<PathBuf>,
    pub tables: Vec<SpatialDataElement>,
    pub shapes: Vec<SpatialShapesLayer>,
    pub points: Option<SpatialPointsLayer>,
    next_shape_layer_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositiveCellSelectionTarget {
    SegmentationObjects,
    AllObjectLayers,
    ShapeLayer(u64),
}

impl SpatialDataLayers {
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

    fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn load_shapes(&mut self, element: &SpatialDataElement) -> u64 {
        // Shape elements always start as lightweight SpatialShapesLayer wrappers.
        // The actual load step decides whether they stay as raw polylines/points or
        // are promoted into an ObjectsLayer for shared selection/filtering behavior.
        let Some(root) = self.root().map(|p| p.to_path_buf()) else {
            return 0;
        };
        let Some(rel) = element.rel_parquet.clone() else {
            return 0;
        };
        let path = root.join(rel);
        let id = self.next_shape_layer_id.max(1);
        self.next_shape_layer_id = id.wrapping_add(1).max(1);
        let layer = SpatialShapesLayer::new(
            id,
            None,
            None,
            format!("Shapes: {}", element.name),
            path,
            element.transform,
        );
        self.shapes.push(layer);
        id
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
        self.shapes.push(SpatialShapesLayer::new(
            id,
            Some(external_id),
            Some(external_resource_id),
            name,
            parquet_path,
            transform,
        ));
        id
    }

    pub fn load_points_with_image_size(
        &mut self,
        element: &SpatialDataElement,
        max_points: usize,
        image_size_world: Option<[f32; 2]>,
    ) {
        // Points are rebuilt from the parquet source each time because preparation
        // derives world-space bounds, feature caches, and optional image scaling
        // from the current metadata instead of storing a second normalized copy.
        let Some(root) = self.root().map(|p| p.to_path_buf()) else {
            return;
        };
        let Some(rel) = element.rel_parquet.clone() else {
            return;
        };
        let path = root.join(rel);
        let layer = SpatialPointsLayer::new(
            format!("Points: {}", element.name),
            path,
            element.transform,
            element.feature_key.clone(),
            max_points,
            image_size_world,
        );
        self.points = Some(layer);
    }

    pub fn tick(&mut self) {
        for s in &mut self.shapes {
            s.tick();
        }
        if let Some(p) = self.points.as_mut() {
            p.tick();
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
