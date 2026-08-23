use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

#[derive(Debug, Clone)]
pub struct ControlSegmentationGeoJsonResource {
    pub path: PathBuf,
    pub downsample_factor: f32,
    pub polylines: Arc<Vec<Vec<[f32; 2]>>>,
    pub segment_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentationGeoJsonLoadSpec {
    pub document_generation: u64,
    pub operation_generation: u64,
    pub path: PathBuf,
    pub downsample_factor: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentationGeoJsonModel {
    path: Option<PathBuf>,
    downsample_factor: f32,
    operation_generation: u64,
    resource_generation: u64,
    resource: Option<Arc<ControlSegmentationGeoJsonResource>>,
    pending: bool,
    status: String,
}

impl Default for SegmentationGeoJsonModel {
    fn default() -> Self {
        Self {
            path: None,
            downsample_factor: 1.0,
            operation_generation: 0,
            resource_generation: 0,
            resource: None,
            pending: false,
            status: String::new(),
        }
    }
}

impl SegmentationGeoJsonModel {
    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "path":self.path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "downsample_factor":self.downsample_factor,
            "generation":self.resource_generation,
            "pending":self.pending,
            "loaded":self.resource.is_some(),
            "segment_count":self.resource.as_ref().map_or(0, |resource| resource.segment_count),
            "status":self.status,
        })
    }

    pub(crate) fn resource(&self) -> Option<Arc<ControlSegmentationGeoJsonResource>> {
        self.resource.clone()
    }

    pub(crate) fn prepare_load(
        &mut self,
        document_generation: u64,
        params: &Value,
    ) -> Result<SegmentationGeoJsonLoadSpec, ControlError> {
        let path = params
            .get("path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .or_else(|| self.path.clone())
            .ok_or_else(|| {
                ControlError::invalid_params(
                    "viewer.segmentation_geojson.source.load",
                    "path is required when no source is configured",
                )
            })?;
        let downsample_factor = params
            .get("downsample_factor")
            .and_then(Value::as_f64)
            .unwrap_or(self.downsample_factor as f64) as f32;
        if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
            return Err(ControlError::invalid_params(
                "viewer.segmentation_geojson.source.load",
                "downsample_factor must be finite and greater than zero",
            ));
        }
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.path = Some(path.clone());
        self.downsample_factor = downsample_factor;
        self.pending = true;
        self.status = format!("Loading: {}", path.to_string_lossy());
        Ok(SegmentationGeoJsonLoadSpec {
            document_generation,
            operation_generation: self.operation_generation,
            path,
            downsample_factor,
        })
    }

    pub(crate) fn finish_load(
        &mut self,
        spec: &SegmentationGeoJsonLoadSpec,
        resource: ControlSegmentationGeoJsonResource,
    ) -> bool {
        if self.operation_generation != spec.operation_generation {
            return false;
        }
        self.resource_generation = self.resource_generation.wrapping_add(1).max(1);
        self.path = Some(resource.path.clone());
        self.downsample_factor = resource.downsample_factor;
        self.status = format!("Loaded {} segments.", resource.segment_count);
        self.resource = Some(Arc::new(resource));
        self.pending = false;
        true
    }

    pub(crate) fn fail_load(
        &mut self,
        spec: &SegmentationGeoJsonLoadSpec,
        message: impl Into<String>,
    ) -> bool {
        if self.operation_generation != spec.operation_generation {
            return false;
        }
        self.pending = false;
        self.status = message.into();
        true
    }

    pub(crate) fn clear(&mut self) -> Value {
        let changed = self.path.is_some() || self.resource.is_some() || self.pending;
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.resource_generation = self.resource_generation.wrapping_add(1).max(1);
        self.path = None;
        self.resource = None;
        self.pending = false;
        self.status.clear();
        json!({"changed":changed,"source":self.snapshot()})
    }

    pub(crate) fn ensure_current_document(
        &self,
        current_document_generation: u64,
        spec: &SegmentationGeoJsonLoadSpec,
    ) -> Result<(), ControlError> {
        if current_document_generation != spec.document_generation
            || self.operation_generation != spec.operation_generation
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "segmentation GeoJSON load was superseded",
            ));
        }
        Ok(())
    }
}
