use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::{any::Any, fmt};

use serde_json::{Map, Value, json};

#[derive(Debug, Clone)]
pub struct ControlObjectFeature {
    pub id: String,
    pub bbox_world: [f32; 4],
    pub centroid_world: [f32; 2],
    /// Polygon rings retained in the immutable shared resource for renderer-independent
    /// geometry queries. They are never copied into render projections.
    pub polygons_world: Arc<Vec<Vec<[f32; 2]>>>,
    /// Point geometry used when the object has no usable polygon outline.
    pub point_position_world: Option<[f32; 2]>,
    pub area_px: f32,
    pub perimeter_px: f32,
    pub properties: Map<String, Value>,
}

#[derive(Clone)]
pub struct ControlObjectResource {
    pub source: PathBuf,
    pub downsample_factor: f32,
    pub features: Arc<Vec<ControlObjectFeature>>,
    pub property_names: Arc<Vec<String>>,
    /// Full-source summaries for finite numeric values, computed while the resource is built.
    pub numeric_summaries: Arc<BTreeMap<String, ControlObjectNumericSummary>>,
    /// Optional renderer-native, immutable preload produced by the same worker. The canonical
    /// model never inspects it; a compatible renderer may downcast and install it without
    /// reparsing the source after frames resume.
    pub renderer_payload: Option<Arc<dyn Any + Send + Sync>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlObjectNumericSummary {
    pub minimum: f64,
    pub maximum: f64,
    pub positive_minimum: Option<f64>,
    pub positive_count: usize,
    pub numeric_count: usize,
    pub missing_count: usize,
}

impl fmt::Debug for ControlObjectResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ControlObjectResource")
            .field("source", &self.source)
            .field("downsample_factor", &self.downsample_factor)
            .field("feature_count", &self.features.len())
            .field("property_names", &self.property_names)
            .field("numeric_summaries", &self.numeric_summaries)
            .field("has_renderer_payload", &self.renderer_payload.is_some())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ControlObjectFilterResult {
    /// Canonical, renderer-independent filter model suitable for persistence and projection.
    pub model: Value,
    /// Object indices that satisfy the filter, in stable source order.
    pub matching_indices: Arc<Vec<usize>>,
    pub active: bool,
}

impl ControlObjectResource {
    pub fn descriptor_json(&self, generation: u64) -> Value {
        json!({
            "generation": generation,
            "source": self.source.to_string_lossy(),
            "downsample_factor": self.downsample_factor,
            "object_count": self.features.len(),
            "property_count": self.property_names.len(),
            "properties": self.property_names.as_ref(),
            "numeric_properties": self.numeric_summaries.keys().collect::<Vec<_>>(),
            "model_ready": true,
            "resources_ready": true,
        })
    }

    pub fn property_value(&self, feature_index: usize, property: &str) -> Option<Value> {
        let feature = self.features.get(feature_index)?;
        if property == "id" {
            return Some(Value::String(feature.id.clone()));
        }
        feature.properties.get(property).cloned()
    }

    pub fn build_numeric_summaries(
        features: &[ControlObjectFeature],
        property_names: &[String],
    ) -> Arc<BTreeMap<String, ControlObjectNumericSummary>> {
        let mut summaries = BTreeMap::new();
        for property in property_names
            .iter()
            .filter(|property| property.as_str() != "id")
        {
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            let mut numeric_count = 0usize;
            let mut positive_minimum = f64::INFINITY;
            let mut positive_count = 0usize;
            for feature in features {
                let Some(value) = feature
                    .properties
                    .get(property)
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                else {
                    continue;
                };
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                numeric_count += 1;
                if value > 0.0 {
                    positive_minimum = positive_minimum.min(value);
                    positive_count += 1;
                }
            }
            if numeric_count > 0 {
                summaries.insert(
                    property.clone(),
                    ControlObjectNumericSummary {
                        minimum,
                        maximum,
                        positive_minimum: positive_minimum.is_finite().then_some(positive_minimum),
                        positive_count,
                        numeric_count,
                        missing_count: features.len().saturating_sub(numeric_count),
                    },
                );
            }
        }
        Arc::new(summaries)
    }

    pub fn numeric_summary(&self, property: &str) -> Option<ControlObjectNumericSummary> {
        self.numeric_summaries.get(property).copied()
    }

    pub fn renderer_payload<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.renderer_payload.as_ref()?.downcast_ref::<T>()
    }

    pub fn renderer_payload_identity(&self) -> Option<usize> {
        self.renderer_payload
            .as_ref()
            .map(|payload| Arc::as_ptr(payload) as *const () as usize)
    }
}

pub trait ObjectResourceLoader: Send + Sync + 'static {
    fn load(&self, path: PathBuf, downsample_factor: f32) -> anyhow::Result<ControlObjectResource>;

    fn load_with_options(
        &self,
        path: PathBuf,
        downsample_factor: f32,
        _options: Option<Value>,
    ) -> anyhow::Result<ControlObjectResource> {
        self.load(path, downsample_factor)
    }

    fn evaluate_filter(
        &self,
        _resource: Arc<ControlObjectResource>,
        _model: Value,
    ) -> anyhow::Result<ControlObjectFilterResult> {
        anyhow::bail!("object filter evaluator is unavailable")
    }
}

impl<F> ObjectResourceLoader for F
where
    F: Fn(PathBuf, f32) -> anyhow::Result<ControlObjectResource> + Send + Sync + 'static,
{
    fn load(&self, path: PathBuf, downsample_factor: f32) -> anyhow::Result<ControlObjectResource> {
        self(path, downsample_factor)
    }
}
