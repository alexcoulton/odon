use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::{any::Any, fmt};

use eframe::egui;
use serde_json::{Map, Value, json};

/// Immutable, column-oriented property access shared by the renderer and control model.
///
/// Implementations retain typed columns and materialize JSON only for values that are actually
/// requested. This avoids a `serde_json::Map` allocation for every GeoParquet row.
pub trait ControlObjectPropertySource: Send + Sync + fmt::Debug + 'static {
    fn value_json_at(&self, object_index: usize, property: &str) -> Option<Value>;

    fn f64_at(&self, object_index: usize, property: &str) -> Option<f64> {
        self.value_json_at(object_index, property)?.as_f64()
    }

    fn label_at(&self, object_index: usize, property: &str) -> Option<String> {
        match self.value_json_at(object_index, property)? {
            Value::String(value) => Some(value),
            Value::Number(value) => Some(value.to_string()),
            Value::Bool(value) => Some(value.to_string()),
            Value::Null | Value::Array(_) | Value::Object(_) => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct EmptyControlObjectPropertySource;

impl ControlObjectPropertySource for EmptyControlObjectPropertySource {
    fn value_json_at(&self, _object_index: usize, _property: &str) -> Option<Value> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct ControlObjectF32Column {
    values: Arc<Vec<f32>>,
    validity: Arc<Vec<u64>>,
}

impl ControlObjectF32Column {
    pub fn from_optional_values(values: impl IntoIterator<Item = Option<f32>>) -> Self {
        let values = values.into_iter();
        let (lower_bound, _) = values.size_hint();
        let mut dense = Vec::with_capacity(lower_bound);
        let mut validity = Vec::with_capacity(lower_bound.div_ceil(u64::BITS as usize));
        for value in values {
            let index = dense.len();
            if index % u64::BITS as usize == 0 {
                validity.push(0);
            }
            if let Some(value) = value {
                validity[index / u64::BITS as usize] |= 1u64 << (index % u64::BITS as usize);
                dense.push(value);
            } else {
                dense.push(0.0);
            }
        }
        Self {
            values: Arc::new(dense),
            validity: Arc::new(validity),
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        let value = *self.values.get(index)?;
        let word = self.validity.get(index / u64::BITS as usize)?;
        ((*word & (1u64 << (index % u64::BITS as usize))) != 0).then_some(value)
    }
}

/// A compact immutable overlay used for generated numeric properties such as measurements.
#[derive(Debug)]
pub struct ControlObjectPropertyOverlay {
    base: Arc<dyn ControlObjectPropertySource>,
    f32_columns: BTreeMap<String, ControlObjectF32Column>,
}

impl ControlObjectPropertyOverlay {
    pub fn new(base: Arc<dyn ControlObjectPropertySource>) -> Self {
        Self {
            base,
            f32_columns: BTreeMap::new(),
        }
    }

    pub fn insert_f32(&mut self, name: String, column: ControlObjectF32Column) {
        self.f32_columns.insert(name, column);
    }
}

impl ControlObjectPropertySource for ControlObjectPropertyOverlay {
    fn value_json_at(&self, object_index: usize, property: &str) -> Option<Value> {
        if let Some(column) = self.f32_columns.get(property) {
            return column
                .get(object_index)
                .map(f64::from)
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number);
        }
        self.base.value_json_at(object_index, property)
    }

    fn f64_at(&self, object_index: usize, property: &str) -> Option<f64> {
        self.f32_columns
            .get(property)
            .and_then(|column| column.get(object_index))
            .map(f64::from)
            .or_else(|| self.base.f64_at(object_index, property))
    }

    fn label_at(&self, object_index: usize, property: &str) -> Option<String> {
        self.f32_columns
            .get(property)
            .and_then(|column| column.get(object_index))
            .map(|value| value.to_string())
            .or_else(|| self.base.label_at(object_index, property))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetainedMemoryComponent {
    pub payload_capacity_bytes: u64,
    pub container_bytes: u64,
    pub allocation_count: u64,
    /// Retained heap allocations whose payload size is hidden behind a standard-library or
    /// third-party container API. These are counted, but deliberately not invented as bytes.
    pub opaque_allocation_count: u64,
    pub logical_element_count: u64,
}

impl RetainedMemoryComponent {
    pub fn retained_bytes(self) -> u64 {
        self.payload_capacity_bytes
            .saturating_add(self.container_bytes)
    }

    pub fn merge(&mut self, other: Self) {
        self.payload_capacity_bytes = self
            .payload_capacity_bytes
            .saturating_add(other.payload_capacity_bytes);
        self.container_bytes = self.container_bytes.saturating_add(other.container_bytes);
        self.allocation_count = self.allocation_count.saturating_add(other.allocation_count);
        self.opaque_allocation_count = self
            .opaque_allocation_count
            .saturating_add(other.opaque_allocation_count);
        self.logical_element_count = self
            .logical_element_count
            .saturating_add(other.logical_element_count);
    }

    fn snapshot(self) -> Value {
        json!({
            "retained_bytes": self.retained_bytes(),
            "payload_capacity_bytes": self.payload_capacity_bytes,
            "container_bytes": self.container_bytes,
            "allocation_count": self.allocation_count,
            "opaque_allocation_count": self.opaque_allocation_count,
            "logical_element_count": self.logical_element_count,
        })
    }
}

/// Capacity-based accounting for retained CPU object data.
///
/// This deliberately excludes allocator size-class rounding, allocator metadata and
/// fragmentation. Those are process-level costs and must be compared with the operating
/// system's heap report. Opaque retained allocations are counted so an allocator report can be
/// correlated without presenting an implementation-dependent estimate as an exact byte total.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ControlObjectMemoryDiagnostics {
    pub components: BTreeMap<String, RetainedMemoryComponent>,
}

impl ControlObjectMemoryDiagnostics {
    pub fn add_component(&mut self, name: impl Into<String>, component: RetainedMemoryComponent) {
        self.components
            .entry(name.into())
            .or_default()
            .merge(component);
    }

    pub fn merge(&mut self, other: &Self) {
        for (name, component) in &other.components {
            self.add_component(name.clone(), *component);
        }
    }

    pub fn total(&self) -> RetainedMemoryComponent {
        self.components.values().copied().fold(
            RetainedMemoryComponent::default(),
            |mut total, component| {
                total.merge(component);
                total
            },
        )
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "measurement": "retained_cpu_object_capacity",
            "total": self.total().snapshot(),
            "components": self.components.iter().map(|(name, component)| {
                (name.clone(), component.snapshot())
            }).collect::<Map<String, Value>>(),
            "excludes": [
                "allocator_size_class_rounding",
                "allocator_metadata",
                "allocator_fragmentation",
                "temporary_worker_allocations",
                "serde_json_btree_node_payload_bytes",
                "columnar_property_store",
                "analysis_value_caches",
                "gpu_buffers_and_textures",
            ],
        })
    }
}

#[derive(Debug, Clone)]
pub struct ControlObjectFeature {
    pub id: String,
    pub polygons_world: Vec<Vec<egui::Pos2>>,
    /// Point geometry used when the object has no usable polygon outline.
    pub point_position_world: Option<egui::Pos2>,
    pub bbox_world: egui::Rect,
    pub area_px: f32,
    pub perimeter_px: f32,
    pub centroid_world: egui::Pos2,
    /// Row-oriented fallback retained for GeoJSON and CSV. GeoParquet keeps this empty.
    pub inline_properties: Map<String, Value>,
    pub source_row_index: Option<usize>,
}

#[derive(Clone)]
pub struct ControlObjectResource {
    pub source: PathBuf,
    pub downsample_factor: f32,
    pub features: Arc<Vec<ControlObjectFeature>>,
    pub property_names: Arc<Vec<String>>,
    /// Shared typed columns. Row-shaped JSON is materialized only by bounded API requests.
    pub property_source: Arc<dyn ControlObjectPropertySource>,
    /// Full-source summaries for finite numeric values, computed while the resource is built.
    pub numeric_summaries: Arc<BTreeMap<String, ControlObjectNumericSummary>>,
    /// Capacity-based retained CPU object-data accounting captured when the resource is built.
    pub memory_diagnostics: Arc<ControlObjectMemoryDiagnostics>,
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
            .field("property_source", &self.property_source)
            .field("numeric_summaries", &self.numeric_summaries)
            .field("memory_diagnostics", &self.memory_diagnostics)
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
            "cpu_geometry_memory": self.memory_diagnostics.snapshot(),
            "model_ready": true,
            "resources_ready": true,
        })
    }

    pub fn property_value(&self, feature_index: usize, property: &str) -> Option<Value> {
        let feature = self.features.get(feature_index)?;
        if property == "id" {
            return Some(Value::String(feature.id.clone()));
        }
        self.property_source
            .value_json_at(feature_index, property)
            .or_else(|| feature.inline_properties.get(property).cloned())
    }

    pub fn property_f64(&self, feature_index: usize, property: &str) -> Option<f64> {
        self.property_source
            .f64_at(feature_index, property)
            .or_else(|| {
                self.features
                    .get(feature_index)?
                    .inline_properties
                    .get(property)?
                    .as_f64()
            })
    }

    pub fn property_label(&self, feature_index: usize, property: &str) -> Option<String> {
        if property == "id" {
            return self
                .features
                .get(feature_index)
                .map(|feature| feature.id.clone());
        }
        self.property_source
            .label_at(feature_index, property)
            .or_else(|| {
                let value = self
                    .features
                    .get(feature_index)?
                    .inline_properties
                    .get(property)?;
                match value {
                    Value::String(value) => Some(value.clone()),
                    Value::Number(value) => Some(value.to_string()),
                    Value::Bool(value) => Some(value.to_string()),
                    Value::Null | Value::Array(_) | Value::Object(_) => None,
                }
            })
    }

    pub fn materialize_properties(&self, feature_index: usize) -> Option<Map<String, Value>> {
        let feature = self.features.get(feature_index)?;
        let mut properties = feature.inline_properties.clone();
        for property in self.property_names.iter() {
            if property == "id" {
                continue;
            }
            if let Some(value) = self.property_source.value_json_at(feature_index, property)
                && !value.is_null()
            {
                properties.insert(property.clone(), value);
            }
        }
        properties.insert("id".to_string(), Value::String(feature.id.clone()));
        Some(properties)
    }

    pub fn build_numeric_summaries(
        features: &[ControlObjectFeature],
        property_names: &[String],
        property_source: &dyn ControlObjectPropertySource,
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
            for (feature_index, feature) in features.iter().enumerate() {
                let Some(value) = property_source
                    .f64_at(feature_index, property)
                    .or_else(|| {
                        feature
                            .inline_properties
                            .get(property)
                            .and_then(Value::as_f64)
                    })
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
