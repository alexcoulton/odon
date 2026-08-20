use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{ControlError, ControlErrorKind, EventHub};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ownership {
    Session,
    Project,
    User,
}

impl Default for Ownership {
    fn default() -> Self {
        Self::Session
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateSpace {
    pub axes: Vec<String>,
    #[serde(default)]
    pub units: Vec<String>,
    #[serde(default)]
    pub scale: Vec<f64>,
    #[serde(default)]
    pub translation: Vec<f64>,
    #[serde(default)]
    pub reference_layer_id: Option<String>,
}

impl CoordinateSpace {
    fn validate(&self) -> Result<(), ControlError> {
        if self.axes.is_empty() || self.axes.iter().any(|axis| axis.trim().is_empty()) {
            return Err(invalid(
                "coordinate_space.axes must contain non-empty axis names",
            ));
        }
        if self
            .axes
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != self.axes.len()
        {
            return Err(invalid("coordinate_space axis names must be unique"));
        }
        for (name, length) in [
            ("units", self.units.len()),
            ("scale", self.scale.len()),
            ("translation", self.translation.len()),
        ] {
            if length != 0 && length != self.axes.len() {
                return Err(invalid(format!(
                    "coordinate_space.{name} must be empty or match axes length"
                )));
            }
        }
        if self
            .scale
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || self.translation.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "coordinate-space scale must be positive and all transforms must be finite",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterDataResource {
    #[serde(default)]
    pub resource_id: Option<String>,
    pub uri: String,
    pub format: String,
    #[serde(default)]
    pub ownership: Ownership,
    pub coordinate_space: CoordinateSpace,
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DataResourceSnapshot {
    pub resource_id: String,
    pub uri: String,
    pub format: String,
    pub ownership: Ownership,
    pub owner_session_id: String,
    pub coordinate_space: CoordinateSpace,
    pub metadata: BTreeMap<String, Value>,
    pub provenance: BTreeMap<String, Value>,
    pub revision: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AddLayer {
    #[serde(default)]
    pub layer_id: Option<String>,
    pub name: String,
    pub kind: String,
    pub data_resource_id: String,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f64,
    #[serde(default)]
    pub ownership: Ownership,
    #[serde(default)]
    pub style: BTreeMap<String, Value>,
    #[serde(default)]
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerSnapshot {
    pub layer_id: String,
    pub name: String,
    pub kind: String,
    pub data_resource_id: String,
    pub visible: bool,
    pub opacity: f64,
    pub ownership: Ownership,
    pub owner_session_id: String,
    pub style: BTreeMap<String, Value>,
    pub provenance: BTreeMap<String, Value>,
    pub order: usize,
    pub revision: u64,
}

#[derive(Debug, Default)]
struct State {
    resources: HashMap<String, DataResourceSnapshot>,
    layers: Vec<LayerSnapshot>,
}

#[derive(Debug)]
pub struct ResourceRegistry {
    state: Mutex<State>,
    events: Arc<EventHub>,
}

impl ResourceRegistry {
    pub fn shared(events: Arc<EventHub>) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State::default()),
            events,
        })
    }

    pub fn register_resource(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<DataResourceSnapshot, ControlError> {
        let request: RegisterDataResource = serde_json::from_value(params)
            .map_err(|error| invalid(format!("invalid data resource: {error}")))?;
        request.coordinate_space.validate()?;
        if request.uri.trim().is_empty() {
            return Err(invalid("resource URI must not be empty"));
        }
        if !matches!(
            request.format.as_str(),
            "ome-zarr" | "zarr" | "arrow-ipc" | "parquet" | "geoparquet" | "geojson"
        ) {
            return Err(invalid("unsupported data resource format"));
        }
        let resource_id = allocate_id("resource", request.resource_id)?;
        let mut state = self.state.lock().expect("resource registry poisoned");
        if state.resources.contains_key(&resource_id) {
            return Err(conflict("resource_id", &resource_id));
        }
        let revision = self.events.next_revision();
        let snapshot = DataResourceSnapshot {
            resource_id: resource_id.clone(),
            uri: request.uri,
            format: request.format,
            ownership: request.ownership,
            owner_session_id: session_id.to_string(),
            coordinate_space: request.coordinate_space,
            metadata: request.metadata,
            provenance: request.provenance,
            revision,
        };
        state
            .resources
            .insert(resource_id.clone(), snapshot.clone());
        drop(state);
        self.publish(
            "data.resources.added",
            &resource_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn list_resources(&self) -> Vec<DataResourceSnapshot> {
        let mut resources = self
            .state
            .lock()
            .expect("resource registry poisoned")
            .resources
            .values()
            .cloned()
            .collect::<Vec<_>>();
        resources.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
        resources
    }

    pub fn get_resource(&self, id: &str) -> Result<DataResourceSnapshot, ControlError> {
        self.state
            .lock()
            .expect("resource registry poisoned")
            .resources
            .get(id)
            .cloned()
            .ok_or_else(|| not_found("data resource", id))
    }

    pub fn remove_resource(&self, id: &str, session_id: &str) -> Result<(), ControlError> {
        let mut state = self.state.lock().expect("resource registry poisoned");
        if state
            .layers
            .iter()
            .any(|layer| layer.data_resource_id == id)
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "a data resource referenced by a layer cannot be removed",
            )
            .with_data(json!({"resource_id": id})));
        }
        let resource = state
            .resources
            .get(id)
            .ok_or_else(|| not_found("data resource", id))?;
        ensure_owner(resource.ownership, &resource.owner_session_id, session_id)?;
        state.resources.remove(id);
        let revision = self.events.next_revision();
        drop(state);
        self.events.publish(
            "data.resources.removed",
            id,
            revision,
            json!({"resource_id": id}),
            Some(session_id.to_string()),
            None,
        );
        Ok(())
    }

    pub fn add_layer(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<LayerSnapshot, ControlError> {
        let request: AddLayer = serde_json::from_value(params)
            .map_err(|error| invalid(format!("invalid layer: {error}")))?;
        if request.name.trim().is_empty() {
            return Err(invalid("layer name must not be empty"));
        }
        if !matches!(
            request.kind.as_str(),
            "image" | "labels" | "objects" | "points" | "shapes" | "mask" | "annotations"
        ) {
            return Err(invalid("unsupported layer kind"));
        }
        if !request.opacity.is_finite() || !(0.0..=1.0).contains(&request.opacity) {
            return Err(invalid("layer opacity must be between 0 and 1"));
        }
        let layer_id = allocate_id("layer", request.layer_id)?;
        let mut state = self.state.lock().expect("resource registry poisoned");
        let resource = state
            .resources
            .get(&request.data_resource_id)
            .ok_or_else(|| not_found("data resource", &request.data_resource_id))?;
        ensure_resource_reference(resource, request.ownership, session_id)?;
        if state.layers.iter().any(|layer| layer.layer_id == layer_id) {
            return Err(conflict("layer_id", &layer_id));
        }
        let revision = self.events.next_revision();
        let snapshot = LayerSnapshot {
            layer_id: layer_id.clone(),
            name: request.name,
            kind: request.kind,
            data_resource_id: request.data_resource_id,
            visible: request.visible,
            opacity: request.opacity,
            ownership: request.ownership,
            owner_session_id: session_id.to_string(),
            style: request.style,
            provenance: request.provenance,
            order: state.layers.len(),
            revision,
        };
        state.layers.push(snapshot.clone());
        drop(state);
        self.publish(
            "viewer.layers.added",
            &layer_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn list_layers(&self) -> Vec<LayerSnapshot> {
        self.state
            .lock()
            .expect("resource registry poisoned")
            .layers
            .clone()
    }

    pub fn project_manifest(&self) -> (Vec<Value>, Vec<Value>) {
        let state = self.state.lock().expect("resource registry poisoned");
        let resources = state
            .resources
            .values()
            .filter(|resource| resource.ownership == Ownership::Project)
            .filter_map(|resource| serde_json::to_value(resource).ok())
            .map(strip_runtime_fields)
            .collect();
        let layers = state
            .layers
            .iter()
            .filter(|layer| layer.ownership == Ownership::Project)
            .filter_map(|layer| serde_json::to_value(layer).ok())
            .map(strip_runtime_fields)
            .collect();
        (resources, layers)
    }

    pub fn replace_project_manifest(
        &self,
        resources: &[Value],
        layers: &[Value],
    ) -> Result<(), ControlError> {
        let validation = ResourceRegistry::shared(EventHub::shared());
        for resource in resources {
            validation
                .register_resource(project_owned_request(resource, true)?, "project:persisted")?;
        }
        for layer in layers {
            validation.add_layer(project_owned_request(layer, false)?, "project:persisted")?;
        }
        let validated_resources = validation.list_resources();
        let validated_layers = validation.list_layers();

        let mut state = self.state.lock().expect("resource registry poisoned");
        if validated_resources.iter().any(|resource| {
            state
                .resources
                .get(&resource.resource_id)
                .is_some_and(|existing| existing.ownership != Ownership::Project)
        }) || validated_layers.iter().any(|layer| {
            state.layers.iter().any(|existing| {
                existing.layer_id == layer.layer_id && existing.ownership != Ownership::Project
            })
        }) {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                "project control resources conflict with an active session resource ID",
            ));
        }
        let old_project_resources = state
            .resources
            .values()
            .filter(|resource| resource.ownership == Ownership::Project)
            .map(|resource| resource.resource_id.clone())
            .collect::<HashSet<_>>();
        state.layers.retain(|layer| {
            layer.ownership != Ownership::Project
                && !old_project_resources.contains(&layer.data_resource_id)
        });
        state
            .resources
            .retain(|_, resource| resource.ownership != Ownership::Project);
        let revision = self.events.next_revision();
        for mut resource in validated_resources {
            resource.revision = revision;
            resource.owner_session_id = "project:persisted".into();
            state
                .resources
                .insert(resource.resource_id.clone(), resource);
        }
        let first_order = state.layers.len();
        for (index, mut layer) in validated_layers.into_iter().enumerate() {
            layer.revision = revision;
            layer.owner_session_id = "project:persisted".into();
            layer.order = first_order + index;
            state.layers.push(layer);
        }
        drop(state);
        self.events.publish(
            "project.control_resources.loaded",
            "project:active",
            revision,
            json!({
                "resource_count": resources.len(),
                "layer_count": layers.len(),
            }),
            None,
            None,
        );
        Ok(())
    }

    pub fn get_layer(&self, id: &str) -> Result<LayerSnapshot, ControlError> {
        self.state
            .lock()
            .expect("resource registry poisoned")
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .cloned()
            .ok_or_else(|| not_found("layer", id))
    }

    pub fn update_layer(
        &self,
        id: &str,
        params: &Value,
        session_id: &str,
    ) -> Result<LayerSnapshot, ControlError> {
        let object = params
            .as_object()
            .ok_or_else(|| invalid("layer update must be an object"))?;
        if let Some(field) = object.keys().find(|field| {
            !matches!(
                field.as_str(),
                "visible" | "opacity" | "name" | "style" | "data_resource_id" | "if_revision"
            )
        }) {
            return Err(invalid(format!("unknown layer update field '{field}'")));
        }
        let mut state = self.state.lock().expect("resource registry poisoned");
        let index = state
            .layers
            .iter()
            .position(|layer| layer.layer_id == id)
            .ok_or_else(|| not_found("layer", id))?;
        if let Some(resource_id) = params.get("data_resource_id").and_then(Value::as_str) {
            let resource = state
                .resources
                .get(resource_id)
                .ok_or_else(|| not_found("data resource", resource_id))?;
            ensure_resource_reference(resource, state.layers[index].ownership, session_id)?;
        }
        let mut layer = state.layers[index].clone();
        ensure_owner(layer.ownership, &layer.owner_session_id, session_id)?;
        if let Some(expected) = params.get("if_revision") {
            let expected = expected
                .as_u64()
                .ok_or_else(|| invalid("if_revision must be an unsigned integer"))?;
            if expected != layer.revision {
                return Err(ControlError::new(
                    ControlErrorKind::Conflict,
                    "layer revision conflict",
                )
                .with_data(json!({
                    "layer_id": id,
                    "expected_revision": expected,
                    "current_revision": layer.revision,
                })));
            }
        }
        if let Some(value) = params.get("visible").and_then(Value::as_bool) {
            layer.visible = value;
        }
        if let Some(value) = params.get("opacity") {
            let value = value
                .as_f64()
                .ok_or_else(|| invalid("opacity must be a number"))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(invalid("opacity must be between 0 and 1"));
            }
            layer.opacity = value;
        }
        if let Some(value) = params.get("name").and_then(Value::as_str) {
            if value.trim().is_empty() {
                return Err(invalid("layer name must not be empty"));
            }
            layer.name = value.to_string();
        }
        if let Some(value) = params.get("style") {
            layer.style = serde_json::from_value(value.clone())
                .map_err(|error| invalid(format!("style must be an object: {error}")))?;
        }
        if let Some(value) = params.get("data_resource_id").and_then(Value::as_str) {
            layer.data_resource_id = value.to_string();
        }
        let revision = self.events.next_revision();
        layer.revision = revision;
        let snapshot = layer.clone();
        state.layers[index] = layer;
        drop(state);
        self.publish("viewer.layers.changed", id, revision, &snapshot, session_id);
        Ok(snapshot)
    }

    pub fn remove_layer(&self, id: &str, session_id: &str) -> Result<(), ControlError> {
        let mut state = self.state.lock().expect("resource registry poisoned");
        let index = state
            .layers
            .iter()
            .position(|layer| layer.layer_id == id)
            .ok_or_else(|| not_found("layer", id))?;
        let layer = &state.layers[index];
        ensure_owner(layer.ownership, &layer.owner_session_id, session_id)?;
        state.layers.remove(index);
        for (order, layer) in state.layers.iter_mut().enumerate() {
            layer.order = order;
        }
        let revision = self.events.next_revision();
        drop(state);
        self.events.publish(
            "viewer.layers.removed",
            id,
            revision,
            json!({"layer_id": id}),
            Some(session_id.to_string()),
            None,
        );
        Ok(())
    }

    pub fn reorder_layers(
        &self,
        order: &[String],
        session_id: &str,
    ) -> Result<Vec<LayerSnapshot>, ControlError> {
        let mut state = self.state.lock().expect("resource registry poisoned");
        if order.len() != state.layers.len()
            || state
                .layers
                .iter()
                .any(|layer| !order.contains(&layer.layer_id))
        {
            return Err(invalid("order must contain every layer ID exactly once"));
        }
        for layer in &state.layers {
            ensure_owner(layer.ownership, &layer.owner_session_id, session_id)?;
        }
        let mut by_id = state
            .layers
            .drain(..)
            .map(|layer| (layer.layer_id.clone(), layer))
            .collect::<HashMap<_, _>>();
        state.layers = order
            .iter()
            .enumerate()
            .map(|(index, id)| {
                let mut layer = by_id.remove(id).expect("validated layer order");
                layer.order = index;
                layer
            })
            .collect();
        let revision = self.events.next_revision();
        for layer in &mut state.layers {
            layer.revision = revision;
        }
        let snapshot = state.layers.clone();
        drop(state);
        self.events.publish(
            "viewer.layers.reordered",
            "viewer:active",
            revision,
            json!({"order": order}),
            Some(session_id.to_string()),
            None,
        );
        Ok(snapshot)
    }

    pub fn cleanup_session(&self, session_id: &str) {
        let mut state = self.state.lock().expect("resource registry poisoned");
        let removed_layers = state
            .layers
            .iter()
            .filter(|layer| {
                layer.ownership == Ownership::Session && layer.owner_session_id == session_id
            })
            .map(|layer| layer.layer_id.clone())
            .collect::<Vec<_>>();
        state.layers.retain(|layer| {
            layer.ownership != Ownership::Session || layer.owner_session_id != session_id
        });
        let removed_resources = state
            .resources
            .values()
            .filter(|resource| {
                resource.ownership == Ownership::Session && resource.owner_session_id == session_id
            })
            .map(|resource| resource.resource_id.clone())
            .collect::<Vec<_>>();
        state.resources.retain(|_, resource| {
            resource.ownership != Ownership::Session || resource.owner_session_id != session_id
        });
        for (order, layer) in state.layers.iter_mut().enumerate() {
            layer.order = order;
        }
        drop(state);
        if removed_layers.is_empty() && removed_resources.is_empty() {
            return;
        }
        let revision = self.events.next_revision();
        for id in removed_layers {
            self.events.publish(
                "viewer.layers.removed",
                &id,
                revision,
                json!({"layer_id": id, "reason": "session_disconnected"}),
                Some(session_id.to_string()),
                None,
            );
        }
        for id in removed_resources {
            self.events.publish(
                "data.resources.removed",
                &id,
                revision,
                json!({"resource_id": id, "reason": "session_disconnected"}),
                Some(session_id.to_string()),
                None,
            );
        }
    }

    fn publish(
        &self,
        event: &str,
        source: &str,
        revision: u64,
        snapshot: &impl Serialize,
        session_id: &str,
    ) {
        self.events.publish(
            event,
            source,
            revision,
            serde_json::to_value(snapshot).unwrap_or_else(|_| json!({})),
            Some(session_id.to_string()),
            None,
        );
    }
}

fn allocate_id(prefix: &str, requested: Option<String>) -> Result<String, ControlError> {
    if let Some(requested) = requested {
        if requested.trim().is_empty() || requested.chars().any(char::is_whitespace) {
            return Err(invalid(format!(
                "{prefix}_id must not be empty or contain whitespace"
            )));
        }
        return Ok(requested);
    }
    crate::control::discovery::random_uuid_like()
        .map(|id| format!("{prefix}:{id}"))
        .map_err(|error| ControlError::new(ControlErrorKind::Internal, error.to_string()))
}

fn strip_runtime_fields(mut value: Value) -> Value {
    if let Some(object) = value.as_object_mut() {
        object.remove("owner_session_id");
        object.remove("revision");
        object.remove("order");
    }
    value
}

fn project_owned_request(value: &Value, resource: bool) -> Result<Value, ControlError> {
    let mut value = value.clone();
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("project control descriptors must be objects"))?;
    object.insert("ownership".into(), json!("project"));
    object.remove("owner_session_id");
    object.remove("revision");
    if !resource {
        object.remove("order");
    }
    Ok(value)
}

fn ensure_owner(ownership: Ownership, owner: &str, session: &str) -> Result<(), ControlError> {
    if ownership == Ownership::Session && owner != session {
        Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "session-owned resources can only be changed by their owner",
        ))
    } else {
        Ok(())
    }
}

fn ensure_resource_reference(
    resource: &DataResourceSnapshot,
    layer_ownership: Ownership,
    session_id: &str,
) -> Result<(), ControlError> {
    if resource.ownership == Ownership::Session && resource.owner_session_id != session_id {
        return Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "a session-owned data resource is private to its owner",
        )
        .with_data(json!({"resource_id": resource.resource_id})));
    }
    if layer_ownership != Ownership::Session && resource.ownership == Ownership::Session {
        return Err(ControlError::new(
            ControlErrorKind::Conflict,
            "a durable layer cannot reference a session-owned data resource",
        )
        .with_data(json!({
            "resource_id": resource.resource_id,
            "layer_ownership": layer_ownership,
        })));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn not_found(kind: &str, id: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("{kind} '{id}' was not found"),
    )
    .with_data(json!({"resource_id": id}))
}

fn conflict(field: &str, id: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::Conflict,
        format!("{field} '{id}' already exists"),
    )
    .with_data(json!({field: id}))
}

fn default_true() -> bool {
    true
}
fn default_opacity() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource() -> Value {
        json!({
            "resource_id": "resource:labels",
            "uri": "file:///tmp/labels.zarr",
            "format": "ome-zarr",
            "coordinate_space": {
                "axes": ["y", "x"], "units": ["micrometer", "micrometer"],
                "scale": [0.5, 0.5], "translation": [0.0, 0.0]
            }
        })
    }

    #[test]
    fn resources_and_layers_validate_ownership_and_lifecycle() {
        let registry = ResourceRegistry::shared(EventHub::shared());
        registry
            .register_resource(resource(), "one")
            .expect("resource");
        let layer = registry
            .add_layer(
                json!({
                    "layer_id": "layer:labels", "name": "Labels", "kind": "labels",
                    "data_resource_id": "resource:labels"
                }),
                "one",
            )
            .expect("layer");
        assert_eq!(layer.opacity, 1.0);
        assert!(
            registry
                .update_layer("layer:labels", &json!({"opacity": 2}), "one")
                .is_err()
        );
        assert!(registry.remove_layer("layer:labels", "two").is_err());
        registry.cleanup_session("one");
        assert!(registry.list_layers().is_empty());
        assert!(registry.list_resources().is_empty());
        assert!(registry.events.revision() > 0);
    }

    #[test]
    fn durable_layers_cannot_reference_temporary_or_foreign_resources() {
        let registry = ResourceRegistry::shared(EventHub::shared());
        registry
            .register_resource(resource(), "one")
            .expect("resource");
        assert!(
            registry
                .add_layer(
                    json!({
                        "name": "Durable", "kind": "labels",
                        "data_resource_id": "resource:labels", "ownership": "project"
                    }),
                    "one",
                )
                .is_err()
        );
        assert!(
            registry
                .add_layer(
                    json!({
                        "name": "Foreign", "kind": "labels",
                        "data_resource_id": "resource:labels"
                    }),
                    "two",
                )
                .is_err()
        );
    }

    #[test]
    fn project_owned_resources_roundtrip_through_project_manifest() {
        let source = ResourceRegistry::shared(EventHub::shared());
        let mut project_resource = resource();
        project_resource["ownership"] = json!("project");
        source
            .register_resource(project_resource, "one")
            .expect("project resource");
        source
            .add_layer(
                json!({
                    "layer_id": "layer:labels", "name": "Labels", "kind": "labels",
                    "data_resource_id": "resource:labels", "ownership": "project"
                }),
                "one",
            )
            .expect("project layer");
        let (resources, layers) = source.project_manifest();
        assert_eq!(resources.len(), 1);
        assert!(resources[0].get("revision").is_none());

        let restored = ResourceRegistry::shared(EventHub::shared());
        restored
            .replace_project_manifest(&resources, &layers)
            .expect("restore project manifest");
        assert_eq!(restored.list_resources().len(), 1);
        assert_eq!(restored.list_layers()[0].ownership, Ownership::Project);
    }
}
