use serde_json::{Value, json};
use std::time::{Duration, Instant};

use super::ControlError;
use super::registry::{self, MethodDescriptor, RequestShape};

mod requests;
#[cfg(test)]
mod tests;

use requests::validate_params;

#[derive(Debug, Clone)]
pub struct ControlCommand {
    descriptor: Option<&'static MethodDescriptor>,
    service_method: Option<ServiceMethod>,
    params: Value,
    if_revision: Option<u64>,
    decoded_at: Instant,
}

#[derive(Debug, Clone, Copy)]
struct ServiceMethod {
    name: &'static str,
    mutates: bool,
    starts_task: bool,
}

impl ControlCommand {
    pub fn decode(method: &str, params: Value) -> Result<Self, ControlError> {
        let descriptor = registry::method(method);
        let service_method = descriptor
            .is_none()
            .then(|| {
                registry::PROTOCOL_METHODS
                    .iter()
                    .find(|entry| entry.0 == method && is_actor_service_method(method))
                    .map(|entry| ServiceMethod {
                        name: entry.0,
                        mutates: entry.3,
                        starts_task: entry.4,
                    })
            })
            .flatten();
        if descriptor.is_none() && service_method.is_none() {
            return Err(ControlError::new(
                super::ControlErrorKind::MethodNotFound,
                format!("unknown Odon control method '{method}'"),
            )
            .with_data(json!({"method": method})));
        }
        let mut params = if params.is_null() { json!({}) } else { params };
        if !params.is_object() {
            return Err(ControlError::invalid_params(
                method,
                "params must be an object",
            ));
        }
        let if_revision = params
            .as_object_mut()
            .and_then(|object| object.remove("if_revision"))
            .map(|value| {
                value.as_u64().ok_or_else(|| {
                    ControlError::invalid_params(method, "if_revision must be an unsigned integer")
                })
            })
            .transpose()?;
        let mutates = descriptor
            .map(|descriptor| descriptor.mutates)
            .or_else(|| service_method.map(|descriptor| descriptor.mutates))
            .expect("a command descriptor was resolved");
        if if_revision.is_some() && !mutates {
            return Err(ControlError::invalid_params(
                method,
                "if_revision is only valid for mutating methods",
            ));
        }
        validate_params(
            method,
            descriptor.map_or(RequestShape::Object, |descriptor| descriptor.request_shape),
            &params,
        )?;
        Ok(Self {
            descriptor,
            service_method,
            params,
            if_revision,
            decoded_at: Instant::now(),
        })
    }

    pub fn method(&self) -> &'static str {
        self.descriptor
            .map(|descriptor| descriptor.name)
            .or_else(|| self.service_method.map(|descriptor| descriptor.name))
            .expect("control command retains its descriptor")
    }

    pub fn params(&self) -> &Value {
        &self.params
    }

    pub fn mutates(&self) -> bool {
        self.descriptor
            .map(|descriptor| descriptor.mutates)
            .or_else(|| self.service_method.map(|descriptor| descriptor.mutates))
            .expect("control command retains its descriptor")
    }

    pub fn starts_task(&self) -> bool {
        self.descriptor
            .map(|descriptor| descriptor.starts_task)
            .or_else(|| self.service_method.map(|descriptor| descriptor.starts_task))
            .expect("control command retains its descriptor")
    }

    pub fn event_name(&self) -> Option<&'static str> {
        self.descriptor.and_then(|descriptor| descriptor.event)
    }

    pub fn available_in(&self) -> &'static [&'static str] {
        self.descriptor.map_or(
            &["project", "single", "mosaic", "transition"],
            |descriptor| descriptor.available_in,
        )
    }

    pub fn if_revision(&self) -> Option<u64> {
        self.if_revision
    }

    pub fn queue_age(&self) -> Duration {
        self.decoded_at.elapsed()
    }
}

fn is_actor_service_method(method: &str) -> bool {
    matches!(
        method,
        "data.resources.register"
            | "data.resources.list"
            | "data.resources.get"
            | "data.resources.remove"
            | "viewer.layers.add"
            | "viewer.layers.list"
            | "viewer.layers.get"
            | "viewer.layers.update"
            | "viewer.layers.remove"
            | "viewer.layers.reorder"
    )
}
