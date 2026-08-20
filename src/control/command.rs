use serde::Deserialize;
use serde_json::{Value, json};

use super::ControlError;
use super::registry::{self, MethodDescriptor, RequestShape};

#[derive(Debug, Clone)]
pub struct ControlCommand {
    descriptor: &'static MethodDescriptor,
    params: Value,
    if_revision: Option<u64>,
}

impl ControlCommand {
    pub fn decode(method: &str, params: Value) -> Result<Self, ControlError> {
        let descriptor = registry::method(method).ok_or_else(|| {
            ControlError::new(
                super::ControlErrorKind::MethodNotFound,
                format!("unknown Odon control method '{method}'"),
            )
            .with_data(json!({"method": method}))
        })?;
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
        if if_revision.is_some() && !descriptor.mutates {
            return Err(ControlError::invalid_params(
                method,
                "if_revision is only valid for mutating methods",
            ));
        }
        validate_params(method, descriptor.request_shape, &params)?;
        Ok(Self {
            descriptor,
            params,
            if_revision,
        })
    }

    pub fn method(&self) -> &'static str {
        self.descriptor.name
    }

    pub fn params(&self) -> &Value {
        &self.params
    }

    pub fn mutates(&self) -> bool {
        self.descriptor.mutates
    }

    pub fn if_revision(&self) -> Option<u64> {
        self.if_revision
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetSidePanelsRequest {
    left: Option<bool>,
    right: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetSmoothPixelsRequest {
    smooth: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ChannelSelector {
    Name(String),
    Index(usize),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetVisibleChannelsRequest {
    channels: Vec<ChannelSelector>,
    #[serde(default)]
    mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetCameraRequest {
    center_world_lvl0: Option<[f64; 2]>,
    center_x: Option<f64>,
    center_y: Option<f64>,
    zoom: Option<f64>,
    zoom_screen_per_lvl0_px: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureScreenshotRequest {
    path: Option<String>,
}

fn validate_params(method: &str, shape: RequestShape, params: &Value) -> Result<(), ControlError> {
    let invalid = |error: serde_json::Error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    };
    match shape {
        RequestShape::Empty => {
            if params.as_object().is_some_and(|object| !object.is_empty()) {
                return Err(ControlError::invalid_params(
                    method,
                    "this method does not accept parameters",
                ));
            }
        }
        RequestShape::SetSidePanels => {
            let request: SetSidePanelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.left.is_none() && request.right.is_none() {
                return Err(ControlError::invalid_params(
                    method,
                    "left and/or right is required",
                ));
            }
        }
        RequestShape::SetSmoothPixels => {
            let request: SetSmoothPixelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let _ = request.smooth;
        }
        RequestShape::SetVisibleChannels => {
            let request: SetVisibleChannelsRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request.channels.is_empty() && request.mode.as_deref() != Some("only") {
                return Err(ControlError::invalid_params(
                    method,
                    "channels must not be empty unless mode is 'only'",
                ));
            }
            if let Some(mode) = request.mode.as_deref()
                && !matches!(mode, "only" | "show" | "hide")
            {
                return Err(ControlError::invalid_params(
                    method,
                    "mode must be 'only', 'show', or 'hide'",
                ));
            }
            for channel in request.channels {
                match channel {
                    ChannelSelector::Name(name) if name.trim().is_empty() => {
                        return Err(ControlError::invalid_params(
                            method,
                            "channel names must not be empty",
                        ));
                    }
                    ChannelSelector::Name(_) => {}
                    ChannelSelector::Index(index) => {
                        let _ = index;
                    }
                }
            }
        }
        RequestShape::SetCamera => {
            let request: SetCameraRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            let values = request
                .center_world_lvl0
                .into_iter()
                .flatten()
                .chain(request.center_x)
                .chain(request.center_y);
            if values.into_iter().any(|value| !value.is_finite()) {
                return Err(ControlError::invalid_params(
                    method,
                    "camera coordinates must be finite",
                ));
            }
            for zoom in [request.zoom, request.zoom_screen_per_lvl0_px]
                .into_iter()
                .flatten()
            {
                if !zoom.is_finite() || zoom <= 0.0 {
                    return Err(ControlError::invalid_params(
                        method,
                        "zoom must be finite and greater than zero",
                    ));
                }
            }
        }
        RequestShape::CaptureScreenshot => {
            let request: CaptureScreenshotRequest =
                serde_json::from_value(params.clone()).map_err(invalid)?;
            if request
                .path
                .as_deref()
                .is_some_and(|path| path.trim().is_empty())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "path must not be empty",
                ));
            }
        }
        RequestShape::Object => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_commands_validate_representative_parameters() {
        assert!(ControlCommand::decode("get_camera", json!({})).is_ok());
        assert!(ControlCommand::decode("get_camera", json!({"extra": true})).is_err());
        assert!(ControlCommand::decode("set_side_panels", json!({})).is_err());
        assert!(ControlCommand::decode("set_side_panels", json!({"left": false})).is_ok());
        assert!(
            ControlCommand::decode(
                "set_visible_channels",
                json!({"channels": ["DAPI", 2], "mode": "only"})
            )
            .is_ok()
        );
        assert!(ControlCommand::decode("set_camera", json!({"zoom": 0.0})).is_err());
        let command = ControlCommand::decode("set_camera", json!({"zoom": 2.0, "if_revision": 4}))
            .expect("revision precondition");
        assert_eq!(command.if_revision(), Some(4));
        assert_eq!(command.params(), &json!({"zoom": 2.0}));
        assert!(ControlCommand::decode("get_camera", json!({"if_revision": 4})).is_err());
    }
}
