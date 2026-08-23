use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

use super::*;

pub(super) fn required_id(params: &Value) -> Result<u64, ControlError> {
    params
        .get("id")
        .or_else(|| params.get("layer_id"))
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("mask layer id is required"))
}

pub(super) fn required_index(params: &Value) -> Result<usize, ControlError> {
    params
        .get("index")
        .or_else(|| params.get("polygon_index"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("polygon index is required"))
}

pub(super) fn optional_nonempty_string(
    params: &Value,
    name: &str,
) -> Result<Option<String>, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| invalid(format!("mask layer {name} must not be empty")))
        })
        .transpose()
}

pub(super) fn optional_bool(params: &Value, name: &str) -> Result<Option<bool>, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid(format!("{name} must be a boolean")))
        })
        .transpose()
}

pub(super) fn optional_bounded_f32(
    params: &Value,
    name: &str,
    min: f32,
    max: f32,
    inclusive_min: bool,
) -> Result<Option<f32>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let value = value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| value as f32)
                .ok_or_else(|| invalid(format!("{name} must be finite")))?;
            let lower = if inclusive_min {
                value >= min
            } else {
                value > min
            };
            if !lower || value > max {
                return Err(invalid(format!("{name} is outside its allowed range")));
            }
            Ok(value)
        })
        .transpose()
}

pub(super) fn optional_rgb(params: &Value, name: &str) -> Result<Option<[u8; 3]>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let values = value
                .as_array()
                .filter(|values| values.len() == 3)
                .ok_or_else(|| invalid(format!("{name} must contain three integers")))?;
            Ok([
                u8::try_from(
                    values[0]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid red value"))?,
                )
                .map_err(|_| invalid("invalid red value"))?,
                u8::try_from(
                    values[1]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid green value"))?,
                )
                .map_err(|_| invalid("invalid green value"))?,
                u8::try_from(
                    values[2]
                        .as_u64()
                        .ok_or_else(|| invalid("invalid blue value"))?,
                )
                .map_err(|_| invalid("invalid blue value"))?,
            ])
        })
        .transpose()
}

pub(super) fn optional_pair(params: &Value, name: &str) -> Result<Option<[f32; 2]>, ControlError> {
    params
        .get(name)
        .map(|value| {
            let values = value
                .as_array()
                .filter(|values| values.len() == 2)
                .ok_or_else(|| invalid(format!("{name} must contain two finite numbers")))?;
            let x = values[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("{name}[0] must be finite")))?
                as f32;
            let y = values[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("{name}[1] must be finite")))?
                as f32;
            Ok([x, y])
        })
        .transpose()
}

pub(super) fn usize_param(
    params: &Value,
    name: &str,
    default: usize,
) -> Result<usize, ControlError> {
    params
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid(format!("{name} must be a non-negative integer")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

pub(super) fn parse_vertices(
    params: &Value,
    layer_offset: [f32; 2],
) -> Result<Vec<[f32; 2]>, ControlError> {
    let values = params
        .get("vertices")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("vertices is required"))?;
    if values.len() < 3 {
        return Err(invalid("vertices must contain at least three points"));
    }
    let world = match params
        .get("coordinate_space")
        .and_then(Value::as_str)
        .unwrap_or("world")
    {
        "world" => true,
        "local" => false,
        _ => return Err(invalid("coordinate_space must be 'world' or 'local'")),
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| invalid(format!("vertices[{index}] must be [x, y]")))?;
            let x = pair[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("vertices[{index}][0] must be finite")))?
                as f32;
            let y = pair[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("vertices[{index}][1] must be finite")))?
                as f32;
            Ok(if world {
                [x - layer_offset[0], y - layer_offset[1]]
            } else {
                [x, y]
            })
        })
        .collect()
}

pub(super) fn close_polygon(mut vertices: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if vertices.first() != vertices.last()
        && let Some(first) = vertices.first().copied()
    {
        vertices.push(first);
    }
    vertices
}

pub(super) fn unique_vertex_count(polygon: &[[f32; 2]]) -> usize {
    polygon.len().saturating_sub(usize::from(
        polygon.len() > 1 && polygon.first() == polygon.last(),
    ))
}

pub(super) fn polygon_json(layer: &ProjectMaskLayer, index: usize, polygon: &[[f32; 2]]) -> Value {
    json!({
        "index": index,
        "vertices_local": polygon,
        "vertices_world": polygon.iter().map(|point| [
            point[0] + layer.offset_world[0],
            point[1] + layer.offset_world[1],
        ]).collect::<Vec<_>>(),
    })
}

pub(super) fn parse_projection_selection(
    value: Option<&Value>,
    layers: &[ProjectMaskLayer],
) -> Result<Option<MaskSelection>, ControlError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let layer_id = value
        .get("layer_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("renderer mask selection has no layer_id"))?;
    let polygon_index = value
        .get("polygon_index")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("renderer mask selection has no polygon_index"))?;
    let layer = layers
        .iter()
        .find(|layer| layer.id == layer_id)
        .ok_or_else(|| invalid("renderer mask selection references an unknown layer"))?;
    let polygon = layer
        .polygons_world
        .get(polygon_index)
        .ok_or_else(|| invalid("renderer mask selection references an unknown polygon"))?;
    let vertex_index = value
        .get("vertex_index")
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|index| *index < unique_vertex_count(polygon))
                .ok_or_else(|| invalid("renderer mask selection references an unknown vertex"))
        })
        .transpose()?;
    Ok(Some(MaskSelection {
        layer_id,
        polygon_index,
        vertex_index,
    }))
}

pub(super) fn validate_layers(layers: &[ProjectMaskLayer]) -> Result<(), ControlError> {
    let mut ids = std::collections::HashSet::new();
    for layer in layers {
        if layer.id == 0 || !ids.insert(layer.id) {
            return Err(invalid("mask layer IDs must be unique positive integers"));
        }
        if layer.name.trim().is_empty() {
            return Err(invalid(format!(
                "mask layer {} has an empty name",
                layer.id
            )));
        }
        if !layer.opacity.is_finite() || !(0.0..=1.0).contains(&layer.opacity) {
            return Err(invalid(format!(
                "mask layer {} opacity must be between 0 and 1",
                layer.id
            )));
        }
        if !layer.width_screen_px.is_finite() || layer.width_screen_px <= 0.0 {
            return Err(invalid(format!(
                "mask layer {} width_screen_px must be greater than zero",
                layer.id
            )));
        }
        if layer.offset_world.iter().any(|value| !value.is_finite()) {
            return Err(invalid(format!(
                "mask layer {} offset_world must be finite",
                layer.id
            )));
        }
        if layer.display_mode.as_deref().is_some_and(|mode| {
            !matches!(
                mode,
                "outline_only"
                    | "outline"
                    | "translucent_fill"
                    | "fill_outline"
                    | "semi_transparent_fill"
                    | "filled_preview"
                    | "mask_preview"
            )
        }) {
            return Err(invalid(format!(
                "mask layer {} has an unknown display_mode",
                layer.id
            )));
        }
        for polygon in &layer.polygons_world {
            if polygon.len() < 2
                || polygon
                    .iter()
                    .flatten()
                    .any(|coordinate| !coordinate.is_finite())
            {
                return Err(invalid(format!(
                    "mask layer {} contains invalid polygon geometry",
                    layer.id
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

pub(super) fn not_found(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::ResourceNotFound, message)
}
