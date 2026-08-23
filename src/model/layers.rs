use std::collections::HashSet;

use serde_json::{Map, Value, json};

use crate::control::{ControlError, ControlErrorKind};
use crate::data::annotations::ProjectAnnotationLayerState;
use crate::data::ome::ChannelInfo;
use crate::data::project_config::ProjectMaskLayer;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeLayerModel {
    pub(crate) layer_id: String,
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) stack: String,
    pub(crate) available: bool,
    pub(crate) visible: bool,
    pub(crate) offset_world: [f32; 2],
    pub(crate) loaded_offset_world: [f32; 2],
    pub(crate) presentation: Value,
}

impl NativeLayerModel {
    fn from_snapshot(value: &Value) -> Result<Self, ControlError> {
        let layer_id = required_string(value, "layer_id")?;
        let kind = required_string(value, "kind")?;
        let name = required_string(value, "name")?;
        let stack = required_string(value, "stack")?;
        if !matches!(stack.as_str(), "channels" | "overlays") {
            return Err(invalid(
                "native layer stack must be 'channels' or 'overlays'",
            ));
        }
        let visible = value
            .get("visible")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' has no visibility")))?;
        let offset_world =
            finite_pair(value.get("offset_world"), "offset_world")?.unwrap_or([0.0, 0.0]);
        let loaded_offset_world =
            finite_pair(value.get("loaded_offset_world"), "loaded_offset_world")?
                .unwrap_or(offset_world);
        let presentation = value.get("presentation").cloned().unwrap_or(Value::Null);
        Ok(Self {
            layer_id,
            kind,
            name,
            stack,
            available: value
                .get("available")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            visible,
            offset_world,
            loaded_offset_world,
            presentation,
        })
    }

    pub(crate) fn snapshot(&self, order: usize, active: bool) -> Value {
        json!({
            "layer_id": self.layer_id,
            "kind": self.kind,
            "name": self.name,
            "stack": self.stack,
            "order": order,
            "active": active,
            "visible": self.visible,
            "available": self.available,
            "offset_world": self.offset_world,
            "presentation": self.presentation,
        })
    }

    fn apply_presentation(&mut self, patch: &Value) -> Result<bool, ControlError> {
        let object = patch
            .as_object()
            .ok_or_else(|| invalid("presentation must be an object"))?;
        validate_presentation_patch(object)?;
        let before = self.presentation.clone();
        let mut presentation = self.presentation.as_object().cloned().unwrap_or_default();
        for (key, value) in object {
            if !matches!(
                key.as_str(),
                "viewport_id" | "id" | "layer_id" | "if_revision" | "if_presentation_revision"
            ) {
                presentation.insert(key.clone(), value.clone());
            }
        }
        if let Some(visible) = presentation.get("visible").and_then(Value::as_bool) {
            self.visible = visible;
        }
        self.presentation = Value::Object(presentation);
        Ok(before != self.presentation)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NativeLayersModel {
    layers: Vec<NativeLayerModel>,
    active_layer_id: Option<String>,
}

impl NativeLayersModel {
    pub(crate) fn channels(channels: &[ChannelInfo]) -> Self {
        let layers = channels
            .iter()
            .map(|channel| NativeLayerModel {
                layer_id: format!("channel:{}", channel.index),
                kind: "channel".to_string(),
                name: channel.name.clone(),
                stack: "channels".to_string(),
                available: true,
                visible: channel.visible,
                offset_world: [0.0, 0.0],
                loaded_offset_world: [0.0, 0.0],
                presentation: json!({
                    "visible": channel.visible,
                    "color_rgb": channel.color_rgb,
                    "window": channel.window.map(|(min, max)| json!({"min":min,"max":max})),
                }),
            })
            .collect::<Vec<_>>();
        Self {
            active_layer_id: layers.first().map(|layer| layer.layer_id.clone()),
            layers,
        }
    }

    pub(crate) fn restore(value: &Value) -> Result<Self, ControlError> {
        let values = value
            .as_array()
            .ok_or_else(|| invalid("native_layers must be an array"))?;
        let mut parsed = values
            .iter()
            .map(NativeLayerModel::from_snapshot)
            .collect::<Result<Vec<_>, _>>()?;
        let mut ids = HashSet::new();
        if parsed
            .iter()
            .any(|layer| !ids.insert(layer.layer_id.clone()))
        {
            return Err(invalid("native layer IDs must be unique"));
        }
        parsed.sort_by_key(|layer| {
            let stack = usize::from(layer.stack == "overlays");
            let order = values
                .iter()
                .find(|value| {
                    value.get("layer_id").and_then(Value::as_str) == Some(&layer.layer_id)
                })
                .and_then(|value| value.get("order"))
                .and_then(Value::as_u64)
                .unwrap_or(u64::MAX);
            (stack, order)
        });
        let active_layer_id = values
            .iter()
            .find(|value| value.get("active").and_then(Value::as_bool) == Some(true))
            .and_then(|value| value.get("layer_id"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| parsed.first().map(|layer| layer.layer_id.clone()));
        Ok(Self {
            layers: parsed,
            active_layer_id,
        })
    }

    pub(crate) fn merge_missing(&mut self, value: &Value) -> Result<bool, ControlError> {
        let incoming = Self::restore(value)?;
        let mut changed = false;
        for layer in incoming.layers {
            if self.get(&layer.layer_id).is_none() {
                self.layers.push(layer);
                changed = true;
            }
        }
        Ok(changed)
    }

    pub(crate) fn snapshots(&self) -> Vec<Value> {
        self.layers
            .iter()
            .enumerate()
            .map(|(order, layer)| {
                let stack_order = self.layers[..order]
                    .iter()
                    .filter(|candidate| candidate.stack == layer.stack)
                    .count();
                layer.snapshot(
                    stack_order,
                    self.active_layer_id.as_deref() == Some(layer.layer_id.as_str()),
                )
            })
            .collect()
    }

    pub(crate) fn get(&self, layer_id: &str) -> Option<&NativeLayerModel> {
        self.layers.iter().find(|layer| layer.layer_id == layer_id)
    }

    pub(crate) fn active_layer_id(&self) -> Option<&str> {
        self.active_layer_id.as_deref()
    }

    fn get_mut(&mut self, layer_id: &str) -> Result<&mut NativeLayerModel, ControlError> {
        self.layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))
    }

    pub(crate) fn set_active(&mut self, layer_id: &str) -> Result<bool, ControlError> {
        let layer = self
            .get(layer_id)
            .ok_or_else(|| invalid(format!("native layer '{layer_id}' is not loaded")))?;
        if !layer.available {
            return Err(invalid("native layer is not currently available"));
        }
        let changed = self.active_layer_id.as_deref() != Some(layer_id);
        self.active_layer_id = Some(layer_id.to_string());
        Ok(changed)
    }

    pub(crate) fn set_visibility(
        &mut self,
        layer_id: &str,
        visible: bool,
    ) -> Result<bool, ControlError> {
        let layer = self.get_mut(layer_id)?;
        let changed = layer.visible != visible;
        layer.visible = visible;
        if let Some(presentation) = layer.presentation.as_object_mut() {
            presentation.insert("visible".to_string(), Value::Bool(visible));
        }
        Ok(changed)
    }

    pub(crate) fn set_presentation(
        &mut self,
        layer_id: &str,
        presentation: &Value,
    ) -> Result<bool, ControlError> {
        self.get_mut(layer_id)?.apply_presentation(presentation)
    }

    pub(crate) fn set_order(
        &mut self,
        stack: &str,
        layer_ids: &[String],
    ) -> Result<bool, ControlError> {
        if !matches!(stack, "channels" | "overlays") {
            return Err(invalid("stack must be 'channels' or 'overlays'"));
        }
        let current = self
            .layers
            .iter()
            .filter(|layer| layer.stack == stack)
            .map(|layer| layer.layer_id.clone())
            .collect::<Vec<_>>();
        let requested = layer_ids.iter().cloned().collect::<HashSet<_>>();
        if requested.len() != layer_ids.len()
            || current.len() != layer_ids.len()
            || current.iter().any(|id| !requested.contains(id))
        {
            return Err(invalid(format!(
                "layers must contain every loaded {stack} layer exactly once"
            )));
        }
        if current == layer_ids {
            return Ok(false);
        }
        let mut reordered = Vec::with_capacity(self.layers.len());
        if stack == "overlays" {
            reordered.extend(
                self.layers
                    .iter()
                    .filter(|layer| layer.stack == "channels")
                    .cloned(),
            );
        }
        for id in layer_ids {
            reordered.push(
                self.get(id)
                    .expect("validated native layer remains present")
                    .clone(),
            );
        }
        if stack == "channels" {
            reordered.extend(
                self.layers
                    .iter()
                    .filter(|layer| layer.stack == "overlays")
                    .cloned(),
            );
        }
        self.layers = reordered;
        Ok(true)
    }

    pub(crate) fn set_offset(
        &mut self,
        layer_id: &str,
        offset: [f32; 2],
    ) -> Result<bool, ControlError> {
        let layer = self.get_mut(layer_id)?;
        let changed = layer.offset_world != offset;
        layer.offset_world = offset;
        Ok(changed)
    }

    pub(crate) fn reset_offset(&mut self, layer_id: &str) -> Result<bool, ControlError> {
        let layer = self.get_mut(layer_id)?;
        let changed = layer.offset_world != layer.loaded_offset_world;
        layer.offset_world = layer.loaded_offset_world;
        Ok(changed)
    }

    pub(crate) fn replace(&mut self, value: &Value) -> Result<bool, ControlError> {
        let replacement = Self::restore(value)?;
        let changed = *self != replacement;
        *self = replacement;
        Ok(changed)
    }

    pub(crate) fn set_primary_objects(&mut self, loaded: bool) -> bool {
        let before = self.clone();
        self.layers
            .retain(|layer| layer.layer_id != "segmentation_objects");
        if loaded {
            self.layers.push(NativeLayerModel {
                layer_id: "segmentation_objects".to_string(),
                kind: "segmentation_objects".to_string(),
                name: "Objects".to_string(),
                stack: "overlays".to_string(),
                available: true,
                visible: false,
                offset_world: [0.0, 0.0],
                loaded_offset_world: [0.0, 0.0],
                presentation: json!({
                    "visible":false,
                    "opacity":0.75,
                    "width_screen_px":1.25,
                    "color_rgb":[255,255,255],
                    "fill_cells":false,
                    "fill_opacity":0.30,
                    "selected_fill_opacity":0.70,
                    "show_selection_overlay":true,
                    "fast_rendering":true,
                    "color_property":"",
                    "color_level_overrides":{},
                }),
            });
        } else if self.active_layer_id.as_deref() == Some("segmentation_objects") {
            self.active_layer_id = self.layers.first().map(|layer| layer.layer_id.clone());
        }
        *self != before
    }

    pub(crate) fn set_spatial_object_layer(
        &mut self,
        layer_id: &str,
        name: &str,
        loaded: bool,
    ) -> bool {
        let before = self.clone();
        self.layers.retain(|layer| layer.layer_id != layer_id);
        if loaded {
            self.layers.push(NativeLayerModel {
                layer_id: layer_id.to_string(),
                kind: "spatial_shape".to_string(),
                name: name.to_string(),
                stack: "overlays".to_string(),
                available: true,
                visible: true,
                offset_world: [0.0, 0.0],
                loaded_offset_world: [0.0, 0.0],
                presentation: json!({
                    "visible":true,
                    "opacity":0.75,
                    "width_screen_px":1.0,
                    "color_rgb":[0,255,120],
                    "objects":{
                        "visible":true,
                        "opacity":0.75,
                        "width_screen_px":1.0,
                        "color_rgb":[0,255,120],
                        "fill_cells":false,
                        "fill_opacity":0.30,
                        "selected_fill_opacity":0.70,
                        "show_selection_overlay":true,
                        "fast_rendering":true,
                        "color_property":"",
                        "color_level_overrides":{},
                        "filter":{
                            "mode":"simple",
                            "logic":"all",
                            "clauses":[{"enabled":true,"property":"id","query":""}],
                        },
                    },
                }),
            });
        } else if self.active_layer_id.as_deref() == Some(layer_id) {
            self.active_layer_id = self.layers.first().map(|layer| layer.layer_id.clone());
        }
        *self != before
    }

    pub(crate) fn set_segmentation_labels(&mut self, loaded: bool, visible: bool) -> bool {
        let before = self.clone();
        self.layers
            .retain(|layer| layer.layer_id != "segmentation_labels");
        if loaded {
            self.layers.push(NativeLayerModel {
                layer_id: "segmentation_labels".to_string(),
                kind: "segmentation_labels".to_string(),
                name: "Segmentation labels".to_string(),
                stack: "overlays".to_string(),
                available: true,
                visible,
                offset_world: [0.0, 0.0],
                loaded_offset_world: [0.0, 0.0],
                presentation: json!({
                    "visible":visible,
                    "opacity":0.75,
                    "width_screen_px":0.0,
                    "color_rgb":[0,255,0],
                }),
            });
        } else if self.active_layer_id.as_deref() == Some("segmentation_labels") {
            self.active_layer_id = self.layers.first().map(|layer| layer.layer_id.clone());
        }
        *self != before
    }

    pub(crate) fn sync_masks(
        &mut self,
        masks: &[ProjectMaskLayer],
        active_mask_id: Option<u64>,
    ) -> bool {
        let before = self.clone();
        let baselines = self
            .layers
            .iter()
            .filter(|layer| layer.kind == "mask")
            .map(|layer| (layer.layer_id.clone(), layer.loaded_offset_world))
            .collect::<std::collections::HashMap<_, _>>();
        self.layers.retain(|layer| layer.kind != "mask");
        self.layers.extend(masks.iter().map(|mask| {
            let layer_id = format!("mask:{}", mask.id);
            NativeLayerModel {
                loaded_offset_world: baselines
                    .get(&layer_id)
                    .copied()
                    .unwrap_or(mask.offset_world),
                layer_id,
                kind: "mask".to_string(),
                name: mask.name.clone(),
                stack: "overlays".to_string(),
                available: true,
                visible: mask.visible,
                offset_world: mask.offset_world,
                presentation: json!({
                    "visible":mask.visible,
                    "opacity":mask.opacity,
                    "width_screen_px":mask.width_screen_px,
                    "display_mode":mask.display_mode.as_deref().unwrap_or("outline_only"),
                    "color_rgb":mask.color_rgb,
                }),
            }
        }));
        if let Some(id) = active_mask_id {
            self.active_layer_id = Some(format!("mask:{id}"));
        } else if self
            .active_layer_id
            .as_deref()
            .is_some_and(|layer_id| layer_id.starts_with("mask:"))
        {
            self.active_layer_id = self.layers.first().map(|layer| layer.layer_id.clone());
        }
        *self != before
    }

    pub(crate) fn sync_annotations(&mut self, annotations: &[ProjectAnnotationLayerState]) -> bool {
        let before = self.clone();
        self.layers.retain(|layer| layer.kind != "annotation");
        self.layers
            .extend(annotations.iter().map(|annotation| NativeLayerModel {
                layer_id: format!("annotation:{}", annotation.id),
                kind: "annotation".to_string(),
                name: annotation.name.clone(),
                stack: "overlays".to_string(),
                available: true,
                visible: annotation.visible,
                offset_world: annotation.offset_world,
                loaded_offset_world: [0.0, 0.0],
                presentation: json!({
                    "visible": annotation.visible,
                    "opacity": annotation.opacity,
                    "radius_screen_px": annotation.radius_screen_px,
                    "stroke_width": annotation.stroke_width,
                    "stroke_color_rgb": annotation.stroke_color_rgb,
                    "stroke_color_alpha": annotation.stroke_color_alpha,
                }),
            }));
        if self.active_layer_id.as_deref().is_some_and(|id| {
            id.starts_with("annotation:") && !self.layers.iter().any(|layer| layer.layer_id == id)
        }) {
            self.active_layer_id = self.layers.first().map(|layer| layer.layer_id.clone());
        }
        *self != before
    }
}

fn required_string(value: &Value, name: &str) -> Result<String, ControlError> {
    value
        .get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid(format!("native layer has no valid {name}")))
}

fn finite_pair(value: Option<&Value>, name: &str) -> Result<Option<[f32; 2]>, ControlError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| invalid(format!("{name} must contain exactly two numbers")))?;
    let pair = [
        values[0]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid(format!("{name} values must be finite numbers")))?
            as f32,
        values[1]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| invalid(format!("{name} values must be finite numbers")))?
            as f32,
    ];
    Ok(Some(pair))
}

fn validate_presentation_patch(patch: &Map<String, Value>) -> Result<(), ControlError> {
    if let Some(value) = patch.get("visible")
        && value.as_bool().is_none()
    {
        return Err(invalid("visible must be a boolean"));
    }
    if let Some(value) = patch.get("opacity")
        && value
            .as_f64()
            .is_none_or(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(invalid("opacity must be a finite number between 0 and 1"));
    }
    if let Some(value) = patch.get("width_screen_px")
        && value
            .as_f64()
            .is_none_or(|value| !value.is_finite() || !(0.0..=100.0).contains(&value))
    {
        return Err(invalid(
            "width_screen_px must be a finite number between 0 and 100",
        ));
    }
    if let Some(value) = patch.get("color_rgb") {
        let valid = value.as_array().is_some_and(|values| {
            values.len() == 3
                && values
                    .iter()
                    .all(|value| value.as_u64().is_some_and(|value| value <= 255))
        });
        if !valid {
            return Err(invalid(
                "color_rgb must contain three integers from 0 to 255",
            ));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_mutate_and_replace_native_layers() {
        let input = json!([
            {"layer_id":"channel:0","kind":"channel","name":"DAPI","stack":"channels","order":0,"active":true,"visible":true,"available":true,"offset_world":[2.0,3.0],"loaded_offset_world":[1.0,1.5],"presentation":{"visible":true,"color_rgb":[1,2,3]}},
            {"layer_id":"segmentation_objects","kind":"segmentation_objects","name":"Objects","stack":"overlays","order":0,"active":false,"visible":false,"available":true,"offset_world":[0.0,0.0],"presentation":{"visible":false,"opacity":0.5}}
        ]);
        let mut layers = NativeLayersModel::restore(&input).unwrap();
        assert!(layers.set_visibility("segmentation_objects", true).unwrap());
        assert!(layers.set_active("segmentation_objects").unwrap());
        assert!(layers.set_offset("channel:0", [9.0, 8.0]).unwrap());
        assert!(layers.reset_offset("channel:0").unwrap());
        let snapshots = layers.snapshots();
        assert_eq!(snapshots[0]["offset_world"], json!([1.0, 1.5]));
        assert_eq!(snapshots[1]["active"], true);
        assert_eq!(snapshots[1]["presentation"]["visible"], true);
    }
}
