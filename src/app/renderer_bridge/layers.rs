use super::super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn control_native_layer_kind(id: LayerId) -> &'static str {
        match id {
            LayerId::Channel(_) => "channel",
            LayerId::SpatialImage(_) => "spatial_image",
            LayerId::SegmentationLabels => "segmentation_labels",
            LayerId::SegmentationGeoJson => "segmentation_geojson",
            LayerId::SegmentationObjects => "segmentation_objects",
            LayerId::Mask(_) => "mask",
            LayerId::Points => "points",
            LayerId::Annotation(_) => "annotation",
            LayerId::SpatialShape(_) => "spatial_shape",
            LayerId::SpatialPoints => "spatial_points",
            LayerId::XeniumCells => "xenium_cells",
            LayerId::XeniumTranscripts => "xenium_transcripts",
        }
    }

    pub(in crate::app) fn control_native_layer_snapshot(
        &self,
        id: LayerId,
        stack: &str,
        order: usize,
    ) -> serde_json::Value {
        let offset = self.layer_offset_world(id);
        let loaded_offset = self
            .loaded_layer_offsets_world
            .get(&id)
            .copied()
            .unwrap_or(offset);
        serde_json::json!({
            "layer_id": Self::layer_id_storage_key(id),
            "kind": Self::control_native_layer_kind(id),
            "name": self.layer_display_name(id),
            "stack": stack,
            "order": order,
            "active": self.active_layer == id,
            "visible": self.layer_visible_value(id).unwrap_or(false),
            "available": self.layer_is_available(id),
            "offset_world": [offset.x, offset.y],
            "loaded_offset_world": [loaded_offset.x, loaded_offset.y],
            "presentation": self.control_native_layer_presentation(id),
        })
    }

    pub(in crate::app) fn control_native_layer_presentation(
        &self,
        id: LayerId,
    ) -> serde_json::Value {
        match id {
            LayerId::Channel(index) => self
                .channels
                .get(index)
                .map(|channel| {
                    serde_json::json!({
                        "visible": channel.visible,
                        "color_rgb": channel.color_rgb,
                        "window": channel.window.map(|(min, max)| serde_json::json!({
                            "min": min,
                            "max": max,
                        })),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "current_z_level0": layer.current_z_level0,
                        "channels": layer.channels.iter().map(|channel| serde_json::json!({
                            "index": channel.index,
                            "name": channel.name,
                            "visible": channel.visible,
                            "color_rgb": channel.color_rgb,
                            "window": channel.window.map(|(min, max)| [min, max]),
                        })).collect::<Vec<_>>(),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SegmentationLabels => serde_json::json!({
                "visible": self.cells_outlines_visible,
                "opacity": self.cells_outlines_opacity,
                "width_screen_px": self.cells_outlines_width_px,
                "color_rgb": self.cells_outlines_color_rgb,
            }),
            LayerId::SegmentationGeoJson => serde_json::json!({
                "visible": self.seg_geojson.visible,
                "opacity": self.seg_geojson.opacity,
                "width_screen_px": self.seg_geojson.width_screen_px,
                "color_rgb": self.seg_geojson.color_rgb,
            }),
            LayerId::SegmentationObjects => ViewerViewportState::object_layer_presentation_json(
                &ObjectLayerViewportPresentation::capture(&self.seg_objects),
            ),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "display_mode": layer.display_mode.storage_key(),
                        "color_rgb": layer.color_rgb,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::Points => serde_json::json!({
                "visible": self.cell_points.visible,
                "style": ViewerViewportState::points_style_json(&self.cell_points.style),
            }),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": {
                            "radius_screen_px": layer.style.radius_screen_px,
                            "opacity": layer.style.opacity,
                            "stroke_width": layer.style.stroke.width,
                            "stroke_color_rgba": ViewerViewportState::color_json(layer.style.stroke.color),
                        },
                        "category_styles": layer.category_styles.iter().map(|category| serde_json::json!({
                            "name": category.name,
                            "visible": category.visible,
                            "color_rgba": ViewerViewportState::color_json(category.color),
                            "shape": category.shape.storage_key(),
                        })).collect::<Vec<_>>(),
                        "continuous_shape": layer.continuous_shape.storage_key(),
                        "continuous_range": layer.continuous_range.map(|(min, max)| [min, max]),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "color_rgb": layer.color_rgb,
                        "objects": layer.object_layer().map(|objects| {
                            ViewerViewportState::object_layer_presentation_json(
                                &ObjectLayerViewportPresentation::capture(objects),
                            )
                        }),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialPoints => self
                .spatial_layers
                .points
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": ViewerViewportState::points_style_json(&layer.style),
                        "threshold": layer.threshold,
                        "max_render_points_total": layer.max_render_points_total,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::XeniumCells => self
                .xenium_layers
                .cells
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "color_rgb": layer.color_rgb,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": ViewerViewportState::points_style_json(&layer.style),
                        "gene_query": layer.gene_query,
                        "max_render_points_total": layer.max_render_points_total,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
        }
    }

    pub fn control_native_layer_snapshot_list(&self) -> serde_json::Value {
        let mut layers = self
            .channel_layer_order
            .iter()
            .copied()
            .enumerate()
            .map(|(order, idx)| {
                self.control_native_layer_snapshot(LayerId::Channel(idx), "channels", order)
            })
            .collect::<Vec<_>>();
        layers.extend(
            self.overlay_layer_order
                .iter()
                .copied()
                .enumerate()
                .map(|(order, id)| self.control_native_layer_snapshot(id, "overlays", order)),
        );
        serde_json::Value::Array(layers)
    }

    #[cfg(test)]
    pub(in crate::app) fn control_native_layer_id_from_params(
        &self,
        params: &serde_json::Value,
    ) -> Result<LayerId, String> {
        let Some(raw) = params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(serde_json::Value::as_str)
        else {
            return Err("layer_id is required".to_string());
        };
        let Some(id) = self.parse_layer_id_storage_key(raw) else {
            return Err(format!("unknown native layer '{raw}'"));
        };
        let exists = match id {
            LayerId::Channel(idx) => self.channel_layer_order.contains(&idx),
            _ => self.overlay_layer_order.contains(&id),
        };
        exists
            .then_some(id)
            .ok_or_else(|| format!("native layer '{raw}' is not loaded"))
    }

    #[cfg(test)]
    pub fn control_get_native_layer(&self, params: &serde_json::Value) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let (stack, order) = match id {
            LayerId::Channel(idx) => (
                "channels",
                self.channel_layer_order
                    .iter()
                    .position(|candidate| *candidate == idx)
                    .unwrap_or_default(),
            ),
            _ => (
                "overlays",
                self.overlay_layer_order
                    .iter()
                    .position(|candidate| *candidate == id)
                    .unwrap_or_default(),
            ),
        };
        self.control_native_layer_snapshot(id, stack, order)
    }

    #[cfg(test)]
    pub fn control_set_active_native_layer(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        if !self.layer_is_available(id) {
            return serde_json::json!({"error": "native layer is not currently available"});
        }
        let changed = self.active_layer != id;
        self.set_active_layer_local(id);
        serde_json::json!({
            "changed": changed,
            "layer": self.control_get_native_layer(params),
        })
    }

    #[cfg(test)]
    pub fn control_set_native_layer_visibility(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(visible) = params.get("visible").and_then(serde_json::Value::as_bool) else {
            return serde_json::json!({"error": "visible is required"});
        };
        let before = self.layer_visible_value(id);
        let Some(target) = self.layer_visible_mut(id) else {
            return serde_json::json!({"error": "native layer has no visibility state"});
        };
        *target = visible;
        let changed = before != Some(visible);
        if changed {
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "layer": self.control_get_native_layer(params),
        })
    }

    #[cfg(test)]
    pub fn control_set_native_layer_presentation(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let presentation = params.get("presentation").unwrap_or(params);
        let Some(presentation_object) = presentation.as_object() else {
            return serde_json::json!({"error": "presentation must be an object"});
        };
        let presentation = serde_json::Value::Object(presentation_object.clone());
        let before = self.control_native_layer_presentation(id);

        let invalid_unit = |name: &str| {
            presentation.get(name).is_some_and(|value| {
                value
                    .as_f64()
                    .is_none_or(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            })
        };
        if invalid_unit("opacity") {
            return serde_json::json!({"error": "opacity must be a finite number between 0 and 1"});
        }
        if presentation.get("width_screen_px").is_some_and(|value| {
            value
                .as_f64()
                .is_none_or(|v| !v.is_finite() || v < 0.0 || v > 100.0)
        }) {
            return serde_json::json!({
                "error": "width_screen_px must be a finite number between 0 and 100",
            });
        }
        if presentation.get("color_rgb").is_some()
            && presentation
                .get("color_rgb")
                .and_then(ViewerViewportState::rgb_from_json)
                .is_none()
        {
            return serde_json::json!({
                "error": "color_rgb must contain three integers from 0 to 255",
            });
        }

        let apply_result = match id {
            LayerId::Channel(index) => {
                let Some(channel) = self.channels.get_mut(index) else {
                    return serde_json::json!({"error": "channel layer is not loaded"});
                };
                if let Some(value) = presentation.get("visible") {
                    let Some(value) = value.as_bool() else {
                        return serde_json::json!({"error": "visible must be a boolean"});
                    };
                    channel.visible = value;
                }
                if let Some(color) = presentation
                    .get("color_rgb")
                    .and_then(ViewerViewportState::rgb_from_json)
                {
                    channel.color_rgb = color;
                }
                if let Some(window) = presentation.get("window") {
                    let values = if let Some(values) = window.as_array().filter(|v| v.len() == 2) {
                        (values[0].as_f64(), values[1].as_f64())
                    } else {
                        (
                            window.get("min").and_then(serde_json::Value::as_f64),
                            window.get("max").and_then(serde_json::Value::as_f64),
                        )
                    };
                    let (Some(min), Some(max)) = values else {
                        return serde_json::json!({
                            "error": "window must be [min, max] or an object containing min and max",
                        });
                    };
                    if !min.is_finite() || !max.is_finite() || max <= min {
                        return serde_json::json!({
                            "error": "window values must be finite and max must be greater than min",
                        });
                    }
                    channel.window = Some((min as f32, max as f32));
                }
                Ok(())
            }
            LayerId::SegmentationLabels => {
                if let Some(value) = presentation.get("visible") {
                    let Some(value) = value.as_bool() else {
                        return serde_json::json!({"error": "visible must be a boolean"});
                    };
                    self.cells_outlines_visible = value;
                }
                if let Some(value) = presentation
                    .get("opacity")
                    .and_then(serde_json::Value::as_f64)
                {
                    self.cells_outlines_opacity = value as f32;
                }
                if let Some(value) = presentation
                    .get("width_screen_px")
                    .and_then(serde_json::Value::as_f64)
                {
                    self.cells_outlines_width_px = value as f32;
                }
                if let Some(color) = presentation
                    .get("color_rgb")
                    .and_then(ViewerViewportState::rgb_from_json)
                {
                    self.cells_outlines_color_rgb = color;
                }
                Ok(())
            }
            LayerId::SegmentationObjects => {
                let result = self.seg_objects.control_set_style_json(&presentation);
                if let Some(error) = result.get("error").and_then(serde_json::Value::as_str) {
                    Err(error.to_string())
                } else {
                    Ok(())
                }
            }
            other => {
                let mut state = ViewerViewportState::capture(self);
                let mut with_id = presentation_object.clone();
                let payload = match other {
                    LayerId::SpatialImage(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"spatial_images": [with_id]})
                    }
                    LayerId::SegmentationGeoJson => {
                        serde_json::json!({"segmentation_geojson": presentation})
                    }
                    LayerId::Mask(id) => {
                        if let Some(mode) = presentation.get("display_mode") {
                            let Some(mode) =
                                mode.as_str().and_then(MaskDisplayMode::from_storage_key)
                            else {
                                return serde_json::json!({
                                    "error": "display_mode must be outline_only, translucent_fill, or filled_preview",
                                });
                            };
                            with_id.insert(
                                "display_mode".to_string(),
                                serde_json::json!(mode.storage_key()),
                            );
                        }
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"masks": [with_id]})
                    }
                    LayerId::Points => serde_json::json!({"cell_points": presentation}),
                    LayerId::Annotation(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"annotations": [with_id]})
                    }
                    LayerId::SpatialShape(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"spatial_shapes": [with_id]})
                    }
                    LayerId::SpatialPoints => serde_json::json!({"spatial_points": presentation}),
                    LayerId::XeniumCells => serde_json::json!({"xenium_cells": presentation}),
                    LayerId::XeniumTranscripts => {
                        serde_json::json!({"xenium_transcripts": presentation})
                    }
                    LayerId::Channel(_)
                    | LayerId::SegmentationLabels
                    | LayerId::SegmentationObjects => unreachable!(),
                };
                state
                    .apply_project_presentation_json(&payload)
                    .map(|()| state.apply(self))
            }
        };
        if let Err(error) = apply_result {
            return serde_json::json!({"error": error});
        }
        let changed = before != self.control_native_layer_presentation(id);
        if changed {
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "layer": self.control_get_native_layer(params),
        })
    }

    #[cfg(test)]
    pub fn control_set_native_layer_order(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let Some(stack) = params.get("stack").and_then(serde_json::Value::as_str) else {
            return serde_json::json!({"error": "stack is required"});
        };
        let Some(values) = params.get("layers").and_then(serde_json::Value::as_array) else {
            return serde_json::json!({"error": "layers is required"});
        };
        let parsed = values
            .iter()
            .map(|value| {
                let raw = value
                    .as_str()
                    .ok_or_else(|| "layer IDs must be strings".to_string())?;
                self.parse_layer_id_storage_key(raw)
                    .ok_or_else(|| format!("unknown native layer '{raw}'"))
            })
            .collect::<Result<Vec<_>, _>>();
        let parsed = match parsed {
            Ok(parsed) => parsed,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let mut unique = parsed.clone();
        unique.sort_by_key(|id| Self::layer_id_storage_key(*id));
        unique.dedup();
        if unique.len() != parsed.len() {
            return serde_json::json!({"error": "layers must not contain duplicates"});
        }
        let changed = match stack {
            "channels" => {
                let indices = parsed
                    .iter()
                    .map(|id| match id {
                        LayerId::Channel(idx) => Ok(*idx),
                        _ => Err("channels stack accepts only channel layers"),
                    })
                    .collect::<Result<Vec<_>, _>>();
                let indices = match indices {
                    Ok(indices) => indices,
                    Err(error) => return serde_json::json!({"error": error}),
                };
                if indices.len() != self.channel_layer_order.len()
                    || !self
                        .channel_layer_order
                        .iter()
                        .all(|idx| indices.contains(idx))
                {
                    return serde_json::json!({"error": "channels must contain every loaded channel exactly once"});
                }
                let changed = self.channel_layer_order != indices;
                self.channel_layer_order = indices;
                changed
            }
            "overlays" => {
                if parsed.len() != self.overlay_layer_order.len()
                    || !self
                        .overlay_layer_order
                        .iter()
                        .all(|id| parsed.contains(id))
                    || parsed.iter().any(|id| matches!(id, LayerId::Channel(_)))
                {
                    return serde_json::json!({"error": "layers must contain every loaded overlay exactly once"});
                }
                let changed = self.overlay_layer_order != parsed;
                self.overlay_layer_order = parsed;
                changed
            }
            _ => return serde_json::json!({"error": "stack must be 'channels' or 'overlays'"}),
        };
        if changed {
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "layers": self.control_native_layer_snapshot_list(),
        })
    }

    #[cfg(test)]
    pub fn control_set_native_layer_offset(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let Some(values) = params
            .get("offset_world")
            .and_then(serde_json::Value::as_array)
            .filter(|values| values.len() == 2)
        else {
            return serde_json::json!({"error": "offset_world must contain exactly two numbers"});
        };
        let Some(x) = values[0].as_f64().filter(|value| value.is_finite()) else {
            return serde_json::json!({"error": "offset_world values must be finite numbers"});
        };
        let Some(y) = values[1].as_f64().filter(|value| value.is_finite()) else {
            return serde_json::json!({"error": "offset_world values must be finite numbers"});
        };
        let before = self.layer_offset_world(id);
        let Some(target) = self.layer_offset_world_mut(id) else {
            return serde_json::json!({"error": "native layer does not support translation"});
        };
        *target = egui::vec2(x as f32, y as f32);
        let changed = *target != before;
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "layer": self.control_get_native_layer(params),
        })
    }

    #[cfg(test)]
    pub fn control_reset_native_layer_offset(
        &mut self,
        params: &serde_json::Value,
    ) -> serde_json::Value {
        let id = match self.control_native_layer_id_from_params(params) {
            Ok(id) => id,
            Err(error) => return serde_json::json!({"error": error}),
        };
        let baseline = self
            .loaded_layer_offsets_world
            .get(&id)
            .copied()
            .unwrap_or(egui::Vec2::ZERO);
        let before = self.layer_offset_world(id);
        let Some(target) = self.layer_offset_world_mut(id) else {
            return serde_json::json!({"error": "native layer does not support translation"});
        };
        *target = baseline;
        let changed = before != baseline;
        if changed {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        serde_json::json!({
            "changed": changed,
            "layer": self.control_get_native_layer(params),
        })
    }
}
