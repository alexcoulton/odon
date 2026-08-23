use super::*;

impl OmeZarrViewerApp {
    fn projected_native_layer_id(&self, raw: &str) -> Result<LayerId, String> {
        let id = self
            .parse_layer_id_storage_key(raw)
            .ok_or_else(|| format!("unknown actor native layer '{raw}'"))?;
        let loaded = match id {
            LayerId::Channel(index) => self.channel_layer_order.contains(&index),
            _ => self.overlay_layer_order.contains(&id),
        };
        loaded
            .then_some(id)
            .ok_or_else(|| format!("actor native layer '{raw}' is not loaded"))
    }

    fn apply_native_layer_visibility_projection(
        &mut self,
        id: LayerId,
        visible: bool,
    ) -> Result<(), String> {
        let before = self.layer_visible_value(id);
        let target = self
            .layer_visible_mut(id)
            .ok_or_else(|| "actor native layer has no visibility state".to_string())?;
        *target = visible;
        if before != Some(visible) {
            self.bump_render_id();
        }
        Ok(())
    }

    fn apply_native_layer_presentation_projection(
        &mut self,
        id: LayerId,
        presentation: &serde_json::Value,
    ) -> Result<(), String> {
        let presentation = presentation
            .as_object()
            .ok_or_else(|| "actor native layer presentation is not an object".to_string())?;
        let presentation = serde_json::Value::Object(presentation.clone());
        let before = self.control_native_layer_presentation(id);

        match id {
            LayerId::Channel(index) => {
                let channel = self
                    .channels
                    .get_mut(index)
                    .ok_or_else(|| "actor channel layer is not loaded".to_string())?;
                if let Some(value) = presentation.get("visible") {
                    channel.visible = value
                        .as_bool()
                        .ok_or_else(|| "actor channel visibility is not a boolean".to_string())?;
                }
                if let Some(value) = presentation.get("color_rgb") {
                    channel.color_rgb = ViewerViewportState::rgb_from_json(value)
                        .ok_or_else(|| "actor channel color_rgb is invalid".to_string())?;
                }
                if let Some(window) = presentation.get("window") {
                    channel.window = if window.is_null() {
                        None
                    } else {
                        let values = if let Some(values) =
                            window.as_array().filter(|values| values.len() == 2)
                        {
                            (values[0].as_f64(), values[1].as_f64())
                        } else {
                            (
                                window.get("min").and_then(serde_json::Value::as_f64),
                                window.get("max").and_then(serde_json::Value::as_f64),
                            )
                        };
                        let (Some(min), Some(max)) = values else {
                            return Err("actor channel window is invalid".to_string());
                        };
                        if !min.is_finite() || !max.is_finite() || max <= min {
                            return Err("actor channel window is invalid".to_string());
                        }
                        Some((min as f32, max as f32))
                    };
                }
            }
            LayerId::SegmentationLabels => {
                if let Some(value) = presentation.get("visible") {
                    self.cells_outlines_visible = value.as_bool().ok_or_else(|| {
                        "actor segmentation-label visibility is not a boolean".to_string()
                    })?;
                }
                if let Some(value) = presentation.get("opacity") {
                    self.cells_outlines_opacity = value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                        "actor segmentation-label opacity is invalid".to_string()
                    })? as f32;
                }
                if let Some(value) = presentation.get("width_screen_px") {
                    self.cells_outlines_width_px = value
                        .as_f64()
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            "actor segmentation-label width_screen_px is invalid".to_string()
                        })? as f32;
                }
                if let Some(value) = presentation.get("color_rgb") {
                    self.cells_outlines_color_rgb = ViewerViewportState::rgb_from_json(value)
                        .ok_or_else(|| {
                            "actor segmentation-label color_rgb is invalid".to_string()
                        })?;
                }
            }
            LayerId::SegmentationObjects => {
                self.seg_objects
                    .apply_actor_style_projection_json(&presentation)?;
            }
            other => {
                let mut state = ViewerViewportState::capture(self);
                let mut with_id = presentation
                    .as_object()
                    .expect("projection presentation remains an object")
                    .clone();
                let payload = match other {
                    LayerId::SpatialImage(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"spatial_images":[with_id]})
                    }
                    LayerId::SegmentationGeoJson => {
                        serde_json::json!({"segmentation_geojson":presentation})
                    }
                    LayerId::Mask(id) => {
                        if let Some(mode) = presentation.get("display_mode") {
                            let mode = mode
                                .as_str()
                                .and_then(MaskDisplayMode::from_storage_key)
                                .ok_or_else(|| "actor mask display_mode is invalid".to_string())?;
                            with_id.insert(
                                "display_mode".to_string(),
                                serde_json::json!(mode.storage_key()),
                            );
                        }
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"masks":[with_id]})
                    }
                    LayerId::Points => serde_json::json!({"cell_points":presentation}),
                    LayerId::Annotation(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"annotations":[with_id]})
                    }
                    LayerId::SpatialShape(id) => {
                        with_id.insert("id".to_string(), serde_json::json!(id));
                        serde_json::json!({"spatial_shapes":[with_id]})
                    }
                    LayerId::SpatialPoints => {
                        serde_json::json!({"spatial_points":presentation})
                    }
                    LayerId::XeniumCells => serde_json::json!({"xenium_cells":presentation}),
                    LayerId::XeniumTranscripts => {
                        serde_json::json!({"xenium_transcripts":presentation})
                    }
                    LayerId::Channel(_)
                    | LayerId::SegmentationLabels
                    | LayerId::SegmentationObjects => unreachable!(),
                };
                state.apply_project_presentation_json(&payload)?;
                state.apply(self);
            }
        }

        if before != self.control_native_layer_presentation(id) {
            self.bump_render_id();
        }
        Ok(())
    }

    fn apply_native_layer_offset_projection(
        &mut self,
        id: LayerId,
        offset: &serde_json::Value,
    ) -> Result<(), String> {
        let values = offset
            .as_array()
            .filter(|values| values.len() == 2)
            .ok_or_else(|| "actor native layer offset_world is invalid".to_string())?;
        let x = values[0]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| "actor native layer offset_world is invalid".to_string())?;
        let y = values[1]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| "actor native layer offset_world is invalid".to_string())?;
        let before = self.layer_offset_world(id);
        let target = self
            .layer_offset_world_mut(id)
            .ok_or_else(|| "actor native layer does not support translation".to_string())?;
        *target = egui::vec2(x as f32, y as f32);
        if *target != before {
            self.hist_dirty = true;
            self.bump_render_id();
        }
        Ok(())
    }

    fn apply_native_layer_order_projection(
        &mut self,
        stack: &str,
        ordered: Vec<LayerId>,
    ) -> Result<(), String> {
        let mut unique = ordered.clone();
        unique.sort_by_key(|id| Self::layer_id_storage_key(*id));
        unique.dedup();
        if unique.len() != ordered.len() {
            return Err(format!("actor native {stack} order contains duplicates"));
        }
        let changed = match stack {
            "channels" => {
                let indices = ordered
                    .into_iter()
                    .map(|id| match id {
                        LayerId::Channel(index) => Ok(index),
                        _ => Err("actor channels order contains an overlay".to_string()),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if indices.len() != self.channel_layer_order.len()
                    || !self
                        .channel_layer_order
                        .iter()
                        .all(|index| indices.contains(index))
                {
                    return Err("actor channels order is incomplete".to_string());
                }
                let changed = self.channel_layer_order != indices;
                self.channel_layer_order = indices;
                changed
            }
            "overlays" => {
                if ordered.len() != self.overlay_layer_order.len()
                    || ordered.iter().any(|id| matches!(id, LayerId::Channel(_)))
                    || !self
                        .overlay_layer_order
                        .iter()
                        .all(|id| ordered.contains(id))
                {
                    return Err("actor overlays order is incomplete".to_string());
                }
                let changed = self.overlay_layer_order != ordered;
                self.overlay_layer_order = ordered;
                changed
            }
            _ => return Err(format!("unknown actor native layer stack '{stack}'")),
        };
        if changed {
            self.bump_render_id();
        }
        Ok(())
    }

    fn apply_active_native_layer_projection(&mut self, id: LayerId) {
        self.active_layer = id;
        if self
            .selected_mask_polygon
            .is_some_and(|selection| id != LayerId::Mask(selection.layer_id))
        {
            self.clear_mask_polygon_selection();
        }
        if let LayerId::Channel(index) = id {
            self.selected_channel = index.min(self.channels.len().saturating_sub(1));
            self.hist_dirty = true;
        } else {
            self.selected_channel_group_id = None;
        }
    }

    pub(super) fn apply_control_actor_native_layers_projection(
        &mut self,
        projection: &serde_json::Value,
    ) -> Result<(), String> {
        let layers = projection
            .as_array()
            .ok_or_else(|| "actor native_layers projection is not an array".to_string())?;
        for layer in layers {
            if layer.get("available").and_then(serde_json::Value::as_bool) == Some(false) {
                continue;
            }
            let raw_id = layer
                .get("layer_id")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| "actor native layer has no layer_id".to_string())?;
            let id = self.projected_native_layer_id(raw_id)?;
            if let Some(name) = layer.get("name").and_then(serde_json::Value::as_str)
                && let LayerId::Annotation(annotation_id) = id
                && let Some(annotation) = self
                    .annotation_layers
                    .iter_mut()
                    .find(|annotation| annotation.id == annotation_id)
            {
                annotation.name = name.to_string();
            }
            if let Some(presentation) = layer.get("presentation") {
                self.apply_native_layer_presentation_projection(id, presentation)
                    .map_err(|error| {
                        format!("actor native layer '{raw_id}' presentation failed: {error}")
                    })?;
            } else if let Some(visible) = layer.get("visible") {
                let visible = visible.as_bool().ok_or_else(|| {
                    format!("actor native layer '{raw_id}' visibility is not a boolean")
                })?;
                self.apply_native_layer_visibility_projection(id, visible)?;
            }
            if let Some(offset) = layer.get("offset_world") {
                self.apply_native_layer_offset_projection(id, offset)
                    .map_err(|error| {
                        format!("actor native layer '{raw_id}' offset failed: {error}")
                    })?;
            }
        }

        for stack in ["channels", "overlays"] {
            let mut ordered = layers
                .iter()
                .filter(|layer| {
                    layer.get("stack").and_then(serde_json::Value::as_str) == Some(stack)
                })
                .filter_map(|layer| {
                    Some((
                        layer.get("order").and_then(serde_json::Value::as_u64)?,
                        layer.get("layer_id").and_then(serde_json::Value::as_str)?,
                    ))
                })
                .collect::<Vec<_>>();
            if ordered.is_empty() {
                continue;
            }
            ordered.sort_by_key(|(order, _)| *order);
            let mut ordered = ordered
                .into_iter()
                .map(|(_, raw)| self.projected_native_layer_id(raw))
                .collect::<Result<Vec<_>, _>>()?;
            // Renderer-discovered compatibility descriptors can precede the actor's first
            // observation. Keep them after the canonical actor order until the next projection.
            let compatibility = match stack {
                "channels" => self
                    .channel_layer_order
                    .iter()
                    .copied()
                    .map(LayerId::Channel)
                    .collect::<Vec<_>>(),
                "overlays" => self.overlay_layer_order.clone(),
                _ => unreachable!(),
            };
            for id in compatibility {
                if !ordered.contains(&id) {
                    ordered.push(id);
                }
            }
            self.apply_native_layer_order_projection(stack, ordered)?;
        }

        if let Some(raw) = layers
            .iter()
            .find(|layer| layer.get("active").and_then(serde_json::Value::as_bool) == Some(true))
            .and_then(|layer| layer.get("layer_id"))
            .and_then(serde_json::Value::as_str)
        {
            let id = self.projected_native_layer_id(raw)?;
            self.apply_active_native_layer_projection(id);
        }
        Ok(())
    }
}
