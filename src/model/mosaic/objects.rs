//! Mosaic object-resource loading lifecycle and projection state.

use super::*;

impl MosaicModel {
    pub(super) fn object_style_snapshot(&self) -> Result<Value, ControlError> {
        self.require_resource()?;
        Ok(json!({"style":self.object_style}))
    }

    pub(super) fn set_object_style(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        let patch = params.get("style").unwrap_or(params);
        let patch = patch
            .as_object()
            .ok_or_else(|| invalid("mosaic object style must be an object"))?;
        let mut next = self.object_style.clone();
        let next_object = next.as_object_mut().expect("mosaic object style object");
        for (key, value) in patch {
            match key.as_str() {
                "opacity" | "fill_opacity" | "selected_fill_opacity" => {
                    let number = value
                        .as_f64()
                        .filter(|number| number.is_finite() && (0.0..=1.0).contains(number))
                        .ok_or_else(|| invalid(format!("{key} must be between 0 and 1")))?;
                    next_object.insert(key.clone(), json!(number));
                }
                "width_screen_px" => {
                    let number = value
                        .as_f64()
                        .filter(|number| number.is_finite() && *number >= 0.0)
                        .ok_or_else(|| invalid("width_screen_px must be non-negative"))?;
                    next_object.insert(key.clone(), json!(number));
                }
                "downsample_factor" => {
                    let number = value
                        .as_f64()
                        .filter(|number| number.is_finite() && *number > 0.0)
                        .ok_or_else(|| invalid("downsample_factor must be greater than zero"))?;
                    next_object.insert(key.clone(), json!(number));
                }
                "fill_cells" => {
                    value
                        .as_bool()
                        .ok_or_else(|| invalid("fill_cells must be boolean"))?;
                    next_object.insert(key.clone(), value.clone());
                }
                "color_rgb" => {
                    let values = value
                        .as_array()
                        .filter(|values| values.len() == 3)
                        .ok_or_else(|| invalid("color_rgb must contain three bytes"))?;
                    if values
                        .iter()
                        .any(|value| value.as_u64().is_none_or(|value| value > 255))
                    {
                        return Err(invalid("color_rgb must contain three bytes"));
                    }
                    next_object.insert(key.clone(), value.clone());
                }
                "color_property_key" => {
                    value
                        .as_str()
                        .ok_or_else(|| invalid("color_property_key must be a string"))?;
                    next_object.insert(key.clone(), value.clone());
                }
                "color_level_overrides" => {
                    value
                        .as_object()
                        .ok_or_else(|| invalid("color_level_overrides must be an object"))?;
                    next_object.insert(key.clone(), value.clone());
                }
                "style" | "item_id" | "roi_id" => {}
                _ => {
                    return Err(invalid(format!(
                        "unknown mosaic object style field '{key}'"
                    )));
                }
            }
        }
        let changed = next != self.object_style;
        self.object_style = next;
        Ok(json!({"changed":changed,"style":self.object_style}))
    }

    pub(super) fn object_selection_projection(&self) -> Value {
        Value::Object(
            self.object_selections
                .iter()
                .map(|(item_id, selection)| (item_id.to_string(), selection.projection_json()))
                .collect(),
        )
    }

    fn object_item_id(&self, params: &Value) -> Result<usize, ControlError> {
        if let Some(item_id) = params.get("item_id").and_then(Value::as_u64) {
            let item_id = usize::try_from(item_id).map_err(|_| invalid("item_id is too large"))?;
            if self.items.iter().any(|item| item.id == item_id) {
                return Ok(item_id);
            }
            return Err(invalid("mosaic object item_id was not found"));
        }
        let roi_id = params
            .get("roi_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("item_id or roi_id is required"))?;
        self.items
            .iter()
            .find(|item| item.roi_id == roi_id)
            .map(|item| item.id)
            .ok_or_else(|| invalid("mosaic object roi_id was not found"))
    }

    pub(super) fn object_selection_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        self.require_resource()?;
        if params.get("item_id").is_none() && params.get("roi_id").is_none() {
            return Ok(json!({"items":self.object_selection_projection()}));
        }
        let item_id = self.object_item_id(params)?;
        let selection = self.object_selections.get(&item_id);
        Ok(json!({
            "item_id":item_id,
            "selection":selection.map_or_else(
                || ObjectSelectionModel::default().snapshot(self.object_resources.get(&item_id).map(AsRef::as_ref), 256),
                |selection| selection.snapshot(self.object_resources.get(&item_id).map(AsRef::as_ref), 256)
            )
        }))
    }

    pub(super) fn replace_object_selection(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        let item_id = self.object_item_id(params)?;
        if params
            .get("clear_others")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            for (other_item_id, selection) in &mut self.object_selections {
                if *other_item_id != item_id {
                    selection.clear(None, 0);
                }
            }
        }
        let resource = self.object_resources.get(&item_id).cloned();
        let selection = self.object_selections.entry(item_id).or_default();
        let response = selection.replace_transaction(resource.as_deref(), params, 256)?;
        Ok(json!({"item_id":item_id,"result":response}))
    }

    pub(super) fn clear_object_selections(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_resource()?;
        if params.get("item_id").is_some() || params.get("roi_id").is_some() {
            let item_id = self.object_item_id(params)?;
            let resource = self.object_resources.get(&item_id).cloned();
            let result = self
                .object_selections
                .entry(item_id)
                .or_default()
                .clear(resource.as_deref(), 256);
            return Ok(json!({"item_id":item_id,"result":result}));
        }
        let changed = self
            .object_selections
            .values()
            .any(|selection| !selection.selected_indices().is_empty());
        for selection in self.object_selections.values_mut() {
            selection.clear(None, 0);
        }
        Ok(json!({"changed":changed,"items":self.object_selection_projection()}))
    }

    pub(crate) fn prepare_object_load(
        &mut self,
        params: &Value,
        downsample_factor: f32,
    ) -> Result<MosaicObjectLoadSpec, ControlError> {
        self.require_resource()?;
        if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
            return Err(invalid(
                "downsample_factor must be finite and greater than zero",
            ));
        }
        let explicit_item_ids = params
            .get("item_ids")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| invalid("item_ids must contain non-negative integers"))
                    })
                    .collect::<Result<HashSet<_>, _>>()
            })
            .transpose()?;
        let explicit_roi_ids = params
            .get("roi_ids")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_string)
                            .ok_or_else(|| invalid("roi_ids must contain strings"))
                    })
                    .collect::<Result<HashSet<_>, _>>()
            })
            .transpose()?;
        let scope = params.get("scope").and_then(Value::as_str);
        if scope.is_some_and(|scope| !matches!(scope, "selected" | "all")) {
            return Err(invalid("scope must be 'selected' or 'all'"));
        }
        let scope_all = scope == Some("all");
        let selected = |item: &&MosaicItemModel| {
            scope_all
                || explicit_item_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(&item.id))
                || explicit_roi_ids
                    .as_ref()
                    .is_some_and(|ids| ids.contains(&item.roi_id))
                || (explicit_item_ids.is_none()
                    && explicit_roi_ids.is_none()
                    && !scope_all
                    && self.selected_ids.contains(&item.id))
        };
        let items = self
            .items
            .iter()
            .filter(selected)
            .filter_map(|item| {
                item.segmentation_path
                    .as_ref()
                    .map(|path| (item.id, path.clone()))
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "None of the requested mosaic ROIs has an object segmentation source.",
            ));
        }
        self.cancel_object_load("Superseded by a newer mosaic object load");
        self.object_operation_generation = self.object_operation_generation.wrapping_add(1).max(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.object_pending_ids = items.iter().map(|(id, _)| *id).collect();
        self.object_failures.clear();
        self.object_cancel = Some(Arc::clone(&cancel));
        self.object_status = format!("Loading objects for {} mosaic ROI(s)", items.len());
        Ok(MosaicObjectLoadSpec {
            resource_generation: self.resource_generation(),
            operation_generation: self.object_operation_generation,
            downsample_factor,
            items,
            cancel,
        })
    }

    pub(crate) fn finish_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        result: MosaicObjectLoadResult,
    ) -> Option<Value> {
        if !self.object_spec_is_current(spec) {
            return None;
        }
        let requested = spec.items.len();
        for (id, resource) in result.loaded {
            self.object_resources.insert(id, resource);
            self.object_pending_ids.remove(&id);
        }
        for (id, error) in result.failures {
            self.object_failures.insert(id, error);
            self.object_pending_ids.remove(&id);
        }
        let cancelled = result.cancelled || spec.is_cancelled();
        self.object_pending_ids.clear();
        self.object_cancel = None;
        self.object_status = if cancelled {
            "Mosaic object loading cancelled".to_string()
        } else if self.object_failures.is_empty() {
            format!("Loaded objects for {requested} mosaic ROI(s)")
        } else {
            format!(
                "Loaded objects for {} of {requested} mosaic ROI(s)",
                requested.saturating_sub(self.object_failures.len())
            )
        };
        Some(json!({
            "settled":true,
            "cancelled":cancelled,
            "requested":requested,
            "loaded":requested.saturating_sub(self.object_failures.len()),
            "failed":self.object_failures.len(),
            "state":self.object_state(),
        }))
    }

    pub(crate) fn fail_object_load(
        &mut self,
        spec: &MosaicObjectLoadSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.object_spec_is_current(spec) {
            return false;
        }
        self.object_pending_ids.clear();
        self.object_cancel = None;
        self.object_status = message.into();
        true
    }

    pub(crate) fn cancel_object_load(&mut self, message: impl Into<String>) -> usize {
        let cancelled = self.object_pending_ids.len();
        if let Some(cancel) = self.object_cancel.take() {
            cancel.store(true, AtomicOrdering::Relaxed);
        }
        self.object_pending_ids.clear();
        if cancelled > 0 {
            self.object_status = message.into();
        }
        cancelled
    }

    pub(crate) fn cancel_object_load_response(&mut self) -> Result<Value, ControlError> {
        self.require_resource()?;
        let cancelled = self.cancel_object_load("Mosaic object loading cancelled");
        Ok(json!({
            "cancelled_requests":cancelled,
            "in_flight_cancelled":cancelled > 0,
            "state":self.object_state(),
        }))
    }

    pub(super) fn object_spec_is_current(&self, spec: &MosaicObjectLoadSpec) -> bool {
        self.resource_generation() == spec.resource_generation
            && self.object_operation_generation == spec.operation_generation
    }

    pub(super) fn object_state(&self) -> Value {
        let items = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let resource = self.object_resources.get(&item.id);
                json!({
                    "index":index,
                    "item_id":item.id,
                    "roi_id":item.roi_id,
                    "selected":self.selected_ids.contains(&item.id),
                    "available":item.segmentation_path.is_some(),
                    "path":item.segmentation_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
                    "requested":self.object_pending_ids.contains(&item.id),
                    "loaded":resource.is_some(),
                    "object_count":resource.map_or(0, |resource| resource.features.len()),
                    "error":self.object_failures.get(&item.id),
                })
            })
            .collect::<Vec<_>>();
        json!({
            "generation":self.object_operation_generation,
            "requested_count":self.object_pending_ids.len(),
            "requested_loading":self.object_pending_ids.len(),
            "settled":self.object_pending_ids.is_empty(),
            "loaded_count":self.object_resources.len(),
            "failed_count":self.object_failures.len(),
            "status":self.object_status,
            "items":items,
        })
    }
}
