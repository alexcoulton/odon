//! Object properties, targeting, selection, spatial queries, and filters.

use super::*;

impl AppModel {
    pub(super) fn object_properties_list(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.list";
        let target = self.resolve_object_target(params)?;
        let resource = self.object_resource_for_target(target, method)?;
        let offset = bounded_offset(params, "offset")?;
        let limit = bounded_limit(params, 200)?;
        let total = resource.property_names.len();
        let columns = resource
            .property_names
            .iter()
            .skip(offset)
            .take(limit)
            .map(|name| {
                let values = resource
                    .features
                    .iter()
                    .filter_map(|feature| {
                        if name == "id" {
                            Some(Value::String(feature.id.clone()))
                        } else {
                            feature.properties.get(name).cloned()
                        }
                    })
                    .filter(|value| !value.is_null())
                    .collect::<Vec<_>>();
                let kind = object_property_type(&values);
                let categorical = values
                    .iter()
                    .map(Value::to_string)
                    .collect::<HashSet<_>>()
                    .len()
                    <= 256;
                json!({
                    "name": name,
                    "loaded": true,
                    "loading": false,
                    "type": kind,
                    "numeric": matches!(kind, "integer" | "number"),
                    "categorical": categorical,
                })
            })
            .collect::<Vec<_>>();
        let mut response = json!({
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(columns.len()) < total,
            "columns": columns,
        });
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_property_load(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.load";
        let target = self.resolve_object_target(params)?;
        let resource = self.object_resource_for_target(target, method)?;
        let property = required_nonempty_string(params, &["property", "name"], "property")?;
        if !resource.property_names.iter().any(|name| name == property) {
            return Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("unknown object property '{property}'"),
            ));
        }
        let mut response = json!({"property": property, "loaded": true, "loading": false});
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_property_values(&self, params: &Value) -> Result<Value, ControlError> {
        let method = "viewer.objects.properties.values";
        let target = self.resolve_object_target(params)?;
        let resource = self.object_resource_for_target(target, method)?;
        let property = required_nonempty_string(params, &["property", "name"], "property")?;
        if !resource.property_names.iter().any(|name| name == property) {
            return Err(ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("unknown object property '{property}'"),
            ));
        }
        let offset = bounded_offset(params, "offset")?;
        let limit = bounded_limit(params, 200)?;
        let total = resource.features.len();
        let values = resource
            .features
            .iter()
            .enumerate()
            .skip(offset)
            .take(limit)
            .map(|(index, feature)| {
                json!({
                    "index": index,
                    "id": feature.id,
                    "value": resource.property_value(index, property).unwrap_or(Value::Null),
                })
            })
            .collect::<Vec<_>>();
        let mut response = json!({
            "property": property,
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": offset.saturating_add(values.len()) < total,
            "values": values,
        });
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn resolve_object_target(
        &self,
        params: &Value,
    ) -> Result<ObjectTarget, ControlError> {
        let dataset = self.dataset()?;
        match params
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("segmentation_objects")
        {
            "objects" | "segmentation_objects" | "primary" => Ok(ObjectTarget::Primary),
            "spatial_shape" => {
                let id = params
                    .get("layer_id")
                    .or_else(|| params.get("id"))
                    .and_then(Value::as_u64)
                    .ok_or_else(|| invalid("target='spatial_shape' requires numeric layer_id"))?;
                dataset
                    .secondary_object_layers
                    .contains_key(&id)
                    .then_some(ObjectTarget::SpatialShape(id))
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::ResourceNotFound,
                            format!("spatial shape layer {id} was not found or has no objects"),
                        )
                    })
            }
            "active" => {
                let viewport = &dataset.workspace.active().state;
                let active = viewport.native_layers.active_layer_id().unwrap_or_default();
                if active == "segmentation_objects" {
                    return dataset
                        .object_resource
                        .is_some()
                        .then_some(ObjectTarget::Primary)
                        .ok_or_else(|| {
                            ControlError::new(
                                ControlErrorKind::NotReady,
                                "active segmentation object layer is empty",
                            )
                        });
                }
                if let Some(id) = active
                    .strip_prefix("spatial_shape:")
                    .and_then(|value| value.parse::<u64>().ok())
                    && dataset.secondary_object_layers.contains_key(&id)
                {
                    return Ok(ObjectTarget::SpatialShape(id));
                }
                let primary_visible = viewport
                    .objects
                    .get("visible")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if primary_visible && dataset.object_resource.is_some() {
                    Ok(ObjectTarget::Primary)
                } else {
                    Err(ControlError::new(
                        ControlErrorKind::NotReady,
                        "active layer does not provide selectable objects in the current view",
                    ))
                }
            }
            target => Err(invalid(format!(
                "unknown object selection target '{target}'"
            ))),
        }
    }

    pub(super) fn object_resource_for_target(
        &self,
        target: ObjectTarget,
        method: &str,
    ) -> Result<&ControlObjectResource, ControlError> {
        let dataset = self.dataset()?;
        match target {
            ObjectTarget::Primary => dataset.object_resource.as_deref(),
            ObjectTarget::SpatialShape(id) => dataset
                .secondary_object_layers
                .get(&id)
                .map(|layer| layer.resource.as_ref()),
        }
        .ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                format!("{method} requires object data to be loaded"),
            )
            .with_data(json!({
                "method": method,
                "target": target.response_name(),
                "layer_id": match target { ObjectTarget::SpatialShape(id) => Some(id), _ => None },
                "required_readiness": ["object_resource"],
            }))
        })
    }

    pub(super) fn object_resource_arc_for_target(
        &self,
        target: ObjectTarget,
        method: &str,
    ) -> Result<Arc<ControlObjectResource>, ControlError> {
        Ok(Arc::new(
            self.object_resource_for_target(target, method)?.clone(),
        ))
    }

    pub(super) fn object_resource_generation_for_target(
        &self,
        target: ObjectTarget,
    ) -> Result<u64, ControlError> {
        match target {
            ObjectTarget::Primary => Ok(self.installed_object_resource_generation),
            ObjectTarget::SpatialShape(id) => self
                .dataset()?
                .secondary_object_layers
                .get(&id)
                .map(|layer| layer.generation)
                .ok_or_else(|| object_target_not_found(ObjectTarget::SpatialShape(id))),
        }
    }

    pub(super) fn object_selection_for_target(
        &self,
        target: ObjectTarget,
    ) -> Result<&ObjectSelectionModel, ControlError> {
        let dataset = self.dataset()?;
        match target {
            ObjectTarget::Primary => Ok(&dataset.object_selection),
            ObjectTarget::SpatialShape(id) => dataset
                .secondary_object_layers
                .get(&id)
                .map(|layer| &layer.selection)
                .ok_or_else(|| object_target_not_found(ObjectTarget::SpatialShape(id))),
        }
    }

    pub(super) fn object_selection_mut_for_target(
        &mut self,
        target: ObjectTarget,
    ) -> Result<&mut ObjectSelectionModel, ControlError> {
        let dataset = self.dataset_mut()?;
        match target {
            ObjectTarget::Primary => Ok(&mut dataset.object_selection),
            ObjectTarget::SpatialShape(id) => dataset
                .secondary_object_layers
                .get_mut(&id)
                .map(|layer| &mut layer.selection)
                .ok_or_else(|| object_target_not_found(ObjectTarget::SpatialShape(id))),
        }
    }

    pub(super) fn decorate_object_target(value: &mut Value, target: ObjectTarget) {
        let Some(object) = value.as_object_mut() else {
            return;
        };
        object.insert(
            "target".to_string(),
            Value::String(target.response_name().to_string()),
        );
        if let ObjectTarget::SpatialShape(id) = target {
            object.insert("layer_id".to_string(), json!(id));
        }
    }

    pub(super) fn object_selection_get(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource = self.object_resource_for_target(target, "viewer.objects.get_selection")?;
        let mut response = json!({
            "selection":self.object_selection_for_target(target)?.snapshot(Some(resource), limit),
        });
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_clear(&mut self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource =
            self.object_resource_arc_for_target(target, "viewer.objects.clear_selection")?;
        let mut response = self
            .object_selection_mut_for_target(target)?
            .clear(Some(resource.as_ref()), limit);
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_select_ids(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource =
            self.object_resource_arc_for_target(target, "viewer.objects.selection.select_ids")?;
        let mut response = self.object_selection_mut_for_target(target)?.select_ids(
            resource.as_ref(),
            params,
            limit,
        )?;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_select_filtered(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        if params.get("filter_query").is_some() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "standalone object filter selection must be evaluated by a resource worker",
            ));
        }
        let limit = bounded_limit(params, 200)?;
        let resource = self
            .object_resource_arc_for_target(target, "viewer.objects.selection.select_filtered")?;
        let dataset = self.dataset()?;
        let explicit_viewport = params.get("viewport_id").and_then(Value::as_str);
        let use_all = params
            .get("use_all_objects")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let use_active = params
            .get("use_active_viewport_filter")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let source_count = usize::from(explicit_viewport.is_some())
            + usize::from(use_all)
            + usize::from(use_active);
        if source_count > 1 {
            return Err(invalid("select_filtered accepts exactly one filter source"));
        }
        if source_count == 0 && dataset.workspace.len() > 1 {
            return Err(invalid(
                "multi-viewport filtered selection requires viewport_id, filter_query, use_all_objects=true, or use_active_viewport_filter=true",
            ));
        }
        let (indices, revision) = if use_all {
            (None, 0)
        } else {
            let slot = if let Some(id) = explicit_viewport {
                let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
                dataset.workspace.get(&id).ok_or_else(|| not_found(&id))?
            } else {
                dataset.workspace.active()
            };
            let (indices, active, revision) = slot
                .state
                .object_filter_state(target)
                .ok_or_else(|| object_target_not_found(target))?;
            (active.then(|| indices.as_ref().clone()), revision)
        };
        let mut response = self
            .object_selection_mut_for_target(target)?
            .select_filtered(
                resource.as_ref(),
                indices.as_deref(),
                revision,
                params,
                limit,
            )?;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(crate) fn begin_object_selection_filter_evaluation(
        &mut self,
        params: &Value,
    ) -> Result<
        (
            u64,
            u64,
            u64,
            u64,
            ObjectTarget,
            Arc<ControlObjectResource>,
            Value,
            String,
            usize,
        ),
        ControlError,
    > {
        let target = self.resolve_object_target(params)?;
        let query = params
            .get("filter_query")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("filter_query must be a string"))?;
        let conflicting_source = params.get("viewport_id").is_some()
            || params
                .get("use_all_objects")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || params
                .get("use_active_viewport_filter")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        if conflicting_source {
            return Err(invalid("select_filtered accepts exactly one filter source"));
        }
        let mode = params
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("replace");
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return Err(invalid(
                "selection mode must be replace, add, remove, or toggle",
            ));
        }
        let limit = bounded_limit(params, 200)?;
        let resource = self
            .object_resource_arc_for_target(target, "viewer.objects.selection.select_filtered")?;
        let resource_generation = self.object_resource_generation_for_target(target)?;
        let selection_generation = self.object_selection_for_target(target)?.generation();
        self.object_selection_filter_operation_generation = self
            .object_selection_filter_operation_generation
            .wrapping_add(1)
            .max(1);
        let operation_generation = self.object_selection_filter_operation_generation;
        self.pending_object_selection_filters
            .insert(target, operation_generation);
        let operation_scope = format!("selection:{}", target.layer_id());
        self.readiness.begin_scoped(
            OperationKind::ObjectFilter,
            &operation_scope,
            operation_generation,
            "Evaluating object selection filter",
        );
        Ok((
            self.document_generation,
            resource_generation,
            selection_generation,
            operation_generation,
            target,
            resource,
            json!({"mode":"query","query":query}),
            mode.to_string(),
            limit,
        ))
    }

    pub(crate) fn fail_object_selection_filter_for_generation(
        &mut self,
        target: ObjectTarget,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if self.pending_object_selection_filters.get(&target).copied() == Some(generation) {
            self.pending_object_selection_filters.remove(&target);
            self.readiness.fail_scoped(
                OperationKind::ObjectFilter,
                &format!("selection:{}", target.layer_id()),
                generation,
                message,
            );
            return true;
        }
        false
    }

    pub(crate) fn cancel_object_selection_filter_for_generation(
        &mut self,
        target: ObjectTarget,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if self.pending_object_selection_filters.get(&target).copied() == Some(generation) {
            self.pending_object_selection_filters.remove(&target);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                &format!("selection:{}", target.layer_id()),
                generation,
                message,
            );
            return true;
        }
        false
    }

    pub(crate) fn install_object_selection_filter_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        selection_generation: u64,
        operation_generation: u64,
        target: ObjectTarget,
        result: ControlObjectFilterResult,
        mode: &str,
        limit: usize,
    ) -> Option<Value> {
        if self.pending_object_selection_filters.get(&target).copied() != Some(operation_generation)
        {
            return None;
        }
        if document_generation != self.document_generation
            || self.object_resource_generation_for_target(target).ok()? != resource_generation
            || self.object_selection_for_target(target).ok()?.generation() != selection_generation
        {
            self.pending_object_selection_filters.remove(&target);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                &format!("selection:{}", target.layer_id()),
                operation_generation,
                "Object selection filter superseded by newer state",
            );
            return None;
        }
        self.pending_object_selection_filters.remove(&target);
        let resource = self
            .object_resource_arc_for_target(target, "viewer.objects.selection.select_filtered")
            .ok()?;
        let params = json!({"mode":mode});
        let mut response = self
            .object_selection_mut_for_target(target)
            .ok()?
            .select_filtered(
                resource.as_ref(),
                Some(result.matching_indices.as_ref()),
                operation_generation,
                &params,
                limit,
            )
            .ok()?;
        Self::decorate_object_target(&mut response, target);
        self.readiness.finish_scoped(
            OperationKind::ObjectFilter,
            &format!("selection:{}", target.layer_id()),
            operation_generation,
            "Object selection filter ready",
        );
        Some(response)
    }

    pub(super) fn object_selection_query_rect(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let rect = self.object_world_rect(params)?;
        let resource = self.object_resource_for_target(target, "viewer.objects.query_rect")?;
        let visible = self.object_query_filter(params, target)?;
        let selection = self.object_selection_for_target(target)?;
        let mut response = json!({
            "query":selection.query_rect(resource, rect, visible.as_deref(), limit),
            "selection":selection.snapshot(Some(resource), limit),
        });
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_select_rect(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let rect = self.object_world_rect(params)?;
        let resource = self.object_resource_arc_for_target(target, "viewer.objects.select_rect")?;
        let visible = self.object_query_filter(params, target)?;
        let mut response = self.object_selection_mut_for_target(target)?.select_rect(
            resource.as_ref(),
            rect,
            visible.as_deref(),
            params,
            limit,
        )?;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_world_rect(&self, params: &Value) -> Result<[f32; 4], ControlError> {
        let Some(values) = params.get("screen_rect").and_then(Value::as_array) else {
            return parse_world_rect(params);
        };
        if values.len() != 4 {
            return Err(invalid("screen_rect must contain four finite numbers"));
        }
        let mut screen = [0.0_f32; 4];
        for (index, value) in values.iter().enumerate() {
            screen[index] = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid("screen_rect must contain four finite numbers"))?
                as f32;
        }
        let viewport = self.selection_viewport(params)?;
        let zoom = viewport.zoom.max(0.000_01);
        let screen_center = [
            viewport.screen_origin[0] + viewport.logical_size[0] * 0.5,
            viewport.screen_origin[1] + viewport.logical_size[1] * 0.5,
        ];
        let first = [
            viewport.center[0] + (screen[0] - screen_center[0]) / zoom,
            viewport.center[1] + (screen[1] - screen_center[1]) / zoom,
        ];
        let second = [
            viewport.center[0] + (screen[2] - screen_center[0]) / zoom,
            viewport.center[1] + (screen[3] - screen_center[1]) / zoom,
        ];
        Ok([
            first[0].min(second[0]),
            first[1].min(second[1]),
            first[0].max(second[0]),
            first[1].max(second[1]),
        ])
    }

    pub(super) fn object_selection_query_lasso(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let points = parse_world_points(params)?;
        let resource = self.object_resource_for_target(target, "viewer.objects.query_lasso")?;
        let visible = self.object_query_filter(params, target)?;
        let mut response = self.object_selection_for_target(target)?.query_lasso(
            resource,
            &points,
            visible.as_deref(),
            limit,
        );
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_select_lasso(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let points = parse_world_points(params)?;
        let resource =
            self.object_resource_arc_for_target(target, "viewer.objects.select_lasso")?;
        let visible = self.object_query_filter(params, target)?;
        let mut response = self.object_selection_mut_for_target(target)?.select_lasso(
            resource.as_ref(),
            &points,
            visible.as_deref(),
            params,
            limit,
        )?;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_query_view(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.resolve_object_target(params)?;
        let viewport = self.selection_viewport(params)?;
        let half_width = viewport.logical_size[0] / viewport.zoom.max(0.000_01) * 0.5;
        let half_height = viewport.logical_size[1] / viewport.zoom.max(0.000_01) * 0.5;
        let rect = [
            viewport.center[0] - half_width,
            viewport.center[1] - half_height,
            viewport.center[0] + half_width,
            viewport.center[1] + half_height,
        ];
        let mut scoped = params.as_object().cloned().unwrap_or_default();
        scoped.insert("world_rect".to_string(), json!(rect));
        self.object_selection_query_rect(&Value::Object(scoped))
    }

    pub(super) fn object_selection_focus(&mut self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let resource = self.object_resource_arc_for_target(target, "viewer.objects.focus.set")?;
        let (response, bounds) = self
            .object_selection_mut_for_target(target)?
            .focus(resource.as_ref(), params)?;
        if let Some(bounds) = bounds {
            self.fit_selection_bounds(params, bounds)?;
        }
        let mut response = response;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_clear_focus(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let mut response = self.object_selection_mut_for_target(target)?.clear_focus();
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn object_selection_replace(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let limit = bounded_limit(params, 200)?;
        let resource =
            self.object_resource_arc_for_target(target, "viewer.objects.selection.state.replace")?;
        let mut response = self
            .object_selection_mut_for_target(target)?
            .replace_transaction(Some(resource.as_ref()), params, limit)?;
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(super) fn selection_viewport(
        &self,
        params: &Value,
    ) -> Result<&ViewportModel, ControlError> {
        let workspace = &self.dataset()?.workspace;
        if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
            return workspace
                .get(&id)
                .map(|slot| &slot.state)
                .ok_or_else(|| not_found(&id));
        }
        Ok(&workspace.active().state)
    }

    pub(super) fn object_query_filter(
        &self,
        params: &Value,
        target: ObjectTarget,
    ) -> Result<Option<Vec<usize>>, ControlError> {
        let workspace = &self.dataset()?.workspace;
        let slot = if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            let id = ViewportId::new(id).map_err(|error| invalid(error.to_string()))?;
            workspace.get(&id).ok_or_else(|| not_found(&id))?
        } else {
            workspace.active()
        };
        let (indices, active, _) = slot
            .state
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        Ok(active.then(|| indices.as_ref().clone()))
    }

    pub(super) fn fit_selection_bounds(
        &mut self,
        params: &Value,
        bounds: [f32; 4],
    ) -> Result<(), ControlError> {
        if !params.get("fit").and_then(Value::as_bool).unwrap_or(true) {
            return Ok(());
        }
        let id = if let Some(id) = params.get("viewport_id").and_then(Value::as_str) {
            ViewportId::new(id).map_err(|error| invalid(error.to_string()))?
        } else {
            self.dataset()?.workspace.active_id().clone()
        };
        let dataset = self.dataset_mut()?;
        let links = dataset.workspace.links();
        let target = dataset
            .workspace
            .get_mut(&id)
            .ok_or_else(|| not_found(&id))?;
        let [x0, y0, x1, y1] = bounds;
        target.state.center = [(x0 + x1) * 0.5, (y0 + y1) * 0.5];
        let width = (x1 - x0).abs().max(32.0);
        let height = (y1 - y0).abs().max(32.0);
        target.state.zoom = ((target.state.logical_size[0] / width)
            .min(target.state.logical_size[1] / height)
            * 0.84)
            .clamp(0.000_01, 5000.0);
        let state = target.state.clone();
        let _ = dataset.workspace.bump_navigation_revision(&id);
        if links.camera {
            propagate_camera(&mut dataset.workspace, &id, &state);
        }
        Ok(())
    }

    pub(crate) fn begin_object_filter_evaluation(
        &mut self,
        params: &Value,
    ) -> Result<
        (
            u64,
            u64,
            u64,
            String,
            ObjectTarget,
            u64,
            Arc<ControlObjectResource>,
            Value,
        ),
        ControlError,
    > {
        let object_target = self.resolve_object_target(params)?;
        let scoped_params = if params.get("viewport_id").is_some() || params.get("id").is_some() {
            params.clone()
        } else {
            self.active_scoped_params(params)?
        };
        self.check_viewport_revision(&scoped_params)?;
        let viewport_id = Self::viewport_id(&scoped_params)?;
        let (presentation_revision, resource) = {
            let dataset = self.dataset()?;
            let viewport = dataset
                .workspace
                .get(&viewport_id)
                .ok_or_else(|| not_found(&viewport_id))?;
            let resource =
                self.object_resource_arc_for_target(object_target, "viewer.objects.set_filter")?;
            (viewport.presentation_revision, resource)
        };
        let resource_generation = self.object_resource_generation_for_target(object_target)?;
        self.object_filter_operation_generation = self
            .object_filter_operation_generation
            .wrapping_add(1)
            .max(1);
        let operation_generation = self.object_filter_operation_generation;
        self.pending_object_filters
            .insert((viewport_id.clone(), object_target), operation_generation);
        let operation_scope = format!("{}:{}", viewport_id.as_str(), object_target.layer_id());
        self.readiness.begin_scoped(
            OperationKind::ObjectFilter,
            &operation_scope,
            operation_generation,
            format!("Evaluating object filter for {}", viewport_id.as_str()),
        );
        Ok((
            self.document_generation,
            resource_generation,
            operation_generation,
            viewport_id.as_str().to_string(),
            object_target,
            presentation_revision,
            resource,
            scoped_params,
        ))
    }

    pub(crate) fn fail_object_filter_for_generation(
        &mut self,
        viewport_id: &str,
        target: ObjectTarget,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let Ok(viewport_id) = ViewportId::new(viewport_id) else {
            return false;
        };
        let key = (viewport_id.clone(), target);
        if self.pending_object_filters.get(&key).copied() != Some(operation_generation) {
            return false;
        }
        self.pending_object_filters.remove(&key);
        self.readiness.fail_scoped(
            OperationKind::ObjectFilter,
            &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
            operation_generation,
            message,
        );
        true
    }

    pub(crate) fn cancel_object_filter_for_generation(
        &mut self,
        viewport_id: &str,
        target: ObjectTarget,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        let Ok(viewport_id) = ViewportId::new(viewport_id) else {
            return false;
        };
        let key = (viewport_id.clone(), target);
        if self.pending_object_filters.get(&key).copied() != Some(operation_generation) {
            return false;
        }
        self.pending_object_filters.remove(&key);
        self.readiness.cancel_scoped(
            OperationKind::ObjectFilter,
            &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
            operation_generation,
            message,
        );
        true
    }

    pub(crate) fn install_object_filter_for_generation(
        &mut self,
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: &str,
        target: ObjectTarget,
        expected_presentation_revision: u64,
        result: ControlObjectFilterResult,
    ) -> Option<Value> {
        let viewport_id = ViewportId::new(viewport_id).ok()?;
        let key = (viewport_id.clone(), target);
        if self.pending_object_filters.get(&key).copied() != Some(operation_generation) {
            return None;
        }
        if document_generation != self.document_generation
            || self.object_resource_generation_for_target(target).ok()? != resource_generation
        {
            self.pending_object_filters.remove(&key);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
                operation_generation,
                "Object filter superseded by newer resource state",
            );
            return None;
        }
        let (total_count, presentation_matches) = {
            let dataset = self.dataset.as_ref()?;
            let viewport = dataset.workspace.get(&viewport_id)?;
            (
                self.object_resource_for_target(target, "viewer.objects.set_filter")
                    .ok()?
                    .features
                    .len(),
                viewport.presentation_revision == expected_presentation_revision,
            )
        };
        if !presentation_matches {
            self.pending_object_filters.remove(&key);
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
                operation_generation,
                "Object filter superseded by newer viewport presentation",
            );
            return None;
        }
        self.pending_object_filters.remove(&key);
        let dataset = self.dataset.as_mut()?;
        let active_before = dataset.workspace.active().state.clone();
        let viewport = dataset.workspace.get_mut(&viewport_id)?;
        viewport.state.replace_object_filter_state(
            target,
            result.model,
            result.matching_indices,
            result.active,
        )?;
        let snapshot =
            object_filter_snapshot_for_target(&viewport.state, target, total_count).ok()?;
        let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        let active_changed =
            presentation_changed(&active_before, &dataset.workspace.active().state);
        let response = viewport_response(
            &dataset.workspace,
            &viewport_id,
            snapshot,
            vec![viewport_id.clone()],
            active_changed,
        );
        self.readiness.finish_scoped(
            OperationKind::ObjectFilter,
            &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
            operation_generation,
            "Object filter ready",
        );
        Some(response)
    }

    pub(super) fn get_object_filter(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let viewport_id = Self::viewport_id(params)?;
        let dataset = self.dataset()?;
        let viewport = dataset
            .workspace
            .get(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        Ok(viewport_response(
            &dataset.workspace,
            &viewport_id,
            object_filter_snapshot_for_target(
                &viewport.state,
                target,
                self.object_resource_for_target(target, "viewer.objects.get_filter")?
                    .features
                    .len(),
            )?,
            vec![viewport_id.clone()],
            false,
        ))
    }

    pub(super) fn clear_object_filter(&mut self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let viewport_id = Self::viewport_id(params)?;
        let total_count = self
            .object_resource_for_target(target, "viewer.objects.clear_filter")?
            .features
            .len();
        if let Some(generation) = self
            .pending_object_filters
            .remove(&(viewport_id.clone(), target))
        {
            self.readiness.cancel_scoped(
                OperationKind::ObjectFilter,
                &format!("{}:{}", viewport_id.as_str(), target.layer_id()),
                generation,
                "Object filter cleared",
            );
        }
        let workspace = &mut self.dataset_mut()?.workspace;
        let active_before = workspace.active().state.clone();
        let viewport = workspace
            .get_mut(&viewport_id)
            .ok_or_else(|| not_found(&viewport_id))?;
        viewport
            .state
            .replace_object_filter_state(
                target,
                default_object_filter_model(),
                Arc::new(Vec::new()),
                false,
            )
            .ok_or_else(|| object_target_not_found(target))?;
        let snapshot = object_filter_snapshot_for_target(&viewport.state, target, total_count)?;
        let _ = workspace.bump_presentation_revision(&viewport_id);
        let active_changed = presentation_changed(&active_before, &workspace.active().state);
        Ok(viewport_response(
            workspace,
            &viewport_id,
            snapshot,
            vec![viewport_id.clone()],
            active_changed,
        ))
    }
}
