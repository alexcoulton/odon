use super::*;

impl AppModel {
    pub(in crate::model::app) fn get_camera(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &self.dataset()?.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            control_camera_json(&slot.state),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_camera(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let workspace = &mut self.dataset_mut()?.workspace;
        let links = workspace.links();
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        let mut state = before.clone();
        if let Some(center) = params
            .get("center_world_lvl0")
            .and_then(Value::as_array)
            .filter(|v| v.len() == 2)
        {
            state.center = [
                center[0]
                    .as_f64()
                    .ok_or_else(|| invalid("camera center must be numeric"))?
                    as f32,
                center[1]
                    .as_f64()
                    .ok_or_else(|| invalid("camera center must be numeric"))?
                    as f32,
            ];
        }
        if let Some(x) = params.get("center_x").and_then(Value::as_f64) {
            if !x.is_finite() {
                return Err(invalid("camera center_x must be finite"));
            }
            state.center[0] = x as f32;
        }
        if let Some(y) = params.get("center_y").and_then(Value::as_f64) {
            if !y.is_finite() {
                return Err(invalid("camera center_y must be finite"));
            }
            state.center[1] = y as f32;
        }
        if let Some(zoom) = params
            .get("zoom_screen_per_lvl0_px")
            .or_else(|| params.get("zoom"))
            .and_then(Value::as_f64)
        {
            if !zoom.is_finite() || zoom <= 0.0 {
                return Err(invalid("zoom must be finite and greater than zero"));
            }
            state.zoom = (zoom as f32).clamp(0.000_01, 5000.0);
        }
        if !state.center.iter().all(|value| value.is_finite()) {
            return Err(invalid("camera center must be finite"));
        }
        target.state = state.clone();
        let changed = camera_changed(&before, &state);
        let _ = workspace.bump_navigation_revision(&id);
        if links.camera && changed {
            propagate_camera(workspace, &id, &state);
        }
        let affected = if links.camera && changed {
            workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = workspace.active().state.clone();
        Ok(viewport_response(
            workspace,
            &id,
            control_camera_json(&state),
            affected,
            camera_changed(&active_before, &active_after),
        ))
    }

    pub(in crate::model::app) fn fit_viewport(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let dataset = self.dataset_mut()?;
        let links = dataset.workspace.links();
        let active_before = dataset.workspace.active().state.clone();
        let target = dataset
            .workspace
            .get_mut(&id)
            .ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        fit_camera(&mut target.state, dataset.world_size);
        let state = target.state.clone();
        let changed = camera_changed(&before, &state);
        let _ = dataset.workspace.bump_navigation_revision(&id);
        if links.camera && changed {
            propagate_camera(&mut dataset.workspace, &id, &state);
        }
        let affected = if links.camera && changed {
            dataset
                .workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = dataset.workspace.active().state.clone();
        Ok(viewport_response(
            &dataset.workspace,
            &id,
            control_camera_json(&state),
            affected,
            camera_changed(&active_before, &active_after),
        ))
    }

    pub(in crate::model::app) fn get_plane(&self, params: &Value) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let dataset = self.dataset()?;
        let workspace = &dataset.workspace;
        let slot = workspace.get(&id).ok_or_else(|| not_found(&id))?;
        Ok(viewport_response(
            workspace,
            &id,
            control_plane_json(
                &slot.state,
                dataset.plane_extents,
                dataset.orthogonal_planes,
            ),
            vec![id.clone()],
            false,
        ))
    }

    pub(in crate::model::app) fn set_plane(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let id = Self::viewport_id(params)?;
        let requested_mode = params
            .get("mode")
            .and_then(Value::as_str)
            .map(normalize_plane_mode)
            .transpose()?;
        let requested_slice = params.get("slice").and_then(Value::as_u64);
        let dataset = self.dataset_mut()?;
        if requested_mode.is_some_and(|mode| mode != "xy") && !dataset.orthogonal_planes {
            return Err(invalid(format!(
                "{} view is not available for this dataset",
                requested_mode
                    .expect("checked as some")
                    .to_ascii_uppercase()
            )));
        }
        let plane_extents = dataset.plane_extents;
        let orthogonal_planes = dataset.orthogonal_planes;
        let workspace = &mut dataset.workspace;
        let links = workspace.links();
        let active_before = workspace.active().state.clone();
        let target = workspace.get_mut(&id).ok_or_else(|| not_found(&id))?;
        let before = target.state.clone();
        if let Some(mode) = requested_mode {
            target.state.plane_mode = mode.to_string();
        }
        if let Some(slice) = requested_slice {
            set_current_plane_slice(&mut target.state, slice, plane_extents);
        } else {
            clamp_current_plane_slice(&mut target.state, plane_extents);
        }
        let state = target.state.clone();
        let changed = plane_changed(&before, &state);
        let _ = workspace.bump_navigation_revision(&id);
        if links.plane && changed {
            for slot in workspace
                .viewports()
                .iter()
                .filter(|slot| slot.id != id)
                .map(|slot| slot.id.clone())
                .collect::<Vec<_>>()
            {
                if let Some(other) = workspace.get_mut(&slot) {
                    other.state.plane_mode = state.plane_mode.clone();
                    other.state.plane_slices = state.plane_slices;
                }
                let _ = workspace.bump_navigation_revision(&slot);
            }
        }
        let affected = if links.plane && changed {
            workspace
                .viewports()
                .iter()
                .map(|slot| slot.id.clone())
                .collect()
        } else {
            vec![id.clone()]
        };
        let active_after = workspace.active().state.clone();
        Ok(viewport_response(
            workspace,
            &id,
            json!({
                "changed": changed,
                "plane": control_plane_json(&state, plane_extents, orthogonal_planes),
            }),
            affected,
            plane_changed(&active_before, &active_after),
        ))
    }
}
