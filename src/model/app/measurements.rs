//! Measurement configuration, worker generations, and result publication.

use super::*;

impl AppModel {
    pub fn measurement_generation(&self) -> u64 {
        self.measurement.generation()
    }

    pub fn measurement_projection_state(&self) -> Value {
        self.measurement_snapshot(&json!({}))
            .unwrap_or_else(|_| json!({}))
    }

    pub(crate) fn configure_measurement(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.resolve_object_target(params)?;
        let levels = self.dataset()?.descriptor.levels.len();
        self.measurement.configure(params, levels)?;
        self.measurement_snapshot(params)
    }

    pub(crate) fn prepare_measurement(
        &mut self,
        params: &Value,
    ) -> Result<MeasurementSpec, ControlError> {
        let target = self.resolve_object_target(params)?;
        self.configure_measurement(params)?;
        let dataset = self.dataset()?;
        if dataset.descriptor.render_kind == DatasetRenderKind::LabelMask {
            return Err(invalid(
                "image measurements are unavailable for label-mask root datasets",
            ));
        }
        if dataset.plane_extents[0] > 1 {
            return Err(invalid(
                "image measurements are currently unavailable for OME-Zarr z-stacks",
            ));
        }
        let resource = self.object_resource_arc_for_target(target, "viewer.measurements.start")?;
        if resource
            .features
            .iter()
            .all(|feature| feature.polygons_world.is_empty())
        {
            return Err(invalid(
                "bulk polygon measurements are unavailable for point-only object layers",
            ));
        }
        let viewport = &dataset.workspace.active().state;
        let (filter_indices, filter_active, _) = viewport
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        let target_indices = if self.measurement.filtered_only && filter_active {
            filter_indices.as_ref().clone()
        } else {
            (0..resource.features.len()).collect()
        };
        if target_indices.is_empty() {
            return Err(invalid("no target cells available for measurement"));
        }
        let document_generation = self.document_generation;
        let resource_generation = self.object_resource_generation_for_target(target)?;
        let operation_generation = self.measurement.begin(target_indices.len());
        self.readiness.begin(
            OperationKind::Measurement,
            operation_generation,
            "Measuring object intensities",
        );
        Ok(MeasurementSpec {
            document_generation,
            resource_generation,
            operation_generation,
            target,
            level: self.measurement.level,
            metric: self.measurement.metric,
            prefix: self.measurement.prefix.clone(),
            resource,
            target_indices: Arc::new(target_indices),
        })
    }

    pub(crate) fn install_measurement(
        &mut self,
        spec: &MeasurementSpec,
        resource: ControlObjectResource,
        measured: usize,
    ) -> Option<Value> {
        if spec.document_generation != self.document_generation
            || self.object_resource_generation_for_target(spec.target).ok()
                != Some(spec.resource_generation)
            || !self.measurement.finish(spec.operation_generation, measured)
        {
            return None;
        }
        self.object_resource_generation = self.object_resource_generation.wrapping_add(1).max(1);
        let installed_generation = self.object_resource_generation;
        match spec.target {
            ObjectTarget::Primary => {
                self.installed_object_resource_generation = installed_generation;
                self.dataset.as_mut()?.object_resource = Some(Arc::new(resource));
            }
            ObjectTarget::SpatialShape(id) => {
                let layer = self
                    .dataset
                    .as_mut()?
                    .secondary_object_layers
                    .get_mut(&id)?;
                layer.generation = installed_generation;
                layer.resource = Arc::new(resource);
            }
        }
        self.readiness.finish(
            OperationKind::Measurement,
            spec.operation_generation,
            "Measurements ready",
        );
        Some(
            json!({"started":true,"completed":true,"measurement":self.measurement_snapshot(&object_target_params(spec.target)).ok()?}),
        )
    }

    pub(crate) fn fail_measurement(
        &mut self,
        spec: &MeasurementSpec,
        message: impl Into<String>,
    ) -> bool {
        if spec.document_generation != self.document_generation
            || self.object_resource_generation_for_target(spec.target).ok()
                != Some(spec.resource_generation)
        {
            return false;
        }
        let message = message.into();
        if !self
            .measurement
            .fail(spec.operation_generation, message.clone())
        {
            return false;
        }
        self.readiness.fail(
            OperationKind::Measurement,
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn cancel_measurement(&mut self, params: &Value) -> Result<Value, ControlError> {
        self.resolve_object_target(params)?;
        self.dataset()?;
        let generation = self.measurement.generation;
        let cancelled = self.measurement.cancel();
        if cancelled {
            self.readiness.cancel(
                OperationKind::Measurement,
                generation,
                "Measurement cancelled",
            );
        }
        Ok(json!({"cancelled":cancelled,"status":self.measurement.status}))
    }

    pub(crate) fn object_export_columns_snapshot(
        &self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let resource = self.object_resource_for_target(target, "exports.objects.columns")?;
        let columns = object_export_columns(resource, self.analysis_state_for_target(target));
        let mut response = json!({"columns":columns,"total":columns.len()});
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(crate) fn object_export_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        self.object_resource_for_target(target, "exports.objects.get_state")?;
        let mut response = self.object_export.snapshot();
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }
}
