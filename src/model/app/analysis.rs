//! Analysis configuration, warmup, retained generations, and results.

use super::*;

impl AppModel {
    pub fn analysis_state(&self) -> &Value {
        self.analysis_state_for_target(ObjectTarget::Primary)
    }

    pub fn analysis_generation(&self) -> u64 {
        self.analysis_generation_for_target(ObjectTarget::Primary)
    }

    pub(crate) fn analysis_state_for_target(&self, target: ObjectTarget) -> &Value {
        self.analyses
            .get(&target)
            .unwrap_or_else(|| panic!("analysis model for {} is not installed", target.layer_id()))
            .state()
    }

    pub(crate) fn analysis_generation_for_target(&self, target: ObjectTarget) -> u64 {
        self.analyses
            .get(&target)
            .unwrap_or_else(|| panic!("analysis model for {} is not installed", target.layer_id()))
            .generation()
    }

    fn analysis_for_target_mut(&mut self, target: ObjectTarget) -> &mut AnalysisModel {
        self.analyses.entry(target).or_default()
    }

    pub(crate) fn analysis_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let dataset = self.dataset()?;
        let resource = self.object_resource_for_target(target, "viewer.analysis.get")?;
        let analysis = self
            .analyses
            .get(&target)
            .ok_or_else(|| object_target_not_found(target))?;
        let viewport = &dataset.workspace.active().state;
        let (filter_indices, filter_active, _) = viewport
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        let mut response = json!({
            "state":analysis.state(),
            "generation":analysis.generation(),
            "numeric_properties":numeric_object_properties(resource),
            "warmup":analysis.warmup_snapshot(),
            "active_channel":viewport.channels.get(viewport.active_channel).map(|channel| channel.name.as_str()),
            "filtered":filter_active,
            "filtered_count":if filter_active { filter_indices.len() } else { resource.features.len() },
            "object_count":resource.features.len(),
        });
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }

    pub(crate) fn set_analysis_state(&mut self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        self.object_resource_for_target(target, "viewer.analysis.set")?;
        self.analysis_for_target_mut(target).replace(params)?;
        self.analysis_snapshot(params)
    }

    pub(crate) fn prepare_analysis_resource_operation(
        &mut self,
        params: &Value,
        scope: &str,
    ) -> Result<AnalysisResourceSpec, ControlError> {
        let target = self.resolve_object_target(params)?;
        let dataset = self.dataset()?;
        let resource = self.object_resource_arc_for_target(target, "viewer.analysis")?;
        let viewport = &dataset.workspace.active().state;
        let (filter_indices, filtered, _) = viewport
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        let indices = filtered.then(|| Arc::clone(filter_indices));
        let document_generation = self.document_generation;
        let resource_generation = self.object_resource_generation_for_target(target)?;
        let operation_scope = format!("{scope}:{}", target.layer_id());
        let operation_generation = self.analysis_for_target_mut(target).begin(&operation_scope);
        self.readiness.begin_scoped(
            OperationKind::Analysis,
            &operation_scope,
            operation_generation,
            format!("Running {scope}"),
        );
        Ok(AnalysisResourceSpec {
            document_generation,
            resource_generation,
            operation_generation,
            operation_scope,
            target,
            resource,
            indices,
            filtered,
        })
    }

    pub(crate) fn analysis_operation_is_current(&self, spec: &AnalysisResourceSpec) -> bool {
        self.mode == ModelMode::Single
            && spec.document_generation == self.document_generation
            && self.object_resource_generation_for_target(spec.target).ok()
                == Some(spec.resource_generation)
            && self.analyses.get(&spec.target).is_some_and(|analysis| {
                analysis.is_current(&spec.operation_scope, spec.operation_generation)
            })
    }

    pub(crate) fn finish_analysis_operation(&mut self, spec: &AnalysisResourceSpec) -> bool {
        if !self.analysis_operation_is_current(spec) {
            return false;
        }
        self.readiness.finish_scoped(
            OperationKind::Analysis,
            &spec.operation_scope,
            spec.operation_generation,
            "Analysis ready",
        );
        true
    }

    pub(crate) fn fail_analysis_operation(
        &mut self,
        spec: &AnalysisResourceSpec,
        message: impl Into<String>,
    ) -> bool {
        if !self.analysis_operation_is_current(spec) {
            return false;
        }
        self.analysis_for_target_mut(spec.target).fail_warmup();
        self.readiness.fail_scoped(
            OperationKind::Analysis,
            &spec.operation_scope,
            spec.operation_generation,
            message,
        );
        true
    }

    pub(crate) fn begin_analysis_warmup(
        &mut self,
        params: &Value,
    ) -> Result<AnalysisResourceSpec, ControlError> {
        let spec = self.prepare_analysis_resource_operation(params, "analysis_warmup")?;
        let total = numeric_object_properties(&spec.resource).len();
        self.analysis_for_target_mut(spec.target)
            .begin_warmup(total);
        Ok(spec)
    }

    pub(crate) fn finish_analysis_warmup(
        &mut self,
        spec: &AnalysisResourceSpec,
        completed: usize,
    ) -> Option<Value> {
        if !self.finish_analysis_operation(spec) {
            return None;
        }
        let analysis = self.analysis_for_target_mut(spec.target);
        analysis.finish_warmup(completed);
        Some(analysis.warmup_snapshot())
    }

    pub(crate) fn analysis_warmup_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        self.object_resource_for_target(target, "viewer.analysis.warmup.get")?;
        Ok(self
            .analyses
            .get(&target)
            .ok_or_else(|| object_target_not_found(target))?
            .warmup_snapshot())
    }

    pub(crate) fn install_analysis_preset(
        &mut self,
        spec: &AnalysisResourceSpec,
        state: Value,
        path: &Path,
    ) -> Option<Result<Value, ControlError>> {
        if !self.finish_analysis_operation(spec) {
            return None;
        }
        if let Err(error) = self
            .analysis_for_target_mut(spec.target)
            .install_imported_state(state)
        {
            return Some(Err(error));
        }
        let analysis = self
            .analyses
            .get(&spec.target)
            .expect("analysis target remains installed");
        Some(Ok(json!({
            "imported":true,
            "path":path.to_string_lossy(),
            "call_count":analysis.state()["threshold_elements"].as_array().map_or(0, Vec::len),
        })))
    }

    pub(crate) fn measurement_snapshot(&self, params: &Value) -> Result<Value, ControlError> {
        let target = self.resolve_object_target(params)?;
        let dataset = self.dataset()?;
        let resource = self.object_resource_for_target(target, "viewer.measurements.get")?;
        let viewport = &dataset.workspace.active().state;
        let (filter_indices, filter_active, _) = viewport
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        let target_count = if self.measurement.filtered_only && filter_active {
            filter_indices.len()
        } else {
            resource.features.len()
        };
        let properties = resource
            .property_names
            .iter()
            .filter(|property| property.starts_with(&self.measurement.prefix))
            .cloned()
            .collect();
        let mut response = self
            .measurement
            .snapshot(&dataset.descriptor, target_count, properties);
        Self::decorate_object_target(&mut response, target);
        Ok(response)
    }
}
