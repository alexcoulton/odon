use super::*;
use crate::model::{AnnotationLoadResult, AnnotationLoadSpec};

impl AppModel {
    pub(crate) fn annotation_id(params: &Value) -> Result<u64, ControlError> {
        params
            .get("layer_id")
            .or_else(|| params.get("id"))
            .and_then(Value::as_u64)
            .filter(|id| *id > 0)
            .ok_or_else(|| invalid("annotation layer_id is required"))
    }
    pub(crate) fn annotation_snapshot(&self) -> Value {
        self.annotations.snapshot()
    }

    pub(crate) fn annotation_layer_snapshot(&self, id: u64) -> Result<Value, ControlError> {
        self.require_annotation_mode()?;
        self.annotations.layer_snapshot(id)
    }

    pub(crate) fn create_annotation_layer(
        &mut self,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_annotation_mode()?;
        let response = self.annotations.create(params)?;
        self.sync_annotation_native_layers();
        if let Some(id) = response.get("id").and_then(Value::as_u64)
            && let Some(dataset) = self.dataset.as_mut()
        {
            for viewport in dataset.workspace.viewports_mut() {
                let _ = viewport
                    .state
                    .native_layers
                    .set_active(&format!("annotation:{id}"));
            }
        }
        self.sync_annotations_to_project()?;
        Ok(response)
    }

    pub(crate) fn update_annotation_layer(
        &mut self,
        id: u64,
        params: &Value,
    ) -> Result<Value, ControlError> {
        self.require_annotation_mode()?;
        let response = self.annotations.update(id, params)?;
        if !response
            .get("pending")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            self.readiness.cancel_scope_pending(
                OperationKind::Annotations,
                &id.to_string(),
                "Annotation source configuration changed",
            );
        }
        self.sync_annotation_native_layers();
        self.sync_annotations_to_project()?;
        Ok(response)
    }

    pub(crate) fn delete_annotation_layer(&mut self, id: u64) -> Result<Value, ControlError> {
        self.require_annotation_mode()?;
        let response = self.annotations.delete(id)?;
        self.readiness.cancel_scope_pending(
            OperationKind::Annotations,
            &id.to_string(),
            "Annotation layer was deleted",
        );
        self.sync_annotation_native_layers();
        self.sync_annotations_to_project()?;
        Ok(response)
    }

    pub(crate) fn clear_annotation_source(&mut self, id: u64) -> Result<Value, ControlError> {
        self.require_annotation_mode()?;
        let response = self.annotations.clear_source(id)?;
        self.readiness.cancel_scope_pending(
            OperationKind::Annotations,
            &id.to_string(),
            "Annotation source was cleared",
        );
        self.sync_annotations_to_project()?;
        Ok(response)
    }

    pub(crate) fn prepare_annotation_load(
        &mut self,
        id: u64,
        params: &Value,
        load_dataset: bool,
    ) -> Result<AnnotationLoadSpec, ControlError> {
        self.require_annotation_mode()?;
        let mut spec =
            self.annotations
                .begin_load(self.document_generation, id, params, load_dataset)?;
        if spec.path.is_relative()
            && let Some(project_dir) = self
                .project
                .snapshot()
                .saved_path
                .as_deref()
                .and_then(std::path::Path::parent)
        {
            spec.path = project_dir.join(&spec.path);
        }
        self.sync_annotations_to_project()?;
        self.readiness.begin_scoped(
            OperationKind::Annotations,
            id.to_string(),
            spec.operation_generation,
            if load_dataset {
                format!("Loading annotation layer {id}")
            } else {
                format!("Inspecting annotation layer {id}")
            },
        );
        Ok(spec)
    }

    pub(crate) fn prepare_restored_annotation_loads(&mut self) -> Vec<AnnotationLoadSpec> {
        self.annotations
            .restorable_layer_ids()
            .into_iter()
            .filter_map(|id| self.prepare_annotation_load(id, &json!({}), true).ok())
            .collect()
    }

    pub(crate) fn install_annotation_load(
        &mut self,
        spec: &AnnotationLoadSpec,
        result: AnnotationLoadResult,
    ) -> Option<Value> {
        if spec.document_generation != self.document_generation {
            return None;
        }
        let response = self.annotations.finish_load(spec, result)?;
        self.readiness.finish_scoped(
            OperationKind::Annotations,
            &spec.layer_id.to_string(),
            spec.operation_generation,
            "Annotation resource ready",
        );
        self.sync_annotation_native_layers();
        let _ = self.sync_annotations_to_project();
        Some(response)
    }

    pub(crate) fn fail_annotation_load(
        &mut self,
        spec: &AnnotationLoadSpec,
        message: String,
    ) -> bool {
        if spec.document_generation != self.document_generation
            || !self.annotations.fail_load(spec, message.clone())
        {
            return false;
        }
        self.readiness.fail_scoped(
            OperationKind::Annotations,
            &spec.layer_id.to_string(),
            spec.operation_generation,
            message,
        )
    }

    pub(crate) fn restore_annotation_states(
        &mut self,
        states: Vec<crate::data::annotations::ProjectAnnotationLayerState>,
    ) -> Result<(), ControlError> {
        self.annotations.restore(states)?;
        self.sync_annotation_native_layers();
        Ok(())
    }

    fn require_annotation_mode(&self) -> Result<(), ControlError> {
        if matches!(self.mode, ModelMode::Single | ModelMode::Mosaic) {
            Ok(())
        } else {
            Err(ControlError::new(
                ControlErrorKind::WrongMode,
                "Annotations require an open dataset or mosaic",
            ))
        }
    }

    fn sync_annotation_native_layers(&mut self) {
        let states = self.annotations.states();
        if let Some(dataset) = self.dataset.as_mut() {
            for viewport in dataset.workspace.viewports_mut() {
                viewport.state.native_layers.sync_annotations(&states);
            }
        } else if self.mode == ModelMode::Mosaic {
            self.mosaic.sync_annotation_layers(&states);
        }
    }

    fn sync_annotations_to_project(&mut self) -> Result<(), ControlError> {
        let states = self.annotations.states();
        match self.mode {
            ModelMode::Single => {
                let source_key = self.dataset()?.descriptor.source.source_key();
                let mut view = self
                    .project
                    .roi_view_state_json(&source_key)
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                view.as_object_mut()
                    .ok_or_else(|| invalid("project ROI view state must be an object"))?
                    .insert("annotation_layers".to_string(), json!(states));
                self.project.set_roi_view_state_json(&source_key, view)
            }
            ModelMode::Mosaic => self.project.set_mosaic_annotation_layers(states),
            _ => Ok(()),
        }
    }
}
