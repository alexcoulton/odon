//! Mask readiness, import/export, persistence, and native-layer synchronization.

use super::*;

impl AppModel {
    pub fn mask_generation(&self) -> Result<(u64, u64), ControlError> {
        Ok((self.document_generation, self.dataset()?.masks.generation()))
    }

    pub fn begin_mask_import_operation(&mut self) -> Result<(u64, u64, u64, String), ControlError> {
        let (document_generation, mask_generation) = self.mask_generation()?;
        let (operation_generation, scope) = self.begin_mask_io_operation("import");
        self.readiness.begin_scoped(
            OperationKind::MaskIo,
            &scope,
            operation_generation,
            "Importing mask GeoJSON",
        );
        Ok((
            document_generation,
            mask_generation,
            operation_generation,
            scope,
        ))
    }

    pub fn begin_mask_export_operation(&mut self) -> Result<(u64, String), ControlError> {
        self.dataset()?;
        let (operation_generation, scope) = self.begin_mask_io_operation("export");
        self.readiness.begin_scoped(
            OperationKind::MaskIo,
            &scope,
            operation_generation,
            "Exporting mask GeoJSON",
        );
        Ok((operation_generation, scope))
    }

    pub fn begin_mask_append_operation(
        &mut self,
    ) -> Result<(u64, u64, u64, String, Vec<ProjectMaskLayer>), ControlError> {
        if self
            .readiness
            .has_pending_scoped_prefix(OperationKind::MaskIo, "append:")
        {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "a project mask append is already in progress",
            ));
        }
        let (document_generation, mask_generation) = self.mask_generation()?;
        let layers = self.dataset()?.masks.appendable_layers();
        if layers.is_empty() {
            return Err(invalid("no drawn masks to save"));
        }
        let (operation_generation, scope) = self.begin_mask_io_operation("append");
        self.readiness.begin_scoped(
            OperationKind::MaskIo,
            &scope,
            operation_generation,
            "Appending drawn masks to project GeoJSON",
        );
        Ok((
            document_generation,
            mask_generation,
            operation_generation,
            scope,
            layers,
        ))
    }

    pub fn fail_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .fail_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    pub fn cancel_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .cancel_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    pub fn finish_mask_io_for_generation(
        &mut self,
        scope: &str,
        operation_generation: u64,
        message: impl Into<String>,
    ) -> bool {
        self.readiness
            .finish_scoped(OperationKind::MaskIo, scope, operation_generation, message)
    }

    pub(super) fn begin_mask_io_operation(&mut self, direction: &str) -> (u64, String) {
        self.mask_io_operation_generation =
            self.mask_io_operation_generation.wrapping_add(1).max(1);
        let generation = self.mask_io_operation_generation;
        (generation, format!("{direction}:{generation}"))
    }

    pub fn mask_export_layers(
        &self,
        layer_id: Option<u64>,
    ) -> Result<Vec<crate::data::project_config::ProjectMaskLayer>, ControlError> {
        self.dataset()?.masks.export_layers(layer_id)
    }

    pub fn install_imported_masks_for_generation(
        &mut self,
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: &str,
        name: String,
        editable: bool,
        replace_layer_id: Option<u64>,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: PathBuf,
    ) -> Option<Value> {
        if !self.readiness.is_pending_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
        ) {
            return None;
        }
        if document_generation != self.document_generation
            || self
                .dataset
                .as_ref()
                .is_none_or(|dataset| dataset.masks.generation() != mask_generation)
        {
            self.readiness.cancel_scoped(
                OperationKind::MaskIo,
                operation_scope,
                operation_generation,
                "Mask import superseded by newer document or mask state",
            );
            return None;
        }
        let dataset = self.dataset.as_mut()?;
        let response = dataset.masks.install_imported_layer(
            name,
            editable,
            replace_layer_id,
            polygons_world,
            source_geojson,
        )?;
        Self::sync_mask_native_layers(dataset);
        self.readiness.finish_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
            "Mask import ready",
        );
        Some(response)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reconcile_appended_masks_for_generation(
        &mut self,
        document_generation: u64,
        starting_mask_generation: u64,
        operation_generation: u64,
        operation_scope: &str,
        saved_layers: &[ProjectMaskLayer],
        name: String,
        polygons_world: Vec<Vec<[f32; 2]>>,
        source_geojson: PathBuf,
    ) -> Option<Value> {
        if !self.readiness.is_pending_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
        ) {
            return None;
        }
        if document_generation != self.document_generation {
            self.readiness.finish_scoped(
                OperationKind::MaskIo,
                operation_scope,
                operation_generation,
                "Mask GeoJSON saved after the originating document was closed",
            );
            return Some(json!({
                "saved": true,
                "applied_to_current_document": false,
                "path": source_geojson.to_string_lossy(),
            }));
        }

        let source = self.dataset().ok()?.descriptor.source.clone();
        let current_mask_generation = self.dataset().ok()?.masks.generation();
        let response = self.dataset_mut().ok()?.masks.reconcile_appended_file(
            saved_layers,
            name,
            polygons_world,
            source_geojson.clone(),
        );
        let layers = self.dataset().ok()?.masks.export_layers(None).ok()?;
        self.project
            .sync_mask_layers_for_source(source, layers)
            .ok()?;
        self.project_initialized = true;
        let dataset = self.dataset_mut().ok()?;
        dataset.masks.mark_persisted();
        Self::sync_mask_native_layers(dataset);
        self.readiness.finish_scoped(
            OperationKind::MaskIo,
            operation_scope,
            operation_generation,
            "Drawn masks appended and reloaded",
        );
        Some(json!({
            "saved": true,
            "applied_to_current_document": true,
            "concurrent_mask_edits_reconciled": current_mask_generation != starting_mask_generation,
            "path": source_geojson.to_string_lossy(),
            "mask": response,
            "persistence": self.mask_persistence_state().ok()?,
        }))
    }

    pub(super) fn sync_mask_native_layers(dataset: &mut DatasetModel) {
        let projection = dataset.masks.projection_json();
        let active = projection.get("active_layer_id").and_then(Value::as_u64);
        let masks = dataset.masks.export_layers(None).unwrap_or_default();
        for viewport in dataset.workspace.viewports_mut() {
            viewport.state.native_layers.sync_masks(&masks, active);
        }
    }

    pub(super) fn mask_persistence_state(&self) -> Result<Value, ControlError> {
        let dataset = self.dataset()?;
        let local_path = dataset.descriptor.source.local_path();
        Ok(json!({
            "dirty": dataset.masks.dirty(),
            "dataset_local": local_path.is_some(),
            "dataset_path": local_path.map(|path| path.to_string_lossy().into_owned()),
            "project_path": self.project_snapshot().saved_path.map(|path| path.to_string_lossy().into_owned()),
            "live_layer_count": dataset.masks.export_layers(None)?.len(),
            "persisted_layer_count": self.project.mask_layer_count_for_source(&dataset.descriptor.source),
        }))
    }

    pub(crate) fn sync_masks_to_project(&mut self) -> Result<Value, ControlError> {
        let (source, layers) = {
            let dataset = self.dataset()?;
            if dataset.descriptor.source.local_path().is_none() {
                return Err(invalid("mask project persistence requires a local dataset"));
            }
            (
                dataset.descriptor.source.clone(),
                dataset.masks.export_layers(None)?,
            )
        };
        self.project.sync_mask_layers_for_source(source, layers)?;
        self.project_initialized = true;
        self.dataset_mut()?.masks.mark_persisted();
        Ok(json!({
            "synced": true,
            "persistence": self.mask_persistence_state()?,
        }))
    }
}
