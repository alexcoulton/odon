//! Object-export configuration, worker generations, and result publication.

use super::*;

impl AppModel {
    pub fn object_export_generation(&self) -> u64 {
        self.object_export.generation()
    }

    pub fn object_export_projection_state(&self) -> Value {
        self.object_export.snapshot()
    }

    pub(crate) fn prepare_object_export(
        &mut self,
        params: &Value,
        path: PathBuf,
        forced_format: Option<ObjectExportFormat>,
    ) -> Result<ObjectExportSpec, ControlError> {
        let target = self.resolve_object_target(params)?;
        let dataset = self.dataset()?;
        let resource = self.object_resource_arc_for_target(target, "exports.objects.start")?;
        let format = forced_format.unwrap_or_else(|| {
            match params
                .get("format")
                .and_then(Value::as_str)
                .or_else(|| path.extension().and_then(|extension| extension.to_str()))
                .unwrap_or("csv")
                .to_ascii_lowercase()
                .as_str()
            {
                "parquet" | "geoparquet" => ObjectExportFormat::GeoParquet,
                _ => ObjectExportFormat::Csv,
            }
        });
        if forced_format.is_none()
            && let Some(format_name) = params.get("format").and_then(Value::as_str)
            && !matches!(format_name, "csv" | "parquet" | "geoparquet")
        {
            return Err(invalid("format must be 'csv' or 'geoparquet'"));
        }
        let overwrite = match params.get("overwrite") {
            Some(value) => value
                .as_bool()
                .ok_or_else(|| invalid("overwrite must be a boolean"))?,
            None => false,
        };
        let available = object_export_columns(&resource, self.analysis_state_for_target(target));
        let columns = match params.get("columns") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| invalid("columns must be an array of names"))?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_string)
                        .ok_or_else(|| invalid("columns must contain strings"))
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => available.clone(),
        };
        if columns.is_empty() {
            return Err(invalid("at least one export column is required"));
        }
        let mut unique_columns = HashSet::new();
        for column in &columns {
            if !unique_columns.insert(column) {
                return Err(invalid(format!("duplicate export column '{column}'")));
            }
            if !available.contains(column) {
                return Err(invalid(format!("unknown export column '{column}'")));
            }
        }
        let scope = params.get("scope").and_then(Value::as_str).unwrap_or("all");
        let viewport = &dataset.workspace.active().state;
        let (filter_indices, filter_active, _) = viewport
            .object_filter_state(target)
            .ok_or_else(|| object_target_not_found(target))?;
        let mut row_indices = match scope {
            "all" => (0..resource.features.len()).collect::<Vec<_>>(),
            "filtered" if filter_active => filter_indices.as_ref().clone(),
            "filtered" => (0..resource.features.len()).collect::<Vec<_>>(),
            "selected" => self
                .object_selection_for_target(target)?
                .selected_indices()
                .into_iter()
                .collect::<Vec<_>>(),
            _ => return Err(invalid("scope must be 'all', 'filtered', or 'selected'")),
        };
        row_indices.sort_unstable();
        if row_indices.is_empty() {
            return Err(invalid(format!(
                "the '{scope}' export scope contains no objects"
            )));
        }
        let selected_indices = self.object_selection_for_target(target)?.selected_indices();
        let document_generation = self.document_generation;
        let resource_generation = self.object_resource_generation_for_target(target)?;
        let operation_generation = self.object_export.begin(&path, row_indices.len())?;
        self.readiness.begin(
            OperationKind::ObjectExport,
            operation_generation,
            format!("Exporting objects to {}", path.to_string_lossy()),
        );
        Ok(ObjectExportSpec {
            document_generation,
            resource_generation,
            operation_generation,
            target,
            path,
            overwrite,
            format,
            scope: scope.to_string(),
            resource,
            row_indices: Arc::new(row_indices),
            columns: Arc::new(columns),
            selected_indices: Arc::new(selected_indices),
            analysis_state: self.analysis_state_for_target(target).clone(),
        })
    }

    pub(crate) fn finish_object_export(
        &mut self,
        spec: &ObjectExportSpec,
        result: &ObjectExportResult,
    ) -> Option<Value> {
        if spec.document_generation != self.document_generation
            || self.object_resource_generation_for_target(spec.target).ok()
                != Some(spec.resource_generation)
        {
            return None;
        }
        let output = self.object_export.finish(
            spec.operation_generation,
            &spec.path,
            spec.format,
            result,
        )?;
        self.readiness.finish(
            OperationKind::ObjectExport,
            spec.operation_generation,
            "Object export complete",
        );
        Some(json!({
            "started":true,
            "completed":true,
            "request_id":spec.operation_generation,
            "path":spec.path.to_string_lossy(),
            "format":spec.format.as_str(),
            "scope":spec.scope,
            "object_count":result.object_count,
            "column_count":result.column_count,
            "bytes":result.bytes,
            "output":output,
        }))
    }

    pub(crate) fn fail_object_export(
        &mut self,
        spec: &ObjectExportSpec,
        message: impl Into<String>,
    ) -> bool {
        if spec.document_generation != self.document_generation
            || self.object_resource_generation_for_target(spec.target).ok()
                != Some(spec.resource_generation)
        {
            return false;
        }
        let message = message.into();
        if !self.object_export.fail(spec.operation_generation, &message) {
            return false;
        }
        self.readiness.fail(
            OperationKind::ObjectExport,
            spec.operation_generation,
            message,
        );
        true
    }
}
