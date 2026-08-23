use super::*;

impl AppModel {
    pub(crate) fn prepare_segmentation_geojson_load(
        &mut self,
        params: &Value,
    ) -> Result<SegmentationGeoJsonLoadSpec, ControlError> {
        let document_generation = self.document_generation;
        let spec = self
            .dataset_mut()?
            .segmentation_geojson
            .prepare_load(document_generation, params)?;
        self.readiness.begin(
            OperationKind::SegmentationGeoJson,
            spec.operation_generation,
            format!("Loading segmentation GeoJSON {}", spec.path.display()),
        );
        Ok(spec)
    }

    pub(crate) fn finish_segmentation_geojson_load(
        &mut self,
        spec: &SegmentationGeoJsonLoadSpec,
        resource: ControlSegmentationGeoJsonResource,
    ) -> Option<Value> {
        self.dataset()
            .ok()?
            .segmentation_geojson
            .ensure_current_document(self.document_generation, spec)
            .ok()?;
        if !self
            .dataset_mut()
            .ok()?
            .segmentation_geojson
            .finish_load(spec, resource)
        {
            return None;
        }
        self.readiness.finish(
            OperationKind::SegmentationGeoJson,
            spec.operation_generation,
            "Segmentation GeoJSON ready",
        );
        Some(self.segmentation_geojson_snapshot().ok()?)
    }

    pub(crate) fn fail_segmentation_geojson_load(
        &mut self,
        spec: &SegmentationGeoJsonLoadSpec,
        message: impl Into<String>,
    ) -> bool {
        let message = message.into();
        if self.document_generation != spec.document_generation
            || !self
                .dataset_mut()
                .is_ok_and(|dataset| dataset.segmentation_geojson.fail_load(spec, &message))
        {
            return false;
        }
        self.readiness.fail(
            OperationKind::SegmentationGeoJson,
            spec.operation_generation,
            message,
        )
    }

    pub(crate) fn segmentation_geojson_snapshot(&self) -> Result<Value, ControlError> {
        Ok(json!({
            "mode":"single",
            "source":self.dataset()?.segmentation_geojson.snapshot(),
        }))
    }

    pub(crate) fn segmentation_geojson_resource(
        &self,
    ) -> Option<Arc<ControlSegmentationGeoJsonResource>> {
        self.dataset
            .as_ref()
            .and_then(|dataset| dataset.segmentation_geojson.resource())
    }

    pub(crate) fn clear_segmentation_geojson(&mut self) -> Result<Value, ControlError> {
        let result = self.dataset_mut()?.segmentation_geojson.clear();
        self.readiness.cancel_kind_pending(
            OperationKind::SegmentationGeoJson,
            "Segmentation GeoJSON source cleared",
        );
        Ok(json!({"mode":"single","result":result}))
    }
}
