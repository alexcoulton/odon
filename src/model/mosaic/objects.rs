//! Mosaic object-resource loading lifecycle and projection state.

use super::*;

impl MosaicModel {
    pub(crate) fn prepare_object_load(
        &mut self,
        downsample_factor: f32,
    ) -> Result<MosaicObjectLoadSpec, ControlError> {
        self.require_resource()?;
        if !downsample_factor.is_finite() || downsample_factor <= 0.0 {
            return Err(invalid(
                "downsample_factor must be finite and greater than zero",
            ));
        }
        if self.selected_ids.is_empty() {
            return Err(invalid("Select at least one mosaic ROI first."));
        }
        let items = self
            .items
            .iter()
            .filter(|item| self.selected_ids.contains(&item.id))
            .filter_map(|item| {
                item.segmentation_path
                    .as_ref()
                    .map(|path| (item.id, path.clone()))
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "None of the selected mosaic ROIs has an object segmentation source.",
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
