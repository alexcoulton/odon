//! Project persistence generations, canonical installation, and intensity specs.

use super::*;

impl AppModel {
    pub fn bootstrap_project_from_renderer(&mut self, snapshot: ProjectModelSnapshot) -> bool {
        if self.project_initialized {
            return false;
        }
        self.project.replace(snapshot);
        self.project_initialized = true;
        true
    }

    pub fn begin_project_operation(&mut self, description: impl Into<String>) -> u64 {
        self.cancel_pending_deep_link_apply("Superseded by project transaction");
        if self.project_roi_open_pending {
            let generation = self.project_roi_open_generation;
            self.project_roi_open_pending = false;
            self.readiness.cancel(
                OperationKind::ProjectRoiOpen,
                generation,
                "Superseded by project transaction",
            );
        }
        self.project_operation_generation =
            self.project_operation_generation.wrapping_add(1).max(1);
        self.project_operation_pending = true;
        self.readiness.begin(
            OperationKind::ProjectIo,
            self.project_operation_generation,
            description,
        );
        self.project_operation_generation
    }

    pub fn project_operation_is_current(&self, generation: u64) -> bool {
        self.project_operation_pending
            && generation == self.project_operation_generation
            && self
                .readiness
                .is_pending(OperationKind::ProjectIo, generation)
    }

    pub fn finish_project_operation_for_generation(&mut self, generation: u64) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        true
    }

    pub fn replace_project_rois_from_samplesheet_for_generation(
        &mut self,
        generation: u64,
        rois: Vec<crate::data::project_config::ProjectRoi>,
    ) -> Result<Option<Value>, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(None);
        }
        let response = self.project.replace_rois_from_samplesheet(rois)?;
        self.project_initialized = true;
        self.finish_project_operation_for_generation(generation);
        Ok(Some(response))
    }

    pub fn add_discovered_project_roots_for_generation(
        &mut self,
        generation: u64,
        roots: Vec<PathBuf>,
    ) -> Result<Option<(usize, Value)>, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(None);
        }
        let response = self.project.add_discovered_roots(roots)?;
        self.project_initialized = true;
        self.finish_project_operation_for_generation(generation);
        Ok(Some(response))
    }

    pub fn install_project_for_generation(
        &mut self,
        generation: u64,
        path: PathBuf,
        config: crate::data::project_config::ProjectConfig,
        state: Value,
    ) -> Result<bool, ControlError> {
        if !self.project_operation_is_current(generation) {
            return Ok(false);
        }
        self.project.install_loaded(path, config, state)?;
        self.project_initialized = true;
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        Ok(true)
    }

    pub fn finish_project_save_for_generation(
        &mut self,
        generation: u64,
        path: PathBuf,
        saved_config_generation: u64,
    ) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project.mark_saved(path, saved_config_generation);
        self.project_operation_pending = false;
        self.readiness
            .finish(OperationKind::ProjectIo, generation, "Ready");
        true
    }

    pub fn fail_project_operation(&mut self, generation: u64, message: impl Into<String>) -> bool {
        if !self.project_operation_is_current(generation) {
            return false;
        }
        self.project_operation_pending = false;
        self.readiness
            .fail(OperationKind::ProjectIo, generation, message);
        true
    }

    pub fn project_persistence_payload(&self) -> Result<(Value, u64), ControlError> {
        self.project.persistence_payload()
    }

    pub fn update_project_manifest(&mut self, resources: Vec<Value>, layers: Vec<Value>) -> bool {
        self.project.update_manifest(resources, layers)
    }

    pub fn channel_intensity_spec(
        &self,
        dataset: &OmeZarrDataset,
        params: &Value,
    ) -> Result<ChannelIntensitySpec, ControlError> {
        let viewport = &self.dataset()?.workspace.active().state;
        let channel_index = if params.as_object().is_some_and(|object| !object.is_empty())
            && channel_selector_from_params(params).is_ok()
        {
            resolve_channel(&viewport.channels, channel_selector_from_params(params)?)?
        } else {
            viewport.active_channel
        };
        let channel = viewport
            .channels
            .get(channel_index)
            .ok_or_else(|| invalid(format!("channel index {channel_index} is out of range")))?;
        let level0 = dataset
            .levels
            .first()
            .ok_or_else(|| invalid("dataset has no pyramid levels"))?;
        let requested_level = params
            .get("level")
            .and_then(Value::as_u64)
            .and_then(|level| usize::try_from(level).ok());
        let level_index = requested_level
            .unwrap_or_else(|| dataset.levels.len().saturating_sub(1))
            .min(dataset.levels.len().saturating_sub(1));
        let level = dataset
            .levels
            .get(level_index)
            .ok_or_else(|| invalid(format!("level {level_index} is out of range")))?;
        let (vertical, horizontal, slice_dimension) = match viewport.plane_mode.as_str() {
            "xy" => (dataset.dims.y, dataset.dims.x, dataset.dims.z),
            "xz" => (
                dataset
                    .dims
                    .z
                    .ok_or_else(|| invalid("current view plane has no display axes"))?,
                dataset.dims.x,
                Some(dataset.dims.y),
            ),
            "yz" => (
                dataset
                    .dims
                    .z
                    .ok_or_else(|| invalid("current view plane has no display axes"))?,
                dataset.dims.y,
                Some(dataset.dims.x),
            ),
            _ => return Err(invalid("current view plane has no display axes")),
        };
        if vertical >= level.shape.len() || horizontal >= level.shape.len() {
            return Err(invalid("display axes are outside image shape"));
        }
        let slice_level0 = current_plane_slice(viewport);
        let mapped_slice = slice_dimension
            .and_then(|dimension| map_level0_axis_index(level0, level, dimension, slice_level0));
        let mut ranges = Vec::with_capacity(level.shape.len());
        for dimension in 0..level.shape.len() {
            let length = level.shape[dimension];
            if Some(dimension) == dataset.dims.c {
                let selected = (channel.index as u64).min(length.saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if Some(dimension) == slice_dimension {
                let selected = mapped_slice.unwrap_or(0).min(length.saturating_sub(1));
                ranges.push(selected..selected.saturating_add(1));
            } else if dimension == vertical || dimension == horizontal {
                ranges.push(0..length);
            } else {
                ranges.push(0..length.min(1));
            }
        }
        Ok(ChannelIntensitySpec {
            channel_index,
            channel_name: channel.name.clone(),
            level_number: level.index,
            downsample: level.downsample,
            zarr_path: format!("/{}", level.path.trim_start_matches('/')),
            dtype: level.dtype.clone(),
            ranges,
        })
    }

    pub fn document_generation(&self) -> u64 {
        self.document_generation
    }
}
