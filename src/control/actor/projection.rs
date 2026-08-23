pub struct RenderDocument {
    pub generation: u64,
    pub opened: ControlOpenedDocument,
}

impl RenderDocument {
    pub fn path(&self) -> Option<&std::path::Path> {
        self.opened.descriptor.source.local_path()
    }

    pub fn dataset(&self) -> &OmeZarrDataset {
        self.opened.resource.dataset()
    }

    pub fn store(&self) -> &Arc<dyn ReadableStorageTraits> {
        self.opened.resource.store()
    }
}

pub struct RenderProjection {
    pub revision: u64,
    pub mode: ModelMode,
    pub document_generation: u64,
    pub document: Option<Arc<RenderDocument>>,
    pub project: ProjectModelSnapshot,
    pub project_object_preload: crate::model::ProjectObjectPreloadProjection,
    pub settings: AppSettings,
    pub screenshot_preferences: ScreenshotPreferences,
    pub tile_loading_policy: TileLoadingPolicy,
    pub pinned_levels: Vec<Arc<ControlPinnedLevelResource>>,
    pub channel_compute_generation: u64,
    pub channel_compute_state: Value,
    pub threshold_preview_generation: u64,
    pub threshold_preview_pending: bool,
    pub threshold_preview: Option<Arc<ControlThresholdPreviewResource>>,
    pub threshold_preview_state: Value,
    pub analysis_generation: u64,
    pub analysis_state: Value,
    pub measurement_generation: u64,
    pub measurement_state: Value,
    pub object_export_generation: u64,
    pub object_export_state: Value,
    pub mosaic_resource_generation: u64,
    pub mosaic_resource: Option<Arc<ControlMosaicResource>>,
    pub mosaic_state: Value,
    pub mosaic_object_resources: Vec<(usize, Arc<ControlObjectResource>)>,
    pub mosaic_pinned_levels: Vec<(usize, Arc<ControlPinnedLevelResource>)>,
    pub workspace: Option<Value>,
    pub object_resource: Option<Arc<ControlObjectResource>>,
    pub segmentation_geojson_resource: Option<Arc<ControlSegmentationGeoJsonResource>>,
    pub secondary_object_layers: Vec<crate::model::ControlSecondaryObjectProjection>,
    pub annotation_layers: Vec<crate::model::ControlAnnotationLayerProjection>,
    pub label_resource: Option<Arc<ControlLabelResource>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformEffect {
    CloseWindow { quit: bool },
}

pub(super) fn publish_projection(
    model: &mut AppModel,
    document: Option<Arc<RenderDocument>>,
    tx: &Sender<RenderProjection>,
    coalesce_rx: &Receiver<RenderProjection>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    let revision = model.mark_projection_dirty();
    let mut projection = RenderProjection {
        revision,
        mode: model.mode(),
        document_generation: model.document_generation(),
        document,
        project: model.project_snapshot(),
        project_object_preload: model.project_object_preload_projection(),
        settings: model.settings().clone(),
        screenshot_preferences: model.screenshot_preferences().clone(),
        tile_loading_policy: model.tile_loading_policy().clone(),
        pinned_levels: model.pinned_level_resources(),
        channel_compute_generation: model.channel_compute_generation(),
        channel_compute_state: model.channel_compute_state(),
        threshold_preview_generation: model.threshold_preview_generation(),
        threshold_preview_pending: model.threshold_preview_pending(),
        threshold_preview: model.threshold_preview_resource(),
        threshold_preview_state: model
            .threshold_preview_snapshot()
            .unwrap_or_else(|_| json!({})),
        analysis_generation: model.analysis_generation(),
        analysis_state: model.analysis_state().clone(),
        measurement_generation: model.measurement_generation(),
        measurement_state: model.measurement_projection_state(),
        object_export_generation: model.object_export_generation(),
        object_export_state: model.object_export_projection_state(),
        mosaic_resource_generation: model.mosaic_resource_generation(),
        mosaic_resource: model.mosaic_resource(),
        mosaic_state: model.mosaic_projection_state(),
        mosaic_object_resources: model.mosaic_object_resources(),
        mosaic_pinned_levels: model.mosaic_pinned_level_resources(),
        workspace: model.render_workspace_snapshot(),
        object_resource: model.object_resource(),
        segmentation_geojson_resource: model.segmentation_geojson_resource(),
        secondary_object_layers: model.secondary_object_projections(),
        annotation_layers: model.annotation_projections(),
        label_resource: model.label_resource(),
    };
    let mut coalesced = false;
    loop {
        match tx.try_send(projection) {
            Ok(()) => {
                diagnostics.projection_published(revision, coalesced);
                wake_ui();
                return;
            }
            Err(crossbeam_channel::TrySendError::Full(returned)) => {
                projection = returned;
                let _ = coalesce_rx.try_recv();
                coalesced = true;
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => return,
        }
    }
}
use super::*;
