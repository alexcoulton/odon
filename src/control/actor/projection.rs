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
    pub threshold_preview_generation: u64,
    pub threshold_preview_pending: bool,
    pub threshold_preview: Option<Arc<ControlThresholdPreviewResource>>,
    pub threshold_preview_state: Value,
    pub analysis_generation: u64,
    pub analysis_state: Value,
    pub workspace: Option<Value>,
    pub object_resource: Option<Arc<ControlObjectResource>>,
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
        threshold_preview_generation: model.threshold_preview_generation(),
        threshold_preview_pending: model.threshold_preview_pending(),
        threshold_preview: model.threshold_preview_resource(),
        threshold_preview_state: model
            .threshold_preview_snapshot()
            .unwrap_or_else(|_| json!({})),
        analysis_generation: model.analysis_generation(),
        analysis_state: model.analysis_state().clone(),
        workspace: model.render_workspace_snapshot(),
        object_resource: model.object_resource(),
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
