pub struct RenderDocument {
    pub generation: u64,
    pub opened: OpenedDocument<OmeZarrDocumentResource>,
}

impl RenderDocument {
    pub fn path(&self) -> Option<&std::path::Path> {
        self.opened.descriptor.source.local_path()
    }

    pub fn dataset(&self) -> &OmeZarrDataset {
        &self.opened.resource.dataset
    }

    pub fn store(&self) -> &Arc<dyn ReadableStorageTraits> {
        &self.opened.resource.store
    }
}

pub struct RenderProjection {
    pub revision: u64,
    pub mode: ModelMode,
    pub document_generation: u64,
    pub document: Option<Arc<RenderDocument>>,
    pub project: ProjectModelSnapshot,
    pub settings: AppSettings,
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
        settings: model.settings().clone(),
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
