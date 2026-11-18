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

/// Content fingerprints of the JSON sections that are expensive for the renderer to apply.
///
/// A fingerprint covers everything the renderer consumes from its section except navigation
/// (cameras and navigation revisions), which travels typed in [`RenderProjection::navigation`],
/// and it also covers the identity of the shared resources applied together with the section.
/// Two projections with equal fingerprints carry the same section, so the renderer may skip
/// re-applying it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProjectionFingerprints {
    pub workspace: u64,
    pub mosaic_state: u64,
}

/// Hashes a JSON value.
pub fn json_fingerprint(value: &Value, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    match value {
        Value::Null => 0u8.hash(hasher),
        Value::Bool(value) => (1u8, value).hash(hasher),
        Value::Number(number) => {
            2u8.hash(hasher);
            if let Some(value) = number.as_u64() {
                (0u8, value).hash(hasher);
            } else if let Some(value) = number.as_i64() {
                (1u8, value).hash(hasher);
            } else {
                (2u8, number.as_f64().unwrap_or(f64::NAN).to_bits()).hash(hasher);
            }
        }
        Value::String(value) => (3u8, value).hash(hasher),
        Value::Array(values) => {
            (4u8, values.len()).hash(hasher);
            for value in values {
                json_fingerprint(value, hasher);
            }
        }
        Value::Object(map) => {
            5u8.hash(hasher);
            for (key, value) in map {
                key.hash(hasher);
                json_fingerprint(value, hasher);
            }
            6u8.hash(hasher);
        }
    }
}

/// Hashes the members of a JSON object except the top-level keys in `skip`.
pub fn json_object_fingerprint(
    value: &Value,
    skip: &[&str],
    hasher: &mut impl std::hash::Hasher,
) {
    use std::hash::Hash;
    let Some(map) = value.as_object() else {
        json_fingerprint(value, hasher);
        return;
    };
    for (key, value) in map {
        if !skip.contains(&key.as_str()) {
            key.hash(hasher);
            json_fingerprint(value, hasher);
        }
    }
}

fn arc_identity<T>(resource: Option<&Arc<T>>, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    resource
        .map(|resource| Arc::as_ptr(resource) as usize)
        .hash(hasher);
}

impl ProjectionFingerprints {
    fn of(projection: &RenderProjection) -> Self {
        use std::hash::{Hash, Hasher};
        let workspace = projection.workspace.as_ref().map_or(0, |workspace| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            // `performance` only echoes the renderer's own observation back to it, and each
            // viewport's camera and navigation revision travel in `navigation`.
            json_object_fingerprint(workspace, &["performance", "viewports"], &mut hasher);
            for viewport in workspace
                .get("viewports")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                json_object_fingerprint(
                    viewport,
                    &["camera", "navigation_revision"],
                    &mut hasher,
                );
                0xffu8.hash(&mut hasher);
            }
            arc_identity(projection.object_resource.as_ref(), &mut hasher);
            arc_identity(
                projection.segmentation_geojson_resource.as_ref(),
                &mut hasher,
            );
            arc_identity(projection.label_resource.as_ref(), &mut hasher);
            hasher.finish()
        });
        let mosaic_state = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            json_object_fingerprint(&projection.mosaic_state, &["camera"], &mut hasher);
            json_fingerprint(projection.memory_state.as_ref(), &mut hasher);
            for (item_id, resource) in &projection.mosaic_object_resources {
                item_id.hash(&mut hasher);
                arc_identity(Some(resource), &mut hasher);
            }
            for (item_id, resource) in &projection.mosaic_pinned_levels {
                item_id.hash(&mut hasher);
                arc_identity(Some(resource), &mut hasher);
            }
            hasher.finish()
        };
        Self {
            workspace,
            mosaic_state,
        }
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
    pub command_surface: Value,
    pub shell: Value,
    pub screenshot_preferences: ScreenshotPreferences,
    pub tile_loading_policy: TileLoadingPolicy,
    pub memory_state: Arc<Value>,
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
    pub navigation: crate::model::ProjectedNavigation,
    pub fingerprints: ProjectionFingerprints,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformEffect {
    CloseWindow {
        quit: bool,
    },
    InvokeNativeCommand {
        command_id: String,
        action: String,
        checked: Option<bool>,
    },
    InvokeControlCommand {
        command_id: String,
        method: String,
        params: Value,
    },
}

pub(super) fn publish_projection(
    model: &mut AppModel,
    document: Option<Arc<RenderDocument>>,
    tx: &Sender<RenderProjection>,
    coalesce_rx: &Receiver<RenderProjection>,
    wake_ui: &UiWake,
    diagnostics: &ActorDiagnostics,
) {
    model.apply_startup_shell_layout_if_needed();
    // Keep legacy compatibility fields current before every projection, but do not rewrite the
    // actor-owned desired layout from their narrower panel/tab vocabulary. Explicit legacy UI
    // mutations use `sync_active_shell_domain_to_layout` and deliberately update both domains.
    let _ = model.sync_active_shell_domain();
    let revision = model.mark_projection_dirty();
    let mut projection = RenderProjection {
        revision,
        mode: model.mode(),
        document_generation: model.document_generation(),
        document,
        project: model.project_snapshot(),
        project_object_preload: model.project_object_preload_projection(),
        settings: model.settings().clone(),
        command_surface: {
            let mut surface = model.command_surface_projection();
            surface["active_mode"] = json!(model.mode().as_str());
            surface
        },
        shell: model.shell_projection(),
        screenshot_preferences: model.screenshot_preferences().clone(),
        tile_loading_policy: model.tile_loading_policy().clone(),
        memory_state: model.memory_projection_state(),
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
        navigation: model.projected_navigation(),
        fingerprints: ProjectionFingerprints::default(),
    };
    projection.fingerprints = ProjectionFingerprints::of(&projection);
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
