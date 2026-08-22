use super::*;

pub(super) struct DeepLinkResolveWorkerResult {
    pub(super) request: DeepLinkRequest,
    pub(super) resolution: Result<DeepLinkResolution, String>,
}

pub(super) enum LoadCompletion {
    DatasetInspect {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        result: DatasetInspection,
    },
    DeepLinkResolve {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        result: DeepLinkResolveWorkerResult,
    },
    OmeZarr {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        result: anyhow::Result<(
            OpenedDocument<OmeZarrDocumentResource>,
            Vec<String>,
            Option<ControlLabelResource>,
        )>,
    },
    Tiff {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        z: usize,
        t: usize,
        result: anyhow::Result<OpenedDocument<AlternateDocumentResource>>,
    },
    SpatialData {
        generation: u64,
        request: OdonControlRequest,
        result: anyhow::Result<(
            OpenedDocument<AlternateDocumentResource>,
            crate::data::document::SpatialDataOpenIdentity,
        )>,
    },
    Xenium {
        generation: u64,
        request: OdonControlRequest,
        result: anyhow::Result<(
            OpenedDocument<AlternateDocumentResource>,
            crate::data::document::XeniumOpenIdentity,
        )>,
    },
    RemoteList {
        session_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        result: anyhow::Result<crate::data::remote_store::S3BrowseListing>,
    },
    RemoteOpen {
        generation: u64,
        session_generation: Option<u64>,
        request: OdonControlRequest,
        identity: RemoteOpenIdentity,
        result: anyhow::Result<(
            OpenedDocument<OmeZarrDocumentResource>,
            Vec<String>,
            Option<ControlLabelResource>,
        )>,
    },
    ChannelIntensity {
        generation: u64,
        request: OdonControlRequest,
        result: anyhow::Result<Value>,
    },
    ProjectOpen {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        result: anyhow::Result<(ProjectConfig, Value)>,
    },
    ProjectSave {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        saved_config_generation: u64,
        platform_effect: Option<PlatformEffect>,
        result: anyhow::Result<()>,
    },
    SettingsSave {
        generation: u64,
        request: OdonControlRequest,
        settings: AppSettings,
        response: Value,
        result: anyhow::Result<PathBuf>,
    },
    SettingsPersist {
        generation: u64,
        settings: AppSettings,
        response: Value,
        result: anyhow::Result<PathBuf>,
    },
    SamplesheetInspect {
        request: OdonControlRequest,
        result: Value,
    },
    SamplesheetImport {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        result: anyhow::Result<Vec<ProjectRoi>>,
    },
    SamplesheetExport {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        result: anyhow::Result<u64>,
    },
    ProjectDiscovery {
        generation: u64,
        request: OdonControlRequest,
        root: PathBuf,
        result: anyhow::Result<Vec<PathBuf>>,
    },
    ObjectResource {
        document_generation: u64,
        resource_generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        downsample_factor: f32,
        result: anyhow::Result<ControlObjectResource>,
    },
    Labels {
        document_generation: u64,
        label_generation: u64,
        request: OdonControlRequest,
        name: String,
        result: anyhow::Result<ControlLabelResource>,
    },
    ObjectFilter {
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: String,
        expected_presentation_revision: u64,
        request: OdonControlRequest,
        result: anyhow::Result<ControlObjectFilterResult>,
    },
    ObjectSelectionFilter {
        document_generation: u64,
        resource_generation: u64,
        selection_generation: u64,
        operation_generation: u64,
        request: OdonControlRequest,
        mode: String,
        limit: usize,
        result: anyhow::Result<ControlObjectFilterResult>,
    },
    MaskImport {
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        name: String,
        editable: bool,
        result: anyhow::Result<Vec<Vec<[f32; 2]>>>,
    },
    MaskExport {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        layer_id: Option<u64>,
        layer_count: usize,
        polygon_count: usize,
        result: anyhow::Result<u64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CompletionDomain {
    Opening,
    Project,
    Resources,
    Objects,
    Masks,
}

impl LoadCompletion {
    pub(super) const fn domain(&self) -> CompletionDomain {
        match self {
            Self::DatasetInspect { .. }
            | Self::DeepLinkResolve { .. }
            | Self::OmeZarr { .. }
            | Self::Tiff { .. }
            | Self::SpatialData { .. }
            | Self::Xenium { .. }
            | Self::RemoteList { .. }
            | Self::RemoteOpen { .. }
            | Self::ChannelIntensity { .. } => CompletionDomain::Opening,
            Self::ProjectOpen { .. }
            | Self::ProjectSave { .. }
            | Self::SettingsSave { .. }
            | Self::SettingsPersist { .. }
            | Self::SamplesheetInspect { .. }
            | Self::SamplesheetImport { .. }
            | Self::SamplesheetExport { .. }
            | Self::ProjectDiscovery { .. } => CompletionDomain::Project,
            Self::ObjectResource { .. } | Self::Labels { .. } => CompletionDomain::Resources,
            Self::ObjectFilter { .. } | Self::ObjectSelectionFilter { .. } => {
                CompletionDomain::Objects
            }
            Self::MaskImport { .. } | Self::MaskExport { .. } => CompletionDomain::Masks,
        }
    }
}

pub(super) enum LoadJob {
    DatasetInspect {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
    },
    DeepLinkResolve {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        deep_link: DeepLinkRequest,
        current_project: ProjectModelSnapshot,
    },
    OmeZarr {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
    },
    Tiff {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        z: usize,
        t: usize,
    },
    SpatialData {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        options: crate::data::document::SpatialDataOpenOptions,
    },
    Xenium {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        options: crate::data::document::XeniumOpenOptions,
    },
    RemoteList {
        session_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        credentials: crate::data::remote_store::S3SessionCredentials,
        prefix: String,
    },
    RemoteOpen {
        generation: u64,
        session_generation: Option<u64>,
        request: OdonControlRequest,
        spec: RemoteOpenSpec,
    },
    ChannelIntensity {
        generation: u64,
        request: OdonControlRequest,
        document: Arc<RenderDocument>,
        spec: ChannelIntensitySpec,
    },
    ProjectOpen {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
    },
    ProjectSave {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        payload: Value,
        saved_config_generation: u64,
        platform_effect: Option<PlatformEffect>,
    },
    SettingsSave {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        settings: AppSettings,
        response: Value,
    },
    SettingsPersist {
        generation: u64,
        path: PathBuf,
        settings: AppSettings,
        response: Value,
    },
    SamplesheetInspect {
        request: OdonControlRequest,
        path: PathBuf,
        offset: usize,
        limit: usize,
    },
    SamplesheetImport {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        default_dataset: String,
    },
    SamplesheetExport {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        rois: Vec<ProjectRoi>,
        overwrite: bool,
    },
    ProjectDiscovery {
        generation: u64,
        request: OdonControlRequest,
        root: PathBuf,
    },
    ObjectResource {
        document_generation: u64,
        resource_generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
        downsample_factor: f32,
        options: Option<Value>,
    },
    Labels {
        document_generation: u64,
        label_generation: u64,
        request: OdonControlRequest,
        document: Arc<RenderDocument>,
        name: String,
    },
    ObjectFilter {
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: String,
        expected_presentation_revision: u64,
        request: OdonControlRequest,
        resource: Arc<ControlObjectResource>,
        model: Value,
    },
    ObjectSelectionFilter {
        document_generation: u64,
        resource_generation: u64,
        selection_generation: u64,
        operation_generation: u64,
        request: OdonControlRequest,
        resource: Arc<ControlObjectResource>,
        model: Value,
        mode: String,
        limit: usize,
    },
    MaskImport {
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        name: String,
        editable: bool,
        downsample_factor: f32,
    },
    MaskExport {
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        layer_id: Option<u64>,
        layers: Vec<ProjectMaskLayer>,
        overwrite: bool,
    },
}
