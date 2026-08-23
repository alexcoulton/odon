use super::*;

pub(super) struct DeepLinkResolveWorkerResult {
    pub(super) request: DeepLinkRequest,
    pub(super) resolution: Result<DeepLinkResolution, String>,
}

pub(super) struct DeepLinkApplySpec {
    pub(super) deep_link: DeepLinkRequest,
    pub(super) current_project: ProjectModelSnapshot,
    pub(super) cached_object: Option<(PathBuf, Arc<ControlObjectResource>)>,
    pub(super) s3_session: Option<(u64, crate::data::remote_store::S3SessionCredentials)>,
    pub(super) current_document: Option<ControlOpenedDocument>,
    pub(super) current_resources: Option<DeepLinkCurrentResources>,
}

pub(super) struct DeepLinkApplyWorkerResult {
    pub(super) deep_link: DeepLinkRequest,
    pub(super) project: ProjectModelSnapshot,
    pub(super) project_source: String,
    pub(super) opened: ProjectRoiOpenWorkerResult,
    pub(super) object_filter: Option<ControlObjectFilterResult>,
}

pub(super) enum MemoryPinWorkerOutcome {
    Confirmation {
        risk: &'static str,
        projected_bytes: u64,
        available_bytes: u64,
    },
    Loaded(ControlPinnedLevelResource),
}

pub(super) struct MemoryPinWorkerResult {
    pub(super) system: Option<SystemMemorySnapshot>,
    pub(super) outcome: MemoryPinWorkerOutcome,
}

pub(super) enum MosaicMemoryPinWorkerOutcome {
    Confirmation {
        risk: &'static str,
        projected_bytes: u64,
        available_bytes: u64,
    },
    Loaded(MosaicMemoryPinResult),
}

pub(super) struct MaskAppendWorkerResult {
    pub(super) bytes: u64,
    pub(super) appended_polygon_count: usize,
    pub(super) polygons_world: Vec<Vec<[f32; 2]>>,
}

pub(super) struct MosaicMemoryPinWorkerResult {
    pub(super) system: Option<SystemMemorySnapshot>,
    pub(super) outcome: MosaicMemoryPinWorkerOutcome,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AnalysisComputeKind {
    Histogram,
    ThresholdSuggestions,
    Warmup,
}

pub(super) struct ProjectObjectPreloadWorkerResult {
    pub(super) sources: Vec<ProjectObjectPreloadSource>,
    pub(super) resources: Vec<(PathBuf, ControlObjectResource)>,
    pub(super) failures: Vec<(PathBuf, String)>,
}

pub(super) struct ProjectRoiOpenSpec {
    pub(super) roi: ProjectRoi,
    pub(super) source: DatasetSource,
    pub(super) saved_view: Option<Value>,
    pub(super) object_path: Option<PathBuf>,
    pub(super) cached_object: Option<Arc<ControlObjectResource>>,
    pub(super) s3_session: Option<(u64, crate::data::remote_store::S3SessionCredentials)>,
    pub(super) requested_label: Option<String>,
}

pub(super) struct ProjectRoiOpenWorkerResult {
    pub(super) opened: ControlOpenedDocument,
    pub(super) roi: ProjectRoi,
    pub(super) saved_view: Option<Value>,
    pub(super) label_available: Vec<String>,
    pub(super) label_resource: Option<ControlLabelResource>,
    pub(super) object_resource: Option<Arc<ControlObjectResource>>,
    pub(super) s3_session_generation: Option<u64>,
    pub(super) reuse_current: bool,
}

pub(super) struct ProjectViewApplyWorkerResult {
    pub(super) object_resource: Option<Arc<ControlObjectResource>>,
    pub(super) label_resource: Option<Arc<ControlLabelResource>>,
}

pub(super) struct MosaicOpenWorkerResult {
    pub(super) resource: ControlMosaicResource,
    pub(super) s3_session_generation: Option<u64>,
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
    DeepLinkApply {
        operation_generation: u64,
        guard: DeepLinkApplyGuard,
        request: OdonControlRequest,
        result: anyhow::Result<DeepLinkApplyWorkerResult>,
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
    ProjectRoiOpen {
        operation_generation: u64,
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        result: anyhow::Result<ProjectRoiOpenWorkerResult>,
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
    ScreenshotSettingsValidate {
        generation: u64,
        request: OdonControlRequest,
        preferences: ScreenshotPreferences,
        result: anyhow::Result<()>,
    },
    ScreenshotWrite {
        request: OdonControlRequest,
        spec: ScreenshotWriteSpec,
        result: anyhow::Result<u64>,
    },
    ProjectViewApply {
        request: OdonControlRequest,
        spec: ProjectViewApplySpec,
        result: anyhow::Result<ProjectViewApplyWorkerResult>,
    },
    MemoryPin {
        request: OdonControlRequest,
        spec: MemoryPinSpec,
        result: anyhow::Result<MemoryPinWorkerResult>,
    },
    MosaicMemoryPin {
        request: OdonControlRequest,
        spec: MosaicMemoryPinSpec,
        result: anyhow::Result<MosaicMemoryPinWorkerResult>,
    },
    ThresholdLoad {
        request: OdonControlRequest,
        spec: ThresholdPreviewLoadSpec,
        result: anyhow::Result<ControlThresholdPreviewResource>,
    },
    ThresholdRecompute {
        request: OdonControlRequest,
        spec: ThresholdPreviewRecomputeSpec,
        result: anyhow::Result<ControlThresholdPreviewResource>,
    },
    ThresholdApply {
        request: OdonControlRequest,
        spec: ThresholdPreviewApplySpec,
        result: anyhow::Result<Vec<Vec<[f32; 2]>>>,
    },
    AnalysisCompute {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        kind: AnalysisComputeKind,
        result: anyhow::Result<Value>,
    },
    AnalysisPresetImport {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        path: PathBuf,
        result: anyhow::Result<Value>,
    },
    AnalysisPresetExport {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        path: PathBuf,
        result: anyhow::Result<usize>,
    },
    Measurement {
        request: OdonControlRequest,
        spec: MeasurementSpec,
        result: anyhow::Result<(ControlObjectResource, usize)>,
    },
    ObjectExport {
        request: OdonControlRequest,
        spec: ObjectExportSpec,
        result: anyhow::Result<ObjectExportResult>,
    },
    MosaicOpen {
        generation: u64,
        request: OdonControlRequest,
        result: anyhow::Result<MosaicOpenWorkerResult>,
    },
    MosaicObjects {
        request: OdonControlRequest,
        spec: MosaicObjectLoadSpec,
        result: anyhow::Result<MosaicObjectLoadResult>,
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
    ProjectObjectSourceScan {
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        result: anyhow::Result<Vec<ProjectObjectPreloadSource>>,
    },
    ProjectObjectPreload {
        generation: u64,
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        result: anyhow::Result<ProjectObjectPreloadWorkerResult>,
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
    Annotations {
        request: Option<OdonControlRequest>,
        spec: AnnotationLoadSpec,
        result: anyhow::Result<AnnotationLoadResult>,
    },
    ObjectFilter {
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: String,
        target: ObjectTarget,
        expected_presentation_revision: u64,
        request: OdonControlRequest,
        result: anyhow::Result<ControlObjectFilterResult>,
    },
    ObjectSelectionFilter {
        document_generation: u64,
        resource_generation: u64,
        selection_generation: u64,
        operation_generation: u64,
        target: ObjectTarget,
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
        replace_layer_id: Option<u64>,
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
    MaskAppend {
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        name: String,
        saved_layers: Vec<ProjectMaskLayer>,
        result: anyhow::Result<MaskAppendWorkerResult>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CompletionDomain {
    Opening,
    Project,
    Resources,
    Objects,
    Masks,
    Mosaic,
    Presentation,
}

impl LoadCompletion {
    pub(super) const fn allowed_during_presentation_barrier(&self) -> bool {
        matches!(self, Self::ScreenshotWrite { .. })
    }

    pub(super) const fn domain(&self) -> CompletionDomain {
        match self {
            Self::DatasetInspect { .. }
            | Self::DeepLinkResolve { .. }
            | Self::DeepLinkApply { .. }
            | Self::OmeZarr { .. }
            | Self::Tiff { .. }
            | Self::SpatialData { .. }
            | Self::Xenium { .. }
            | Self::ProjectRoiOpen { .. }
            | Self::RemoteList { .. }
            | Self::RemoteOpen { .. }
            | Self::ChannelIntensity { .. } => CompletionDomain::Opening,
            Self::ProjectOpen { .. }
            | Self::ProjectSave { .. }
            | Self::SettingsSave { .. }
            | Self::SettingsPersist { .. }
            | Self::ScreenshotSettingsValidate { .. }
            | Self::SamplesheetInspect { .. }
            | Self::SamplesheetImport { .. }
            | Self::SamplesheetExport { .. }
            | Self::ProjectDiscovery { .. }
            | Self::ProjectObjectSourceScan { .. }
            | Self::ProjectObjectPreload { .. } => CompletionDomain::Project,
            Self::ProjectViewApply { .. } => CompletionDomain::Project,
            Self::ObjectResource { .. }
            | Self::Labels { .. }
            | Self::Annotations { .. }
            | Self::MemoryPin { .. }
            | Self::ThresholdLoad { .. }
            | Self::ThresholdRecompute { .. }
            | Self::ThresholdApply { .. } => CompletionDomain::Resources,
            Self::AnalysisCompute { .. }
            | Self::AnalysisPresetImport { .. }
            | Self::AnalysisPresetExport { .. }
            | Self::Measurement { .. }
            | Self::ObjectExport { .. } => CompletionDomain::Objects,
            Self::ObjectFilter { .. } | Self::ObjectSelectionFilter { .. } => {
                CompletionDomain::Objects
            }
            Self::MaskImport { .. } | Self::MaskExport { .. } | Self::MaskAppend { .. } => {
                CompletionDomain::Masks
            }
            Self::MosaicOpen { .. } | Self::MosaicObjects { .. } | Self::MosaicMemoryPin { .. } => {
                CompletionDomain::Mosaic
            }
            Self::ScreenshotWrite { .. } => CompletionDomain::Presentation,
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
    DeepLinkApply {
        operation_generation: u64,
        guard: DeepLinkApplyGuard,
        request: OdonControlRequest,
        spec: DeepLinkApplySpec,
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
    ProjectRoiOpen {
        operation_generation: u64,
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        spec: ProjectRoiOpenSpec,
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
    ScreenshotSettingsValidate {
        generation: u64,
        request: OdonControlRequest,
        preferences: ScreenshotPreferences,
    },
    ScreenshotWrite {
        request: OdonControlRequest,
        spec: ScreenshotWriteSpec,
        pixels: PresentationPixels,
    },
    ProjectViewApply {
        request: OdonControlRequest,
        spec: ProjectViewApplySpec,
        document: Arc<RenderDocument>,
    },
    MemoryPin {
        request: OdonControlRequest,
        document: Arc<RenderDocument>,
        spec: MemoryPinSpec,
    },
    MosaicMemoryPin {
        request: OdonControlRequest,
        spec: MosaicMemoryPinSpec,
    },
    ThresholdLoad {
        request: OdonControlRequest,
        document: Arc<RenderDocument>,
        spec: ThresholdPreviewLoadSpec,
    },
    ThresholdRecompute {
        request: OdonControlRequest,
        spec: ThresholdPreviewRecomputeSpec,
    },
    ThresholdApply {
        request: OdonControlRequest,
        spec: ThresholdPreviewApplySpec,
    },
    AnalysisCompute {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        kind: AnalysisComputeKind,
        params: Value,
    },
    AnalysisPresetImport {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        path: PathBuf,
    },
    AnalysisPresetExport {
        request: OdonControlRequest,
        spec: AnalysisResourceSpec,
        path: PathBuf,
        overwrite: bool,
        state: Value,
    },
    Measurement {
        request: OdonControlRequest,
        document: Arc<RenderDocument>,
        spec: MeasurementSpec,
    },
    ObjectExport {
        request: OdonControlRequest,
        spec: ObjectExportSpec,
    },
    MosaicSamplesheet {
        generation: u64,
        request: OdonControlRequest,
        path: PathBuf,
    },
    MosaicProject {
        generation: u64,
        request: OdonControlRequest,
        rois: Vec<ProjectRoi>,
        project_dir: Option<PathBuf>,
        s3_session: Option<(u64, crate::data::remote_store::S3SessionCredentials)>,
    },
    MosaicObjects {
        request: OdonControlRequest,
        spec: MosaicObjectLoadSpec,
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
    ProjectObjectSourceScan {
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        candidates: Vec<PathBuf>,
    },
    ProjectObjectPreload {
        generation: u64,
        scope: ProjectObjectPreloadScope,
        request: OdonControlRequest,
        settings: ProjectObjectPreloadSettings,
        candidates: Vec<PathBuf>,
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
    Annotations {
        request: Option<OdonControlRequest>,
        spec: AnnotationLoadSpec,
    },
    ObjectFilter {
        document_generation: u64,
        resource_generation: u64,
        operation_generation: u64,
        viewport_id: String,
        target: ObjectTarget,
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
        target: ObjectTarget,
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
        replace_layer_id: Option<u64>,
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
    MaskAppend {
        document_generation: u64,
        mask_generation: u64,
        operation_generation: u64,
        operation_scope: String,
        request: OdonControlRequest,
        path: PathBuf,
        name: String,
        downsample_factor: f32,
        roi_root: String,
        saved_layers: Vec<ProjectMaskLayer>,
    },
}
