//! Project object discovery, preload, and ROI-open worker computations.

use super::*;

pub(in crate::control::actor) fn scan_project_object_sources(
    candidates: Vec<PathBuf>,
) -> anyhow::Result<Vec<ProjectObjectPreloadSource>> {
    let mut sources = Vec::new();
    for path in candidates {
        match fs::metadata(&path) {
            Ok(metadata) => sources.push(ProjectObjectPreloadSource {
                path,
                bytes: metadata.len(),
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(anyhow::anyhow!(
                    "could not inspect object source {}: {error}",
                    path.display()
                ));
            }
        }
    }
    Ok(sources)
}

pub(in crate::control::actor) fn preload_project_objects(
    object_loader: Option<&dyn ObjectResourceLoader>,
    request: &OdonControlRequest,
    settings: ProjectObjectPreloadSettings,
    candidates: Vec<PathBuf>,
) -> anyhow::Result<ProjectObjectPreloadWorkerResult> {
    let loader =
        object_loader.ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
    let sources = scan_project_object_sources(candidates)?;
    if sources.is_empty() {
        anyhow::bail!(
            "project has no preload-eligible Parquet or GeoParquet segmentation paths on disk"
        );
    }
    let mut resources = Vec::new();
    let mut failures = Vec::new();
    let total = sources.len();
    for (index, source) in sources.iter().enumerate() {
        if request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
        {
            break;
        }
        match loader.load_with_options(source.path.clone(), 1.0, Some(settings.worker_options())) {
            Ok(resource) => resources.push((source.path.clone(), resource)),
            Err(error) => failures.push((source.path.clone(), error.to_string())),
        }
        if let Some(task_id) = request.task_id.as_deref() {
            let _ = request.task_registry.progress(
                task_id,
                Some((index + 1) as f64 / total as f64),
                format!("preloaded {} of {total} project object sources", index + 1),
            );
        }
    }
    Ok(ProjectObjectPreloadWorkerResult {
        sources,
        resources,
        failures,
    })
}

pub(in crate::control::actor) fn open_project_roi_on_worker(
    spec: ProjectRoiOpenSpec,
    object_loader: Option<&dyn ObjectResourceLoader>,
    dataset_inspector: &dyn DatasetInspector,
    remote_backend: &dyn RemoteDatasetBackend,
    alternate_backend: &dyn AlternateDatasetBackend,
) -> anyhow::Result<ProjectRoiOpenWorkerResult> {
    let ProjectRoiOpenSpec {
        roi,
        source,
        saved_view,
        object_path,
        cached_object,
        s3_session,
        requested_label,
    } = spec;
    let mut label_available = Vec::new();
    let mut label_resource = None;
    let mut s3_session_generation = None;
    let opened = match &source {
        DatasetSource::Local(requested_path) => {
            let path = normalize_local_dataset_path(requested_path)
                .unwrap_or_else(|| requested_path.clone());
            match classify_local_dataset_path(&path) {
                Some(LocalDatasetKind::OmeZarr) => {
                    let (opened, available, root) =
                        complete_ome_zarr_resource(open_local_ome_zarr(&path)?, Vec::new());
                    label_available = available;
                    label_resource = root;
                    opened.into_control()
                }
                Some(LocalDatasetKind::Tiff) => {
                    alternate_backend.open_tiff(&path, 0, 0)?.into_control()
                }
                Some(LocalDatasetKind::Xenium) => {
                    let options = crate::data::document::XeniumOpenOptions {
                        imagery: "auto".to_string(),
                        load_cells: true,
                        load_transcripts: true,
                    };
                    alternate_backend
                        .open_xenium(&path, &options)?
                        .0
                        .into_control()
                }
                None => {
                    let inspection = dataset_inspector.inspect(&path);
                    if inspection.kind
                        != Some(crate::data::document::DatasetInspectionKind::SpatialData)
                    {
                        anyhow::bail!(
                            "project ROI source {} is not a supported dataset",
                            path.display()
                        );
                    }
                    let image = inspection
                        .elements
                        .as_ref()
                        .into_iter()
                        .flatten()
                        .find(|element| element.kind == "image")
                        .map(|element| element.name.clone())
                        .ok_or_else(|| {
                            anyhow::anyhow!("SpatialData source has no image element")
                        })?;
                    let options = crate::data::document::SpatialDataOpenOptions {
                        image,
                        extra_images: Vec::new(),
                        labels: None,
                        shapes: Vec::new(),
                        points: None,
                        points_max: 200_000,
                    };
                    alternate_backend
                        .open_spatialdata(&path, &options)?
                        .0
                        .into_control()
                }
            }
        }
        DatasetSource::Http { base_url } => {
            let (opened, available, root) =
                complete_ome_zarr_resource(remote_backend.open_http(base_url)?, Vec::new());
            label_available = available;
            label_resource = root;
            opened.into_control()
        }
        DatasetSource::S3 { prefix, .. } => {
            let (generation, credentials) = s3_session
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("S3 session credentials are unavailable"))?;
            let (opened, available, root) = complete_ome_zarr_resource(
                remote_backend.open_s3(credentials, prefix)?,
                Vec::new(),
            );
            s3_session_generation = Some(*generation);
            label_available = available;
            label_resource = root;
            opened.into_control()
        }
    };

    if opened.descriptor.kind == crate::data::document::DocumentKind::OmeZarr
        && let Some(selected) =
            requested_label.or_else(|| saved_project_label_name(saved_view.as_ref()))
    {
        if !label_available.contains(&selected) {
            label_available.push(selected.clone());
        }
        let root_matches = label_resource
            .as_ref()
            .is_some_and(|resource| resource.dataset.label_name == selected);
        if !root_matches {
            label_resource =
                match LabelZarrDataset::try_open(Arc::clone(opened.resource.store()), &selected)? {
                    Some(dataset) => Some(ControlLabelResource {
                        dataset,
                        store: Arc::clone(opened.resource.store()),
                    }),
                    None => anyhow::bail!("saved project label '{selected}' was not found"),
                };
        }
    }

    let object_resource = match (cached_object, object_path) {
        (Some(resource), _) => Some(resource),
        (None, Some(path)) => {
            let loader = object_loader
                .ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
            let settings = ProjectObjectPreloadSettings::default();
            Some(Arc::new(loader.load_with_options(
                path,
                1.0,
                Some(settings.worker_options()),
            )?))
        }
        (None, None) => None,
    };

    Ok(ProjectRoiOpenWorkerResult {
        opened,
        roi,
        saved_view,
        label_available,
        label_resource,
        object_resource,
        s3_session_generation,
        reuse_current: false,
    })
}

pub(in crate::control::actor) fn saved_project_label_name(view: Option<&Value>) -> Option<String> {
    let view = view?;
    view.get("segmentation")
        .and_then(|segmentation| segmentation.get("label_name"))
        .and_then(Value::as_str)
        .or_else(|| {
            let workspace = view.get("workspace")?;
            let active = workspace.get("active_viewport_id")?.as_str()?;
            workspace
                .get("viewports")?
                .as_array()?
                .iter()
                .find(|viewport| viewport.get("id").and_then(Value::as_str) == Some(active))?
                .get("segmentation")?
                .get("label_name")?
                .as_str()
        })
        .map(str::to_string)
}

pub(in crate::control::actor) fn complete_ome_zarr_resource(
    opened: OpenedDocument<OmeZarrDocumentResource>,
    mut available: Vec<String>,
) -> (
    OpenedDocument<OmeZarrDocumentResource>,
    Vec<String>,
    Option<ControlLabelResource>,
) {
    let dataset = &opened.resource.dataset;
    let root_resource = dataset.is_root_label_mask().then(|| {
        let labels = LabelZarrDataset::from_root_dataset(dataset);
        if !available.contains(&labels.label_name) {
            available.push(labels.label_name.clone());
        }
        ControlLabelResource {
            dataset: labels,
            store: Arc::clone(&opened.resource.store),
        }
    });
    (opened, available, root_resource)
}
