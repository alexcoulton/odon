use super::*;
use std::collections::HashMap;
use std::path::Path;

pub(super) fn begin_mosaic_open(
    model: &mut AppModel,
    remote_session: &RemoteSessionState,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return;
    }
    let job = match request.command.method() {
        "datasets.open_mosaic_samplesheet" => {
            let Some(path) = request
                .command
                .params()
                .get("path")
                .and_then(Value::as_str)
                .map(expand_path)
            else {
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::invalid_params(
                        "datasets.open_mosaic_samplesheet",
                        "path is required",
                    ),
                );
                return;
            };
            let source = path.to_string_lossy().into_owned();
            let generation = model.begin_mosaic_open(&source);
            LoadJob::MosaicSamplesheet {
                generation,
                request,
                path,
            }
        }
        "project.rois.open_selected_mosaic" => {
            let project = model.project_snapshot();
            let selected = project.selected_source_keys.iter().collect::<HashSet<_>>();
            let rois = project
                .rois
                .iter()
                .filter(|roi| {
                    roi.source_key()
                        .as_ref()
                        .is_some_and(|key| selected.contains(key))
                })
                .cloned()
                .collect::<Vec<_>>();
            if rois.len() < 2 {
                reject_actor_request(
                    request,
                    diagnostics,
                    ControlError::invalid_params(
                        "project.rois.open_selected_mosaic",
                        "select at least two ROIs to open a mosaic",
                    ),
                );
                return;
            }
            let needs_s3 = rois
                .iter()
                .any(|roi| matches!(roi.dataset_source(), Some(DatasetSource::S3 { .. })));
            let s3_session = if needs_s3 {
                match remote_session.credentials() {
                    Ok(session) => Some(session),
                    Err(error) => {
                        reject_actor_request(request, diagnostics, error);
                        return;
                    }
                }
            } else {
                None
            };
            let project_dir = project
                .saved_path
                .as_ref()
                .and_then(|path| path.parent())
                .map(Path::to_path_buf);
            let generation = model.begin_mosaic_open("selected project ROIs");
            LoadJob::MosaicProject {
                generation,
                request,
                rois,
                project_dir,
                s3_session,
            }
        }
        method => {
            reject_actor_request(
                request,
                diagnostics,
                ControlError::new(
                    ControlErrorKind::MethodNotFound,
                    format!("unsupported mosaic opening method '{method}'"),
                ),
            );
            return;
        }
    };
    match load_job_tx.try_send(job) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            let job = error.into_inner();
            let (generation, request) = match job {
                LoadJob::MosaicSamplesheet {
                    generation,
                    request,
                    ..
                }
                | LoadJob::MosaicProject {
                    generation,
                    request,
                    ..
                } => (generation, request),
                _ => unreachable!("mosaic opening submission returns its own job"),
            };
            model.fail_mosaic_open(generation, "Mosaic worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
        }
    }
}

pub(super) fn begin_mosaic_object_load(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let spec = match model.prepare_mosaic_object_load(request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::MosaicObjects { request, spec }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::MosaicObjects { request, spec } = error.into_inner() else {
                unreachable!("mosaic object submission returns its own job")
            };
            model.fail_mosaic_object_load(&spec, "Mosaic object worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}

pub(super) fn open_mosaic_samplesheet_on_worker(
    generation: u64,
    path: &Path,
    columns: Option<usize>,
) -> anyhow::Result<MosaicOpenWorkerResult> {
    let sheet = load_samplesheet_csv(path)?;
    let base_dir = path.parent().map(Path::to_path_buf);
    let mut items = Vec::with_capacity(sheet.rows.len());
    let mut failures = Vec::new();
    for row in sheet.rows {
        match open_local_ome_zarr(&row.path) {
            Ok(document) => {
                let id = items.len();
                let segmentation_path = segmentation_path_from_metadata(&row.meta, &base_dir);
                items.push(ControlMosaicItemResource {
                    id,
                    roi_id: if row.id.trim().is_empty() {
                        document.descriptor.source.display_name()
                    } else {
                        row.id
                    },
                    metadata: row.meta,
                    document: document.into_control(),
                    segmentation_path,
                });
            }
            Err(error) => failures.push(format!("{}: {error}", row.path.display())),
        }
    }
    if items.is_empty() {
        anyhow::bail!(
            "failed to open any ROIs from samplesheet {}{}",
            path.display(),
            failure_suffix(&failures)
        );
    }
    Ok(MosaicOpenWorkerResult {
        resource: ControlMosaicResource {
            generation,
            source: path.to_string_lossy().into_owned(),
            base_dir,
            initial_columns: columns,
            metadata_columns: Arc::new(sheet.meta_columns),
            items: Arc::new(items),
        },
        s3_session_generation: None,
    })
}

pub(super) fn open_mosaic_project_on_worker(
    generation: u64,
    rois: Vec<ProjectRoi>,
    project_dir: Option<PathBuf>,
    s3_session: Option<(u64, crate::data::remote_store::S3SessionCredentials)>,
    remote_backend: &dyn RemoteDatasetBackend,
) -> anyhow::Result<MosaicOpenWorkerResult> {
    let mut items = Vec::with_capacity(rois.len());
    let mut metadata_columns = BTreeSet::new();
    let mut failures = Vec::new();
    for roi in rois {
        let Some(source) = roi.dataset_source().map(|source| match source {
            DatasetSource::Local(path) if path.is_relative() => DatasetSource::Local(
                project_dir
                    .as_ref()
                    .map(|directory| directory.join(&path))
                    .unwrap_or(path),
            ),
            source => source,
        }) else {
            failures.push(format!("{}: no dataset source", roi.id));
            continue;
        };
        let opened = match &source {
            DatasetSource::Local(path) => open_local_ome_zarr(path),
            DatasetSource::Http { base_url } => remote_backend.open_http(base_url),
            DatasetSource::S3 { prefix, .. } => {
                let (_, credentials) = s3_session
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("S3 session credentials are unavailable"))?;
                remote_backend.open_s3(credentials, prefix)
            }
        };
        match opened {
            Ok(document) => {
                let id = items.len();
                let mut metadata = roi.meta.clone();
                let segmentation_path = roi.segpath.as_ref().map(|path| {
                    if path.is_relative() {
                        project_dir
                            .as_ref()
                            .map(|directory| directory.join(path))
                            .unwrap_or_else(|| path.clone())
                    } else {
                        path.clone()
                    }
                });
                if let Some(path) = roi.segpath.as_ref() {
                    metadata.insert("segpath".to_string(), path.to_string_lossy().into_owned());
                }
                metadata_columns.extend(
                    metadata
                        .keys()
                        .filter(|key| !key.trim().is_empty())
                        .cloned(),
                );
                items.push(ControlMosaicItemResource {
                    id,
                    roi_id: roi
                        .display_name
                        .filter(|name| !name.trim().is_empty())
                        .unwrap_or(roi.id),
                    metadata,
                    document: document.into_control(),
                    segmentation_path,
                });
            }
            Err(error) => failures.push(format!("{}: {error}", roi.id)),
        }
    }
    if items.len() < 2 {
        anyhow::bail!(
            "need at least 2 valid ROIs to open mosaic{}",
            failure_suffix(&failures)
        );
    }
    Ok(MosaicOpenWorkerResult {
        resource: ControlMosaicResource {
            generation,
            source: "selected project ROIs".to_string(),
            base_dir: project_dir,
            initial_columns: None,
            metadata_columns: Arc::new(metadata_columns.into_iter().collect()),
            items: Arc::new(items),
        },
        s3_session_generation: s3_session.as_ref().map(|(generation, _)| *generation),
    })
}

pub(super) fn load_mosaic_objects_on_worker(
    spec: &MosaicObjectLoadSpec,
    object_loader: Option<&dyn ObjectResourceLoader>,
) -> anyhow::Result<MosaicObjectLoadResult> {
    let loader =
        object_loader.ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
    let mut loaded = Vec::new();
    let mut failures = Vec::new();
    for (item_id, path) in &spec.items {
        if spec.is_cancelled() {
            break;
        }
        match loader.load(path.clone(), spec.downsample_factor) {
            Ok(resource) => loaded.push((*item_id, Arc::new(resource))),
            Err(error) => failures.push((*item_id, error.to_string())),
        }
    }
    Ok(MosaicObjectLoadResult {
        loaded,
        failures,
        cancelled: spec.is_cancelled(),
    })
}

fn segmentation_path_from_metadata(
    metadata: &HashMap<String, String>,
    base_dir: &Option<PathBuf>,
) -> Option<PathBuf> {
    let raw = metadata.get("segpath")?.trim();
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    if path.is_relative() {
        Some(
            base_dir
                .as_ref()
                .map(|directory| directory.join(&path))
                .unwrap_or(path),
        )
    } else {
        Some(path)
    }
}

fn failure_suffix(failures: &[String]) -> String {
    failures
        .first()
        .map(|failure| format!("; first failure: {failure}"))
        .unwrap_or_default()
}
