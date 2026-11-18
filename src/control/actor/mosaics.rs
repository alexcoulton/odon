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

pub(super) fn begin_mosaic_memory_pin(
    model: &mut AppModel,
    request: OdonControlRequest,
    load_job_tx: &Sender<LoadJob>,
    diagnostics: &ActorDiagnostics,
) -> bool {
    if load_job_tx.is_full() {
        reject_worker_submission(request, diagnostics);
        return false;
    }
    let spec = match model.prepare_mosaic_memory_pin(request.command.params()) {
        Ok(spec) => spec,
        Err(error) => {
            reject_actor_request(request, diagnostics, error);
            return false;
        }
    };
    match load_job_tx.try_send(LoadJob::MosaicMemoryPin { request, spec }) {
        Ok(()) => {
            diagnostics.workers_started.fetch_add(1, Ordering::Relaxed);
            true
        }
        Err(error) => {
            let LoadJob::MosaicMemoryPin { request, spec } = error.into_inner() else {
                unreachable!("mosaic memory submission returns its own job")
            };
            model.fail_mosaic_memory_pin(&spec, "Mosaic memory worker queue is unavailable");
            reject_worker_submission(request, diagnostics);
            false
        }
    }
}

/// Makes every item's `roi_id` unique within the mosaic.
///
/// The ROI id addresses an item in commands (focus, object loading) and keys per-ROI state such
/// as continuous colour domains, so two items must never share one. Display names often do
/// collide: every ROI of a multi-sample project can be called `ROI1.ome.zarr`. A colliding name
/// is qualified with the item's own identifier when one is available (the project ROI id), and
/// any name that still collides gets a numeric suffix. Unique names are left untouched.
fn disambiguate_roi_ids(items: &mut [ControlMosaicItemResource], qualifiers: &[Option<String>]) {
    let mut counts = std::collections::HashMap::<String, usize>::new();
    for item in items.iter() {
        *counts.entry(item.roi_id.clone()).or_default() += 1;
    }
    for (item, qualifier) in items.iter_mut().zip(qualifiers) {
        if counts[&item.roi_id] > 1
            && let Some(qualifier) = qualifier
                .as_deref()
                .map(str::trim)
                .filter(|qualifier| !qualifier.is_empty() && *qualifier != item.roi_id)
        {
            item.roi_id = format!("{} [{qualifier}]", item.roi_id);
        }
    }
    let mut taken = HashSet::<String>::new();
    for index in 0..items.len() {
        if taken.insert(items[index].roi_id.clone()) {
            continue;
        }
        let base = items[index].roi_id.clone();
        let mut suffix = 2usize;
        let unique = loop {
            let candidate = format!("{base} ({suffix})");
            if !taken.contains(&candidate)
                && !items.iter().any(|other| other.roi_id == candidate)
            {
                break candidate;
            }
            suffix += 1;
        };
        taken.insert(unique.clone());
        items[index].roi_id = unique;
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
    let qualifiers = vec![None; items.len()];
    disambiguate_roi_ids(&mut items, &qualifiers);
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
    let mut project_ids = Vec::new();
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
                project_ids.push(Some(roi.id.clone()));
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
    disambiguate_roi_ids(&mut items, &project_ids);
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

/// Upper bound on object sources decoded at the same time.
const MAX_CONCURRENT_OBJECT_LOADS: usize = 4;
/// Loading is IO-bound for part of each file, so two at a time pays off even on two cores.
const MIN_CONCURRENT_OBJECT_LOADS: usize = 2;

/// Loads several object sources at once and returns one outcome per source, in order. A source
/// that was never started, because `should_stop` reported true first, yields `None`.
///
/// Each source is read and decoded on one thread, so loading several concurrently shortens a
/// multi-ROI load almost proportionally. The limit is deliberately modest: every in-flight load
/// holds its file's decoded geometry in memory until it is handed over.
///
/// `should_stop` and `on_finished` run on the calling thread, after each source completes;
/// `on_finished` receives the number of sources completed so far.
pub(super) fn load_object_sources_concurrently(
    loader: &dyn ObjectResourceLoader,
    sources: &[PathBuf],
    downsample_factor: f32,
    options: &Value,
    should_stop: &dyn Fn() -> bool,
    on_finished: &mut dyn FnMut(usize),
) -> Vec<Option<anyhow::Result<ControlObjectResource>>> {
    let total = sources.len();
    let mut outcomes = std::iter::repeat_with(|| None)
        .take(total)
        .collect::<Vec<Option<anyhow::Result<ControlObjectResource>>>>();
    if total == 0 || should_stop() {
        return outcomes;
    }
    let concurrency = thread::available_parallelism()
        .map(|cores| cores.get().div_ceil(2))
        .unwrap_or(MIN_CONCURRENT_OBJECT_LOADS)
        .clamp(MIN_CONCURRENT_OBJECT_LOADS, MAX_CONCURRENT_OBJECT_LOADS)
        .min(total);
    let next_source = std::sync::atomic::AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    thread::scope(|scope| {
        let (outcome_tx, outcome_rx) = crossbeam_channel::unbounded();
        for _ in 0..concurrency {
            let outcome_tx = outcome_tx.clone();
            let (next_source, stop) = (&next_source, &stop);
            scope.spawn(move || {
                loop {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let index = next_source.fetch_add(1, Ordering::AcqRel);
                    let Some(path) = sources.get(index) else {
                        break;
                    };
                    let outcome = loader.load_with_options(
                        path.clone(),
                        downsample_factor,
                        Some(options.clone()),
                    );
                    if outcome_tx.send((index, outcome)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(outcome_tx);
        let mut done = 0usize;
        for (index, outcome) in outcome_rx {
            outcomes[index] = Some(outcome);
            done += 1;
            on_finished(done);
            if should_stop() {
                // Sources already being read finish; no further ones are started.
                stop.store(true, Ordering::Release);
            }
        }
    });
    outcomes
}

pub(super) fn load_mosaic_objects_on_worker(
    spec: &MosaicObjectLoadSpec,
    object_loader: Option<&dyn ObjectResourceLoader>,
) -> anyhow::Result<MosaicObjectLoadResult> {
    let loader =
        object_loader.ok_or_else(|| anyhow::anyhow!("object resource loader is unavailable"))?;
    let paths = spec
        .items
        .iter()
        .map(|(_, path)| path.clone())
        .collect::<Vec<_>>();
    // A mosaic can retain many ROI resources simultaneously. Loading every
    // GeoParquet property into every resource makes memory scale as
    // ROI × cells × columns; wide marker tables can exhaust system RAM
    // before the first frame. Keep geometry resident and let each
    // ObjectsLayer materialize only the active style/filter property.
    let outcomes = load_object_sources_concurrently(
        loader,
        &paths,
        spec.downsample_factor,
        &ProjectObjectPreloadSettings::default().worker_options(),
        &|| spec.is_cancelled(),
        &mut |_| {},
    );
    let mut loaded = Vec::new();
    let mut failures = Vec::new();
    for ((item_id, _), outcome) in spec.items.iter().zip(outcomes) {
        match outcome {
            Some(Ok(resource)) => loaded.push((*item_id, Arc::new(resource))),
            Some(Err(error)) => failures.push((*item_id, error.to_string())),
            None => {}
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

#[cfg(test)]
mod roi_id_tests {
    use super::*;

    fn item(id: usize, roi_id: &str) -> ControlMosaicItemResource {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
        ControlMosaicItemResource {
            id,
            roi_id: roi_id.to_string(),
            metadata: Default::default(),
            document: open_local_ome_zarr(&fixture).expect("fixture").into_control(),
            segmentation_path: None,
        }
    }

    fn ids(items: &[ControlMosaicItemResource]) -> Vec<&str> {
        items.iter().map(|item| item.roi_id.as_str()).collect()
    }

    #[test]
    fn colliding_display_names_are_qualified_and_unique_names_are_untouched() {
        let mut items = vec![
            item(0, "ROI1.ome.zarr"),
            item(1, "ROI1.ome.zarr"),
            item(2, "Tonsil"),
        ];
        let qualifiers = vec![
            Some("sample-a".to_string()),
            Some("sample-b".to_string()),
            Some("sample-c".to_string()),
        ];
        disambiguate_roi_ids(&mut items, &qualifiers);
        assert_eq!(
            ids(&items),
            ["ROI1.ome.zarr [sample-a]", "ROI1.ome.zarr [sample-b]", "Tonsil"]
        );
    }

    #[test]
    fn names_without_a_distinguishing_qualifier_get_numeric_suffixes() {
        let mut items = vec![item(0, "core"), item(1, "core"), item(2, "core (2)"), item(3, "core")];
        disambiguate_roi_ids(&mut items, &[None, None, None, None]);
        let unique = ids(&items).into_iter().collect::<HashSet<_>>();
        assert_eq!(unique.len(), 4, "{:?}", ids(&items));
        assert_eq!(items[0].roi_id, "core");
        assert_eq!(items[2].roi_id, "core (2)");
    }
}
