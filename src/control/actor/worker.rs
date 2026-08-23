use super::*;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

mod analysis;
mod measurements;
mod memory;
mod outputs;
mod projects;
mod thresholds;

pub(super) use analysis::*;
pub(super) use measurements::*;
pub(super) use memory::*;
pub(super) use outputs::*;
pub(super) use projects::*;
pub(super) use thresholds::*;

pub(super) fn spawn_resource_workers(
    load_job_rx: Receiver<LoadJob>,
    load_tx: Sender<LoadCompletion>,
    object_loader: Option<Arc<dyn ObjectResourceLoader>>,
    dataset_inspector: Arc<dyn DatasetInspector>,
    remote_backend: Arc<dyn RemoteDatasetBackend>,
    alternate_backend: Arc<dyn AlternateDatasetBackend>,
) -> anyhow::Result<()> {
    for index in 0..LOAD_WORKERS {
        let jobs = load_job_rx.clone();
        let completions = load_tx.clone();
        let object_loader = object_loader.clone();
        let dataset_inspector = Arc::clone(&dataset_inspector);
        let remote_backend = Arc::clone(&remote_backend);
        let alternate_backend = Arc::clone(&alternate_backend);
        thread::Builder::new()
            .name(format!("odon-resource-worker-{index}"))
            .spawn(move || {
                while let Ok(job) = jobs.recv() {
                    match job {
                        LoadJob::DatasetInspect {
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                        } => {
                            let result = dataset_inspector.inspect(&path);
                            if completions
                                .send(LoadCompletion::DatasetInspect {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::DeepLinkResolve {
                            operation_generation,
                            operation_scope,
                            request,
                            deep_link,
                            current_project,
                        } => {
                            let result = resolve_deep_link_on_worker(deep_link, current_project);
                            if completions
                                .send(LoadCompletion::DeepLinkResolve {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::DeepLinkApply {
                            operation_generation,
                            guard,
                            request,
                            spec,
                        } => {
                            let result = apply_deep_link_on_worker(
                                spec,
                                object_loader.as_deref(),
                                dataset_inspector.as_ref(),
                                remote_backend.as_ref(),
                                alternate_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::DeepLinkApply {
                                    operation_generation,
                                    guard,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::OmeZarr {
                            generation,
                            request,
                            path,
                        } => {
                            let result = open_local_ome_zarr(&path).map(|opened| {
                                complete_ome_zarr_resource(
                                    opened,
                                    discover_label_names_local(&path),
                                )
                            });
                            if completions
                                .send(LoadCompletion::OmeZarr {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Tiff {
                            generation,
                            request,
                            path,
                            z,
                            t,
                        } => {
                            let result = alternate_backend.open_tiff(&path, z, t);
                            if completions
                                .send(LoadCompletion::Tiff {
                                    generation,
                                    request,
                                    path,
                                    z,
                                    t,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SpatialData {
                            generation,
                            request,
                            path,
                            options,
                        } => {
                            let result = alternate_backend.open_spatialdata(&path, &options);
                            if completions
                                .send(LoadCompletion::SpatialData {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Xenium {
                            generation,
                            request,
                            path,
                            options,
                        } => {
                            let result = alternate_backend.open_xenium(&path, &options);
                            if completions
                                .send(LoadCompletion::Xenium {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectRoiOpen {
                            operation_generation,
                            scope,
                            request,
                            spec,
                        } => {
                            let result = open_project_roi_on_worker(
                                spec,
                                object_loader.as_deref(),
                                dataset_inspector.as_ref(),
                                remote_backend.as_ref(),
                                alternate_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::ProjectRoiOpen {
                                    operation_generation,
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::RemoteList {
                            session_generation,
                            operation_generation,
                            operation_scope,
                            request,
                            credentials,
                            prefix,
                        } => {
                            let result =
                                remote_backend
                                    .list_s3(&credentials, &prefix)
                                    .map_err(|error| {
                                        anyhow::anyhow!(
                                            credentials.redact_message(&error.to_string())
                                        )
                                    });
                            if completions
                                .send(LoadCompletion::RemoteList {
                                    session_generation,
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::RemoteOpen {
                            generation,
                            session_generation,
                            request,
                            spec,
                        } => {
                            let (identity, result) = match spec {
                                RemoteOpenSpec::Http { url } => {
                                    let result = remote_backend.open_http(&url).map(|opened| {
                                        complete_ome_zarr_resource(opened, Vec::new())
                                    });
                                    (RemoteOpenIdentity::Http { url }, result)
                                }
                                RemoteOpenSpec::S3 {
                                    credentials,
                                    prefix,
                                } => {
                                    let identity = RemoteOpenIdentity::S3 {
                                        endpoint: credentials.endpoint.clone(),
                                        region: credentials.region.clone(),
                                        bucket: credentials.bucket.clone(),
                                        prefix: prefix.clone(),
                                    };
                                    let result = remote_backend
                                        .open_s3(&credentials, &prefix)
                                        .map(|opened| {
                                            complete_ome_zarr_resource(opened, Vec::new())
                                        })
                                        .map_err(|error| {
                                            anyhow::anyhow!(
                                                credentials.redact_message(&error.to_string())
                                            )
                                        });
                                    (identity, result)
                                }
                            };
                            if completions
                                .send(LoadCompletion::RemoteOpen {
                                    generation,
                                    session_generation,
                                    request,
                                    identity,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ChannelIntensity {
                            generation,
                            operation_generation,
                            request,
                            document,
                            spec,
                        } => {
                            let result = read_channel_intensity_stats(&document, &spec);
                            if completions
                                .send(LoadCompletion::ChannelIntensity {
                                    generation,
                                    operation_generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AutoContrast {
                            spec,
                            request,
                            document,
                        } => {
                            let result = read_auto_contrast(&document, &spec);
                            if completions
                                .send(LoadCompletion::AutoContrast {
                                    spec,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectOpen {
                            generation,
                            request,
                            path,
                        } => {
                            let result = read_project_file(&path);
                            if completions
                                .send(LoadCompletion::ProjectOpen {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectSave {
                            generation,
                            request,
                            path,
                            payload,
                            saved_config_generation,
                            platform_effect,
                        } => {
                            let result = serde_json::to_string_pretty(&payload)
                                .map_err(anyhow::Error::from)
                                .and_then(|text| {
                                    fs::write(&path, text).map_err(anyhow::Error::from)
                                });
                            if completions
                                .send(LoadCompletion::ProjectSave {
                                    generation,
                                    request,
                                    path,
                                    saved_config_generation,
                                    platform_effect,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SettingsSave {
                            generation,
                            request,
                            path,
                            settings,
                            response,
                        } => {
                            let result = settings.save_to(&path).map(|()| path);
                            if completions
                                .send(LoadCompletion::SettingsSave {
                                    generation,
                                    request,
                                    settings,
                                    response,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SettingsPersist {
                            generation,
                            path,
                            settings,
                            response,
                        } => {
                            let result = settings.save_to(&path).map(|()| path);
                            if completions
                                .send(LoadCompletion::SettingsPersist {
                                    generation,
                                    settings,
                                    response,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ScreenshotSettingsValidate {
                            generation,
                            request,
                            preferences,
                        } => {
                            let result = preferences
                                .validate_output_dir()
                                .map_err(|error| anyhow::anyhow!(error.message));
                            if completions
                                .send(LoadCompletion::ScreenshotSettingsValidate {
                                    generation,
                                    request,
                                    preferences,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ScreenshotWrite {
                            request,
                            spec,
                            pixels,
                        } => {
                            let result = write_screenshot_on_worker(&request, &spec, pixels);
                            if completions
                                .send(LoadCompletion::ScreenshotWrite {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectViewApply {
                            request,
                            spec,
                            document,
                        } => {
                            let result = (|| -> anyhow::Result<ProjectViewApplyWorkerResult> {
                                let cancelled = || {
                                    request
                                        .task_id
                                        .as_deref()
                                        .and_then(|task_id| request.task_registry.get(task_id).ok())
                                        .is_some_and(|task| task.state == TaskState::Cancelled)
                                };
                                anyhow::ensure!(
                                    !cancelled(),
                                    "saved-view application was cancelled"
                                );
                                let object_resource = spec
                                    .object_path
                                    .as_ref()
                                    .map(|path| {
                                        object_loader
                                            .as_ref()
                                            .ok_or_else(|| {
                                                anyhow::anyhow!(
                                                    "object resource loader is unavailable"
                                                )
                                            })?
                                            .load_with_options(
                                                path.clone(),
                                                1.0,
                                                Some(
                                                    ProjectObjectPreloadSettings::default()
                                                        .worker_options(),
                                                ),
                                            )
                                            .map(Arc::new)
                                    })
                                    .transpose()?;
                                anyhow::ensure!(
                                    !cancelled(),
                                    "saved-view application was cancelled"
                                );
                                let label_resource = spec
                                    .label_name
                                    .as_deref()
                                    .map(|name| load_label_resource(&document, name).map(Arc::new))
                                    .transpose()?;
                                Ok(ProjectViewApplyWorkerResult {
                                    object_resource,
                                    label_resource,
                                })
                            })();
                            if completions
                                .send(LoadCompletion::ProjectViewApply {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MemoryPin {
                            request,
                            document,
                            spec,
                        } => {
                            let result = load_pinned_memory_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::MemoryPin {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicMemoryPin { request, spec } => {
                            let result = load_mosaic_pinned_memory_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::MosaicMemoryPin {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdLoad {
                            request,
                            document,
                            spec,
                        } => {
                            let result =
                                load_threshold_preview_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdLoad {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdRecompute { request, spec } => {
                            let result = recompute_threshold_preview_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdRecompute {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ThresholdApply { request, spec } => {
                            let result = apply_threshold_preview_on_worker(&spec, &request);
                            if completions
                                .send(LoadCompletion::ThresholdApply {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisCompute {
                            request,
                            spec,
                            kind,
                            params,
                        } => {
                            let result = compute_analysis_on_worker(&spec, kind, &params, &request);
                            if completions
                                .send(LoadCompletion::AnalysisCompute {
                                    request,
                                    spec,
                                    kind,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisPresetImport {
                            request,
                            spec,
                            path,
                        } => {
                            let result = read_analysis_preset_on_worker(&path);
                            if completions
                                .send(LoadCompletion::AnalysisPresetImport {
                                    request,
                                    spec,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::AnalysisPresetExport {
                            request,
                            spec,
                            path,
                            overwrite,
                            state,
                        } => {
                            let result = write_analysis_preset_on_worker(&path, overwrite, &state);
                            if completions
                                .send(LoadCompletion::AnalysisPresetExport {
                                    request,
                                    spec,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Measurement {
                            request,
                            document,
                            spec,
                        } => {
                            let result = measure_objects_on_worker(&document, &spec, &request);
                            if completions
                                .send(LoadCompletion::Measurement {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectExport { request, spec } => {
                            let result =
                                write_object_export(&spec, || worker_request_cancelled(&request));
                            if completions
                                .send(LoadCompletion::ObjectExport {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicSamplesheet {
                            generation,
                            request,
                            path,
                        } => {
                            let columns = request
                                .command
                                .params()
                                .get("columns")
                                .or_else(|| request.command.params().get("cols"))
                                .and_then(Value::as_u64)
                                .and_then(|columns| usize::try_from(columns).ok())
                                .filter(|columns| *columns > 0);
                            let result =
                                open_mosaic_samplesheet_on_worker(generation, &path, columns);
                            if completions
                                .send(LoadCompletion::MosaicOpen {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicProject {
                            generation,
                            request,
                            rois,
                            project_dir,
                            s3_session,
                        } => {
                            let result = open_mosaic_project_on_worker(
                                generation,
                                rois,
                                project_dir,
                                s3_session,
                                remote_backend.as_ref(),
                            );
                            if completions
                                .send(LoadCompletion::MosaicOpen {
                                    generation,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MosaicObjects { request, spec } => {
                            let result =
                                load_mosaic_objects_on_worker(&spec, object_loader.as_deref());
                            if completions
                                .send(LoadCompletion::MosaicObjects {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetInspect {
                            request,
                            path,
                            offset,
                            limit,
                        } => {
                            let result = inspect_samplesheet(&path, offset, limit);
                            if completions
                                .send(LoadCompletion::SamplesheetInspect { request, result })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetImport {
                            generation,
                            request,
                            path,
                            default_dataset,
                        } => {
                            let result = import_samplesheet_rois(&path, &default_dataset);
                            if completions
                                .send(LoadCompletion::SamplesheetImport {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SamplesheetExport {
                            generation,
                            request,
                            path,
                            rois,
                            overwrite,
                        } => {
                            let result = export_samplesheet_rois(&path, &rois, overwrite);
                            if completions
                                .send(LoadCompletion::SamplesheetExport {
                                    generation,
                                    request,
                                    path,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectDiscovery {
                            generation,
                            request,
                            root,
                        } => {
                            let result = discover_omezarr_roots_under(&root);
                            if completions
                                .send(LoadCompletion::ProjectDiscovery {
                                    generation,
                                    request,
                                    root,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectObjectSourceScan {
                            scope,
                            request,
                            candidates,
                        } => {
                            let result = scan_project_object_sources(candidates);
                            if completions
                                .send(LoadCompletion::ProjectObjectSourceScan {
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ProjectObjectPreload {
                            generation,
                            scope,
                            request,
                            settings,
                            candidates,
                        } => {
                            let result = preload_project_objects(
                                object_loader.as_deref(),
                                &request,
                                settings,
                                candidates,
                            );
                            if completions
                                .send(LoadCompletion::ProjectObjectPreload {
                                    generation,
                                    scope,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectResource {
                            document_generation,
                            resource_generation,
                            request,
                            path,
                            downsample_factor,
                            options,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object resource loader is unavailable"),
                                |loader| {
                                    loader.load_with_options(
                                        path.clone(),
                                        downsample_factor,
                                        options,
                                    )
                                },
                            );
                            if completions
                                .send(LoadCompletion::ObjectResource {
                                    document_generation,
                                    resource_generation,
                                    request,
                                    path,
                                    downsample_factor,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::SegmentationGeoJson { request, spec } => {
                            let result = crate::data::segmentation_geojson::load_geojson_polyline_coordinates_world(
                                    &spec.path,
                                    spec.downsample_factor,
                                    crate::data::segmentation_geojson::PolygonRingMode::ExteriorOnly,
                                )
                                .map(|polylines| {
                                    let segment_count = polylines
                                        .iter()
                                        .map(|line| line.len().saturating_sub(1))
                                        .sum();
                                    ControlSegmentationGeoJsonResource {
                                        path: spec.path.clone(),
                                        downsample_factor: spec.downsample_factor,
                                        polylines: Arc::new(polylines),
                                        segment_count,
                                    }
                                });
                            if completions
                                .send(LoadCompletion::SegmentationGeoJson {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Labels {
                            document_generation,
                            label_generation,
                            request,
                            document,
                            name,
                        } => {
                            let result = if document.dataset().is_root_label_mask() {
                                let labels =
                                    LabelZarrDataset::from_root_dataset(document.dataset());
                                if labels.label_name == name {
                                    Ok(ControlLabelResource {
                                        dataset: labels,
                                        store: Arc::clone(document.store()),
                                    })
                                } else {
                                    Err(anyhow::anyhow!(
                                        "top-level label mask is named '{}', not '{}'",
                                        labels.label_name,
                                        name
                                    ))
                                }
                            } else {
                                match LabelZarrDataset::try_open(
                                    Arc::clone(document.store()),
                                    &name,
                                ) {
                                    Ok(Some(dataset)) => Ok(ControlLabelResource {
                                        dataset,
                                        store: Arc::clone(document.store()),
                                    }),
                                    Ok(None) => {
                                        Err(anyhow::anyhow!("no labels/{name} found in this ROI"))
                                    }
                                    Err(error) => Err(error),
                                }
                            };
                            if completions
                                .send(LoadCompletion::Labels {
                                    document_generation,
                                    label_generation,
                                    request,
                                    name,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::Annotations { request, spec } => {
                            let result = (|| -> anyhow::Result<AnnotationLoadResult> {
                                let schema = read_parquet_columns(&spec.path)?;
                                let dataset = if spec.load_dataset {
                                    Some(load_annotations_parquet(
                                        &spec.path,
                                        &spec.roi_id_column,
                                        &spec.x_column,
                                        &spec.y_column,
                                        &spec.value_column,
                                    )?)
                                } else {
                                    None
                                };
                                Ok(AnnotationLoadResult { schema, dataset })
                            })();
                            if completions
                                .send(LoadCompletion::Annotations {
                                    request,
                                    spec,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectFilter {
                            document_generation,
                            resource_generation,
                            operation_generation,
                            viewport_id,
                            target,
                            expected_presentation_revision,
                            request,
                            resource,
                            model,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object filter evaluator is unavailable"),
                                |loader| loader.evaluate_filter(resource, model),
                            );
                            if completions
                                .send(LoadCompletion::ObjectFilter {
                                    document_generation,
                                    resource_generation,
                                    operation_generation,
                                    viewport_id,
                                    target,
                                    expected_presentation_revision,
                                    request,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::ObjectSelectionFilter {
                            document_generation,
                            resource_generation,
                            selection_generation,
                            operation_generation,
                            target,
                            request,
                            resource,
                            model,
                            mode,
                            limit,
                        } => {
                            let result = object_loader.as_ref().map_or_else(
                                || anyhow::bail!("object filter evaluator is unavailable"),
                                |loader| loader.evaluate_filter(resource, model),
                            );
                            if completions
                                .send(LoadCompletion::ObjectSelectionFilter {
                                    document_generation,
                                    resource_generation,
                                    selection_generation,
                                    operation_generation,
                                    target,
                                    request,
                                    mode,
                                    limit,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MaskImport {
                            document_generation,
                            mask_generation,
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                            name,
                            editable,
                            replace_layer_id,
                            downsample_factor,
                        } => {
                            let result =
                                crate::model::load_geojson_mask_polylines(&path, downsample_factor);
                            if completions
                                .send(LoadCompletion::MaskImport {
                                    document_generation,
                                    mask_generation,
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    path,
                                    name,
                                    editable,
                                    replace_layer_id,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MaskExport {
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                            layer_id,
                            layers,
                            overwrite,
                        } => {
                            let layer_count = layers.len();
                            let polygon_count =
                                layers.iter().map(|layer| layer.polygons_world.len()).sum();
                            let result = export_mask_layers_geojson(&path, &layers, overwrite);
                            if completions
                                .send(LoadCompletion::MaskExport {
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    path,
                                    layer_id,
                                    layer_count,
                                    polygon_count,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        LoadJob::MaskAppend {
                            document_generation,
                            mask_generation,
                            operation_generation,
                            operation_scope,
                            request,
                            path,
                            name,
                            downsample_factor,
                            roi_root,
                            saved_layers,
                        } => {
                            let cancelled = || {
                                request
                                    .task_id
                                    .as_deref()
                                    .and_then(|task_id| request.task_registry.get(task_id).ok())
                                    .is_some_and(|task| task.state == TaskState::Cancelled)
                            };
                            let result = append_mask_layers_geojson(
                                &path,
                                &saved_layers,
                                downsample_factor,
                                &roi_root,
                                cancelled,
                            );
                            if completions
                                .send(LoadCompletion::MaskAppend {
                                    document_generation,
                                    mask_generation,
                                    operation_generation,
                                    operation_scope,
                                    request,
                                    path,
                                    name,
                                    saved_layers,
                                    result,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                }
            })?;
    }

    Ok(())
}
