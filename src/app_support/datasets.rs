use std::path::Path;
use std::sync::Arc;

use odon::data::document::{
    AlternateDatasetBackend, AlternateDocumentResource, AlternateIntensityData,
    AlternateIntensityReader, AlternateIntensityRequest, DatasetElementInspection,
    DatasetElementTransform, DatasetInspection, DatasetInspectionKind, DatasetInspectionMetadata,
    DatasetInspector, DocumentDescriptor, DocumentKind, DocumentObjectLayerResource,
    OmeTiffChannelInspection, OmeTiffInspection, OpenedDocument, SpatialDataOpenIdentity,
    SpatialDataOpenOptions, TiffChannelInspection, TiffInspectionMetadata, TiffLevelInspection,
    TiffPlaneInspection, TiffPlanesInspection, XeniumInspectionMetadata, XeniumOpenIdentity,
    XeniumOpenOptions, inspect_core_dataset,
};

use crate::data::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::spatialdata::{SpatialDataElement, discover_spatialdata};
use crate::xenium::{TiffPlaneSelection, TiffPyramid, discover_xenium_explorer};

/// Source-family metadata adapter used by the control actor's bounded workers.
///
/// This service contains no egui, window, or GPU state. It lives in the native binary only because
/// TIFF, SpatialData, and Xenium readers have not all moved into the core library yet.
pub(crate) struct NativeDatasetInspector;

pub(crate) struct NativeAlternateDatasetBackend;

#[derive(Clone)]
pub(crate) struct PreparedSpatialDataDocument {
    pub root: std::path::PathBuf,
    pub image_transform: crate::spatialdata::SpatialDataTransform2,
    pub extra_images: Vec<crate::spatialdata::PreparedSpatialImage>,
    pub labels: Option<SpatialDataElement>,
    pub tables: Vec<SpatialDataElement>,
    pub shapes: Vec<crate::spatialdata::PreparedSpatialShape>,
    pub points: Option<crate::spatialdata::PreparedSpatialPointsLayer>,
}

#[derive(Clone)]
pub(crate) enum PreparedXeniumImagery {
    OmeZarr,
    Tiff(Arc<TiffPyramid>),
}

impl AlternateIntensityReader for TiffPyramid {
    fn read_channel_region(
        &self,
        request: &AlternateIntensityRequest,
    ) -> anyhow::Result<AlternateIntensityData> {
        let (values, width, height) = self.read_channel_region_u16(
            request.level,
            request.channel,
            request.y0,
            request.y1,
            request.x0,
            request.x1,
        )?;
        Ok(AlternateIntensityData {
            values,
            shape: vec![height, width],
        })
    }
}

#[derive(Clone)]
pub(crate) struct PreparedXeniumDocument {
    pub root: std::path::PathBuf,
    pub imagery: PreparedXeniumImagery,
    pub cells: Option<crate::xenium::PreparedXeniumCells>,
    pub transcripts: Option<crate::xenium::PreparedXeniumTranscripts>,
    pub pixel_size_um: f32,
}

fn selected_spatial_element(
    elements: &[SpatialDataElement],
    name: &str,
    kind: &str,
) -> anyhow::Result<SpatialDataElement> {
    elements
        .iter()
        .find(|element| element.name == name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("SpatialData {kind} element '{name}' was not found"))
}

impl AlternateDatasetBackend for NativeAlternateDatasetBackend {
    fn open_tiff(
        &self,
        path: &Path,
        z: usize,
        t: usize,
    ) -> anyhow::Result<OpenedDocument<AlternateDocumentResource>> {
        let pyramid = Arc::new(TiffPyramid::open_with_selection(
            path,
            TiffPlaneSelection { z, t },
        )?);
        pyramid.validate_supported_ome_layout()?;
        let dataset_name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("tiff")
            .to_string();
        let dataset = crate::app::build_tiff_dataset(
            path.to_path_buf(),
            dataset_name,
            pyramid.to_levels_info(),
            pyramid.dims(),
            pyramid.default_channels_named("image"),
            pyramid.abs_max,
            pyramid.physical_pixel_size_xy(),
        );
        let store = crate::app::dummy_local_store_for_path(path)?;
        let descriptor = DocumentDescriptor::from_alternate(&dataset, DocumentKind::Tiff);
        Ok(OpenedDocument {
            descriptor,
            resource: AlternateDocumentResource::new(dataset, store, Arc::clone(&pyramid))
                .with_intensity_reader(pyramid),
        })
    }

    fn open_spatialdata(
        &self,
        path: &Path,
        options: &SpatialDataOpenOptions,
    ) -> anyhow::Result<(
        OpenedDocument<AlternateDocumentResource>,
        SpatialDataOpenIdentity,
    )> {
        let discovery = discover_spatialdata(path)?;
        let image = selected_spatial_element(&discovery.images, &options.image, "image")?;
        let extra_images = options
            .extra_images
            .iter()
            .map(|name| selected_spatial_element(&discovery.images, name, "image"))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .filter(|element| element.name != image.name)
            .collect::<Vec<_>>();
        let labels = options
            .labels
            .as_deref()
            .map(|name| selected_spatial_element(&discovery.labels, name, "label"))
            .transpose()?;
        let shapes = options
            .shapes
            .iter()
            .map(|name| selected_spatial_element(&discovery.shapes, name, "shape"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let points = options
            .points
            .as_deref()
            .map(|name| selected_spatial_element(&discovery.points, name, "points"))
            .transpose()?;
        let image_root = discovery.root.join(&image.rel_group);
        let (dataset, store) = crate::data::ome::OmeZarrDataset::open_local(&image_root)?;
        let descriptor = DocumentDescriptor::from_alternate(&dataset, DocumentKind::SpatialData);
        let identity = SpatialDataOpenIdentity {
            root: discovery.root.clone(),
            image: image.name.clone(),
            extra_images: extra_images
                .iter()
                .map(|element| element.name.clone())
                .collect(),
            labels: labels.as_ref().map(|element| element.name.clone()),
            shapes: shapes.iter().map(|element| element.name.clone()).collect(),
            points: points.as_ref().map(|element| element.name.clone()),
            points_max: options.points_max,
        };
        let prepared_extra_images = extra_images
            .iter()
            .cloned()
            .map(|element| {
                let image_root = discovery.root.join(&element.rel_group);
                let (dataset, store) = crate::data::ome::OmeZarrDataset::open_local(&image_root)?;
                Ok(crate::spatialdata::PreparedSpatialImage {
                    element,
                    dataset,
                    store,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut next_shape_layer_id = 1_u64;
        let mut object_layers = Vec::new();
        let mut prepared_shapes = Vec::new();
        for shape in &shapes {
            let Some(relative_path) = shape.rel_parquet.as_ref() else {
                continue;
            };
            let parquet_path = discovery.root.join(relative_path);
            let render_kind = crate::spatialdata::detect_shapes_render_kind(&parquet_path)?;
            let supports_objects = crate::spatialdata::shapes_support_object_layer(&parquet_path)?;
            let primary = shape.name == "cell_boundaries";
            let actor_object_layer = primary
                || matches!(
                    render_kind,
                    crate::spatialdata::ShapesRenderKind::Points
                        | crate::spatialdata::ShapesRenderKind::Circles
                )
                || (render_kind == crate::spatialdata::ShapesRenderKind::Lines && supports_objects);
            let numeric_layer_id = (!primary).then(|| {
                let id = next_shape_layer_id;
                next_shape_layer_id = next_shape_layer_id.wrapping_add(1).max(1);
                id
            });
            let layer_id = if primary {
                "segmentation_objects".to_string()
            } else {
                format!(
                    "spatial_shape:{}",
                    numeric_layer_id.expect("non-primary ID")
                )
            };
            let transform = shape.transform.relative_to(image.transform);
            if actor_object_layer {
                let resource = crate::objects::load_control_spatialdata_object_resource(
                    parquet_path.clone(),
                    transform,
                )?;
                object_layers.push(DocumentObjectLayerResource {
                    layer_id,
                    name: shape.name.clone(),
                    kind: if primary {
                        "segmentation_objects".to_string()
                    } else {
                        "spatial_shape".to_string()
                    },
                    primary,
                    resource: Arc::new(resource),
                });
            }
            if let Some(id) = numeric_layer_id {
                let mut element = shape.clone();
                element.transform = transform;
                let data = if actor_object_layer {
                    None
                } else {
                    Some(crate::spatialdata::prepare_spatial_shape_data(
                        &parquet_path,
                        transform,
                    )?)
                };
                prepared_shapes.push(crate::spatialdata::PreparedSpatialShape {
                    id,
                    element,
                    object_backed: actor_object_layer,
                    data,
                });
            }
        }
        let prepared_points = points
            .as_ref()
            .map(|element| {
                let relative_path = element.rel_parquet.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("SpatialData points element has no parquet path")
                })?;
                let transform = element.transform.relative_to(image.transform);
                let image_size_world = dataset.levels.first().and_then(|level| {
                    Some([
                        *level.shape.get(dataset.dims.x)? as f32,
                        *level.shape.get(dataset.dims.y)? as f32,
                    ])
                });
                crate::spatialdata::prepare_spatial_points_layer(
                    format!("Points: {}", element.name),
                    discovery.root.join(relative_path),
                    transform,
                    element.feature_key.clone(),
                    options.points_max,
                    image_size_world,
                )
            })
            .transpose()?;
        let payload = PreparedSpatialDataDocument {
            root: discovery.root,
            image_transform: image.transform,
            extra_images: prepared_extra_images,
            labels,
            tables: discovery.tables,
            shapes: prepared_shapes,
            points: prepared_points,
        };
        Ok((
            OpenedDocument {
                descriptor,
                resource: AlternateDocumentResource::new(dataset, store, Arc::new(payload))
                    .with_object_layers(object_layers),
            },
            identity,
        ))
    }

    fn open_xenium(
        &self,
        path: &Path,
        options: &XeniumOpenOptions,
    ) -> anyhow::Result<(
        OpenedDocument<AlternateDocumentResource>,
        XeniumOpenIdentity,
    )> {
        let discovery = discover_xenium_explorer(path)?;
        let selected = match options.imagery.as_str() {
            "ome_zarr" => discovery
                .morphology_mip_omezarr
                .clone()
                .map(|path| ("ome_zarr", path)),
            "tiff" => discovery
                .morphology_mip_tiff
                .clone()
                .map(|path| ("tiff", path)),
            _ => discovery
                .morphology_mip_omezarr
                .clone()
                .map(|path| ("ome_zarr", path))
                .or_else(|| {
                    discovery
                        .morphology_mip_tiff
                        .clone()
                        .map(|path| ("tiff", path))
                }),
        };
        let Some((imagery_kind, imagery_path)) = selected else {
            anyhow::bail!(
                "requested Xenium {} imagery is unavailable",
                options.imagery
            );
        };
        let cells = options
            .load_cells
            .then(|| discovery.cells_zarr_zip.clone())
            .flatten();
        let transcripts = options
            .load_transcripts
            .then(|| discovery.transcripts_zarr_zip.clone())
            .flatten();
        let prepared_cells = cells
            .as_ref()
            .map(|path| {
                crate::xenium::load_cells_outline_bins(
                    path,
                    crate::xenium::XeniumPolygonSet::Cell,
                    discovery.pixel_size_um,
                )
                .map(|bins| crate::xenium::PreparedXeniumCells {
                    path: path.clone(),
                    bins,
                })
            })
            .transpose()?;
        let prepared_transcripts = transcripts
            .as_ref()
            .map(|path| {
                let meta = Arc::new(crate::xenium::load_transcripts_meta(path)?);
                let payload = Arc::new(crate::xenium::load_transcripts_all_points(
                    path,
                    &meta,
                    discovery.pixel_size_um,
                    0,
                )?);
                Ok::<_, anyhow::Error>(crate::xenium::PreparedXeniumTranscripts {
                    path: path.clone(),
                    meta,
                    payload,
                })
            })
            .transpose()?;
        let (dataset, store, imagery) = if imagery_kind == "ome_zarr" {
            let (dataset, store) = crate::data::ome::OmeZarrDataset::open_local(&imagery_path)?;
            (dataset, store, PreparedXeniumImagery::OmeZarr)
        } else {
            let pyramid = Arc::new(TiffPyramid::open_with_selection(
                &imagery_path,
                TiffPlaneSelection { z: 0, t: 0 },
            )?);
            pyramid.validate_supported_ome_layout()?;
            let dataset = crate::app::build_tiff_dataset(
                discovery.root.clone(),
                "xenium".to_string(),
                pyramid.to_levels_info(),
                pyramid.dims(),
                pyramid.default_channels_named("morphology"),
                pyramid.abs_max,
                pyramid.physical_pixel_size_xy(),
            );
            let store = crate::app::dummy_local_store_for_path(&discovery.root)?;
            (dataset, store, PreparedXeniumImagery::Tiff(pyramid))
        };
        let descriptor = DocumentDescriptor::from_alternate(&dataset, DocumentKind::Xenium);
        let identity = XeniumOpenIdentity {
            root: discovery.root.clone(),
            imagery: imagery_kind.to_string(),
            imagery_path,
            cells_loaded: prepared_cells.is_some(),
            transcripts_loaded: prepared_transcripts.is_some(),
            pixel_size_um: discovery.pixel_size_um,
        };
        let payload = PreparedXeniumDocument {
            root: discovery.root,
            imagery,
            cells: prepared_cells,
            transcripts: prepared_transcripts,
            pixel_size_um: discovery.pixel_size_um,
        };
        let intensity_reader = match &payload.imagery {
            PreparedXeniumImagery::Tiff(pyramid) => {
                Some(Arc::clone(pyramid) as Arc<dyn AlternateIntensityReader>)
            }
            PreparedXeniumImagery::OmeZarr => None,
        };
        let mut resource = AlternateDocumentResource::new(dataset, store, Arc::new(payload));
        if let Some(reader) = intensity_reader {
            resource = resource.with_intensity_reader(reader);
        }
        Ok((
            OpenedDocument {
                descriptor,
                resource,
            },
            identity,
        ))
    }
}

impl DatasetInspector for NativeDatasetInspector {
    fn inspect(&self, path: &Path) -> DatasetInspection {
        inspect_native_dataset(path)
    }
}

pub(crate) fn inspect_native_dataset(path: &Path) -> DatasetInspection {
    if !path.exists() {
        return DatasetInspection::missing(path.to_path_buf());
    }
    let normalized = normalize_local_dataset_path(path).unwrap_or_else(|| path.to_path_buf());

    if normalized.is_dir()
        && let Ok(discovery) = discover_spatialdata(&normalized)
    {
        let elements = |kind: &str, values: &[SpatialDataElement]| {
            values
                .iter()
                .map(|element| DatasetElementInspection {
                    kind: kind.to_string(),
                    name: element.name.clone(),
                    path: element.rel_group.clone(),
                    parquet_path: element.rel_parquet.clone(),
                    transform: DatasetElementTransform {
                        scale: element.transform.scale,
                        translation: element.transform.translation,
                    },
                    feature_key: element.feature_key.clone(),
                })
                .collect::<Vec<_>>()
        };
        let mut all = Vec::new();
        all.extend(elements("image", &discovery.images));
        all.extend(elements("label", &discovery.labels));
        all.extend(elements("points", &discovery.points));
        all.extend(elements("shape", &discovery.shapes));
        all.extend(elements("table", &discovery.tables));
        return DatasetInspection {
            kind: Some(DatasetInspectionKind::SpatialData),
            path: discovery.root,
            can_open: Some(!discovery.images.is_empty()),
            metadata: None,
            elements: Some(all),
            error: None,
        };
    }

    match classify_local_dataset_path(&normalized) {
        Some(LocalDatasetKind::OmeZarr) => inspect_core_dataset(&normalized),
        Some(LocalDatasetKind::Tiff) => {
            match TiffPyramid::open_with_selection(&normalized, TiffPlaneSelection { z: 0, t: 0 }) {
                Ok(pyramid) => {
                    let channels = pyramid.default_channels_named("image");
                    DatasetInspection {
                        kind: Some(DatasetInspectionKind::Tiff),
                        path: normalized.clone(),
                        can_open: Some(true),
                        metadata: Some(DatasetInspectionMetadata::Tiff(TiffInspectionMetadata {
                            file_size_bytes: std::fs::metadata(&normalized)
                                .ok()
                                .map(|metadata| metadata.len()),
                            pixel_dtype: pyramid.pixel_dtype,
                            absolute_max: pyramid.abs_max,
                            channel_count: pyramid.channel_count,
                            channels: channels
                                .into_iter()
                                .map(|channel| TiffChannelInspection {
                                    index: channel.index,
                                    name: channel.name,
                                    color_rgb: channel.color_rgb,
                                })
                                .collect(),
                            planes: TiffPlanesInspection {
                                size_z: pyramid.size_z,
                                size_t: pyramid.size_t,
                                default: TiffPlaneInspection { z: 0, t: 0 },
                            },
                            levels: pyramid
                                .levels
                                .iter()
                                .enumerate()
                                .map(|(index, level)| TiffLevelInspection {
                                    index,
                                    width: level.width,
                                    height: level.height,
                                    chunk_width: level.chunk_w,
                                    chunk_height: level.chunk_h,
                                    tiles_x: level.tiles_x,
                                    tiles_y: level.tiles_y,
                                    channels: level.channels,
                                    channel_layout: format!("{:?}", level.channel_layout)
                                        .to_ascii_lowercase(),
                                })
                                .collect(),
                            ome: pyramid.ome.map(|ome| OmeTiffInspection {
                                dimension_order: ome.dimension_order,
                                size_z: ome.size_z,
                                size_t: ome.size_t,
                                size_c: ome.size_c,
                                physical_size_x: ome.physical_size_x,
                                physical_size_x_unit: ome.physical_size_x_unit,
                                physical_size_y: ome.physical_size_y,
                                physical_size_y_unit: ome.physical_size_y_unit,
                                channels: ome
                                    .channels
                                    .into_iter()
                                    .map(|channel| OmeTiffChannelInspection {
                                        name: channel.name,
                                        color_rgb: channel.color_rgb,
                                    })
                                    .collect(),
                            }),
                        })),
                        elements: None,
                        error: None,
                    }
                }
                Err(error) => DatasetInspection::failed(
                    DatasetInspectionKind::Tiff,
                    normalized,
                    format!("failed to inspect TIFF: {error}"),
                ),
            }
        }
        Some(LocalDatasetKind::Xenium) => match discover_xenium_explorer(&normalized) {
            Ok(discovery) => DatasetInspection {
                kind: Some(DatasetInspectionKind::Xenium),
                path: discovery.root,
                can_open: Some(
                    discovery.morphology_mip_omezarr.is_some()
                        || discovery.morphology_mip_tiff.is_some(),
                ),
                metadata: Some(DatasetInspectionMetadata::Xenium(
                    XeniumInspectionMetadata {
                        pixel_size_um: discovery.pixel_size_um,
                        morphology_mip_ome_zarr: discovery.morphology_mip_omezarr,
                        morphology_mip_tiff: discovery.morphology_mip_tiff,
                        transcripts_zarr_zip: discovery.transcripts_zarr_zip,
                        cells_zarr_zip: discovery.cells_zarr_zip,
                    },
                )),
                elements: None,
                error: None,
            },
            Err(error) => DatasetInspection::failed(
                DatasetInspectionKind::Xenium,
                normalized,
                format!("failed to inspect Xenium dataset: {error}"),
            ),
        },
        None => DatasetInspection::failed(
            DatasetInspectionKind::Unsupported,
            normalized,
            "path is not a supported OME-Zarr, TIFF, SpatialData, or Xenium source",
        ),
    }
}
