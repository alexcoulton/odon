use std::path::Path;

use odon::data::document::{
    DatasetElementInspection, DatasetElementTransform, DatasetInspection, DatasetInspectionKind,
    DatasetInspectionMetadata, DatasetInspector, OmeTiffChannelInspection, OmeTiffInspection,
    TiffChannelInspection, TiffInspectionMetadata, TiffLevelInspection, TiffPlaneInspection,
    TiffPlanesInspection, XeniumInspectionMetadata, inspect_core_dataset,
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
