use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use zarrs::storage::ReadableStorageTraits;

use super::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use super::dataset_source::DatasetSource;
use super::ome::{Axis, ChannelInfo, DatasetRenderKind, Dims, LevelInfo, OmeZarrDataset};

/// Semantic dataset families understood by the control model.
///
/// This type deliberately describes the opened document rather than its renderer. A source may
/// use a different renderer/runtime without changing the actor-owned document identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    OmeZarr,
    Tiff,
    SpatialData,
    Xenium,
}

/// Send-compatible metadata installed into the canonical model for an opened document.
#[derive(Debug, Clone)]
pub struct DocumentDescriptor {
    pub kind: DocumentKind,
    pub source: DatasetSource,
    pub name: Option<String>,
    pub axes: Vec<Axis>,
    pub levels: Vec<LevelInfo>,
    pub channels: Vec<ChannelInfo>,
    pub dims: Dims,
    pub abs_max: f32,
    pub render_kind: DatasetRenderKind,
}

impl DocumentDescriptor {
    pub fn from_ome_zarr(dataset: &OmeZarrDataset) -> Self {
        Self {
            kind: DocumentKind::OmeZarr,
            source: dataset.source.clone(),
            name: dataset.multiscale.name.clone(),
            axes: dataset.multiscale.axes.clone(),
            levels: dataset.levels.clone(),
            channels: dataset.channels.clone(),
            dims: dataset.dims.clone(),
            abs_max: dataset.abs_max,
            render_kind: dataset.render_kind,
        }
    }
}

/// A model descriptor paired with a source-specific, Send-compatible resource handle.
///
/// The generic resource keeps this boundary strongly typed. The actor may later wrap supported
/// resource variants in an enum; the semantic descriptor does not need to change when it does.
#[derive(Clone)]
pub struct OpenedDocument<R> {
    pub descriptor: DocumentDescriptor,
    pub resource: R,
}

#[derive(Clone)]
pub struct OmeZarrDocumentResource {
    pub dataset: OmeZarrDataset,
    pub store: Arc<dyn ReadableStorageTraits>,
    /// Keeps source-specific asynchronous infrastructure alive for synchronous storage adapters.
    /// Local and HTTP stores do not need one; S3 stores retain their Tokio runtime here.
    pub runtime_guard: Option<Arc<tokio::runtime::Runtime>>,
}

pub fn open_local_ome_zarr(path: &Path) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
    let (dataset, store) = OmeZarrDataset::open_local(path)?;
    Ok(OpenedDocument {
        descriptor: DocumentDescriptor::from_ome_zarr(&dataset),
        resource: OmeZarrDocumentResource {
            dataset,
            store,
            runtime_guard: None,
        },
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetInspectionKind {
    OmeZarr,
    Tiff,
    SpatialData,
    Xenium,
    Unsupported,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetElementInspection {
    pub kind: String,
    pub name: String,
    pub path: PathBuf,
    pub parquet_path: Option<PathBuf>,
    pub transform: DatasetElementTransform,
    pub feature_key: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DatasetElementTransform {
    pub scale: [f32; 2],
    pub translation: [f32; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct OmeZarrInspectionMetadata {
    pub name: Option<String>,
    pub axes: Vec<AxisInspection>,
    pub level_count: usize,
    pub levels: Vec<OmeZarrLevelInspection>,
    pub channels: Vec<ChannelInspection>,
    pub dimensions: DimensionsInspection,
    pub absolute_max: f32,
}

impl OmeZarrInspectionMetadata {
    pub fn from_descriptor(descriptor: &DocumentDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            axes: descriptor
                .axes
                .iter()
                .map(|axis| AxisInspection {
                    name: axis.name.clone(),
                    unit: axis.unit.clone(),
                })
                .collect(),
            level_count: descriptor.levels.len(),
            levels: descriptor
                .levels
                .iter()
                .map(|level| OmeZarrLevelInspection {
                    index: level.index,
                    path: level.path.clone(),
                    shape: level.shape.clone(),
                    chunks: level.chunks.clone(),
                    dtype: level.dtype.clone(),
                    scale: level.scale.clone(),
                    translation: level.translation.clone(),
                })
                .collect(),
            channels: descriptor
                .channels
                .iter()
                .map(|channel| ChannelInspection {
                    index: channel.index,
                    name: channel.name.clone(),
                    color_rgb: channel.color_rgb,
                    window: channel.window.map(|(min, max)| [min, max]),
                })
                .collect(),
            dimensions: DimensionsInspection {
                ndim: descriptor.dims.ndim,
                c: descriptor.dims.c,
                z: descriptor.dims.z,
                y: descriptor.dims.y,
                x: descriptor.dims.x,
            },
            absolute_max: descriptor.abs_max,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AxisInspection {
    pub name: String,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OmeZarrLevelInspection {
    pub index: usize,
    pub path: String,
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
    pub scale: Vec<f32>,
    pub translation: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChannelInspection {
    pub index: usize,
    pub name: String,
    pub color_rgb: [u8; 3],
    pub window: Option<[f32; 2]>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DimensionsInspection {
    pub ndim: usize,
    pub c: Option<usize>,
    pub z: Option<usize>,
    pub y: usize,
    pub x: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TiffInspectionMetadata {
    pub file_size_bytes: Option<u64>,
    pub pixel_dtype: String,
    pub absolute_max: f32,
    pub channel_count: usize,
    pub channels: Vec<TiffChannelInspection>,
    pub planes: TiffPlanesInspection,
    pub levels: Vec<TiffLevelInspection>,
    pub ome: Option<OmeTiffInspection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TiffChannelInspection {
    pub index: usize,
    pub name: String,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TiffPlanesInspection {
    pub size_z: usize,
    pub size_t: usize,
    pub default: TiffPlaneInspection,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct TiffPlaneInspection {
    pub z: usize,
    pub t: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TiffLevelInspection {
    pub index: usize,
    pub width: u32,
    pub height: u32,
    pub chunk_width: u32,
    pub chunk_height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub channels: usize,
    pub channel_layout: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OmeTiffInspection {
    pub dimension_order: Option<String>,
    pub size_z: Option<usize>,
    pub size_t: Option<usize>,
    pub size_c: Option<usize>,
    pub physical_size_x: Option<f32>,
    pub physical_size_x_unit: Option<String>,
    pub physical_size_y: Option<f32>,
    pub physical_size_y_unit: Option<String>,
    pub channels: Vec<OmeTiffChannelInspection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OmeTiffChannelInspection {
    pub name: Option<String>,
    pub color_rgb: Option<[u8; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct XeniumInspectionMetadata {
    pub pixel_size_um: f32,
    pub morphology_mip_ome_zarr: Option<PathBuf>,
    pub morphology_mip_tiff: Option<PathBuf>,
    pub transcripts_zarr_zip: Option<PathBuf>,
    pub cells_zarr_zip: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum DatasetInspectionMetadata {
    OmeZarr(OmeZarrInspectionMetadata),
    Tiff(TiffInspectionMetadata),
    Xenium(XeniumInspectionMetadata),
}

/// Stable typed result for `datasets.inspect`. Optional fields are omitted to retain the existing
/// wire shape for source families that do not use them.
#[derive(Debug, Clone, Serialize)]
pub struct DatasetInspection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<DatasetInspectionKind>,
    pub path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub can_open: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<DatasetInspectionMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elements: Option<Vec<DatasetElementInspection>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl DatasetInspection {
    pub fn failed(kind: DatasetInspectionKind, path: PathBuf, message: impl Into<String>) -> Self {
        Self {
            kind: Some(kind),
            path,
            can_open: Some(false),
            metadata: None,
            elements: None,
            error: Some(message.into()),
        }
    }

    pub fn missing(path: PathBuf) -> Self {
        Self {
            kind: None,
            path,
            can_open: None,
            metadata: None,
            elements: None,
            error: Some("dataset path does not exist".to_string()),
        }
    }
}

/// Native and headless frontends can provide source-family inspection without giving the actor
/// access to renderer state. Implementations are invoked only on the bounded resource workers.
pub trait DatasetInspector: Send + Sync + 'static {
    fn inspect(&self, path: &Path) -> DatasetInspection;
}

#[derive(Debug, Default)]
pub struct CoreDatasetInspector;

impl DatasetInspector for CoreDatasetInspector {
    fn inspect(&self, path: &Path) -> DatasetInspection {
        inspect_core_dataset(path)
    }
}

pub fn inspect_core_dataset(path: &Path) -> DatasetInspection {
    if !path.exists() {
        return DatasetInspection::missing(path.to_path_buf());
    }
    let normalized = normalize_local_dataset_path(path).unwrap_or_else(|| path.to_path_buf());
    match classify_local_dataset_path(&normalized) {
        Some(LocalDatasetKind::OmeZarr) => match open_local_ome_zarr(&normalized) {
            Ok(opened) => DatasetInspection {
                kind: Some(DatasetInspectionKind::OmeZarr),
                path: normalized,
                can_open: Some(true),
                metadata: Some(DatasetInspectionMetadata::OmeZarr(
                    OmeZarrInspectionMetadata::from_descriptor(&opened.descriptor),
                )),
                elements: None,
                error: None,
            },
            Err(error) => DatasetInspection::failed(
                DatasetInspectionKind::OmeZarr,
                normalized,
                format!("failed to inspect OME-Zarr: {error}"),
            ),
        },
        Some(LocalDatasetKind::Tiff) => DatasetInspection::failed(
            DatasetInspectionKind::Tiff,
            normalized,
            "TIFF inspection requires Odon's native dataset provider",
        ),
        Some(LocalDatasetKind::Xenium) => DatasetInspection::failed(
            DatasetInspectionKind::Xenium,
            normalized,
            "Xenium inspection requires Odon's native dataset provider",
        ),
        None => DatasetInspection::failed(
            DatasetInspectionKind::Unsupported,
            normalized,
            "path is not a supported OME-Zarr, TIFF, SpatialData, or Xenium source",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr")
    }

    #[test]
    fn descriptor_and_inspection_share_ome_zarr_metadata() {
        let opened = open_local_ome_zarr(&fixture()).expect("open fixture");
        let inspection = inspect_core_dataset(&fixture());
        assert_eq!(opened.descriptor.kind, DocumentKind::OmeZarr);
        assert_eq!(opened.descriptor.channels.len(), 5);
        assert_eq!(inspection.kind, Some(DatasetInspectionKind::OmeZarr));
        assert_eq!(inspection.can_open, Some(true));
        let value = serde_json::to_value(inspection).expect("serialize inspection");
        assert_eq!(value["metadata"]["channel_count"], serde_json::Value::Null);
        assert_eq!(value["metadata"]["channels"].as_array().unwrap().len(), 5);
        assert_eq!(value["metadata"]["level_count"], 4);
    }
}
