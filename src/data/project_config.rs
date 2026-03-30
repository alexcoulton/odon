use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::data::dataset_source::DatasetSource;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectConfig {
    /// Explicit list of ROIs/datasets available in this project.
    #[serde(default)]
    pub rois: Vec<ProjectRoi>,

    /// Legacy/global knobs that previously lived in the napari-side YAML.
    ///
    /// These are currently not all used by the Rust viewer, but keeping them in the Project JSON
    /// allows a full migration of the old config.yaml into the new format.
    #[serde(default)]
    pub use_full_res_ome_zarrs: bool,
    #[serde(default)]
    pub tumour_antigen_only_mode: bool,
    #[serde(default)]
    pub secondary_dataset: Option<String>,
    #[serde(default)]
    pub clinical_annotations_parquet: Option<String>,
    #[serde(default)]
    pub cluster_column: Option<String>,
    #[serde(default)]
    pub unlabeled_cluster_value: Option<String>,
    #[serde(default)]
    pub all_clusters_label: Option<String>,
    #[serde(default)]
    pub tma1_orig_cells_parquet: Option<String>,

    /// Optional dataset configurations used by custom panels (cell thresholds, masks, etc).
    ///
    /// These replace the old napari-side YAML config concept.
    #[serde(default)]
    pub datasets: BTreeMap<String, ProjectDatasetConfig>,

    /// Optional default dataset key for UI drop-downs.
    #[serde(default)]
    pub default_dataset: Option<String>,

    /// Optional search roots used to auto-match mosaic segmentation files to ROIs.
    #[serde(default)]
    pub mosaic_segmentation_search_roots: Vec<PathBuf>,

    /// Optional default marker for the Cell Thresholds tab.
    #[serde(default)]
    pub default_threshold_marker: Option<String>,

    /// Persistent UI groupings for channels/annotations (Napari-like layer groups).
    #[serde(default)]
    pub layer_groups: ProjectLayerGroups,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProjectLayerGroups {
    /// One-level groups for channel layers, keyed by channel name.
    #[serde(default)]
    pub channel_groups: Vec<ProjectChannelGroup>,
    /// Per-channel membership config (key: channel name).
    #[serde(default)]
    pub channel_members: HashMap<String, ProjectChannelGroupMember>,

    /// One-level groups for annotation layers, keyed by annotation layer id.
    #[serde(default)]
    pub annotation_groups: Vec<ProjectAnnotationGroup>,
    /// Per-annotation membership config (key: annotation layer id).
    #[serde(default)]
    pub annotation_members: HashMap<u64, ProjectAnnotationGroupMember>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectChannelGroup {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub expanded: bool,
    #[serde(default)]
    pub color_rgb: [u8; 3],
}

impl Default for ProjectChannelGroup {
    fn default() -> Self {
        Self {
            id: 1,
            name: "Group".to_string(),
            expanded: true,
            color_rgb: [255, 255, 255],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectChannelGroupMember {
    pub group_id: u64,
    #[serde(default = "default_true")]
    pub inherit_color: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectAnnotationGroup {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub expanded: bool,
    #[serde(default)]
    pub visible: bool,
    /// Optional tint color applied to members (categorical fill; continuous stroke).
    #[serde(default)]
    pub tint_rgb: Option<[u8; 3]>,
    #[serde(default)]
    pub tint_strength: f32,
}

impl Default for ProjectAnnotationGroup {
    fn default() -> Self {
        Self {
            id: 1,
            name: "Group".to_string(),
            expanded: true,
            visible: true,
            tint_rgb: None,
            tint_strength: 0.35,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProjectAnnotationGroupMember {
    pub group_id: u64,
    #[serde(default = "default_true")]
    pub inherit_tint: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectRoi {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<DatasetSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub segpath: Option<PathBuf>,
    /// User-defined mask layers (editable/read-only) for this ROI.
    ///
    /// These are viewer layers, not an OME-NGFF standard. We store them here so projects are fully
    /// self-contained without relying on external sidecar files.
    #[serde(default)]
    pub mask_layers: Vec<ProjectMaskLayer>,
    /// Optional saved channel layer order for this ROI in the single-image viewer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub channel_order: Vec<usize>,
    /// Arbitrary metadata columns (for mosaic sorting/grouping/labels).
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

impl ProjectRoi {
    pub fn dataset_source(&self) -> Option<DatasetSource> {
        self.source
            .clone()
            .or_else(|| self.path.clone().map(DatasetSource::Local))
    }

    pub fn set_dataset_source(&mut self, source: DatasetSource) {
        self.path = source
            .local_path()
            .map(|p: &std::path::Path| p.to_path_buf());
        self.source = Some(source);
    }

    pub fn source_key(&self) -> Option<String> {
        self.dataset_source()
            .map(|source: DatasetSource| source.source_key())
    }

    pub fn local_path(&self) -> Option<&std::path::Path> {
        self.source
            .as_ref()
            .and_then(DatasetSource::local_path)
            .or_else(|| self.path.as_deref())
    }

    pub fn source_display(&self) -> String {
        self.dataset_source()
            .map(|source| match source {
                DatasetSource::Local(path) => path.to_string_lossy().to_string(),
                DatasetSource::Http { base_url } => base_url,
                DatasetSource::S3 { bucket, prefix, .. } => {
                    if prefix.trim().is_empty() {
                        format!("s3://{bucket}")
                    } else {
                        format!("s3://{bucket}/{}", prefix.trim_matches('/'))
                    }
                }
            })
            .unwrap_or_else(|| "<unconfigured dataset>".to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectMaskLayer {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub visible: bool,
    #[serde(default)]
    pub opacity: f32,
    #[serde(default)]
    pub width_screen_px: f32,
    #[serde(default)]
    pub color_rgb: [u8; 3],
    /// Optional per-layer translation in viewer world coordinates (level-0 pixels).
    #[serde(default)]
    pub offset_world: [f32; 2],
    #[serde(default)]
    pub editable: bool,
    /// Closed polygon rings in viewer world coordinates (level-0 pixels).
    ///
    /// The last vertex may repeat the first; the loader should handle either form.
    #[serde(default)]
    pub polygons_world: Vec<Vec<[f32; 2]>>,
    /// Optional source path if this layer was loaded from an external GeoJSON.
    #[serde(default)]
    pub source_geojson: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectDatasetConfig {
    /// Optional per-ROI "layout" conventions used by some custom panels.
    #[serde(default)]
    pub layout: Option<String>,

    /// Optional metadata paths retained from the legacy YAML (e.g. for mosaic / ROI selection).
    #[serde(default)]
    pub samplesheet_path: Option<String>,

    /// Whether to use all channels or a subset for this dataset (legacy YAML concept).
    #[serde(default)]
    pub channel_mode: Option<String>,

    /// Optional helper paths for ROI selector/masks and parquet-backed point loaders.
    #[serde(default)]
    pub base_dir_full_res: Option<String>,
    #[serde(default)]
    pub base_dir_downsampled: Option<String>,

    /// Optional ROI discovery hints retained from the legacy YAML.
    ///
    /// The Rust viewer uses an explicit ROI list (`ProjectConfig.rois`) at runtime; these are only
    /// kept to allow round-tripping/migration of old YAML configs.
    #[serde(default)]
    pub roi_start: Option<u64>,
    #[serde(default)]
    pub roi_end: Option<u64>,
    #[serde(default)]
    pub sample_id: Option<String>,
    #[serde(default)]
    pub include_all_samples: Option<bool>,

    // Cell threshold points (parquet) config.
    #[serde(default)]
    pub cells_backend: Option<String>,
    #[serde(default)]
    pub cells_parquet: Option<String>,
    #[serde(default)]
    pub cells_parquet_dir: Option<String>,
    #[serde(default)]
    pub cells_parquet_dir_flatfield: Option<String>,
    #[serde(default)]
    pub channels_index_path: Option<String>,
    #[serde(default)]
    pub needs_correction_csv: Option<String>,
    #[serde(default)]
    pub coord_downsample_full_res: Option<f32>,
    #[serde(default)]
    pub coord_downsample_downsampled: Option<f32>,
    #[serde(default)]
    pub subset_channel_labels: Option<Vec<String>>,
    #[serde(default)]
    pub thresholds_csv: Option<String>,
    #[serde(default)]
    pub auto_thresholds_json: Option<String>,

    // Exclusion masks config (optional, if not using per-ROI explicit paths).
    #[serde(default)]
    pub masks_dir: Option<String>,
    #[serde(default)]
    pub masks_downsample_full_res: Option<f32>,
    #[serde(default)]
    pub masks_downsample_downsampled: Option<f32>,
}
