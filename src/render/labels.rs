use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use zarrs::array::Array;
use zarrs::storage::ReadableStorageTraits;

use crate::data::ome::{
    Axis, CoordTransform, Dims, LevelInfo, Multiscale, OmeZarrDataset, RootZattrs,
};
use crate::data::zarr_attrs::{normalize_ngff_attributes, read_node_attributes_store};

pub fn discover_label_names_local(root: &Path) -> Vec<String> {
    let labels_dir = root.join("labels");
    let Ok(rd) = std::fs::read_dir(&labels_dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if !looks_like_zarr_group_dir(&path) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        out.push(name.to_string());
    }
    out.sort();
    out.dedup();
    out
}

fn looks_like_zarr_group_dir(dir: &Path) -> bool {
    // Zarr v2 groups commonly have `.zgroup` and/or `.zattrs`.
    // Zarr v3 groups use a `zarr.json`.
    dir.join(".zgroup").is_file()
        || dir.join(".zattrs").is_file()
        || dir.join("zarr.json").is_file()
}

#[derive(Debug, Clone)]
pub struct LabelZarrDataset {
    pub label_name: String,
    pub levels: Vec<LevelInfo>,
    pub dims: Dims,
}

impl LabelZarrDataset {
    /// Tries to open an OME-NGFF label multiscale at `labels/<label_name>`.
    ///
    /// Returns `Ok(None)` if the expected metadata file is not present.
    pub fn try_open(
        store: std::sync::Arc<dyn ReadableStorageTraits>,
        label_name: &str,
    ) -> anyhow::Result<Option<Self>> {
        let labels_prefix = format!("labels/{label_name}");
        let Some(attrs) = read_node_attributes_store(store.as_ref(), &labels_prefix)? else {
            return Ok(None);
        };
        let attrs = normalize_ngff_attributes(attrs);
        let root_zattrs: RootZattrs = serde_json::from_value(serde_json::Value::Object(attrs))
            .context("failed to parse labels attributes")?;
        let multiscale = root_zattrs
            .multiscales
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("no multiscales found in labels attributes"))?;

        let dims = dims_from_axes(&multiscale.axes)?;
        let levels = load_levels(store, label_name, &multiscale, &dims)?;

        Ok(Some(Self {
            label_name: label_name.to_string(),
            levels,
            dims,
        }))
    }

    pub fn from_root_dataset(dataset: &OmeZarrDataset) -> Self {
        Self {
            label_name: Self::root_label_name(dataset),
            levels: dataset.levels.clone(),
            dims: dataset.dims.clone(),
        }
    }

    pub fn root_label_name(dataset: &OmeZarrDataset) -> String {
        dataset
            .multiscale
            .name
            .clone()
            .or_else(|| dataset.channels.first().map(|c| c.name.clone()))
            .unwrap_or_else(|| "labels".to_string())
    }
}

fn dims_from_axes(axes: &[Axis]) -> anyhow::Result<Dims> {
    let mut c = None;
    let mut z = None;
    let mut y = None;
    let mut x = None;

    for (i, axis) in axes.iter().enumerate() {
        match axis.name.as_str() {
            "c" => c = Some(i),
            "z" => z = Some(i),
            "y" => y = Some(i),
            "x" => x = Some(i),
            _ => {}
        }
    }

    let y = y.ok_or_else(|| anyhow!("axes missing required 'y' dimension"))?;
    let x = x.ok_or_else(|| anyhow!("axes missing required 'x' dimension"))?;

    Ok(Dims {
        c,
        z,
        y,
        x,
        ndim: axes.len(),
    })
}

fn load_levels(
    store: std::sync::Arc<dyn ReadableStorageTraits>,
    label_name: &str,
    multiscale: &Multiscale,
    dims: &Dims,
) -> anyhow::Result<Vec<LevelInfo>> {
    let mut levels = Vec::with_capacity(multiscale.datasets.len());

    let base_scale = dataset_scale(multiscale, 0, dims)?;
    let base_y = base_scale[dims.y];
    let base_x = base_scale[dims.x];

    for (index, ds) in multiscale.datasets.iter().enumerate() {
        let full_path = PathBuf::from("labels").join(label_name).join(&ds.path);
        let zarr_path = format!("/{}", full_path.to_string_lossy().trim_start_matches('/'));
        let array: Array<dyn ReadableStorageTraits> = Array::open(store.clone(), &zarr_path)
            .with_context(|| format!("failed to open label array metadata at {zarr_path}"))?;

        let shape = array.shape().to_vec();
        let chunk_shape = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map(|v| v.into_iter().map(|n| n.get()).collect::<Vec<u64>>())
            .unwrap_or_else(|_| vec![1u64; shape.len()]);

        if shape.len() != dims.ndim || chunk_shape.len() != dims.ndim {
            return Err(anyhow!(
                "label level {} has unexpected dimensionality: shape {:?}, chunks {:?}, expected ndim {}",
                index,
                shape,
                chunk_shape,
                dims.ndim
            ));
        }

        let scale = dataset_scale(multiscale, index, dims)?;
        let translation = dataset_translation(multiscale, index, dims)?;
        let downsample_y = scale[dims.y] / base_y;
        let downsample_x = scale[dims.x] / base_x;
        let downsample = downsample_y.max(downsample_x);

        levels.push(LevelInfo {
            index,
            path: full_path.to_string_lossy().to_string(),
            shape,
            chunks: chunk_shape,
            downsample,
            dtype: format!("{}", array.data_type()),
            scale,
            translation,
        });
    }

    Ok(levels)
}

fn dataset_scale(multiscale: &Multiscale, level: usize, dims: &Dims) -> anyhow::Result<Vec<f32>> {
    let ds = multiscale
        .datasets
        .get(level)
        .ok_or_else(|| anyhow!("missing dataset entry for level {level}"))?;

    for ct in &ds.coordinate_transformations {
        if let CoordTransform::Scale { scale } = ct {
            if scale.len() != dims.ndim {
                return Err(anyhow!(
                    "label level {level} scale has wrong length: got {}, expected {}",
                    scale.len(),
                    dims.ndim
                ));
            }
            return Ok(scale.clone());
        }
    }

    Err(anyhow!(
        "label level {level} missing coordinateTransformations scale"
    ))
}

fn dataset_translation(
    multiscale: &Multiscale,
    level: usize,
    dims: &Dims,
) -> anyhow::Result<Vec<f32>> {
    let ds = multiscale
        .datasets
        .get(level)
        .ok_or_else(|| anyhow!("missing dataset entry for level {level}"))?;

    for ct in &ds.coordinate_transformations {
        if let CoordTransform::Translation { translation } = ct {
            if translation.len() != dims.ndim {
                return Err(anyhow!(
                    "label level {level} translation has wrong length: got {}, expected {}",
                    translation.len(),
                    dims.ndim
                ));
            }
            return Ok(translation.clone());
        }
    }

    Ok(vec![0.0; dims.ndim])
}
