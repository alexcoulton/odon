use std::path::Path;

use anyhow::{Context, anyhow};
use serde::Deserialize;
use serde::de::Deserializer;
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::ReadableStorageTraits;

use crate::data::dataset_source::DatasetSource;
use crate::data::zarr_attrs::normalize_ngff_attributes;

fn de_opt_string_or_number<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(deserializer)?;
    match v {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(serde_json::Value::Number(n)) => Ok(Some(n.to_string())),
        Some(other) => Err(serde::de::Error::custom(format!(
            "expected string or number, got {other}"
        ))),
    }
}

#[derive(Debug, Clone)]
pub struct OmeZarrDataset {
    pub source: DatasetSource,
    pub multiscale: Multiscale,
    pub levels: Vec<LevelInfo>,
    pub channels: Vec<ChannelInfo>,
    pub dims: Dims,
    pub abs_max: f32,
    pub render_kind: DatasetRenderKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetRenderKind {
    Image,
    LabelMask,
}

#[derive(Debug, Clone)]
pub struct Dims {
    pub c: Option<usize>,
    pub y: usize,
    pub x: usize,
    pub ndim: usize,
}

#[derive(Debug, Clone)]
pub struct LevelInfo {
    pub index: usize,
    pub path: String,
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub downsample: f32,
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub struct ChannelInfo {
    pub index: usize,
    pub name: String,
    pub color_rgb: [u8; 3],
    pub window: Option<(f32, f32)>,
    pub visible: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RootZattrs {
    #[serde(default)]
    pub multiscales: Vec<Multiscale>,
    pub omero: Option<Omero>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Multiscale {
    pub version: Option<String>,
    pub name: Option<String>,
    pub axes: Vec<Axis>,
    pub datasets: Vec<MultiscaleDataset>,
    #[serde(default)]
    pub r#type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Axis {
    pub name: String,
    #[serde(rename = "type")]
    pub axis_type: Option<String>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultiscaleDataset {
    pub path: String,
    #[serde(default)]
    #[serde(rename = "coordinateTransformations")]
    pub coordinate_transformations: Vec<CoordTransform>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum CoordTransform {
    #[serde(rename = "scale")]
    Scale { scale: Vec<f32> },
    #[serde(rename = "translation")]
    Translation { translation: Vec<f32> },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Omero {
    pub name: Option<String>,
    #[serde(default)]
    pub channels: Vec<OmeroChannel>,
    pub rdefs: Option<OmeroRdefs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeroChannel {
    // Some producers (e.g. SpatialData) encode `label` as an integer index.
    #[serde(default, deserialize_with = "de_opt_string_or_number")]
    pub label: Option<String>,
    pub name: Option<String>,
    pub window: Option<OmeroWindow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeroWindow {
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeroRdefs {
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ZarrayV2 {
    pub shape: Vec<u64>,
    pub chunks: Vec<u64>,
    pub dtype: String,
}

impl OmeZarrDataset {
    pub fn open_local(
        root: &Path,
    ) -> anyhow::Result<(Self, std::sync::Arc<dyn ReadableStorageTraits>)> {
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to canonicalize dataset root: {root:?}"))?;
        let store: std::sync::Arc<dyn ReadableStorageTraits> =
            std::sync::Arc::new(zarrs::filesystem::FilesystemStore::new(&root)?);
        let ds = Self::open_with_store(DatasetSource::Local(root), store.clone())?;
        Ok((ds, store))
    }

    pub fn open_with_store(
        source: DatasetSource,
        store: std::sync::Arc<dyn ReadableStorageTraits>,
    ) -> anyhow::Result<Self> {
        let attrs = crate::data::zarr_attrs::read_node_attributes_store(store.as_ref(), "")?
            .ok_or_else(|| anyhow!("missing .zattrs or zarr.json at dataset root"))?;
        let attrs = normalize_ngff_attributes(attrs);
        let root_zattrs: RootZattrs = serde_json::from_value(serde_json::Value::Object(attrs))
            .context("failed to parse dataset attributes")?;

        let multiscale = root_zattrs
            .multiscales
            .first()
            .cloned()
            .ok_or_else(|| anyhow!("no multiscales found in dataset attributes"))?;

        let dims = dims_from_axes(&multiscale.axes)?;
        let levels = load_levels(store.clone(), &multiscale, &dims)?;

        let channels = build_channels(root_zattrs.omero.as_ref(), &dims, &levels)?;
        let abs_max = infer_abs_max(&levels, &channels);
        let render_kind = classify_render_kind(&levels, &channels);
        if render_kind == DatasetRenderKind::Image {
            ensure_supported_image_dtypes(&levels)?;
        }

        Ok(Self {
            source,
            multiscale,
            levels,
            channels,
            dims,
            abs_max,
            render_kind,
        })
    }

    pub fn is_root_label_mask(&self) -> bool {
        self.render_kind == DatasetRenderKind::LabelMask
    }
}

fn dims_from_axes(axes: &[Axis]) -> anyhow::Result<Dims> {
    let mut c = None;
    let mut y = None;
    let mut x = None;

    for (i, axis) in axes.iter().enumerate() {
        match axis.name.as_str() {
            "c" => c = Some(i),
            "y" => y = Some(i),
            "x" => x = Some(i),
            _ => {}
        }
    }

    let y = y.ok_or_else(|| anyhow!("axes missing required 'y' dimension"))?;
    let x = x.ok_or_else(|| anyhow!("axes missing required 'x' dimension"))?;

    Ok(Dims {
        c,
        y,
        x,
        ndim: axes.len(),
    })
}

fn load_levels(
    store: std::sync::Arc<dyn ReadableStorageTraits>,
    multiscale: &Multiscale,
    dims: &Dims,
) -> anyhow::Result<Vec<LevelInfo>> {
    let mut levels = Vec::with_capacity(multiscale.datasets.len());

    let base_scale = dataset_scale(multiscale, 0, dims)?;
    let base_y = base_scale[dims.y];
    let base_x = base_scale[dims.x];

    for (index, ds) in multiscale.datasets.iter().enumerate() {
        let zarr_path = format!("/{}", ds.path.trim_start_matches('/'));
        let array: Array<dyn ReadableStorageTraits> = Array::open(store.clone(), &zarr_path)
            .with_context(|| format!("failed to open array metadata at {zarr_path}"))?;

        let shape = array.shape().to_vec();
        let chunk_shape = array
            .chunk_shape(&vec![0u64; shape.len()])
            .map(|v| v.into_iter().map(|n| n.get()).collect::<Vec<u64>>())
            .unwrap_or_else(|_| vec![1u64; shape.len()]);

        if shape.len() != dims.ndim || chunk_shape.len() != dims.ndim {
            return Err(anyhow!(
                "level {} has unexpected dimensionality: shape {:?}, chunks {:?}, expected ndim {}",
                index,
                shape,
                chunk_shape,
                dims.ndim
            ));
        }

        let scale = dataset_scale(multiscale, index, dims)?;
        let downsample_y = scale[dims.y] / base_y;
        let downsample_x = scale[dims.x] / base_x;
        let downsample = downsample_y.max(downsample_x);

        levels.push(LevelInfo {
            index,
            path: ds.path.clone(),
            shape,
            chunks: chunk_shape,
            downsample,
            dtype: format!("{}", array.data_type()),
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
                    "level {level} scale has wrong length: got {}, expected {}",
                    scale.len(),
                    dims.ndim
                ));
            }
            return Ok(scale.clone());
        }
    }

    Err(anyhow!(
        "level {level} missing coordinateTransformations scale"
    ))
}

fn build_channels(
    omero: Option<&Omero>,
    dims: &Dims,
    levels: &[LevelInfo],
) -> anyhow::Result<Vec<ChannelInfo>> {
    let c_dim = match dims.c {
        Some(c_dim) => c_dim,
        None => {
            return Ok(vec![ChannelInfo {
                index: 0,
                name: "channel 0".to_string(),
                color_rgb: default_channel_color(0),
                window: None,
                visible: true,
            }]);
        }
    };

    let level0 = levels
        .iter()
        .find(|l| l.index == 0)
        .ok_or_else(|| anyhow!("missing level 0 info"))?;
    let channel_count = level0
        .shape
        .get(c_dim)
        .copied()
        .ok_or_else(|| anyhow!("level 0 missing 'c' dimension"))? as usize;

    let mut channels: Vec<ChannelInfo> = (0..channel_count)
        .map(|i| ChannelInfo {
            index: i,
            name: format!("channel {i}"),
            color_rgb: default_channel_color(i),
            window: None,
            visible: i == 0,
        })
        .collect();

    if let Some(omero) = omero {
        for (i, ch) in omero.channels.iter().enumerate() {
            if let Some(dst) = channels.get_mut(i) {
                dst.name = ch
                    .name
                    .clone()
                    .or_else(|| ch.label.clone())
                    .unwrap_or_else(|| format!("channel {i}"));
                dst.window = ch.window.as_ref().map(|w| (w.start, w.end));
            }
        }
    }

    for ch in &mut channels {
        if ch.name.to_ascii_lowercase().contains("dapi") {
            ch.color_rgb = [0x00, 0x6a, 0xff];
        }
    }

    Ok(channels)
}

fn default_channel_color(i: usize) -> [u8; 3] {
    // A small, high-contrast cycle (roughly napari-like defaults).
    const COLORS: &[[u8; 3]] = &[
        [0x00, 0xff, 0x00], // green
        [0xff, 0x00, 0xff], // magenta
        [0x00, 0xff, 0xff], // cyan
        [0xff, 0xff, 0x00], // yellow
        [0xff, 0x00, 0x00], // red
        [0x00, 0x00, 0xff], // blue
        [0xff, 0x80, 0x00], // orange
        [0x80, 0x00, 0xff], // purple
    ];
    COLORS[i % COLORS.len()]
}

fn infer_abs_max(levels: &[LevelInfo], channels: &[ChannelInfo]) -> f32 {
    let dtype_max = levels
        .first()
        .and_then(|l| dtype_abs_max(&l.dtype))
        .unwrap_or(65535.0);
    let window_max = channels
        .iter()
        .filter_map(|c| c.window.map(|(_, hi)| hi))
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(0.0f32, f32::max);
    let out = if window_max > 0.0 {
        dtype_max.max(window_max)
    } else {
        dtype_max
    };
    if out.is_finite() && out > 0.0 {
        out
    } else {
        65535.0
    }
}

fn classify_render_kind(levels: &[LevelInfo], channels: &[ChannelInfo]) -> DatasetRenderKind {
    let Some(level0) = levels.first() else {
        return DatasetRenderKind::Image;
    };
    let Some(dtype) = parse_numeric_dtype(&level0.dtype) else {
        return DatasetRenderKind::Image;
    };

    // Single-channel integer arrays wider than 16 bits are far more commonly label IDs than
    // fluorescence intensities in odon's target datasets, so route them through the label path.
    if channels.len() == 1 && dtype.kind == ParsedDtypeKind::Unsigned && dtype.bits > 16 {
        DatasetRenderKind::LabelMask
    } else {
        DatasetRenderKind::Image
    }
}

fn ensure_supported_image_dtypes(levels: &[LevelInfo]) -> anyhow::Result<()> {
    let unsupported: Vec<&str> = levels
        .iter()
        .map(|l| l.dtype.as_str())
        .filter(|dtype| !is_supported_image_dtype(dtype))
        .collect();
    if unsupported.is_empty() {
        return Ok(());
    }

    let mut seen = std::collections::BTreeSet::new();
    let distinct = unsupported
        .into_iter()
        .filter(|dtype| seen.insert(*dtype))
        .collect::<Vec<_>>()
        .join(", ");

    Err(anyhow!(
        "unsupported OME-Zarr intensity dtype(s): {distinct}. odon currently supports uint16 image data; top-level uint32 masks are opened via the label renderer"
    ))
}

pub(crate) fn is_supported_image_dtype(dtype: &str) -> bool {
    matches!(
        parse_numeric_dtype(dtype),
        Some(ParsedDtype {
            kind: ParsedDtypeKind::Unsigned,
            bits: 8 | 16
        })
    )
}

pub(crate) fn retrieve_image_subset_u16(
    array: &Array<dyn ReadableStorageTraits>,
    subset: &ArraySubset,
    dtype: &str,
) -> anyhow::Result<ndarray::ArrayD<u16>> {
    match parse_numeric_dtype(dtype) {
        Some(ParsedDtype {
            kind: ParsedDtypeKind::Unsigned,
            bits: 8,
        }) => {
            let data: ndarray::ArrayD<u8> = array.retrieve_array_subset(subset)?;
            Ok(data.mapv(u16::from))
        }
        Some(ParsedDtype {
            kind: ParsedDtypeKind::Unsigned,
            bits: 16,
        }) => {
            let data: ndarray::ArrayD<u16> = array.retrieve_array_subset(subset)?;
            Ok(data)
        }
        _ => Err(anyhow!(
            "unsupported OME-Zarr intensity dtype for read: {dtype}"
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedDtype {
    kind: ParsedDtypeKind,
    bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedDtypeKind {
    Unsigned,
    Signed,
    Float,
    Bool,
}

fn parse_numeric_dtype(dtype: &str) -> Option<ParsedDtype> {
    for part in dtype.split('/').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if let Some(v) = parse_numeric_dtype_one(part) {
            return Some(v);
        }
    }
    parse_numeric_dtype_one(dtype.trim())
}

fn parse_numeric_dtype_one(dtype: &str) -> Option<ParsedDtype> {
    let d = dtype.trim().to_ascii_lowercase();
    if d.is_empty() {
        return None;
    }

    let shorthand = if d.starts_with('<') || d.starts_with('>') || d.starts_with('|') {
        &d[1..]
    } else {
        d.as_str()
    };
    if shorthand.len() >= 2 {
        let kind = shorthand.chars().next()?;
        if kind == 'u' || kind == 'i' || kind == 'f' {
            let bytes: u32 = shorthand[1..].parse().ok()?;
            return Some(ParsedDtype {
                kind: match kind {
                    'u' => ParsedDtypeKind::Unsigned,
                    'i' => ParsedDtypeKind::Signed,
                    'f' => ParsedDtypeKind::Float,
                    _ => return None,
                },
                bits: bytes.checked_mul(8)?,
            });
        }
    }

    if d == "bool" {
        return Some(ParsedDtype {
            kind: ParsedDtypeKind::Bool,
            bits: 1,
        });
    }

    for (prefix, kind) in [
        ("uint", ParsedDtypeKind::Unsigned),
        ("int", ParsedDtypeKind::Signed),
        ("float", ParsedDtypeKind::Float),
    ] {
        if let Some(rest) = d.strip_prefix(prefix) {
            return Some(ParsedDtype {
                kind,
                bits: rest.parse().ok()?,
            });
        }
    }

    None
}

fn dtype_abs_max(dtype: &str) -> Option<f32> {
    // zarrs may format V3/V2 names like "uint16 / <u2". Try each part.
    for part in dtype.split('/').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        if let Some(v) = dtype_abs_max_one(part) {
            return Some(v);
        }
    }
    dtype_abs_max_one(dtype.trim())
}

fn dtype_abs_max_one(dtype: &str) -> Option<f32> {
    let d = dtype.trim().to_ascii_lowercase();
    if d.is_empty() {
        return None;
    }

    // Zarr V2 numpy-style dtype strings like "<u2", "|u1", ">i4", or sometimes "u2"/"i4".
    {
        let bytes_part = if d.starts_with('<') || d.starts_with('>') || d.starts_with('|') {
            &d[1..]
        } else {
            d.as_str()
        };
        if bytes_part.len() >= 2 {
            let kind = bytes_part.chars().next()?;
            if kind == 'u' || kind == 'i' {
                let bytes: u32 = bytes_part[1..].parse().ok()?;
                let bits = bytes.checked_mul(8)?;
                return match kind {
                    'u' => {
                        if bits >= 128 {
                            Some(f32::MAX)
                        } else {
                            let max = (1u128.checked_shl(bits)?).saturating_sub(1);
                            Some((max as f64).min(f64::from(f32::MAX)) as f32)
                        }
                    }
                    'i' => {
                        if bits == 0 {
                            None
                        } else if bits >= 128 {
                            Some(f32::MAX)
                        } else {
                            let max = (1u128.checked_shl(bits - 1)?).saturating_sub(1);
                            Some((max as f64).min(f64::from(f32::MAX)) as f32)
                        }
                    }
                    _ => None,
                };
            }
        }
    }

    // Common Zarr V3 names like "uint16", "int32", "u16", "i16".
    let parse_bits = |suffix: &str| suffix.parse::<u32>().ok();
    if let Some(bits) = d.strip_prefix("uint").and_then(parse_bits) {
        return Some(uint_bits_to_max(bits));
    }
    if let Some(bits) = d.strip_prefix("int").and_then(parse_bits) {
        return Some(int_bits_to_max(bits));
    }
    if let Some(bits) = d.strip_prefix('u').and_then(parse_bits) {
        return Some(uint_bits_to_max(bits));
    }
    if let Some(bits) = d.strip_prefix('i').and_then(parse_bits) {
        return Some(int_bits_to_max(bits));
    }

    None
}

fn uint_bits_to_max(bits: u32) -> f32 {
    if bits >= 128 {
        return f32::MAX;
    }
    let max = (1u128.checked_shl(bits).unwrap_or(0)).saturating_sub(1);
    (max as f64).min(f64::from(f32::MAX)) as f32
}

fn int_bits_to_max(bits: u32) -> f32 {
    if bits == 0 {
        return 0.0;
    }
    if bits >= 128 {
        return f32::MAX;
    }
    let max = (1u128.checked_shl(bits - 1).unwrap_or(0)).saturating_sub(1);
    (max as f64).min(f64::from(f32::MAX)) as f32
}
