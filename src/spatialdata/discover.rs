use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::data::zarr_attrs::read_node_attributes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDataElementKind {
    Image,
    Label,
    Points,
    Shapes,
    Table,
    Other,
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialDataTransform2 {
    pub scale: [f32; 2],
    pub translation: [f32; 2],
}

impl Default for SpatialDataTransform2 {
    fn default() -> Self {
        Self {
            scale: [1.0, 1.0],
            translation: [0.0, 0.0],
        }
    }
}

impl SpatialDataTransform2 {
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        [
            p[0] * self.scale[0] + self.translation[0],
            p[1] * self.scale[1] + self.translation[1],
        ]
    }

    pub fn relative_to(&self, reference: SpatialDataTransform2) -> Self {
        let ref_sx = reference.scale[0].max(1e-6);
        let ref_sy = reference.scale[1].max(1e-6);
        Self {
            scale: [self.scale[0] / ref_sx, self.scale[1] / ref_sy],
            translation: [
                (self.translation[0] - reference.translation[0]) / ref_sx,
                (self.translation[1] - reference.translation[1]) / ref_sy,
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialDataElement {
    pub kind: SpatialDataElementKind,
    pub name: String,
    /// Path to the element group, relative to the SpatialData root.
    pub rel_group: PathBuf,
    /// Optional parquet payload path relative to root (file for shapes, directory for points).
    pub rel_parquet: Option<PathBuf>,
    pub transform: SpatialDataTransform2,
    /// Optional key for per-point features (e.g. Xenium gene name).
    pub feature_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SpatialDataDiscovery {
    pub root: PathBuf,
    pub images: Vec<SpatialDataElement>,
    pub labels: Vec<SpatialDataElement>,
    pub points: Vec<SpatialDataElement>,
    pub shapes: Vec<SpatialDataElement>,
    pub tables: Vec<SpatialDataElement>,
}

pub fn discover_spatialdata(root: &Path) -> anyhow::Result<SpatialDataDiscovery> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    let mut out = SpatialDataDiscovery {
        root: root.clone(),
        ..Default::default()
    };

    // We treat the container as SpatialData if it has a zarr.json with `spatialdata_attrs`,
    // or if it has any of the standard subtrees.
    let attrs = read_node_attributes(&root).ok().flatten();
    let has_spatialdata_attrs = attrs
        .as_ref()
        .and_then(|a| a.get("spatialdata_attrs"))
        .is_some();

    let has_any_dir = root.join("images").is_dir()
        || root.join("labels").is_dir()
        || root.join("points").is_dir()
        || root.join("shapes").is_dir()
        || root.join("tables").is_dir();

    if !has_spatialdata_attrs && !has_any_dir {
        anyhow::bail!("not a SpatialData container: {}", root.to_string_lossy());
    }

    discover_images(&root, &mut out).context("discover images")?;
    discover_labels(&root, &mut out).context("discover labels")?;
    discover_points(&root, &mut out).context("discover points")?;
    discover_shapes(&root, &mut out).context("discover shapes")?;
    discover_tables(&root, &mut out).context("discover tables")?;

    Ok(out)
}

fn discover_images(root: &Path, out: &mut SpatialDataDiscovery) -> anyhow::Result<()> {
    let images = root.join("images");
    if !images.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&images)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();

        // Only include image groups that look like OME-NGFF.
        let attrs = match read_node_attributes(&path) {
            Ok(Some(a)) => a,
            _ => continue,
        };
        if !(attrs.contains_key("ome") || attrs.contains_key("multiscales")) {
            continue;
        }
        let transform = parse_transform2(&attrs);

        out.images.push(SpatialDataElement {
            kind: SpatialDataElementKind::Image,
            name,
            rel_group: PathBuf::from("images").join(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default(),
            ),
            rel_parquet: None,
            transform,
            feature_key: None,
        });
    }
    out.images.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(())
}

fn discover_points(root: &Path, out: &mut SpatialDataDiscovery) -> anyhow::Result<()> {
    let points = root.join("points");
    if !points.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&points)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("points")
            .to_string();
        let attrs = match read_node_attributes(&path) {
            Ok(Some(a)) => a,
            _ => continue,
        };
        let encoding = attrs
            .get("encoding-type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if encoding != "ngff:points" {
            continue;
        }
        let transform = parse_transform2(&attrs);
        let feature_key = attrs
            .get("spatialdata_attrs")
            .and_then(|v| v.get("feature_key"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let parquet = path.join("points.parquet");
        out.points.push(SpatialDataElement {
            kind: SpatialDataElementKind::Points,
            name,
            rel_group: PathBuf::from("points").join(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default(),
            ),
            rel_parquet: parquet.is_dir().then(|| {
                PathBuf::from("points")
                    .join(path.file_name().unwrap())
                    .join("points.parquet")
            }),
            transform,
            feature_key,
        });
    }
    out.points.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(())
}

fn discover_labels(root: &Path, out: &mut SpatialDataDiscovery) -> anyhow::Result<()> {
    let labels = root.join("labels");
    if !labels.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&labels)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("labels")
            .to_string();
        let attrs = match read_node_attributes(&path) {
            Ok(Some(a)) => a,
            _ => continue,
        };
        let has_multiscale = attrs.contains_key("multiscales")
            || attrs
                .get("ome")
                .and_then(|v| v.get("multiscales"))
                .is_some();
        let is_image_label = attrs
            .get("ome")
            .and_then(|v| v.get("image-label"))
            .is_some();
        if !has_multiscale || !is_image_label {
            continue;
        }
        let transform = parse_transform2(&attrs);
        out.labels.push(SpatialDataElement {
            kind: SpatialDataElementKind::Label,
            name,
            rel_group: PathBuf::from("labels").join(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default(),
            ),
            rel_parquet: None,
            transform,
            feature_key: None,
        });
    }
    out.labels.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(())
}

fn discover_shapes(root: &Path, out: &mut SpatialDataDiscovery) -> anyhow::Result<()> {
    let shapes = root.join("shapes");
    if !shapes.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&shapes)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("shapes")
            .to_string();
        let attrs = match read_node_attributes(&path) {
            Ok(Some(a)) => a,
            _ => continue,
        };
        let encoding = attrs
            .get("encoding-type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if encoding != "ngff:shapes" {
            continue;
        }
        let transform = parse_transform2(&attrs);
        let parquet = path.join("shapes.parquet");
        out.shapes.push(SpatialDataElement {
            kind: SpatialDataElementKind::Shapes,
            name,
            rel_group: PathBuf::from("shapes").join(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default(),
            ),
            rel_parquet: parquet.is_file().then(|| {
                PathBuf::from("shapes")
                    .join(path.file_name().unwrap())
                    .join("shapes.parquet")
            }),
            transform,
            feature_key: None,
        });
    }
    out.shapes.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(())
}

fn discover_tables(root: &Path, out: &mut SpatialDataDiscovery) -> anyhow::Result<()> {
    let tables = root.join("tables");
    if !tables.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(&tables)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("table")
            .to_string();
        out.tables.push(SpatialDataElement {
            kind: SpatialDataElementKind::Table,
            name,
            rel_group: PathBuf::from("tables").join(
                path.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default(),
            ),
            rel_parquet: None,
            transform: SpatialDataTransform2 {
                scale: [1.0, 1.0],
                translation: [0.0, 0.0],
            },
            feature_key: None,
        });
    }
    out.tables.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(())
}

fn parse_transform2(attrs: &serde_json::Map<String, serde_json::Value>) -> SpatialDataTransform2 {
    let mut t = SpatialDataTransform2 {
        scale: [1.0, 1.0],
        translation: [0.0, 0.0],
    };
    let Some(cts) = coordinate_transformations(attrs) else {
        return t;
    };
    for ct in cts {
        let Some(obj) = ct.as_object() else { continue };
        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or_default();
        match ty {
            "scale" => {
                if let Some(scale) = obj.get("scale").and_then(|v| v.as_array()) {
                    if let Some(mapped) = parse_xy_components(obj, attrs, scale) {
                        t.scale = mapped;
                    }
                }
            }
            "translation" => {
                if let Some(tr) = obj.get("translation").and_then(|v| v.as_array()) {
                    if let Some(mapped) = parse_xy_components(obj, attrs, tr) {
                        t.translation = mapped;
                    }
                }
            }
            _ => {}
        }
    }
    t
}

fn coordinate_transformations(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Option<&Vec<serde_json::Value>> {
    attrs
        .get("coordinateTransformations")
        .and_then(|v| v.as_array())
        .or_else(|| {
            attrs
                .get("ome")
                .and_then(|v| v.get("multiscales"))
                .and_then(|v| v.as_array())
                .and_then(|ms| ms.first())
                .and_then(|ms| ms.get("coordinateTransformations"))
                .and_then(|v| v.as_array())
        })
        .or_else(|| {
            attrs
                .get("multiscales")
                .and_then(|v| v.as_array())
                .and_then(|ms| ms.first())
                .and_then(|ms| ms.get("coordinateTransformations"))
                .and_then(|v| v.as_array())
        })
}

fn parse_xy_components(
    transform_obj: &serde_json::Map<String, serde_json::Value>,
    attrs: &serde_json::Map<String, serde_json::Value>,
    values: &[serde_json::Value],
) -> Option<[f32; 2]> {
    let axis_names = transform_axis_names(transform_obj).or_else(|| fallback_axis_names(attrs));
    if let Some(axis_names) = axis_names {
        let mut x = None;
        let mut y = None;
        for (idx, axis_name) in axis_names.iter().enumerate() {
            let Some(value) = values.get(idx).and_then(|v| v.as_f64()) else {
                continue;
            };
            match axis_name.as_str() {
                "x" => x = Some(value as f32),
                "y" => y = Some(value as f32),
                _ => {}
            }
        }
        if let (Some(x), Some(y)) = (x, y) {
            return Some([x, y]);
        }
    }

    if values.len() >= 2 {
        let y = values.get(values.len().saturating_sub(2))?.as_f64()? as f32;
        let x = values.get(values.len().saturating_sub(1))?.as_f64()? as f32;
        return Some([x, y]);
    }

    None
}

fn transform_axis_names(
    transform_obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<Vec<String>> {
    let input = transform_obj.get("input")?.as_object()?;
    let axes = input.get("axes")?.as_array()?;
    Some(
        axes.iter()
            .filter_map(|axis| axis.as_object())
            .filter_map(|axis| axis.get("name").and_then(|v| v.as_str()))
            .map(|name| name.to_ascii_lowercase())
            .collect(),
    )
}

fn fallback_axis_names(attrs: &serde_json::Map<String, serde_json::Value>) -> Option<Vec<String>> {
    let axes = attrs.get("axes")?.as_array()?;
    Some(
        axes.iter()
            .filter_map(|axis| {
                axis.as_str().map(|s| s.to_ascii_lowercase()).or_else(|| {
                    axis.as_object()
                        .and_then(|obj| obj.get("name"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_ascii_lowercase())
                })
            })
            .collect(),
    )
}
