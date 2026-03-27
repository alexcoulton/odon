use std::path::{Path, PathBuf};

use anyhow::Context;

use super::manifest::XeniumManifest;
use crate::{log_debug, log_info};

#[derive(Debug, Clone)]
pub struct XeniumDiscovery {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: XeniumManifest,
    pub morphology_mip_omezarr: Option<PathBuf>,
    pub morphology_mip_tiff: Option<PathBuf>,
    pub transcripts_zarr_zip: Option<PathBuf>,
    pub cells_zarr_zip: Option<PathBuf>,
}

fn looks_like_omezarr_root(root: &Path) -> bool {
    root.is_dir() && (root.join(".zattrs").is_file() || root.join("zarr.json").is_file())
}

fn tif_like_to_omezarr_name(p: &str) -> Option<String> {
    // Common Xenium manifest value: "morphology_mip.ome.tif"
    // Convert to: "morphology_mip.ome.zarr"
    let mut s = p.to_string();
    for suf in [".ome.tif", ".ome.tiff", ".tif", ".tiff"] {
        if s.ends_with(suf) {
            s.truncate(s.len().saturating_sub(suf.len()));
            s.push_str(".ome.zarr");
            return Some(s);
        }
    }
    None
}

pub fn discover_xenium_explorer(root: &Path) -> anyhow::Result<XeniumDiscovery> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let manifest_path = root.join("experiment.xenium");
    if !manifest_path.is_file() {
        anyhow::bail!("not a Xenium Explorer dataset (missing experiment.xenium)");
    }

    log_info!("xenium discover: {}", root.to_string_lossy());
    let bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("failed to read manifest: {manifest_path:?}"))?;
    let manifest: XeniumManifest =
        serde_json::from_slice(&bytes).context("failed to parse experiment.xenium")?;

    let morphology_mip_tiff = manifest
        .images
        .morphology_mip_filepath
        .as_deref()
        .map(|p| root.join(p))
        .filter(|p| p.is_file());

    // Prefer a converted OME-Zarr base image if present (avoids JPEG2000-in-TIFF decoding).
    let morphology_mip_omezarr = manifest
        .images
        .morphology_mip_filepath
        .as_deref()
        .and_then(tif_like_to_omezarr_name)
        .map(|p| root.join(p))
        .filter(|p| looks_like_omezarr_root(p))
        .or_else(|| {
            let p = root.join("morphology_mip.ome.zarr");
            looks_like_omezarr_root(&p).then_some(p)
        });
    let transcripts_zarr_zip = manifest
        .explorer
        .transcripts_zarr_filepath
        .as_deref()
        .map(|p| root.join(p))
        .filter(|p| p.is_file());
    let cells_zarr_zip = manifest
        .explorer
        .cells_zarr_filepath
        .as_deref()
        .map(|p| root.join(p))
        .filter(|p| p.is_file());

    log_debug!(
        "xenium files: morphology_mip_omezarr={:?} morphology_mip_tiff={:?} cells={:?} transcripts={:?} pixel_size_um={}",
        morphology_mip_omezarr,
        morphology_mip_tiff,
        cells_zarr_zip,
        transcripts_zarr_zip,
        manifest.pixel_size
    );

    Ok(XeniumDiscovery {
        root,
        manifest_path,
        manifest,
        morphology_mip_omezarr,
        morphology_mip_tiff,
        transcripts_zarr_zip,
        cells_zarr_zip,
    })
}
