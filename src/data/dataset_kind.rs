use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalDatasetKind {
    OmeZarr,
    Xenium,
    Tiff,
}

pub fn normalize_local_dataset_path(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        return classify_local_dataset_path(path).map(|_| path.to_path_buf());
    }
    if path.is_file() {
        if path.file_name().is_some_and(|n| n == ".zattrs")
            || path.file_name().is_some_and(|n| n == "zarr.json")
            || path.file_name().is_some_and(|n| n == "experiment.xenium")
        {
            return path.parent().map(|p| p.to_path_buf());
        }
        if looks_like_tiff_file(path) {
            return Some(path.to_path_buf());
        }
    }
    None
}

pub fn classify_local_dataset_path(path: &Path) -> Option<LocalDatasetKind> {
    if looks_like_omezarr_root(path) {
        return Some(LocalDatasetKind::OmeZarr);
    }
    if looks_like_xenium_root(path) {
        return Some(LocalDatasetKind::Xenium);
    }
    if looks_like_tiff_file(path) {
        return Some(LocalDatasetKind::Tiff);
    }
    None
}

pub fn can_open_in_mosaic(path: &Path) -> bool {
    matches!(
        normalize_local_dataset_path(path)
            .as_deref()
            .and_then(classify_local_dataset_path),
        Some(LocalDatasetKind::OmeZarr)
    )
}

fn looks_like_omezarr_root(root: &Path) -> bool {
    root.is_dir() && (root.join(".zattrs").is_file() || root.join("zarr.json").is_file())
}

fn looks_like_xenium_root(root: &Path) -> bool {
    root.is_dir() && root.join("experiment.xenium").is_file()
}

fn looks_like_tiff_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "tif" | "tiff"))
}
