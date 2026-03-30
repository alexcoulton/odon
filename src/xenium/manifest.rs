use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct XeniumManifest {
    pub pixel_size: f32,
    #[serde(default)]
    pub images: XeniumImages,
    #[serde(rename = "xenium_explorer_files", default)]
    pub explorer: XeniumExplorerFiles,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct XeniumImages {
    #[serde(default)]
    pub morphology_mip_filepath: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct XeniumExplorerFiles {
    #[serde(default)]
    pub transcripts_zarr_filepath: Option<String>,
    #[serde(default)]
    pub cells_zarr_filepath: Option<String>,
}
