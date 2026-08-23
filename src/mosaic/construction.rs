//! Mosaic construction from CLI, samplesheets, actor resources, local, remote, and project sources.

use super::*;

mod actor_resource;
mod assembly;
mod config;
mod local;
mod project;
mod remote;
mod samplesheet;

use assembly::PreparedMosaicConstruction;

impl MosaicViewerApp {
    pub fn from_args(
        cc: &eframe::CreationContext<'_>,
        args: MosaicCliArgs,
    ) -> anyhow::Result<Self> {
        if let Some(sheet) = args.samplesheet_csv.as_deref() {
            apply_napari_like_dark(&cc.egui_ctx);

            let _gl = cc
                .gl
                .as_ref()
                .context("mosaic mode requires GPU (OpenGL) backend")?;

            return Self::from_samplesheet_context(&cc.egui_ctx, sheet, args.columns);
        }
        Self::from_config(cc, args)
    }

    #[cfg(test)]
    pub fn from_samplesheet_runtime(
        ctx: &egui::Context,
        gpu_available: bool,
        samplesheet_csv: &Path,
        columns: Option<usize>,
    ) -> anyhow::Result<Self> {
        if !gpu_available {
            anyhow::bail!("mosaic mode requires GPU (OpenGL) backend");
        }

        Self::from_samplesheet_context(ctx, samplesheet_csv, columns)
    }
}
