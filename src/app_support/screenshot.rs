use std::path::{Path, PathBuf};

use crossbeam_channel::Sender;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenshotSettings {
    pub include_scale_bar: bool,
    pub include_legend: bool,
    pub scale_bar_scale: f32,
    pub legend_scale: f32,
}

impl Default for ScreenshotSettings {
    fn default() -> Self {
        Self {
            include_scale_bar: true,
            include_legend: true,
            scale_bar_scale: 1.0,
            legend_scale: 1.0,
        }
    }
}

/// Transient native dialog state populated from actor-owned screenshot preferences.
#[derive(Debug, Clone, Default)]
pub struct ScreenshotDialogState {
    pub open: bool,
    pub settings: ScreenshotSettings,
    pub output_dir: Option<PathBuf>,
}

impl ScreenshotDialogState {
    pub fn output_dir(&self) -> Option<&Path> {
        self.output_dir.as_deref()
    }
}

/// One generation-specific pixel request released by the actor to the renderer.
#[derive(Debug, Clone)]
pub struct RendererScreenshotRequest {
    pub settings: ScreenshotSettings,
    pub presentation: PresentationScreenshotReply,
}

#[derive(Debug, Clone)]
pub struct PresentationScreenshotReply {
    pub capture_id: u64,
    pub tx: Sender<odon::control::actor::PresentationCaptureCompletion>,
}
