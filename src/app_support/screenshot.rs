use std::path::{Path, PathBuf};
use std::thread;

use crossbeam_channel::{Receiver, Sender};

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone)]
pub struct ScreenshotRequest {
    pub id: u64,
    pub path: PathBuf,
    pub settings: ScreenshotSettings,
}

#[derive(Debug)]
pub enum ScreenshotWorkerMsg {
    SavePng {
        id: u64,
        path: PathBuf,
        width: usize,
        height: usize,
        rgba_bottom_up: Vec<u8>,
    },
}

#[derive(Debug)]
pub enum ScreenshotWorkerResp {
    Saved {
        id: u64,
        path: PathBuf,
        result: Result<(), String>,
    },
}

#[derive(Debug)]
pub struct ScreenshotWorkerHandle {
    pub tx: Sender<ScreenshotWorkerMsg>,
    pub rx: Receiver<ScreenshotWorkerResp>,
}

impl ScreenshotWorkerHandle {
    pub fn spawn() -> Self {
        let (tx, rx_in) = crossbeam_channel::unbounded::<ScreenshotWorkerMsg>();
        let (tx_out, rx) = crossbeam_channel::unbounded::<ScreenshotWorkerResp>();
        thread::spawn(move || {
            while let Ok(msg) = rx_in.recv() {
                match msg {
                    ScreenshotWorkerMsg::SavePng {
                        id,
                        path,
                        width,
                        height,
                        rgba_bottom_up,
                    } => {
                        let result = save_png_rgba_bottom_up(&path, width, height, &rgba_bottom_up)
                            .map_err(|e| e.to_string());
                        let _ = tx_out.send(ScreenshotWorkerResp::Saved { id, path, result });
                    }
                }
            }
        });
        Self { tx, rx }
    }
}

pub fn next_numbered_screenshot_path(
    dir: &Path,
    default_filename: &str,
) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        dir.is_dir(),
        "Screenshot folder does not exist: {}",
        dir.display()
    );

    let default_name = Path::new(default_filename)
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("odon.screenshot.png");
    let default_path = Path::new(default_name);
    let stem = default_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("odon.screenshot");
    let ext = default_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");

    for idx in 1..=999_999u32 {
        let candidate = dir.join(format!("{stem}.{idx:04}.{ext}"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!(
        "No free screenshot filename found in {} for base {}",
        dir.display(),
        default_name
    );
}

fn save_png_rgba_bottom_up(
    path: &PathBuf,
    width: usize,
    height: usize,
    rgba_bottom_up: &[u8],
) -> anyhow::Result<()> {
    anyhow::ensure!(width > 0 && height > 0, "empty screenshot dimensions");
    anyhow::ensure!(
        rgba_bottom_up.len() == width.saturating_mul(height).saturating_mul(4),
        "unexpected screenshot buffer size"
    );

    let row_bytes = width.saturating_mul(4);
    let mut rgb_top_down = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
    for y in 0..height {
        let src_y = height - 1 - y;
        let src = src_y.saturating_mul(row_bytes);
        let dst = y.saturating_mul(width).saturating_mul(3);
        for x in 0..width {
            let src_px = src + x.saturating_mul(4);
            let dst_px = dst + x.saturating_mul(3);
            rgb_top_down[dst_px] = rgba_bottom_up[src_px];
            rgb_top_down[dst_px + 1] = rgba_bottom_up[src_px + 1];
            rgb_top_down[dst_px + 2] = rgba_bottom_up[src_px + 2];
        }
    }

    let Some(img) = image::RgbImage::from_raw(width as u32, height as u32, rgb_top_down) else {
        anyhow::bail!("failed to create rgb image");
    };
    img.save(path)?;
    Ok(())
}
