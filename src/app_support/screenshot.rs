use std::path::{Path, PathBuf};
use std::thread;

use crossbeam_channel::{Receiver, Sender};

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

#[derive(Debug, Clone)]
pub struct ScreenshotRequest {
    pub id: u64,
    pub path: PathBuf,
    pub settings: ScreenshotSettings,
    pub presentation: Option<PresentationScreenshotReply>,
}

#[derive(Debug, Clone)]
pub struct PresentationScreenshotReply {
    pub capture_id: u64,
    pub tx: Sender<odon::control::actor::PresentationCaptureCompletion>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    struct TestScreenshotDir(PathBuf);

    impl TestScreenshotDir {
        fn new() -> Self {
            static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
            let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "odon-screenshot-tests-{}-{sequence}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create screenshot test directory");
            Self(path)
        }
    }

    impl Drop for TestScreenshotDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_rgba_bottom_up() -> Vec<u8> {
        vec![
            255, 0, 0, 255, 0, 255, 0, 128, // bottom: red, green
            0, 0, 255, 255, 255, 255, 255, 0, // top: blue, white
        ]
    }

    #[test]
    fn screenshot_worker_writes_top_down_rgb_png_and_reports_completion() {
        let dir = TestScreenshotDir::new();
        let path = dir.0.join("capture.png");
        let worker = ScreenshotWorkerHandle::spawn();
        worker
            .tx
            .send(ScreenshotWorkerMsg::SavePng {
                id: 42,
                path: path.clone(),
                width: 2,
                height: 2,
                rgba_bottom_up: test_rgba_bottom_up(),
            })
            .expect("queue screenshot");

        let response = worker
            .rx
            .recv_timeout(Duration::from_secs(2))
            .expect("screenshot completion");
        match response {
            ScreenshotWorkerResp::Saved {
                id,
                path: saved_path,
                result,
            } => {
                assert_eq!(id, 42);
                assert_eq!(saved_path, path);
                result.expect("screenshot save result");
            }
        }

        let image = image::open(&path).expect("open saved PNG");
        assert_eq!(image.dimensions(), (2, 2));
        let rgb = image.to_rgb8();
        assert_eq!(rgb.get_pixel(0, 0).0, [0, 0, 255]);
        assert_eq!(rgb.get_pixel(1, 0).0, [255, 255, 255]);
        assert_eq!(rgb.get_pixel(0, 1).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 1).0, [0, 255, 0]);
    }

    #[test]
    fn numbered_screenshot_paths_skip_existing_files_and_validate_directory() {
        let dir = TestScreenshotDir::new();
        let first = next_numbered_screenshot_path(&dir.0, "sample.screenshot.png")
            .expect("first screenshot path");
        assert_eq!(
            first.file_name().and_then(|name| name.to_str()),
            Some("sample.screenshot.0001.png")
        );
        std::fs::write(&first, []).expect("reserve first screenshot path");
        let second = next_numbered_screenshot_path(&dir.0, "nested/ignored.png")
            .expect("second screenshot path with isolated base");
        assert_eq!(
            second.file_name().and_then(|name| name.to_str()),
            Some("ignored.0001.png")
        );
        let next_sample = next_numbered_screenshot_path(&dir.0, "sample.screenshot.png")
            .expect("next sample screenshot path");
        assert_eq!(
            next_sample.file_name().and_then(|name| name.to_str()),
            Some("sample.screenshot.0002.png")
        );

        let error = next_numbered_screenshot_path(&dir.0.join("missing"), "capture.png")
            .expect_err("missing screenshot directory must fail");
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn screenshot_encoding_rejects_empty_dimensions_and_wrong_buffer_size() {
        let dir = TestScreenshotDir::new();
        let path = dir.0.join("invalid.png");
        assert!(save_png_rgba_bottom_up(&path, 0, 2, &[]).is_err());
        assert!(save_png_rgba_bottom_up(&path, 2, 2, &[0; 15]).is_err());
        assert!(!path.exists());
    }
}
