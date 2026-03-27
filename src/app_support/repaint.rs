use std::sync::OnceLock;
use std::time::Duration;

use eframe::egui;

const MAX_FPS_ENV: &str = "RUST_OZ_MAX_FPS";

pub fn request_repaint_busy(ctx: &egui::Context) {
    if let Some(delay) = max_frame_delay() {
        ctx.request_repaint_after(delay);
    } else {
        ctx.request_repaint();
    }
}

fn max_frame_delay() -> Option<Duration> {
    static MAX_FRAME_DELAY: OnceLock<Option<Duration>> = OnceLock::new();
    *MAX_FRAME_DELAY.get_or_init(|| {
        let fps = std::env::var(MAX_FPS_ENV).ok()?;
        let fps = fps.parse::<f32>().ok()?;
        if !fps.is_finite() || fps <= 0.0 {
            return None;
        }
        Some(Duration::from_secs_f32(1.0 / fps))
    })
}
