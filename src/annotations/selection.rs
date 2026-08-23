use eframe::egui;

use super::AnnotationRoiData;

pub(super) struct PointsRadius;

impl PointsRadius {
    pub(super) fn effective(base_radius_points: f32, zoom_screen_per_world_px: f32) -> f32 {
        let zoom = zoom_screen_per_world_px.max(1e-6);
        (base_radius_points.max(0.0) * zoom.sqrt()).clamp(0.75, 40.0)
    }
}

pub(super) fn pick_nearest_in_roi(
    roi: &AnnotationRoiData,
    local: egui::Pos2,
    radius_local: f32,
) -> Option<usize> {
    if radius_local <= 0.0 {
        return None;
    }
    let r2 = radius_local * radius_local;

    let mut best: Option<(usize, f32)> = None;
    if let Some(bins) = roi.bins_local.as_ref() {
        let rect =
            egui::Rect::from_center_size(local, egui::vec2(radius_local * 2.0, radius_local * 2.0));
        let (x0, y0, x1, y1) = bins.bin_range_for_world_rect(rect);
        for by in y0..=y1 {
            for bx in x0..=x1 {
                let bi = by * bins.bins_w + bx;
                for &idx_u in bins.bin_slice(bi) {
                    let idx = idx_u as usize;
                    let Some(p) = roi.positions_local.get(idx) else {
                        continue;
                    };
                    let dx = p.x - local.x;
                    let dy = p.y - local.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 <= r2 {
                        match best {
                            None => best = Some((idx, d2)),
                            Some((_best_i, best_d2)) if d2 < best_d2 => best = Some((idx, d2)),
                            _ => {}
                        }
                    }
                }
            }
        }
        return best.map(|(i, _)| i);
    }

    // Fallback: scan (bounded).
    let max = roi.count.min(20_000);
    for idx in 0..max {
        let Some(p) = roi.positions_local.get(idx) else {
            continue;
        };
        let dx = p.x - local.x;
        let dy = p.y - local.y;
        let d2 = dx * dx + dy * dy;
        if d2 <= r2 {
            match best {
                None => best = Some((idx, d2)),
                Some((_best_i, best_d2)) if d2 < best_d2 => best = Some((idx, d2)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| i)
}
