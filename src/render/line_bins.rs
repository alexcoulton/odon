use std::sync::Arc;

use eframe::egui;

/// A set of world-space line segments binned into a uniform grid for fast viewport selection.
///
/// Segments are stored as packed `[x0, y0, x1, y1]` in world (level-0 pixel) coordinates.
#[derive(Debug, Clone)]
pub struct LineSegmentsBins {
    pub origin: egui::Pos2,
    pub bin_size: f32,
    pub bins_w: usize,
    pub bins_h: usize,

    /// Packed segments for all bins concatenated (bin order: row-major).
    pub segments: Arc<Vec<[f32; 4]>>,
    /// Per-bin starting index into `segments`.
    pub offsets: Arc<Vec<u32>>,
    /// Per-bin segment count.
    pub counts: Arc<Vec<u32>>,
}

/// World-space line segments with stable object ids, binned into a uniform grid.
///
/// Segments are stored as packed `[x0, y0, x1, y1, object_index]` in world coordinates.
#[derive(Debug, Clone)]
pub struct ObjectLineSegmentsBins {
    pub origin: egui::Pos2,
    pub bin_size: f32,
    pub bins_w: usize,
    pub bins_h: usize,
    pub segments: Arc<Vec<[f32; 5]>>,
    pub offsets: Arc<Vec<u32>>,
    pub counts: Arc<Vec<u32>>,
}

impl LineSegmentsBins {
    pub fn build_from_polylines(polylines: &[Vec<egui::Pos2>], bin_size: f32) -> Option<Self> {
        let bin_size = bin_size.max(1.0);

        // Find bounds.
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for pts in polylines {
            for p in pts {
                if p.x.is_finite() && p.y.is_finite() {
                    any = true;
                    min_x = min_x.min(p.x);
                    min_y = min_y.min(p.y);
                    max_x = max_x.max(p.x);
                    max_y = max_y.max(p.y);
                }
            }
        }
        if !any {
            return None;
        }
        let w = (max_x - min_x).max(1.0);
        let h = (max_y - min_y).max(1.0);
        let bins_w = ((w / bin_size).ceil() as usize).max(1);
        let bins_h = ((h / bin_size).ceil() as usize).max(1);
        let origin = egui::pos2(min_x, min_y);

        let bins_len = bins_w.saturating_mul(bins_h);
        let mut tmp: Vec<Vec<[f32; 4]>> = vec![Vec::new(); bins_len];

        // Populate bins. We insert a segment into every bin its bounding box touches.
        for pts in polylines {
            if pts.len() < 2 {
                continue;
            }
            for win in pts.windows(2) {
                let a = win[0];
                let b = win[1];
                if !(a.x.is_finite() && a.y.is_finite() && b.x.is_finite() && b.y.is_finite()) {
                    continue;
                }
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                if (dx * dx + dy * dy) < 1e-12 {
                    continue;
                }

                let bx0 = a.x.min(b.x);
                let by0 = a.y.min(b.y);
                let bx1 = a.x.max(b.x);
                let by1 = a.y.max(b.y);

                let x0 = ((bx0 - origin.x) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_w - 1) as f32) as usize;
                let y0 = ((by0 - origin.y) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_h - 1) as f32) as usize;
                let x1 = ((bx1 - origin.x) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_w - 1) as f32) as usize;
                let y1 = ((by1 - origin.y) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_h - 1) as f32) as usize;

                let packed = [a.x, a.y, b.x, b.y];
                for by in y0..=y1 {
                    for bx in x0..=x1 {
                        tmp[by * bins_w + bx].push(packed);
                    }
                }
            }
        }

        // Flatten bins.
        let mut offsets: Vec<u32> = vec![0; bins_len];
        let mut counts: Vec<u32> = vec![0; bins_len];
        let total: usize = tmp.iter().map(|b| b.len()).sum();
        let mut segments: Vec<[f32; 4]> = Vec::with_capacity(total);

        for (i, bin) in tmp.into_iter().enumerate() {
            offsets[i] = segments.len() as u32;
            counts[i] = bin.len() as u32;
            segments.extend(bin);
        }

        Some(Self {
            origin,
            bin_size,
            bins_w,
            bins_h,
            segments: Arc::new(segments),
            offsets: Arc::new(offsets),
            counts: Arc::new(counts),
        })
    }

    pub fn bin_range_for_world_rect(&self, rect: egui::Rect) -> (usize, usize, usize, usize) {
        let x0 = ((rect.min.x - self.origin.x) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_w - 1) as f32) as usize;
        let y0 = ((rect.min.y - self.origin.y) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_h - 1) as f32) as usize;
        let x1 = ((rect.max.x - self.origin.x) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_w - 1) as f32) as usize;
        let y1 = ((rect.max.y - self.origin.y) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_h - 1) as f32) as usize;
        (x0, y0, x1, y1)
    }

    pub fn bin_slice(&self, bin_index: usize) -> &[[f32; 4]] {
        let offsets = self.offsets.as_ref();
        let counts = self.counts.as_ref();
        let start = offsets.get(bin_index).copied().unwrap_or(0) as usize;
        let count = counts.get(bin_index).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.segments.len());
        &self.segments[start..end]
    }
}

impl ObjectLineSegmentsBins {
    pub fn build_from_indexed_polylines(
        polylines: &[(usize, Vec<egui::Pos2>)],
        bin_size: f32,
    ) -> Option<Self> {
        let bin_size = bin_size.max(1.0);

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;

        for (_, pts) in polylines {
            for p in pts {
                if p.x.is_finite() && p.y.is_finite() {
                    any = true;
                    min_x = min_x.min(p.x);
                    min_y = min_y.min(p.y);
                    max_x = max_x.max(p.x);
                    max_y = max_y.max(p.y);
                }
            }
        }
        if !any {
            return None;
        }

        let w = (max_x - min_x).max(1.0);
        let h = (max_y - min_y).max(1.0);
        let bins_w = ((w / bin_size).ceil() as usize).max(1);
        let bins_h = ((h / bin_size).ceil() as usize).max(1);
        let origin = egui::pos2(min_x, min_y);

        let bins_len = bins_w.saturating_mul(bins_h);
        let mut tmp: Vec<Vec<[f32; 5]>> = vec![Vec::new(); bins_len];

        for (object_index, pts) in polylines {
            if pts.len() < 2 {
                continue;
            }
            for win in pts.windows(2) {
                let a = win[0];
                let b = win[1];
                if !(a.x.is_finite() && a.y.is_finite() && b.x.is_finite() && b.y.is_finite()) {
                    continue;
                }
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                if (dx * dx + dy * dy) < 1e-12 {
                    continue;
                }

                let bx0 = a.x.min(b.x);
                let by0 = a.y.min(b.y);
                let bx1 = a.x.max(b.x);
                let by1 = a.y.max(b.y);

                let x0 = ((bx0 - origin.x) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_w - 1) as f32) as usize;
                let y0 = ((by0 - origin.y) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_h - 1) as f32) as usize;
                let x1 = ((bx1 - origin.x) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_w - 1) as f32) as usize;
                let y1 = ((by1 - origin.y) / bin_size)
                    .floor()
                    .clamp(0.0, (bins_h - 1) as f32) as usize;

                let packed = [a.x, a.y, b.x, b.y, *object_index as f32];
                for by in y0..=y1 {
                    for bx in x0..=x1 {
                        tmp[by * bins_w + bx].push(packed);
                    }
                }
            }
        }

        let mut offsets: Vec<u32> = vec![0; bins_len];
        let mut counts: Vec<u32> = vec![0; bins_len];
        let total: usize = tmp.iter().map(|b| b.len()).sum();
        let mut segments: Vec<[f32; 5]> = Vec::with_capacity(total);

        for (i, bin) in tmp.into_iter().enumerate() {
            offsets[i] = segments.len() as u32;
            counts[i] = bin.len() as u32;
            segments.extend(bin);
        }

        Some(Self {
            origin,
            bin_size,
            bins_w,
            bins_h,
            segments: Arc::new(segments),
            offsets: Arc::new(offsets),
            counts: Arc::new(counts),
        })
    }

    pub fn bin_range_for_world_rect(&self, rect: egui::Rect) -> (usize, usize, usize, usize) {
        let x0 = ((rect.min.x - self.origin.x) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_w - 1) as f32) as usize;
        let y0 = ((rect.min.y - self.origin.y) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_h - 1) as f32) as usize;
        let x1 = ((rect.max.x - self.origin.x) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_w - 1) as f32) as usize;
        let y1 = ((rect.max.y - self.origin.y) / self.bin_size)
            .floor()
            .clamp(0.0, (self.bins_h - 1) as f32) as usize;
        (x0, y0, x1, y1)
    }

    pub fn bin_slice(&self, bin_index: usize) -> &[[f32; 5]] {
        let start = self.offsets.get(bin_index).copied().unwrap_or(0) as usize;
        let count = self.counts.get(bin_index).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.segments.len());
        &self.segments[start..end]
    }
}
