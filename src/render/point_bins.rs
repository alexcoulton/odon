use eframe::egui;

/// A set of points binned into a uniform grid for fast hover/picking queries.
///
/// Points are not duplicated across bins: each point is assigned to exactly one bin.
#[derive(Debug, Clone)]
pub struct PointIndexBins {
    pub origin: egui::Pos2,
    pub bin_size: f32,
    pub bins_w: usize,
    pub bins_h: usize,

    /// Packed indices for all bins concatenated (bin order: row-major).
    pub indices: Vec<u32>,
    /// Per-bin starting index into `indices`.
    pub offsets: Vec<u32>,
    /// Per-bin point count.
    pub counts: Vec<u32>,
}

impl PointIndexBins {
    pub fn build(points: &[egui::Pos2], bin_size: f32) -> Option<Self> {
        let bin_size = bin_size.max(1.0);

        // Bounds.
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut any = false;
        for p in points {
            if p.x.is_finite() && p.y.is_finite() {
                any = true;
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
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
        let mut counts = vec![0u32; bins_len];

        // Pass 1: counts.
        for p in points {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let bx = ((p.x - origin.x) / bin_size)
                .floor()
                .clamp(0.0, (bins_w - 1) as f32) as usize;
            let by = ((p.y - origin.y) / bin_size)
                .floor()
                .clamp(0.0, (bins_h - 1) as f32) as usize;
            let i = by * bins_w + bx;
            counts[i] = counts[i].saturating_add(1);
        }

        // Prefix sum offsets.
        let mut offsets = vec![0u32; bins_len];
        let mut total = 0u32;
        for (i, c) in counts.iter().copied().enumerate() {
            offsets[i] = total;
            total = total.saturating_add(c);
        }
        let total_usize = total as usize;
        let mut indices = vec![0u32; total_usize];

        // Pass 2: fill.
        let mut cursor = offsets.clone();
        for (idx, p) in points.iter().enumerate() {
            if !(p.x.is_finite() && p.y.is_finite()) {
                continue;
            }
            let bx = ((p.x - origin.x) / bin_size)
                .floor()
                .clamp(0.0, (bins_w - 1) as f32) as usize;
            let by = ((p.y - origin.y) / bin_size)
                .floor()
                .clamp(0.0, (bins_h - 1) as f32) as usize;
            let bi = by * bins_w + bx;
            let w = cursor[bi] as usize;
            if w < indices.len() {
                indices[w] = idx as u32;
            }
            cursor[bi] = cursor[bi].saturating_add(1);
        }

        Some(Self {
            origin,
            bin_size,
            bins_w,
            bins_h,
            indices,
            offsets,
            counts,
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

    pub fn bin_slice(&self, bin_index: usize) -> &[u32] {
        let start = self.offsets.get(bin_index).copied().unwrap_or(0) as usize;
        let count = self.counts.get(bin_index).copied().unwrap_or(0) as usize;
        let end = start.saturating_add(count).min(self.indices.len());
        &self.indices[start..end]
    }
}
