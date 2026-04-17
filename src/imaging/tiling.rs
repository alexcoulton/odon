use eframe::egui;

use crate::data::ome::{Dims, LevelInfo};
use crate::imaging::view_plane::{DisplayAxes, display_downsample};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub tile_y: u64,
    pub tile_x: u64,
}

/// Choose the pyramid level whose pixel footprint is closest to ~1 screen pixel.
///
/// `zoom_screen_per_lvl0_px` is the camera zoom in screen-pixels per level-0 pixel.
/// `extra_scale` is an additional multiplicative scale (e.g. mosaic item scale).
pub fn choose_level_auto(
    levels: &[LevelInfo],
    zoom_screen_per_lvl0_px: f32,
    extra_scale: f32,
) -> usize {
    let target = 1.0f32;
    let mut best = 0usize;
    let mut best_err = f32::INFINITY;
    for l in levels {
        let screen_per_level_px = zoom_screen_per_lvl0_px * extra_scale * l.downsample;
        let err = (screen_per_level_px / target).ln().abs();
        if err < best_err {
            best_err = err;
            best = l.index;
        }
    }
    best
}

/// Coarse->fine draw order: always include the coarsest level, the target level,
/// and an optional "mid" level one step coarser than target.
pub fn levels_to_draw(level_count: usize, target_level: usize) -> Vec<usize> {
    if level_count == 0 {
        return Vec::new();
    }
    let coarsest = level_count.saturating_sub(1);
    let mut levels = Vec::with_capacity(3);
    levels.push(coarsest);
    let mid = target_level.saturating_add(1);
    if mid <= coarsest {
        levels.push(mid);
    }
    levels.push(target_level.min(coarsest));
    levels.sort_unstable_by(|a, b| b.cmp(a)); // coarse -> fine
    levels.dedup();
    levels
}

/// Returns the set of chunk tile coordinates needed to cover a visible region.
///
/// `visible_lvl0` is expressed in level-0 pixel coordinates.
pub fn tiles_needed_lvl0_rect(
    visible_lvl0: egui::Rect,
    level: &LevelInfo,
    dims: &Dims,
    pad_tiles: i64,
) -> Vec<TileCoord> {
    let y_dim = dims.y;
    let x_dim = dims.x;

    let shape_y = level.shape[y_dim] as f32;
    let shape_x = level.shape[x_dim] as f32;
    let bounds_lvl0 = egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(
            shape_x * level.downsample.max(1e-6),
            shape_y * level.downsample.max(1e-6),
        ),
    );
    let visible = visible_lvl0.intersect(bounds_lvl0);
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return Vec::new();
    }

    let inv = 1.0 / level.downsample.max(1e-6);
    let visible_lvl = egui::Rect::from_min_max(
        egui::pos2(visible.min.x * inv, visible.min.y * inv),
        egui::pos2(visible.max.x * inv, visible.max.y * inv),
    );

    let chunk_y = level.chunks[y_dim] as f32;
    let chunk_x = level.chunks[x_dim] as f32;

    let max_tiles_y = ((level.shape[y_dim] + level.chunks[y_dim] - 1) / level.chunks[y_dim]) as i64;
    let max_tiles_x = ((level.shape[x_dim] + level.chunks[x_dim] - 1) / level.chunks[x_dim]) as i64;

    let pad = pad_tiles.max(0);
    let tile_y0 = (visible_lvl.min.y / chunk_y).floor().max(0.0) as i64 - pad;
    let tile_x0 = (visible_lvl.min.x / chunk_x).floor().max(0.0) as i64 - pad;
    let tile_y1 = (visible_lvl.max.y / chunk_y).ceil().max(0.0) as i64 + pad;
    let tile_x1 = (visible_lvl.max.x / chunk_x).ceil().max(0.0) as i64 + pad;

    let y0 = tile_y0.clamp(0, max_tiles_y);
    let y1 = tile_y1.clamp(0, max_tiles_y);
    let x0 = tile_x0.clamp(0, max_tiles_x);
    let x1 = tile_x1.clamp(0, max_tiles_x);

    let mut keys = Vec::new();
    for ty in y0..y1 {
        for tx in x0..x1 {
            keys.push(TileCoord {
                tile_y: ty as u64,
                tile_x: tx as u64,
            });
        }
    }
    keys
}

pub fn tiles_needed_lvl0_rect_for_axes(
    visible_lvl0: egui::Rect,
    level0: &LevelInfo,
    level: &LevelInfo,
    axes: DisplayAxes,
    pad_tiles: i64,
) -> Vec<TileCoord> {
    let y_dim = axes.vertical;
    let x_dim = axes.horizontal;
    let (downsample_y, downsample_x) = display_downsample(
        &Dims {
            c: None,
            z: None,
            y: y_dim,
            x: x_dim,
            ndim: level.shape.len(),
        },
        level0,
        level,
        crate::imaging::view_plane::ViewPlaneMode::Xy,
    )
    .unwrap_or((level.downsample.max(1e-6), level.downsample.max(1e-6)));

    let shape_y = level.shape[y_dim] as f32;
    let shape_x = level.shape[x_dim] as f32;
    let bounds_lvl0 = egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(
            shape_x * downsample_x.max(1e-6),
            shape_y * downsample_y.max(1e-6),
        ),
    );
    let visible = visible_lvl0.intersect(bounds_lvl0);
    if visible.width() <= 0.0 || visible.height() <= 0.0 {
        return Vec::new();
    }

    let visible_lvl = egui::Rect::from_min_max(
        egui::pos2(
            visible.min.x / downsample_x.max(1e-6),
            visible.min.y / downsample_y.max(1e-6),
        ),
        egui::pos2(
            visible.max.x / downsample_x.max(1e-6),
            visible.max.y / downsample_y.max(1e-6),
        ),
    );

    let chunk_y = level.chunks[y_dim] as f32;
    let chunk_x = level.chunks[x_dim] as f32;

    let max_tiles_y = ((level.shape[y_dim] + level.chunks[y_dim] - 1) / level.chunks[y_dim]) as i64;
    let max_tiles_x = ((level.shape[x_dim] + level.chunks[x_dim] - 1) / level.chunks[x_dim]) as i64;

    let pad = pad_tiles.max(0);
    let tile_y0 = (visible_lvl.min.y / chunk_y).floor().max(0.0) as i64 - pad;
    let tile_x0 = (visible_lvl.min.x / chunk_x).floor().max(0.0) as i64 - pad;
    let tile_y1 = (visible_lvl.max.y / chunk_y).ceil().max(0.0) as i64 + pad;
    let tile_x1 = (visible_lvl.max.x / chunk_x).ceil().max(0.0) as i64 + pad;

    let y0 = tile_y0.clamp(0, max_tiles_y);
    let y1 = tile_y1.clamp(0, max_tiles_y);
    let x0 = tile_x0.clamp(0, max_tiles_x);
    let x1 = tile_x1.clamp(0, max_tiles_x);

    let mut keys = Vec::new();
    for ty in y0..y1 {
        for tx in x0..x1 {
            keys.push(TileCoord {
                tile_y: ty as u64,
                tile_x: tx as u64,
            });
        }
    }
    keys
}
