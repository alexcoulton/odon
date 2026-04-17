use std::ops::Range;

use crate::data::ome::{Dims, LevelInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewPlaneMode {
    Xy,
    Xz,
    Yz,
}

impl ViewPlaneMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Xy => "XY",
            Self::Xz => "XZ",
            Self::Yz => "YZ",
        }
    }

    pub fn slice_axis_label(self) -> &'static str {
        match self {
            Self::Xy => "Z",
            Self::Xz => "Y",
            Self::Yz => "X",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ViewPlaneSelection {
    pub mode: ViewPlaneMode,
    pub slice_level0: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayAxes {
    pub vertical: usize,
    pub horizontal: usize,
}

pub fn supported_modes(dims: &Dims) -> Vec<ViewPlaneMode> {
    let mut out = vec![ViewPlaneMode::Xy];
    if dims.z.is_some() {
        out.push(ViewPlaneMode::Xz);
        out.push(ViewPlaneMode::Yz);
    }
    out
}

pub fn display_axes(dims: &Dims, mode: ViewPlaneMode) -> Option<DisplayAxes> {
    match mode {
        ViewPlaneMode::Xy => Some(DisplayAxes {
            vertical: dims.y,
            horizontal: dims.x,
        }),
        ViewPlaneMode::Xz => Some(DisplayAxes {
            vertical: dims.z?,
            horizontal: dims.x,
        }),
        ViewPlaneMode::Yz => Some(DisplayAxes {
            vertical: dims.z?,
            horizontal: dims.y,
        }),
    }
}

pub fn slice_dim(dims: &Dims, mode: ViewPlaneMode) -> Option<usize> {
    match mode {
        ViewPlaneMode::Xy => dims.z,
        ViewPlaneMode::Xz => Some(dims.y),
        ViewPlaneMode::Yz => Some(dims.x),
    }
}

pub fn slice_extent_level0(dims: &Dims, level0: &LevelInfo, mode: ViewPlaneMode) -> Option<u64> {
    let slice_dim = slice_dim(dims, mode)?;
    level0.shape.get(slice_dim).copied()
}

pub fn clamp_selection(
    dims: &Dims,
    level0: &LevelInfo,
    mode: ViewPlaneMode,
    slice_level0: u64,
) -> ViewPlaneSelection {
    let slice_level0 = slice_extent_level0(dims, level0, mode)
        .map(|extent| slice_level0.min(extent.saturating_sub(1)))
        .unwrap_or(0);
    ViewPlaneSelection { mode, slice_level0 }
}

fn axis_spacing(level: &LevelInfo, dim: usize) -> Option<f32> {
    level
        .scale
        .get(dim)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
}

fn axis_translation(level: &LevelInfo, dim: usize) -> f32 {
    level
        .translation
        .get(dim)
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

pub fn axis_downsample(level0: &LevelInfo, level: &LevelInfo, dim: usize) -> f32 {
    let base = axis_spacing(level0, dim).unwrap_or(1.0);
    let current = axis_spacing(level, dim).unwrap_or(base.max(1.0));
    (current / base.max(1e-6)).max(1e-6)
}

pub fn display_downsample(
    dims: &Dims,
    level0: &LevelInfo,
    level: &LevelInfo,
    mode: ViewPlaneMode,
) -> Option<(f32, f32)> {
    let axes = display_axes(dims, mode)?;
    Some((
        axis_downsample(level0, level, axes.vertical),
        axis_downsample(level0, level, axes.horizontal),
    ))
}

pub fn map_level0_slice_to_level(
    dims: &Dims,
    level0: &LevelInfo,
    level: &LevelInfo,
    selection: ViewPlaneSelection,
) -> Option<u64> {
    let slice_dim = slice_dim(dims, selection.mode)?;
    let level0_len = *level0.shape.get(slice_dim)?;
    let level_len = *level.shape.get(slice_dim)?;
    if level0_len == 0 || level_len == 0 {
        return None;
    }

    let base_scale = axis_spacing(level0, slice_dim).unwrap_or(1.0);
    let level_scale = axis_spacing(level, slice_dim).unwrap_or(base_scale.max(1.0));
    let base_translation = axis_translation(level0, slice_dim);
    let level_translation = axis_translation(level, slice_dim);

    let slice0 = selection.slice_level0.min(level0_len.saturating_sub(1));
    let center_world = base_translation + (slice0 as f32 + 0.5) * base_scale;
    let mapped = ((center_world - level_translation) / level_scale).floor();
    let mapped = if mapped.is_finite() { mapped as i64 } else { 0 };
    Some(mapped.clamp(0, level_len.saturating_sub(1) as i64) as u64)
}

pub fn image_subset_ranges_for_view(
    dims: &Dims,
    level0: &LevelInfo,
    level: &LevelInfo,
    channel: Option<u64>,
    row_range: Range<u64>,
    col_range: Range<u64>,
    selection: ViewPlaneSelection,
) -> Option<Vec<Range<u64>>> {
    let axes = display_axes(dims, selection.mode)?;
    let slice_dim = slice_dim(dims, selection.mode);
    let slice_index =
        slice_dim.and_then(|_| map_level0_slice_to_level(dims, level0, level, selection));
    let mut ranges: Vec<Range<u64>> = Vec::with_capacity(level.shape.len());

    for dim in 0..level.shape.len() {
        let dim_len = level.shape[dim];
        if Some(dim) == dims.c {
            let ch = channel.unwrap_or(0).min(dim_len.saturating_sub(1));
            ranges.push(ch..ch.saturating_add(1));
        } else if Some(dim) == slice_dim {
            let slice = slice_index.unwrap_or(0).min(dim_len.saturating_sub(1));
            ranges.push(slice..slice.saturating_add(1));
        } else if dim == axes.vertical {
            let start = row_range.start.min(dim_len);
            let end = row_range.end.min(dim_len).max(start);
            ranges.push(start..end);
        } else if dim == axes.horizontal {
            let start = col_range.start.min(dim_len);
            let end = col_range.end.min(dim_len).max(start);
            ranges.push(start..end);
        } else {
            let end = dim_len.min(1);
            ranges.push(0..end);
        }
    }

    Some(ranges)
}

pub fn local_to_world_scale(dims: &Dims, level0: &LevelInfo, mode: ViewPlaneMode) -> (f32, f32) {
    if matches!(mode, ViewPlaneMode::Xy) {
        return (1.0, 1.0);
    }

    let Some(axes) = display_axes(dims, mode) else {
        return (1.0, 1.0);
    };
    let horizontal = axis_spacing(level0, axes.horizontal)
        .unwrap_or(1.0)
        .max(1e-6);
    let vertical = axis_spacing(level0, axes.vertical).unwrap_or(horizontal);
    (1.0, (vertical / horizontal).max(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(shape: Vec<u64>, scale: Vec<f32>, translation: Vec<f32>) -> LevelInfo {
        LevelInfo {
            index: 0,
            path: "0".to_string(),
            shape: shape.clone(),
            chunks: vec![1; shape.len()],
            downsample: 1.0,
            dtype: "uint16".to_string(),
            scale,
            translation,
        }
    }

    #[test]
    fn maps_level0_slice_to_coarser_level_with_translation() {
        let dims = Dims {
            c: Some(0),
            z: Some(1),
            y: 2,
            x: 3,
            ndim: 4,
        };
        let level0 = level(
            vec![4, 723, 3215, 3598],
            vec![1.0, 8.0, 2.6, 2.6],
            vec![0.0, 0.0, 0.0, 0.0],
        );
        let level2 = level(
            vec![4, 361, 526, 599],
            vec![1.0, 16.0, 15.6, 15.6],
            vec![0.0, 1.0, 1.625, 1.625],
        );

        let selection = clamp_selection(&dims, &level0, ViewPlaneMode::Xy, 10);
        assert_eq!(
            map_level0_slice_to_level(&dims, &level0, &level2, selection),
            Some(5)
        );
    }

    #[test]
    fn builds_subset_ranges_for_xz_view() {
        let dims = Dims {
            c: Some(0),
            z: Some(1),
            y: 2,
            x: 3,
            ndim: 4,
        };
        let level0 = level(
            vec![4, 723, 3215, 3598],
            vec![1.0, 8.0, 2.6, 2.6],
            vec![0.0, 0.0, 0.0, 0.0],
        );
        let level2 = level(
            vec![4, 361, 526, 599],
            vec![1.0, 16.0, 15.6, 15.6],
            vec![0.0, 1.0, 1.625, 1.625],
        );

        let selection = clamp_selection(&dims, &level0, ViewPlaneMode::Xz, 20);
        let ranges = image_subset_ranges_for_view(
            &dims,
            &level0,
            &level2,
            Some(2),
            11..29,
            31..47,
            selection,
        )
        .expect("ranges");
        assert_eq!(ranges, vec![2..3, 11..29, 20..21, 31..47]);
    }
}
