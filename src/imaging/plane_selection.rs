use std::ops::Range;

use crate::data::ome::{Dims, LevelInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneSelection {
    pub z_level0: u64,
}

pub fn level0_z_extent(dims: &Dims, level0: &LevelInfo) -> Option<u64> {
    let z_dim = dims.z?;
    level0.shape.get(z_dim).copied()
}

pub fn plane_selection_for_z(
    dims: &Dims,
    level0: &LevelInfo,
    z_level0: u64,
) -> Option<PlaneSelection> {
    let z_extent = level0_z_extent(dims, level0)?;
    if z_extent == 0 {
        return None;
    }
    Some(PlaneSelection {
        z_level0: z_level0.min(z_extent.saturating_sub(1)),
    })
}

pub fn map_level0_z_to_level(
    dims: &Dims,
    level0: &LevelInfo,
    level: &LevelInfo,
    selection: PlaneSelection,
) -> Option<u64> {
    let z_dim = dims.z?;
    let level0_len = *level0.shape.get(z_dim)?;
    let level_len = *level.shape.get(z_dim)?;
    if level0_len == 0 || level_len == 0 {
        return None;
    }

    let base_scale = level0
        .scale
        .get(z_dim)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1.0);
    let level_scale = level
        .scale
        .get(z_dim)
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(base_scale.max(1.0));
    let base_translation = level0
        .translation
        .get(z_dim)
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);
    let level_translation = level
        .translation
        .get(z_dim)
        .copied()
        .filter(|v| v.is_finite())
        .unwrap_or(0.0);

    let z0 = selection.z_level0.min(level0_len.saturating_sub(1));
    let center_world = base_translation + (z0 as f32 + 0.5) * base_scale;
    let mapped = ((center_world - level_translation) / level_scale).floor();
    let mapped = if mapped.is_finite() { mapped as i64 } else { 0 };
    Some(mapped.clamp(0, level_len.saturating_sub(1) as i64) as u64)
}

pub fn image_subset_ranges(
    dims: &Dims,
    level0: &LevelInfo,
    level: &LevelInfo,
    channel: Option<u64>,
    y_range: Range<u64>,
    x_range: Range<u64>,
    plane: Option<PlaneSelection>,
) -> Vec<Range<u64>> {
    let z_index = plane.and_then(|selection| map_level0_z_to_level(dims, level0, level, selection));
    let mut ranges: Vec<Range<u64>> = Vec::with_capacity(level.shape.len());

    for dim in 0..level.shape.len() {
        let dim_len = level.shape[dim];
        if Some(dim) == dims.c {
            let ch = channel.unwrap_or(0).min(dim_len.saturating_sub(1));
            ranges.push(ch..ch.saturating_add(1));
        } else if Some(dim) == dims.z {
            let z = z_index.unwrap_or(0).min(dim_len.saturating_sub(1));
            ranges.push(z..z.saturating_add(1));
        } else if dim == dims.y {
            ranges.push(
                y_range.start.min(dim_len)
                    ..y_range.end.min(dim_len).max(y_range.start.min(dim_len)),
            );
        } else if dim == dims.x {
            ranges.push(
                x_range.start.min(dim_len)
                    ..x_range.end.min(dim_len).max(x_range.start.min(dim_len)),
            );
        } else {
            let end = dim_len.min(1);
            ranges.push(0..end);
        }
    }

    ranges
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
    fn maps_level0_z_to_coarser_level_with_translation() {
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

        let selection = plane_selection_for_z(&dims, &level0, 10).expect("plane selection");
        assert_eq!(
            map_level0_z_to_level(&dims, &level0, &level2, selection),
            Some(5)
        );
    }

    #[test]
    fn builds_subset_ranges_for_selected_plane() {
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

        let selection = plane_selection_for_z(&dims, &level0, 10);
        let ranges =
            image_subset_ranges(&dims, &level0, &level2, Some(2), 11..29, 31..47, selection);
        assert_eq!(ranges, vec![2..3, 5..6, 11..29, 31..47]);
    }
}
