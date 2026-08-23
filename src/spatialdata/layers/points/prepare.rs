use std::sync::Arc;

use eframe::egui;

use crate::render::point_bins::PointIndexBins;
use crate::spatialdata::PointsMeta;

use super::{
    PreparedSpatialPoints, SpatialAxisMode, SpatialFeatureCache, SpatialPointsPrepareConfig,
    SpatialScaleMode,
};

pub(super) fn prepare_spatial_points_payload(
    payload: crate::spatialdata::PointsPayload,
    config: SpatialPointsPrepareConfig,
) -> anyhow::Result<PreparedSpatialPoints> {
    let raw_xy = Arc::new(payload.xy);
    let meta = Arc::new(payload.meta);
    prepare_spatial_points_from_parts(raw_xy, meta, config)
}

pub(super) fn prepare_spatial_points_from_parts(
    raw_xy: Arc<Vec<[f32; 2]>>,
    meta: Arc<PointsMeta>,
    config: SpatialPointsPrepareConfig,
) -> anyhow::Result<PreparedSpatialPoints> {
    // Normalize the raw parquet payload into viewer-native structures once so the
    // draw path can stay format-agnostic: world coordinates, feature counts, bins,
    // and LOD payloads are all derived here rather than during every frame.
    let raw_len = raw_xy.len();
    let mut feature_counts = meta
        .feature
        .as_ref()
        .map(|feature| vec![0usize; feature.dict.len()])
        .unwrap_or_default();
    if let Some(feature) = meta.feature.as_ref() {
        for &id in &feature.ids {
            if let Some(count) = feature_counts.get_mut(id as usize) {
                *count += 1;
            }
        }
    }

    if raw_len == 0 {
        return Ok(PreparedSpatialPoints {
            raw_xy,
            meta,
            positions_world: Arc::new(Vec::new()),
            values: Arc::new(Vec::new()),
            lod_levels: Arc::new(Vec::new()),
            feature_counts,
            feature_cache: Vec::new(),
            bounds_world: None,
            bins: None,
            last_auto_choice: String::new(),
            loaded_count: 0,
        });
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in raw_xy.iter() {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    let raw_w = (max_x - min_x).abs().max(1e-6);
    let raw_h = (max_y - min_y).abs().max(1e-6);

    let base = config.base_transform.scale;
    let inv = [
        if base[0].abs() > 1e-12 {
            1.0 / base[0]
        } else {
            1.0
        },
        if base[1].abs() > 1e-12 {
            1.0 / base[1]
        } else {
            1.0
        },
    ];
    let identity = [1.0f32, 1.0f32];
    let tr = config.base_transform.translation;

    let scale_candidates: [(&str, [f32; 2]); 3] = [("scale", base), ("inv", inv), ("id", identity)];
    let axis_candidates: [(&str, SpatialAxisMode); 2] =
        [("xy", SpatialAxisMode::XY), ("yx", SpatialAxisMode::YX)];

    let want_scales: Vec<(&str, [f32; 2])> = match config.scale_mode {
        SpatialScaleMode::Identity => vec![("id", identity)],
        SpatialScaleMode::UseScale => vec![("scale", base)],
        SpatialScaleMode::InvertScale => vec![("inv", inv)],
        SpatialScaleMode::Auto => scale_candidates.to_vec(),
    };
    let want_axes: Vec<(&str, SpatialAxisMode)> = match config.axis_mode {
        SpatialAxisMode::XY => vec![("xy", SpatialAxisMode::XY)],
        SpatialAxisMode::YX => vec![("yx", SpatialAxisMode::YX)],
        SpatialAxisMode::Auto => axis_candidates.to_vec(),
    };

    let (mut pick_scale_name, mut pick_scale) = want_scales[0];
    let (mut pick_axis_name, mut pick_axis) = want_axes[0];
    let mut best_score = f32::INFINITY;

    if let Some(img) = config.image_size_world {
        let img_w = img[0].max(1.0);
        let img_h = img[1].max(1.0);
        for (sname, s0) in &want_scales {
            for (aname, a0) in &want_axes {
                let sx = s0[0].abs().max(1e-12);
                let sy = s0[1].abs().max(1e-12);
                let (w, h, min_mapped_x, min_mapped_y, max_mapped_x, max_mapped_y) = match a0 {
                    SpatialAxisMode::XY => {
                        let w = raw_w * sx;
                        let h = raw_h * sy;
                        let minx = min_x * s0[0] + tr[0];
                        let miny = min_y * s0[1] + tr[1];
                        let maxx = max_x * s0[0] + tr[0];
                        let maxy = max_y * s0[1] + tr[1];
                        (
                            w,
                            h,
                            minx.min(maxx),
                            miny.min(maxy),
                            minx.max(maxx),
                            miny.max(maxy),
                        )
                    }
                    SpatialAxisMode::YX => {
                        let w = raw_h * sx;
                        let h = raw_w * sy;
                        let minx = min_y * s0[0] + tr[0];
                        let miny = min_x * s0[1] + tr[1];
                        let maxx = max_y * s0[0] + tr[0];
                        let maxy = max_x * s0[1] + tr[1];
                        (
                            w,
                            h,
                            minx.min(maxx),
                            miny.min(maxy),
                            minx.max(maxx),
                            miny.max(maxy),
                        )
                    }
                    SpatialAxisMode::Auto => unreachable!("auto resolved above"),
                };

                let size_score = (w / img_w).ln().abs() + (h / img_h).ln().abs();
                let off_left = (-min_mapped_x).max(0.0);
                let off_top = (-min_mapped_y).max(0.0);
                let off_right = (max_mapped_x - img_w).max(0.0);
                let off_bottom = (max_mapped_y - img_h).max(0.0);
                let outside_score = (off_left + off_right) / img_w + (off_top + off_bottom) / img_h;
                let origin_score = (min_mapped_x / img_w).abs() + (min_mapped_y / img_h).abs();
                let score = size_score + 0.35 * outside_score + 0.05 * origin_score;
                if score < best_score {
                    best_score = score;
                    pick_scale_name = sname;
                    pick_scale = *s0;
                    pick_axis_name = aname;
                    pick_axis = *a0;
                }
            }
        }
    }

    let last_auto_choice = if config.scale_mode == SpatialScaleMode::Auto
        || config.axis_mode == SpatialAxisMode::Auto
    {
        format!("{pick_scale_name} + {pick_axis_name}")
    } else {
        String::new()
    };

    let s = [
        pick_scale[0] * config.scale_mul,
        pick_scale[1] * config.scale_mul,
    ];

    let mut pos: Vec<egui::Pos2> = Vec::with_capacity(raw_len);
    for p in raw_xy.iter() {
        let (in_x, in_y) = match pick_axis {
            SpatialAxisMode::XY => (p[0], p[1]),
            SpatialAxisMode::YX => (p[1], p[0]),
            SpatialAxisMode::Auto => (p[0], p[1]),
        };
        let x = in_x * s[0] + tr[0];
        let y = in_y * s[1] + tr[1];
        pos.push(egui::pos2(x, y));
    }

    let positions_world = Arc::new(pos);
    let values = Arc::new(vec![1.0f32; raw_len]);
    let lod_levels = Arc::new(Vec::new());

    let feature_cache = if let Some(feature_meta) = meta.feature.as_ref() {
        let mut positions_by_id = vec![Vec::<egui::Pos2>::new(); feature_meta.dict.len()];
        let mut raw_indices_by_id = vec![Vec::<u32>::new(); feature_meta.dict.len()];
        for (raw_i, &feature_id) in feature_meta.ids.iter().enumerate() {
            let feature_i = feature_id as usize;
            let Some(bucket) = positions_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(raw_bucket) = raw_indices_by_id.get_mut(feature_i) else {
                continue;
            };
            let Some(&p) = positions_world.get(raw_i) else {
                continue;
            };
            bucket.push(p);
            raw_bucket.push(raw_i as u32);
        }

        positions_by_id
            .into_iter()
            .zip(raw_indices_by_id)
            .map(|(positions_world, raw_indices)| {
                if positions_world.is_empty() {
                    return None;
                }
                let values = vec![1.0f32; positions_world.len()];
                Some(Arc::new(SpatialFeatureCache {
                    positions_world: Arc::new(positions_world),
                    raw_indices: Arc::new(raw_indices),
                    values: Arc::new(values),
                }))
            })
            .collect()
    } else {
        Vec::new()
    };

    let bounds_world = bounds_of_points(positions_world.as_ref());
    let bins = PointIndexBins::build(positions_world.as_ref(), 256.0).map(Arc::new);

    Ok(PreparedSpatialPoints {
        raw_xy,
        meta,
        positions_world,
        values,
        lod_levels,
        feature_counts,
        feature_cache,
        bounds_world,
        bins,
        last_auto_choice,
        loaded_count: raw_len,
    })
}

pub(super) fn bounds_of_points(points: &[egui::Pos2]) -> Option<egui::Rect> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any = false;
    for p in points {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        any = true;
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    if !any {
        return None;
    }
    Some(egui::Rect::from_min_max(
        egui::pos2(min_x, min_y),
        egui::pos2(max_x, max_y),
    ))
}
