//! Fill meshes, render cache identifiers, and geometry summaries.

use super::*;

pub(in crate::objects) fn visible_selected_cache_generation(
    selection_generation: u64,
    indices: &[usize],
) -> u64 {
    let mut hasher = DefaultHasher::new();
    selection_generation.hash(&mut hasher);
    indices.hash(&mut hasher);
    hasher.finish()
}

pub(in crate::objects) fn object_render_cache_id(namespace: u32, index: u64) -> u64 {
    ((namespace as u64) << 32) | (index & 0xffff_ffff)
}

pub(in crate::objects) fn object_render_cache_id_usize(namespace: u32, index: usize) -> u64 {
    object_render_cache_id(namespace, index.min(u32::MAX as usize) as u64)
}

pub(in crate::objects) fn object_property_render_cache_id(
    namespace: u32,
    property_key: &str,
    index: usize,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    namespace.hash(&mut hasher);
    property_key.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

impl ObjectFillMesh {
    pub(in crate::objects) fn bin_range_for_local_rect(
        &self,
        rect: egui::Rect,
    ) -> (usize, usize, usize, usize) {
        rect_bins(rect, self.origin, self.bin_size, self.bins_w, self.bins_h)
    }
}

pub(in crate::objects) fn build_selection_fill_mesh(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<SelectionFillMesh> {
    let mut tess = FillTessellator::new();
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut bounds: Option<egui::Rect> = None;

    for obj in objects {
        bounds = Some(match bounds {
            Some(acc) => acc.union(obj.bbox_world),
            None => obj.bbox_world,
        });

        for poly in &obj.polygons_world {
            let Some(clean) = cleaned_fill_polygon(poly) else {
                continue;
            };
            let mut builder = Path::builder();
            let first = clean[0];
            builder.begin(point(first.x, first.y));
            for p in &clean[1..] {
                builder.line_to(point(p.x, p.y));
            }
            builder.close();
            let path = builder.build();
            tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| {
                    let pos = vertex.position();
                    [pos.x, pos.y]
                }),
            )?;
        }
    }

    let bounds_local = bounds.context("no selection fill bounds")?;
    if geometry.indices.is_empty() {
        anyhow::bail!("no valid triangles for selection fill");
    }

    let mut triangles = Vec::with_capacity(geometry.indices.len());
    for idx in geometry.indices {
        let vertex = geometry
            .vertices
            .get(idx as usize)
            .copied()
            .context("selection fill index out of range")?;
        triangles.push(vertex);
    }

    Ok(SelectionFillMesh {
        vertices_local: Arc::new(triangles),
        bounds_local,
    })
}

pub(in crate::objects) fn build_object_fill_mesh(
    objects: &[GeoJsonObjectFeature],
) -> anyhow::Result<ObjectFillMesh> {
    let mut tess = FillTessellator::new();
    let mut triangles = Vec::new();
    let mut bounds: Option<egui::Rect> = None;

    for (object_index, obj) in objects.iter().enumerate() {
        bounds = Some(match bounds {
            Some(acc) => acc.union(obj.bbox_world),
            None => obj.bbox_world,
        });

        let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        for poly in &obj.polygons_world {
            let Some(clean) = cleaned_fill_polygon(poly) else {
                continue;
            };
            let mut builder = Path::builder();
            let first = clean[0];
            builder.begin(point(first.x, first.y));
            for p in &clean[1..] {
                builder.line_to(point(p.x, p.y));
            }
            builder.close();
            let path = builder.build();
            tess.tessellate_path(
                &path,
                &FillOptions::default(),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| {
                    let pos = vertex.position();
                    [pos.x, pos.y]
                }),
            )?;
        }

        for idx in geometry.indices {
            let vertex = geometry
                .vertices
                .get(idx as usize)
                .copied()
                .context("object fill index out of range")?;
            triangles.push([vertex[0], vertex[1], object_index as f32]);
        }
    }

    let bounds_local = bounds.context("no object fill bounds")?;
    if triangles.is_empty() {
        anyhow::bail!("no valid triangles for object fill rendering");
    }

    const OBJECT_FILL_BIN_SIZE: f32 = 2048.0;
    let w = bounds_local.width().max(1.0);
    let h = bounds_local.height().max(1.0);
    let bins_w = ((w / OBJECT_FILL_BIN_SIZE).ceil() as usize).max(1);
    let bins_h = ((h / OBJECT_FILL_BIN_SIZE).ceil() as usize).max(1);
    let origin = bounds_local.min;
    let bins_len = bins_w.saturating_mul(bins_h);
    let mut tmp_bins: Vec<Vec<[f32; 3]>> = vec![Vec::new(); bins_len];
    for tri in triangles.chunks_exact(3) {
        let min_x = tri.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
        let min_y = tri.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min);
        let max_x = tri.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
        let max_y = tri.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
        if !(min_x.is_finite() && min_y.is_finite() && max_x.is_finite() && max_y.is_finite()) {
            continue;
        }
        let tri_rect = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
        let (bx0, by0, bx1, by1) =
            rect_bins(tri_rect, origin, OBJECT_FILL_BIN_SIZE, bins_w, bins_h);
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let bin_index = by * bins_w + bx;
                if let Some(bin) = tmp_bins.get_mut(bin_index) {
                    bin.extend_from_slice(tri);
                }
            }
        }
    }

    Ok(ObjectFillMesh {
        vertices_local: Arc::new(triangles),
        bounds_local,
        object_count: objects.len(),
        origin,
        bin_size: OBJECT_FILL_BIN_SIZE,
        bins_w,
        bins_h,
        bin_vertices: tmp_bins.into_iter().map(Arc::new).collect(),
    })
}

pub(in crate::objects) fn cleaned_fill_polygon(poly: &[egui::Pos2]) -> Option<Vec<egui::Pos2>> {
    if poly.len() < 3 {
        return None;
    }
    let mut out = Vec::with_capacity(poly.len());
    for &p in poly {
        if !(p.x.is_finite() && p.y.is_finite()) {
            continue;
        }
        if out.last().copied() == Some(p) {
            continue;
        }
        out.push(p);
    }
    if out.len() >= 2 && out.first() == out.last() {
        out.pop();
    }
    if out.len() < 3 {
        return None;
    }
    let area = polygon_signed_area_local(&out).abs();
    (area > 1e-3).then_some(out)
}

pub(in crate::objects) fn polygon_signed_area_local(points: &[egui::Pos2]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

pub(in crate::objects) fn object_vertex_count(object: &GeoJsonObjectFeature) -> usize {
    object.polygons_world.iter().map(|poly| poly.len()).sum()
}

pub(in crate::objects) fn simplified_polyline_screen_points(
    poly: &[egui::Pos2],
    max_points: usize,
    camera: &crate::camera::Camera,
    local_to_world_offset: egui::Vec2,
    display_transform: SpatialDataTransform2,
    viewport: egui::Rect,
) -> Vec<egui::Pos2> {
    if poly.len() < 2 {
        return Vec::new();
    }

    let step = if max_points == usize::MAX || poly.len() <= max_points {
        1
    } else {
        poly.len().div_ceil(max_points)
    };

    let mut pts = Vec::with_capacity(poly.len().div_ceil(step).saturating_add(1));
    for point in poly.iter().step_by(step).copied() {
        let world = egui::pos2(
            point.x * display_transform.scale[0].max(1e-6)
                + display_transform.translation[0]
                + local_to_world_offset.x,
            point.y * display_transform.scale[1].max(1e-6)
                + display_transform.translation[1]
                + local_to_world_offset.y,
        );
        pts.push(camera.world_to_screen(world, viewport));
    }
    if let Some(last) = poly.last().copied() {
        let last_world = egui::pos2(
            last.x * display_transform.scale[0].max(1e-6)
                + display_transform.translation[0]
                + local_to_world_offset.x,
            last.y * display_transform.scale[1].max(1e-6)
                + display_transform.translation[1]
                + local_to_world_offset.y,
        );
        let last_screen = camera.world_to_screen(last_world, viewport);
        if pts.last().copied() != Some(last_screen) {
            pts.push(last_screen);
        }
    }
    pts
}

pub(in crate::objects) fn summarize_geometry(
    polygons: &[Vec<egui::Pos2>],
) -> Option<(egui::Rect, f32, f32, egui::Pos2)> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut area_sum = 0.0f32;
    let mut perimeter_sum = 0.0f32;
    let mut centroid_num = egui::Vec2::ZERO;
    let mut any = false;

    for poly in polygons {
        if poly.len() < 4 {
            continue;
        }
        for p in poly {
            if p.x.is_finite() && p.y.is_finite() {
                any = true;
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
            }
        }
        for win in poly.windows(2) {
            let a = win[0];
            let b = win[1];
            perimeter_sum += (b - a).length();
        }
        if let Some((area, centroid)) = polygon_area_and_centroid(poly) {
            area_sum += area;
            centroid_num += centroid.to_vec2() * area;
        }
    }

    if !any {
        return None;
    }
    let bbox = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
    let centroid = if area_sum > 1e-6 {
        (centroid_num / area_sum).to_pos2()
    } else {
        bbox.center()
    };
    Some((bbox, area_sum.max(0.0), perimeter_sum.max(0.0), centroid))
}

pub(in crate::objects) fn polygon_area_and_centroid(
    poly: &[egui::Pos2],
) -> Option<(f32, egui::Pos2)> {
    if poly.len() < 4 {
        return None;
    }
    let mut cross_sum = 0.0f32;
    let mut cx_sum = 0.0f32;
    let mut cy_sum = 0.0f32;

    for win in poly.windows(2) {
        let a = win[0];
        let b = win[1];
        let cross = a.x * b.y - b.x * a.y;
        cross_sum += cross;
        cx_sum += (a.x + b.x) * cross;
        cy_sum += (a.y + b.y) * cross;
    }
    let area_signed = cross_sum * 0.5;
    let area = area_signed.abs();
    if area <= 1e-6 {
        return None;
    }
    let denom = 3.0 * cross_sum;
    if denom.abs() <= 1e-6 {
        return None;
    }
    Some((area, egui::pos2(cx_sum / denom, cy_sum / denom)))
}
