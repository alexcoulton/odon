//! Pure rectangle, polygon, and segment selection geometry.

use super::*;

pub(in crate::objects) fn point_in_any_polygon(
    p: egui::Pos2,
    polygons: &[Vec<egui::Pos2>],
) -> bool {
    polygons.iter().any(|poly| point_in_polygon(p, poly))
}

pub(in crate::objects) fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let dy = pj.y - pi.y;
        let intersects = ((pi.y > p.y) != (pj.y > p.y))
            && dy.abs() > 1e-12
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / dy + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub(in crate::objects) fn object_intersects_rect_for_selection(
    object: &ObjectFeature,
    rect: egui::Rect,
) -> bool {
    if !object.polygons_world.is_empty() {
        if !rects_intersect_inclusive(object.bbox_world, rect) {
            return false;
        }
        let mut tested_polygon = false;
        for poly in &object.polygons_world {
            if poly.len() < 3 {
                continue;
            }
            tested_polygon = true;
            if polygon_intersects_rect(poly, rect) {
                return true;
            }
        }
        if tested_polygon {
            return false;
        }
    }

    let point = object.point_position_world.unwrap_or(object.centroid_world);
    rect_contains_point_inclusive(rect, point)
}

pub(in crate::objects) fn polygon_intersects_rect(poly: &[egui::Pos2], rect: egui::Rect) -> bool {
    if poly
        .iter()
        .any(|point| rect_contains_point_inclusive(rect, *point))
    {
        return true;
    }

    if rect_corners(rect)
        .iter()
        .any(|corner| point_in_polygon_or_on_edge(*corner, poly))
    {
        return true;
    }

    let edges = rect_edges(rect);
    for (a, b) in polygon_edges(poly) {
        for (c, d) in edges {
            if segments_intersect_inclusive(a, b, c, d) {
                return true;
            }
        }
    }

    false
}

pub(in crate::objects) fn polygon_edges(poly: &[egui::Pos2]) -> Vec<(egui::Pos2, egui::Pos2)> {
    if poly.len() < 2 {
        return Vec::new();
    }
    let mut edges = poly
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    if poly.len() >= 3 && poly.first() != poly.last() {
        edges.push((*poly.last().unwrap(), poly[0]));
    }
    edges
}

pub(in crate::objects) fn point_in_polygon_or_on_edge(
    point: egui::Pos2,
    poly: &[egui::Pos2],
) -> bool {
    polygon_edges(poly)
        .iter()
        .any(|(a, b)| point_on_segment_inclusive(point, *a, *b))
        || point_in_polygon(point, poly)
}

pub(in crate::objects) fn rect_corners(rect: egui::Rect) -> [egui::Pos2; 4] {
    [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ]
}

pub(in crate::objects) fn rect_edges(rect: egui::Rect) -> [(egui::Pos2, egui::Pos2); 4] {
    let [lt, rt, rb, lb] = rect_corners(rect);
    [(lt, rt), (rt, rb), (rb, lb), (lb, lt)]
}

pub(in crate::objects) fn rects_intersect_inclusive(a: egui::Rect, b: egui::Rect) -> bool {
    a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y
}

pub(in crate::objects) fn segments_intersect_inclusive(
    a: egui::Pos2,
    b: egui::Pos2,
    c: egui::Pos2,
    d: egui::Pos2,
) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);

    if o1.abs() <= 1e-5 && point_on_segment_inclusive(c, a, b) {
        return true;
    }
    if o2.abs() <= 1e-5 && point_on_segment_inclusive(d, a, b) {
        return true;
    }
    if o3.abs() <= 1e-5 && point_on_segment_inclusive(a, c, d) {
        return true;
    }
    if o4.abs() <= 1e-5 && point_on_segment_inclusive(b, c, d) {
        return true;
    }

    ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0))
        && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))
}

pub(in crate::objects) fn point_on_segment_inclusive(
    point: egui::Pos2,
    a: egui::Pos2,
    b: egui::Pos2,
) -> bool {
    orient(a, b, point).abs() <= 1e-5
        && point.x >= a.x.min(b.x) - 1e-5
        && point.x <= a.x.max(b.x) + 1e-5
        && point.y >= a.y.min(b.y) - 1e-5
        && point.y <= a.y.max(b.y) + 1e-5
}

pub(in crate::objects) fn orient(a: egui::Pos2, b: egui::Pos2, c: egui::Pos2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

pub(in crate::objects) fn rect_contains_point_inclusive(
    rect: egui::Rect,
    point: egui::Pos2,
) -> bool {
    point.x >= rect.min.x && point.x <= rect.max.x && point.y >= rect.min.y && point.y <= rect.max.y
}
