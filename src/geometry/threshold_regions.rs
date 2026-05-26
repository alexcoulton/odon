use std::collections::{HashMap, HashSet, VecDeque};

use eframe::egui;
use ndarray::Array2;

pub struct ThresholdRegionMask {
    pub width: usize,
    pub height: usize,
    pub included: Vec<bool>,
}

pub fn extract_threshold_region_mask(
    plane: &Array2<u16>,
    threshold: u16,
    min_component_pixels: usize,
) -> ThresholdRegionMask {
    let (height, width) = plane.dim();
    if width == 0 || height == 0 {
        return ThresholdRegionMask {
            width,
            height,
            included: Vec::new(),
        };
    }

    let mask = plane
        .iter()
        .map(|value| *value >= threshold)
        .collect::<Vec<_>>();
    let mut visited = vec![false; width * height];
    let mut included = vec![false; width * height];

    for start in 0..mask.len() {
        if visited[start] || !mask[start] {
            continue;
        }
        let pixels = component_pixels(start, width, height, &mask, &mut visited);
        if pixels.len() < min_component_pixels.max(1) {
            continue;
        }
        for idx in pixels {
            included[idx] = true;
        }
    }

    ThresholdRegionMask {
        width,
        height,
        included,
    }
}

pub fn threshold_region_mask_to_polygons(mask: &ThresholdRegionMask) -> Vec<Vec<egui::Pos2>> {
    if mask.width == 0 || mask.height == 0 || mask.included.is_empty() {
        return Vec::new();
    }

    let mut visited = vec![false; mask.included.len()];
    let mut out = Vec::new();
    for start in 0..mask.included.len() {
        if visited[start] || !mask.included[start] {
            continue;
        }
        let pixels = component_pixels(start, mask.width, mask.height, &mask.included, &mut visited);
        out.extend(component_to_polygons(mask.width, &pixels));
    }
    out
}

fn component_pixels(
    start: usize,
    width: usize,
    height: usize,
    mask: &[bool],
    visited: &mut [bool],
) -> Vec<usize> {
    let mut queue = VecDeque::new();
    let mut pixels = Vec::new();
    visited[start] = true;
    queue.push_back(start);

    while let Some(idx) = queue.pop_front() {
        pixels.push(idx);
        let x = idx % width;
        let y = idx / width;

        if x > 0 {
            let next = idx - 1;
            if mask[next] && !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
        if x + 1 < width {
            let next = idx + 1;
            if mask[next] && !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
        if y > 0 {
            let next = idx - width;
            if mask[next] && !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
        if y + 1 < height {
            let next = idx + width;
            if mask[next] && !visited[next] {
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }

    pixels
}

type GridPoint = (i32, i32);
type GridEdge = (GridPoint, GridPoint);

fn component_to_polygons(width: usize, pixels: &[usize]) -> Vec<Vec<egui::Pos2>> {
    let pixel_set = pixels.iter().copied().collect::<HashSet<_>>();
    let mut edges = HashSet::<GridEdge>::new();
    let mut outgoing = HashMap::<GridPoint, Vec<GridPoint>>::new();

    for &idx in pixels {
        let x = (idx % width) as i32;
        let y = (idx / width) as i32;

        if !pixel_set.contains(&(idx.saturating_sub(width))) || y == 0 {
            insert_boundary_edge(&mut edges, &mut outgoing, (x, y), (x + 1, y));
        }
        if !pixel_set.contains(&(idx + 1)) || (x as usize + 1 == width) {
            insert_boundary_edge(&mut edges, &mut outgoing, (x + 1, y), (x + 1, y + 1));
        }
        if !pixel_set.contains(&(idx + width)) {
            insert_boundary_edge(&mut edges, &mut outgoing, (x + 1, y + 1), (x, y + 1));
        }
        if !pixel_set.contains(&(idx.saturating_sub(1))) || x == 0 {
            insert_boundary_edge(&mut edges, &mut outgoing, (x, y + 1), (x, y));
        }
    }
    for targets in outgoing.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }

    let mut polygons = Vec::new();
    while let Some(&(start, first_next)) = edges.iter().min() {
        let mut loop_vertices = Vec::<GridPoint>::new();
        let mut prev = start;
        let mut cursor = first_next;
        edges.remove(&(start, first_next));
        loop_vertices.push(start);

        loop {
            if cursor == start {
                break;
            }
            loop_vertices.push(cursor);
            let Some(next) = choose_next_boundary_edge(prev, cursor, &outgoing, &edges) else {
                loop_vertices.clear();
                break;
            };
            edges.remove(&(cursor, next));
            prev = cursor;
            cursor = next;
        }
        if loop_vertices.is_empty() {
            continue;
        }

        let polygon = simplify_collinear_vertices(
            &loop_vertices
                .into_iter()
                .map(|(x, y)| egui::pos2(x as f32, y as f32))
                .collect::<Vec<_>>(),
        );
        if polygon.len() >= 3 {
            polygons.push(polygon);
        }
    }

    polygons
}

fn insert_boundary_edge(
    edges: &mut HashSet<GridEdge>,
    outgoing: &mut HashMap<GridPoint, Vec<GridPoint>>,
    from: GridPoint,
    to: GridPoint,
) {
    if edges.insert((from, to)) {
        outgoing.entry(from).or_default().push(to);
    }
}

fn choose_next_boundary_edge(
    previous: GridPoint,
    cursor: GridPoint,
    outgoing: &HashMap<GridPoint, Vec<GridPoint>>,
    unused_edges: &HashSet<GridEdge>,
) -> Option<GridPoint> {
    let incoming_dir = edge_direction(previous, cursor)?;
    outgoing
        .get(&cursor)?
        .iter()
        .copied()
        .filter(|&candidate| unused_edges.contains(&(cursor, candidate)))
        .min_by_key(|&candidate| {
            let outgoing_dir = edge_direction(cursor, candidate).unwrap_or(incoming_dir);
            (turn_priority(incoming_dir, outgoing_dir), candidate)
        })
}

fn edge_direction(from: GridPoint, to: GridPoint) -> Option<u8> {
    match (to.0 - from.0, to.1 - from.1) {
        (1, 0) => Some(0),
        (0, 1) => Some(1),
        (-1, 0) => Some(2),
        (0, -1) => Some(3),
        _ => None,
    }
}

fn turn_priority(incoming_dir: u8, outgoing_dir: u8) -> u8 {
    match (outgoing_dir + 4 - incoming_dir) % 4 {
        3 => 0, // left turn: separates contours that touch at a single grid vertex.
        0 => 1, // straight
        1 => 2, // right turn
        _ => 3, // reverse
    }
}

fn simplify_collinear_vertices(vertices: &[egui::Pos2]) -> Vec<egui::Pos2> {
    if vertices.len() < 3 {
        return vertices.to_vec();
    }
    let mut out = Vec::new();
    for i in 0..vertices.len() {
        let prev = vertices[(i + vertices.len() - 1) % vertices.len()];
        let cur = vertices[i];
        let next = vertices[(i + 1) % vertices.len()];
        let v1 = cur - prev;
        let v2 = next - cur;
        let cross = v1.x * v2.y - v1.y * v2.x;
        if cross.abs() > 1e-6 {
            out.push(cur);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_from_rows(rows: &[&str]) -> ThresholdRegionMask {
        let height = rows.len();
        let width = rows.first().map(|row| row.len()).unwrap_or(0);
        let included = rows
            .iter()
            .flat_map(|row| {
                assert_eq!(row.len(), width);
                row.chars().map(|ch| ch == '#').collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        ThresholdRegionMask {
            width,
            height,
            included,
        }
    }

    fn assert_axis_aligned_polygon_edges(polygons: &[Vec<egui::Pos2>]) {
        for polygon in polygons {
            assert!(polygon.len() >= 3);
            for idx in 0..polygon.len() {
                let a = polygon[idx];
                let b = polygon[(idx + 1) % polygon.len()];
                assert!(
                    (a.x - b.x).abs() < 1e-6 || (a.y - b.y).abs() < 1e-6,
                    "non-axis-aligned edge from {a:?} to {b:?} in {polygon:?}"
                );
            }
        }
    }

    #[test]
    fn polygonizes_self_touching_notch_without_diagonal_closure() {
        let mask = mask_from_rows(&["###.", "#.#.", ".##.", "...."]);
        let polygons = threshold_region_mask_to_polygons(&mask);

        assert_eq!(polygons.len(), 2);
        assert_axis_aligned_polygon_edges(&polygons);
    }

    #[test]
    fn polygonizes_single_pixel_hole_without_open_rings() {
        let mask = mask_from_rows(&["###", "#.#", "###"]);
        let polygons = threshold_region_mask_to_polygons(&mask);

        assert_eq!(polygons.len(), 2);
        assert_axis_aligned_polygon_edges(&polygons);
    }
}
