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

fn component_to_polygons(width: usize, pixels: &[usize]) -> Vec<Vec<egui::Pos2>> {
    let pixel_set = pixels.iter().copied().collect::<HashSet<_>>();
    let mut edges = HashMap::<(i32, i32), (i32, i32)>::new();

    for &idx in pixels {
        let x = (idx % width) as i32;
        let y = (idx / width) as i32;

        if !pixel_set.contains(&(idx.saturating_sub(width))) || y == 0 {
            edges.insert((x, y), (x + 1, y));
        }
        if !pixel_set.contains(&(idx + 1)) || (x as usize + 1 == width) {
            edges.insert((x + 1, y), (x + 1, y + 1));
        }
        if !pixel_set.contains(&(idx + width)) {
            edges.insert((x + 1, y + 1), (x, y + 1));
        }
        if !pixel_set.contains(&(idx.saturating_sub(1))) || x == 0 {
            edges.insert((x, y + 1), (x, y));
        }
    }

    let mut polygons = Vec::new();
    while let Some((&start, _)) = edges.iter().next() {
        let mut loop_vertices = Vec::<(i32, i32)>::new();
        let mut cursor = start;
        loop {
            loop_vertices.push(cursor);
            let Some(next) = edges.remove(&cursor) else {
                break;
            };
            cursor = next;
            if cursor == start {
                break;
            }
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
