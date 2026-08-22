use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use ndarray::Array2;
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdScope {
    Visible,
    EntireImage,
}

impl ThresholdScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::EntireImage => "entire_image",
        }
    }

    pub(crate) fn layer_label(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::EntireImage => "full",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ControlThresholdPreviewResource {
    pub(crate) generation: u64,
    pub(crate) channel_index: usize,
    pub(crate) channel_name: String,
    pub(crate) scope: ThresholdScope,
    pub(crate) level: usize,
    pub(crate) downsample: f32,
    pub(crate) x0: u64,
    pub(crate) y0: u64,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) values: Arc<Vec<u16>>,
    pub(crate) included: Arc<Vec<bool>>,
    pub(crate) threshold: u16,
    pub(crate) min_component_pixels: usize,
}

impl ControlThresholdPreviewResource {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn channel_index(&self) -> usize {
        self.channel_index
    }

    pub fn channel_name(&self) -> &str {
        &self.channel_name
    }

    pub fn scope(&self) -> ThresholdScope {
        self.scope
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn downsample(&self) -> f32 {
        self.downsample
    }

    pub fn origin(&self) -> [u64; 2] {
        [self.x0, self.y0]
    }

    pub fn size(&self) -> [usize; 2] {
        [self.width, self.height]
    }

    pub fn values(&self) -> &Arc<Vec<u16>> {
        &self.values
    }

    pub fn included(&self) -> &Arc<Vec<bool>> {
        &self.included
    }

    pub fn threshold(&self) -> u16 {
        self.threshold
    }

    pub fn min_component_pixels(&self) -> usize {
        self.min_component_pixels
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "generation":self.generation,
            "channel_index":self.channel_index,
            "channel_name":self.channel_name,
            "scope":self.scope.as_str(),
            "level":self.level,
            "downsample":self.downsample,
            "extent":{"x0":self.x0,"y0":self.y0,"width":self.width,"height":self.height},
            "threshold":self.threshold,
            "source_min":self.values.iter().copied().min(),
            "source_max":self.values.iter().copied().max(),
            "min_component_pixels":self.min_component_pixels,
            "included_pixels":self.included.iter().filter(|included| **included).count(),
            "preview_engine":"cpu",
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ThresholdPreviewModel {
    pub(crate) scope: ThresholdScope,
    pub(crate) full_level: usize,
    pub(crate) min_component_pixels: usize,
    pub(crate) status: String,
    pub(crate) preview: Option<Arc<ControlThresholdPreviewResource>>,
    pub(crate) operation_generation: u64,
}

impl Default for ThresholdPreviewModel {
    fn default() -> Self {
        Self {
            scope: ThresholdScope::Visible,
            full_level: 0,
            min_component_pixels: 1,
            status: String::new(),
            preview: None,
            operation_generation: 0,
        }
    }
}

impl ThresholdPreviewModel {
    pub(crate) fn reset(&mut self, full_level: usize) {
        *self = Self {
            full_level,
            operation_generation: 1,
            ..Self::default()
        };
    }

    pub(crate) fn next_generation(&mut self) -> u64 {
        self.operation_generation = self.operation_generation.wrapping_add(1).max(1);
        self.operation_generation
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "active":self.preview.is_some(),
            "configured_scope":self.scope.as_str(),
            "configured_full_level":self.full_level,
            "configured_min_component_pixels":self.min_component_pixels,
            "status":self.status,
            "preview":self.preview.as_ref().map(|preview| preview.snapshot()),
        })
    }
}

pub(crate) struct ThresholdMask {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) included: Vec<bool>,
}

pub(crate) fn extract_threshold_mask(
    plane: &Array2<u16>,
    threshold: u16,
    min_component_pixels: usize,
) -> ThresholdMask {
    let (height, width) = plane.dim();
    let selected = plane
        .iter()
        .map(|value| *value >= threshold)
        .collect::<Vec<_>>();
    let mut visited = vec![false; selected.len()];
    let mut included = vec![false; selected.len()];
    for start in 0..selected.len() {
        if visited[start] || !selected[start] {
            continue;
        }
        let pixels = component_pixels(start, width, height, &selected, &mut visited);
        if pixels.len() >= min_component_pixels.max(1) {
            for index in pixels {
                included[index] = true;
            }
        }
    }
    ThresholdMask {
        width,
        height,
        included,
    }
}

pub(crate) fn threshold_mask_polygons(mask: &ThresholdMask) -> Vec<Vec<[f32; 2]>> {
    if mask.width == 0 || mask.height == 0 || mask.included.is_empty() {
        return Vec::new();
    }
    let mut visited = vec![false; mask.included.len()];
    let mut polygons = Vec::new();
    for start in 0..mask.included.len() {
        if visited[start] || !mask.included[start] {
            continue;
        }
        let pixels = component_pixels(start, mask.width, mask.height, &mask.included, &mut visited);
        polygons.extend(component_polygons(mask.width, &pixels));
    }
    polygons
}

fn component_pixels(
    start: usize,
    width: usize,
    height: usize,
    mask: &[bool],
    visited: &mut [bool],
) -> Vec<usize> {
    let mut queue = VecDeque::from([start]);
    let mut pixels = Vec::new();
    visited[start] = true;
    while let Some(index) = queue.pop_front() {
        pixels.push(index);
        let x = index % width;
        let y = index / width;
        let neighbors = [
            x.checked_sub(1).map(|_| index - 1),
            (x + 1 < width).then_some(index + 1),
            y.checked_sub(1).map(|_| index - width),
            (y + 1 < height).then_some(index + width),
        ];
        for next in neighbors.into_iter().flatten() {
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

fn component_polygons(width: usize, pixels: &[usize]) -> Vec<Vec<[f32; 2]>> {
    let pixel_set = pixels.iter().copied().collect::<HashSet<_>>();
    let mut edges = HashSet::<GridEdge>::new();
    let mut outgoing = HashMap::<GridPoint, Vec<GridPoint>>::new();
    for &index in pixels {
        let x = (index % width) as i32;
        let y = (index / width) as i32;
        if y == 0 || !pixel_set.contains(&index.saturating_sub(width)) {
            insert_edge(&mut edges, &mut outgoing, (x, y), (x + 1, y));
        }
        if x as usize + 1 == width || !pixel_set.contains(&(index + 1)) {
            insert_edge(&mut edges, &mut outgoing, (x + 1, y), (x + 1, y + 1));
        }
        if !pixel_set.contains(&(index + width)) {
            insert_edge(&mut edges, &mut outgoing, (x + 1, y + 1), (x, y + 1));
        }
        if x == 0 || !pixel_set.contains(&index.saturating_sub(1)) {
            insert_edge(&mut edges, &mut outgoing, (x, y + 1), (x, y));
        }
    }
    for targets in outgoing.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }
    let mut polygons = Vec::new();
    while let Some(&(start, first)) = edges.iter().min() {
        let mut vertices = vec![start];
        let mut previous = start;
        let mut cursor = first;
        edges.remove(&(start, first));
        loop {
            if cursor == start {
                break;
            }
            vertices.push(cursor);
            let Some(next) = next_edge(previous, cursor, &outgoing, &edges) else {
                vertices.clear();
                break;
            };
            edges.remove(&(cursor, next));
            previous = cursor;
            cursor = next;
        }
        let polygon = simplify_polygon(
            &vertices
                .into_iter()
                .map(|(x, y)| [x as f32, y as f32])
                .collect::<Vec<_>>(),
        );
        if polygon.len() >= 3 {
            polygons.push(polygon);
        }
    }
    polygons
}

fn insert_edge(
    edges: &mut HashSet<GridEdge>,
    outgoing: &mut HashMap<GridPoint, Vec<GridPoint>>,
    from: GridPoint,
    to: GridPoint,
) {
    if edges.insert((from, to)) {
        outgoing.entry(from).or_default().push(to);
    }
}

fn next_edge(
    previous: GridPoint,
    cursor: GridPoint,
    outgoing: &HashMap<GridPoint, Vec<GridPoint>>,
    edges: &HashSet<GridEdge>,
) -> Option<GridPoint> {
    let incoming = edge_direction(previous, cursor)?;
    outgoing
        .get(&cursor)?
        .iter()
        .copied()
        .filter(|candidate| edges.contains(&(cursor, *candidate)))
        .min_by_key(|candidate| {
            let outgoing = edge_direction(cursor, *candidate).unwrap_or(incoming);
            let priority = match (outgoing + 4 - incoming) % 4 {
                3 => 0,
                0 => 1,
                1 => 2,
                _ => 3,
            };
            (priority, *candidate)
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

fn simplify_polygon(vertices: &[[f32; 2]]) -> Vec<[f32; 2]> {
    if vertices.len() < 3 {
        return vertices.to_vec();
    }
    (0..vertices.len())
        .filter_map(|index| {
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let current = vertices[index];
            let next = vertices[(index + 1) % vertices.len()];
            let first = [current[0] - previous[0], current[1] - previous[1]];
            let second = [next[0] - current[0], next[1] - current[1]];
            ((first[0] * second[1] - first[1] * second[0]).abs() > 1e-6).then_some(current)
        })
        .collect()
}
