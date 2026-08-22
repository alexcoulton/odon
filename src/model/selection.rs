use std::collections::{BTreeSet, HashSet};

use serde_json::{Value, json};

use crate::control::{ControlError, ControlErrorKind};

use super::{ControlObjectFeature, ControlObjectResource};

#[derive(Debug, Clone)]
pub(crate) struct ObjectSelectionModel {
    selected_indices: BTreeSet<usize>,
    primary_index: Option<usize>,
    generation: u64,
}

impl Default for ObjectSelectionModel {
    fn default() -> Self {
        Self {
            selected_indices: BTreeSet::new(),
            primary_index: None,
            generation: 1,
        }
    }
}

impl ObjectSelectionModel {
    pub(crate) fn reset(&mut self) {
        self.selected_indices.clear();
        self.primary_index = None;
        self.generation = self.generation.wrapping_add(1).max(1);
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn selected_indices(&self) -> HashSet<usize> {
        self.selected_indices.iter().copied().collect()
    }

    pub(crate) fn projection_json(&self) -> Value {
        json!({
            "generation": self.generation,
            "selected_indices": self.selected_indices.iter().copied().collect::<Vec<_>>(),
            "primary_index": self.primary_index,
        })
    }

    pub(crate) fn restore_projection(
        &mut self,
        projection: &Value,
        object_count: usize,
    ) -> Result<(), ControlError> {
        let selected_indices = parse_indices(projection, "selected_indices", object_count)?;
        let primary_index = optional_index(projection, "primary_index")?;
        if primary_index.is_some_and(|index| index >= object_count) {
            return Err(invalid(
                "renderer object selection primary_index is out of range",
            ));
        }
        self.selected_indices = selected_indices.into_iter().collect();
        self.primary_index = primary_index.filter(|index| self.selected_indices.contains(index));
        self.generation = projection
            .get("generation")
            .and_then(Value::as_u64)
            .unwrap_or(self.generation)
            .max(1);
        Ok(())
    }

    pub(crate) fn snapshot(&self, resource: Option<&ControlObjectResource>, limit: usize) -> Value {
        let Some(resource) = resource else {
            return json!({
                "object_count":0,
                "selection_count":0,
                "primary":Value::Null,
                "selected":[],
                "truncated":false,
                "generation":self.generation,
            });
        };
        let selected = self
            .selected_indices
            .iter()
            .take(limit)
            .filter_map(|index| object_entry(resource, *index))
            .collect::<Vec<_>>();
        json!({
            "object_count": resource.features.len(),
            "selection_count": self.selected_indices.len(),
            "primary": self.primary_index.and_then(|index| object_entry(resource, index)),
            "selected": selected,
            "truncated": self.selected_indices.len() > limit,
            "generation": self.generation,
        })
    }

    pub(crate) fn clear(
        &mut self,
        resource: Option<&ControlObjectResource>,
        limit: usize,
    ) -> Value {
        let changed = !self.selected_indices.is_empty() || self.primary_index.is_some();
        if changed {
            self.selected_indices.clear();
            self.primary_index = None;
            self.bump();
        }
        json!({
            "target":"segmentation_objects",
            "changed":changed,
            "selection":self.snapshot(resource, limit),
        })
    }

    pub(crate) fn select_ids(
        &mut self,
        resource: &ControlObjectResource,
        params: &Value,
        limit: usize,
    ) -> Result<Value, ControlError> {
        let values = params
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("ids is required"))?;
        let ids = values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| invalid("ids must contain unique strings"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids.iter().collect::<HashSet<_>>().len() != ids.len() {
            return Err(invalid("ids must contain unique strings"));
        }
        let mut indices = Vec::new();
        let mut missing = Vec::new();
        for id in ids {
            let matches = resource
                .features
                .iter()
                .enumerate()
                .filter_map(|(index, feature)| feature_matches_id(feature, &id).then_some(index))
                .collect::<Vec<_>>();
            if matches.is_empty() {
                missing.push(id);
            } else {
                indices.extend(matches);
            }
        }
        indices.sort_unstable();
        indices.dedup();
        let changed = self.apply_indices(&indices, selection_mode(params)?)?;
        Ok(json!({
            "target":"segmentation_objects",
            "changed":changed,
            "matched_count":indices.len(),
            "missing_ids":missing,
            "selection":self.snapshot(Some(resource), limit),
        }))
    }

    pub(crate) fn select_filtered(
        &mut self,
        resource: &ControlObjectResource,
        visible_indices: Option<&[usize]>,
        filter_revision: u64,
        params: &Value,
        limit: usize,
    ) -> Result<Value, ControlError> {
        let indices = visible_indices
            .map(<[usize]>::to_vec)
            .unwrap_or_else(|| (0..resource.features.len()).collect());
        let changed = self.apply_indices(&indices, selection_mode(params)?)?;
        Ok(json!({
            "target":"segmentation_objects",
            "changed":changed,
            "matched_count":indices.len(),
            "filter_revision":filter_revision,
            "selection":self.snapshot(Some(resource), limit),
        }))
    }

    pub(crate) fn focus(
        &mut self,
        resource: &ControlObjectResource,
        params: &Value,
    ) -> Result<(Value, Option<[f32; 4]>), ControlError> {
        let index = if let Some(index) = optional_index(params, "index")? {
            if index >= resource.features.len() {
                return Err(not_found(format!("object index {index} is out of range")));
            }
            index
        } else if let Some(id) = params.get("id").and_then(Value::as_str) {
            let matches = resource
                .features
                .iter()
                .enumerate()
                .filter_map(|(index, feature)| feature_matches_id(feature, id).then_some(index))
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [index] => *index,
                [] => return Err(not_found(format!("object '{id}' was not found"))),
                _ => return Err(invalid(format!("object '{id}' is ambiguous"))),
            }
        } else {
            return Err(invalid("id or index is required"));
        };
        let before = (self.selected_indices.clone(), self.primary_index);
        self.selected_indices.insert(index);
        self.primary_index = Some(index);
        if before != (self.selected_indices.clone(), self.primary_index) {
            self.bump();
        }
        let fit = params.get("fit").and_then(Value::as_bool).unwrap_or(true);
        Ok((
            json!({
                "target":"segmentation_objects",
                "focused":object_entry(resource, index),
                "selection_count":self.selected_indices.len(),
                "generation":self.generation,
            }),
            fit.then_some(resource.features[index].bbox_world),
        ))
    }

    pub(crate) fn clear_focus(&mut self) -> Value {
        let changed = self.primary_index.take().is_some();
        if changed {
            self.bump();
        }
        json!({
            "target":"segmentation_objects",
            "changed":changed,
            "focused":Value::Null,
            "generation":self.generation,
        })
    }

    pub(crate) fn query_rect(
        &self,
        resource: &ControlObjectResource,
        rect: [f32; 4],
        visible_indices: Option<&[usize]>,
        limit: usize,
    ) -> Value {
        let indices = candidate_indices(resource.features.len(), visible_indices)
            .filter(|index| object_intersects_rect(&resource.features[*index], rect))
            .collect::<Vec<_>>();
        let matches = indices
            .iter()
            .take(limit)
            .filter_map(|index| object_entry(resource, *index))
            .collect::<Vec<_>>();
        json!({
            "world_rect":rect,
            "local_rect":rect,
            "match_count":indices.len(),
            "matches":matches,
            "truncated":indices.len() > limit,
        })
    }

    pub(crate) fn select_rect(
        &mut self,
        resource: &ControlObjectResource,
        rect: [f32; 4],
        visible_indices: Option<&[usize]>,
        params: &Value,
        limit: usize,
    ) -> Result<Value, ControlError> {
        let indices = candidate_indices(resource.features.len(), visible_indices)
            .filter(|index| object_intersects_rect(&resource.features[*index], rect))
            .collect::<Vec<_>>();
        let changed = self.apply_indices(&indices, selection_mode_with_additive(params)?)?;
        Ok(json!({
            "target":"segmentation_objects",
            "result":{
                "changed":changed,
                "query":self.query_rect(resource, rect, visible_indices, limit),
                "selection":self.snapshot(Some(resource), limit),
            },
        }))
    }

    pub(crate) fn query_lasso(
        &self,
        resource: &ControlObjectResource,
        points: &[[f32; 2]],
        visible_indices: Option<&[usize]>,
        limit: usize,
    ) -> Value {
        let indices = candidate_indices(resource.features.len(), visible_indices)
            .filter(|index| point_in_polygon(resource.features[*index].centroid_world, points))
            .collect::<Vec<_>>();
        let matches = indices
            .iter()
            .take(limit)
            .filter_map(|index| object_entry(resource, *index))
            .collect::<Vec<_>>();
        json!({
            "world_points":points,
            "match_count":indices.len(),
            "matches":matches,
            "truncated":indices.len() > limit,
        })
    }

    pub(crate) fn select_lasso(
        &mut self,
        resource: &ControlObjectResource,
        points: &[[f32; 2]],
        visible_indices: Option<&[usize]>,
        params: &Value,
        limit: usize,
    ) -> Result<Value, ControlError> {
        let indices = candidate_indices(resource.features.len(), visible_indices)
            .filter(|index| point_in_polygon(resource.features[*index].centroid_world, points))
            .collect::<Vec<_>>();
        let changed = self.apply_indices(&indices, selection_mode(params)?)?;
        Ok(json!({
            "changed":changed,
            "query":self.query_lasso(resource, points, visible_indices, limit),
            "selection":self.snapshot(Some(resource), limit),
        }))
    }

    pub(crate) fn replace_transaction(
        &mut self,
        resource: Option<&ControlObjectResource>,
        params: &Value,
        limit: usize,
    ) -> Result<Value, ControlError> {
        if let Some(expected) = params.get("expected_generation").and_then(Value::as_u64)
            && expected != self.generation
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!(
                    "object selection generation conflict: expected {expected}, current {}",
                    self.generation
                ),
            ));
        }
        let state = params.get("state").unwrap_or(params);
        let object_count = resource.map_or(0, |resource| resource.features.len());
        let indices = parse_indices(state, "selected_indices", object_count)?;
        let primary = optional_index(state, "primary_index")?;
        if primary.is_some_and(|index| !indices.contains(&index)) {
            return Err(invalid("primary_index must belong to selected_indices"));
        }
        let changed = self
            .selected_indices
            .iter()
            .copied()
            .ne(indices.iter().copied())
            || self.primary_index != primary;
        if changed {
            self.selected_indices = indices.into_iter().collect();
            self.primary_index = primary;
            self.bump();
        }
        Ok(json!({
            "target":"segmentation_objects",
            "changed":changed,
            "selection":self.snapshot(resource, limit),
        }))
    }

    fn apply_indices(&mut self, indices: &[usize], mode: &str) -> Result<bool, ControlError> {
        let before = (self.selected_indices.clone(), self.primary_index);
        match mode {
            "replace" => self.selected_indices = indices.iter().copied().collect(),
            "add" => self.selected_indices.extend(indices.iter().copied()),
            "remove" => {
                for index in indices {
                    self.selected_indices.remove(index);
                }
            }
            "toggle" => {
                for index in indices {
                    if !self.selected_indices.insert(*index) {
                        self.selected_indices.remove(index);
                    }
                }
            }
            _ => {
                return Err(invalid(
                    "selection mode must be replace, add, remove, or toggle",
                ));
            }
        }
        if self
            .primary_index
            .is_none_or(|index| !self.selected_indices.contains(&index))
        {
            self.primary_index = self.selected_indices.first().copied();
        }
        let changed = before != (self.selected_indices.clone(), self.primary_index);
        if changed {
            self.bump();
        }
        Ok(changed)
    }

    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1).max(1);
    }
}

fn object_entry(resource: &ControlObjectResource, index: usize) -> Option<Value> {
    let feature = resource.features.get(index)?;
    Some(json!({
        "index":index,
        "id":feature.id,
        "centroid_world":feature.centroid_world,
        "centroid_local":feature.centroid_world,
        "bbox_world":feature.bbox_world,
        "bbox_local":feature.bbox_world,
        "area_px":feature.area_px,
        "perimeter_px":feature.perimeter_px,
    }))
}

fn feature_matches_id(feature: &ControlObjectFeature, id: &str) -> bool {
    feature.id == id
        || ["cell_id", "id", "object_id", "label", "name"]
            .iter()
            .any(|key| {
                feature
                    .properties
                    .get(*key)
                    .and_then(value_label)
                    .is_some_and(|value| value == id)
            })
}

fn value_label(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn candidate_indices<'a>(
    object_count: usize,
    visible_indices: Option<&'a [usize]>,
) -> Box<dyn Iterator<Item = usize> + 'a> {
    match visible_indices {
        Some(indices) => Box::new(
            indices
                .iter()
                .copied()
                .filter(move |index| *index < object_count),
        ),
        None => Box::new(0..object_count),
    }
}

fn object_intersects_rect(feature: &ControlObjectFeature, rect: [f32; 4]) -> bool {
    let rect = normalize_rect(rect);
    if !feature.polygons_world.is_empty() {
        if !rects_intersect(feature.bbox_world, rect) {
            return false;
        }
        let mut tested_polygon = false;
        for polygon in feature.polygons_world.iter() {
            if polygon.len() < 3 {
                continue;
            }
            tested_polygon = true;
            if polygon_intersects_rect(polygon, rect) {
                return true;
            }
        }
        if tested_polygon {
            return false;
        }
    }
    point_in_rect(
        feature
            .point_position_world
            .unwrap_or(feature.centroid_world),
        rect,
    )
}

fn normalize_rect([x0, y0, x1, y1]: [f32; 4]) -> [f32; 4] {
    [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
}

fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut previous = polygon.len() - 1;
    for current in 0..polygon.len() {
        let a = polygon[current];
        let b = polygon[previous];
        let dy = b[1] - a[1];
        if ((a[1] > point[1]) != (b[1] > point[1]))
            && dy.abs() > 1.0e-12
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / dy + a[0]
        {
            inside = !inside;
        }
        previous = current;
    }
    inside
}

fn polygon_intersects_rect(polygon: &[[f32; 2]], rect: [f32; 4]) -> bool {
    if polygon
        .iter()
        .copied()
        .any(|point| point_in_rect(point, rect))
    {
        return true;
    }
    let corners = rect_corners(rect);
    if corners
        .iter()
        .copied()
        .any(|point| point_in_polygon_or_on_edge(point, polygon))
    {
        return true;
    }
    let rect_edges = polygon_edges(&corners);
    polygon_edges(polygon).into_iter().any(|(a, b)| {
        rect_edges
            .iter()
            .copied()
            .any(|(c, d)| segments_intersect(a, b, c, d))
    })
}

fn polygon_edges(points: &[[f32; 2]]) -> Vec<([f32; 2], [f32; 2])> {
    if points.len() < 2 {
        return Vec::new();
    }
    let mut edges = points
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<Vec<_>>();
    if points.len() >= 3 && points.first() != points.last() {
        edges.push((*points.last().expect("non-empty polygon"), points[0]));
    }
    edges
}

fn rect_corners([x0, y0, x1, y1]: [f32; 4]) -> [[f32; 2]; 4] {
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn rects_intersect(a: [f32; 4], b: [f32; 4]) -> bool {
    let [ax0, ay0, ax1, ay1] = normalize_rect(a);
    let [bx0, by0, bx1, by1] = normalize_rect(b);
    ax0 <= bx1 && ax1 >= bx0 && ay0 <= by1 && ay1 >= by0
}

fn point_in_rect([x, y]: [f32; 2], [x0, y0, x1, y1]: [f32; 4]) -> bool {
    x >= x0 && x <= x1 && y >= y0 && y <= y1
}

fn point_in_polygon_or_on_edge(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    polygon_edges(polygon)
        .iter()
        .any(|(a, b)| point_on_segment(point, *a, *b))
        || point_in_polygon(point, polygon)
}

fn segments_intersect(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    if o1.abs() <= 1.0e-5 && point_on_segment(c, a, b) {
        return true;
    }
    if o2.abs() <= 1.0e-5 && point_on_segment(d, a, b) {
        return true;
    }
    if o3.abs() <= 1.0e-5 && point_on_segment(a, c, d) {
        return true;
    }
    if o4.abs() <= 1.0e-5 && point_on_segment(b, c, d) {
        return true;
    }
    ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0))
        && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))
}

fn point_on_segment(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> bool {
    orient(a, b, point).abs() <= 1.0e-5
        && point[0] >= a[0].min(b[0]) - 1.0e-5
        && point[0] <= a[0].max(b[0]) + 1.0e-5
        && point[1] >= a[1].min(b[1]) - 1.0e-5
        && point[1] <= a[1].max(b[1]) + 1.0e-5
}

fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

pub(crate) fn parse_world_rect(params: &Value) -> Result<[f32; 4], ControlError> {
    let mut rect = [0.0_f32; 4];
    if let Some(values) = params
        .get("world_rect")
        .or_else(|| params.get("rect"))
        .and_then(Value::as_array)
    {
        if values.len() != 4 {
            return Err(invalid("world_rect must contain four finite numbers"));
        }
        for (index, value) in values.iter().enumerate() {
            rect[index] = finite_f32(value, "world_rect must contain four finite numbers")?;
        }
    } else {
        let names = [
            ["min_x", "x0"],
            ["min_y", "y0"],
            ["max_x", "x1"],
            ["max_y", "y1"],
        ];
        for (index, aliases) in names.iter().enumerate() {
            let value = aliases
                .iter()
                .find_map(|name| params.get(*name))
                .ok_or_else(|| invalid("provide world_rect or min_x/min_y/max_x/max_y"))?;
            rect[index] = finite_f32(value, "rectangle coordinates must be finite")?;
        }
    }
    Ok(normalize_rect(rect))
}

fn finite_f32(value: &Value, message: &str) -> Result<f32, ControlError> {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
        .ok_or_else(|| invalid(message))
}

pub(crate) fn parse_world_points(params: &Value) -> Result<Vec<[f32; 2]>, ControlError> {
    let values = params
        .get("world_points")
        .or_else(|| params.get("points"))
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("world_points is required"))?;
    if values.len() < 3 {
        return Err(invalid("world_points must contain at least three points"));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| invalid(format!("world_points[{index}] must be [x, y]")))?;
            let x = pair[0]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("world_points[{index}][0] must be finite")))?
                as f32;
            let y = pair[1]
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| invalid(format!("world_points[{index}][1] must be finite")))?
                as f32;
            Ok([x, y])
        })
        .collect()
}

fn parse_indices(
    params: &Value,
    name: &str,
    object_count: usize,
) -> Result<Vec<usize>, ControlError> {
    let values = params
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{name} is required")))?;
    let mut indices = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|index| *index < object_count)
                .ok_or_else(|| invalid(format!("{name} contains an out-of-range index")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    indices.sort_unstable();
    if indices.windows(2).any(|window| window[0] == window[1]) {
        return Err(invalid(format!("{name} must not contain duplicates")));
    }
    Ok(indices)
}

fn optional_index(params: &Value, name: &str) -> Result<Option<usize>, ControlError> {
    params
        .get(name)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid(format!("{name} must be a non-negative integer or null")))
        })
        .transpose()
}

fn selection_mode(params: &Value) -> Result<&str, ControlError> {
    let mode = params
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("replace");
    if matches!(mode, "replace" | "add" | "remove" | "toggle") {
        Ok(mode)
    } else {
        Err(invalid(
            "selection mode must be replace, add, remove, or toggle",
        ))
    }
}

fn selection_mode_with_additive(params: &Value) -> Result<&str, ControlError> {
    if params.get("mode").is_some() {
        selection_mode(params)
    } else if params
        .get("additive")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        Ok("add")
    } else {
        Ok("replace")
    }
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::InvalidParams, message)
}

fn not_found(message: impl Into<String>) -> ControlError {
    ControlError::new(ControlErrorKind::ResourceNotFound, message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use super::*;

    fn resource() -> ControlObjectResource {
        ControlObjectResource {
            source: PathBuf::from("objects.geojson"),
            downsample_factor: 1.0,
            features: Arc::new(vec![
                ControlObjectFeature {
                    id: "a".to_string(),
                    bbox_world: [0.0, 0.0, 10.0, 10.0],
                    centroid_world: [5.0, 5.0],
                    polygons_world: Arc::new(vec![vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]]),
                    point_position_world: None,
                    area_px: 50.0,
                    perimeter_px: 34.142,
                    properties: json!({"label":"tumour"}).as_object().unwrap().clone(),
                },
                ControlObjectFeature {
                    id: "b".to_string(),
                    bbox_world: [20.0, 20.0, 30.0, 30.0],
                    centroid_world: [25.0, 25.0],
                    polygons_world: Arc::new(Vec::new()),
                    point_position_world: Some([25.0, 25.0]),
                    area_px: 0.0,
                    perimeter_px: 0.0,
                    properties: json!({"label":"immune"}).as_object().unwrap().clone(),
                },
            ]),
            property_names: Arc::new(vec!["id".to_string(), "label".to_string()]),
            renderer_payload: None,
        }
    }

    #[test]
    fn selection_ids_queries_focus_and_conflicts_are_renderer_independent() {
        let resource = resource();
        let mut selection = ObjectSelectionModel::default();
        let selected = selection
            .select_ids(&resource, &json!({"ids":["tumour"]}), 10)
            .unwrap();
        assert_eq!(selected["selection"]["primary"]["id"], "a");
        assert_eq!(
            selection.query_rect(&resource, [19.0, 19.0, 31.0, 31.0], None, 10)["matches"][0]["id"],
            "b"
        );
        assert_eq!(
            selection.query_rect(&resource, [8.0, 8.0, 9.0, 9.0], None, 10)["match_count"],
            0
        );
        let (focused, bounds) = selection.focus(&resource, &json!({"id":"b"})).unwrap();
        assert_eq!(focused["focused"]["id"], "b");
        assert_eq!(bounds, Some([20.0, 20.0, 30.0, 30.0]));
        let stale = selection
            .replace_transaction(
                Some(&resource),
                &json!({"expected_generation":1,"selected_indices":[]}),
                10,
            )
            .unwrap_err();
        assert_eq!(stale.kind, ControlErrorKind::Conflict);
    }
}
