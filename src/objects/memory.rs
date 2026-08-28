//! Capacity-based accounting for retained CPU object geometry.

use super::*;
use odon::model::{ControlObjectMemoryDiagnostics, RetainedMemoryComponent};
use std::mem::size_of;

const ARC_HEADER_BYTES: usize = size_of::<usize>() * 2;

fn bytes_for<T>(capacity: usize) -> u64 {
    capacity.saturating_mul(size_of::<T>()) as u64
}

fn add_vec<T>(component: &mut RetainedMemoryComponent, values: &Vec<T>) {
    component.payload_capacity_bytes = component
        .payload_capacity_bytes
        .saturating_add(bytes_for::<T>(values.capacity()));
    component.logical_element_count = component
        .logical_element_count
        .saturating_add(values.len() as u64);
    component.allocation_count = component
        .allocation_count
        .saturating_add(u64::from(values.capacity() > 0));
}

fn add_arc_vec<T>(component: &mut RetainedMemoryComponent, values: &Arc<Vec<T>>) {
    component.container_bytes = component
        .container_bytes
        .saturating_add((ARC_HEADER_BYTES + size_of::<Vec<T>>()) as u64);
    component.allocation_count = component.allocation_count.saturating_add(1);
    add_vec(component, values);
}

fn add_arc_container<T>(component: &mut RetainedMemoryComponent, _value: &Arc<T>) {
    component.container_bytes = component
        .container_bytes
        .saturating_add((ARC_HEADER_BYTES + size_of::<T>()) as u64);
    component.allocation_count = component.allocation_count.saturating_add(1);
}

fn add_string(component: &mut RetainedMemoryComponent, value: &String) {
    component.payload_capacity_bytes = component
        .payload_capacity_bytes
        .saturating_add(value.capacity() as u64);
    component.logical_element_count = component.logical_element_count.saturating_add(1);
    component.allocation_count = component
        .allocation_count
        .saturating_add(u64::from(value.capacity() > 0));
}

fn add_json_map(
    maps: &mut RetainedMemoryComponent,
    strings: &mut RetainedMemoryComponent,
    values: &serde_json::Map<String, serde_json::Value>,
) {
    maps.logical_element_count = maps
        .logical_element_count
        .saturating_add(values.len() as u64);
    maps.opaque_allocation_count = maps
        .opaque_allocation_count
        .saturating_add(u64::from(!values.is_empty()));
    for (key, value) in values {
        add_string(strings, key);
        add_json_value(maps, strings, value);
    }
}

fn add_json_value(
    maps: &mut RetainedMemoryComponent,
    strings: &mut RetainedMemoryComponent,
    value: &serde_json::Value,
) {
    match value {
        serde_json::Value::String(value) => add_string(strings, value),
        serde_json::Value::Array(values) => {
            maps.payload_capacity_bytes =
                maps.payload_capacity_bytes
                    .saturating_add(bytes_for::<serde_json::Value>(values.capacity()));
            maps.allocation_count = maps
                .allocation_count
                .saturating_add(u64::from(values.capacity() > 0));
            for value in values {
                add_json_value(maps, strings, value);
            }
        }
        serde_json::Value::Object(values) => add_json_map(maps, strings, values),
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

fn line_lods_component(lods: &Vec<ObjectRenderLod>) -> RetainedMemoryComponent {
    let mut component = RetainedMemoryComponent::default();
    component.payload_capacity_bytes = component
        .payload_capacity_bytes
        .saturating_add(bytes_for::<ObjectRenderLod>(lods.capacity()));
    component.logical_element_count = component
        .logical_element_count
        .saturating_add(lods.len() as u64);
    component.allocation_count = component
        .allocation_count
        .saturating_add(u64::from(!lods.is_empty()));
    for lod in lods {
        add_arc_container(&mut component, &lod.bins);
        add_arc_vec(&mut component, &lod.bins.segments);
        add_arc_vec(&mut component, &lod.bins.offsets);
        add_arc_vec(&mut component, &lod.bins.counts);
    }
    component
}

fn selection_lods_component(lods: &Vec<ObjectSelectionRenderLod>) -> RetainedMemoryComponent {
    let mut component = RetainedMemoryComponent::default();
    component.payload_capacity_bytes = component
        .payload_capacity_bytes
        .saturating_add(bytes_for::<ObjectSelectionRenderLod>(lods.capacity()));
    component.logical_element_count = component
        .logical_element_count
        .saturating_add(lods.len() as u64);
    component.allocation_count = component
        .allocation_count
        .saturating_add(u64::from(!lods.is_empty()));
    for lod in lods {
        add_arc_container(&mut component, &lod.bins);
        add_arc_vec(&mut component, &lod.bins.segments);
        add_arc_vec(&mut component, &lod.bins.offsets);
        add_arc_vec(&mut component, &lod.bins.counts);
    }
    component
}

pub(super) fn load_result_memory_diagnostics(
    result: &LoadResult,
) -> ControlObjectMemoryDiagnostics {
    let mut diagnostics = ControlObjectMemoryDiagnostics::default();

    let mut feature_records = RetainedMemoryComponent::default();
    add_arc_vec(&mut feature_records, &result.objects);
    diagnostics.add_component("renderer.canonical_feature_records", feature_records);

    let mut feature_ids = RetainedMemoryComponent::default();
    let mut inline_property_strings = RetainedMemoryComponent::default();
    let mut inline_property_maps = RetainedMemoryComponent::default();
    for object in result.objects.iter() {
        add_string(&mut feature_ids, &object.id);
        add_json_map(
            &mut inline_property_maps,
            &mut inline_property_strings,
            &object.inline_properties,
        );
    }
    diagnostics.add_component("renderer.canonical_feature_ids", feature_ids);
    diagnostics.add_component("renderer.inline_property_maps", inline_property_maps);
    diagnostics.add_component("renderer.inline_property_strings", inline_property_strings);

    let mut polygon_containers = RetainedMemoryComponent::default();
    let mut polygon_points = RetainedMemoryComponent::default();
    for object in result.objects.iter() {
        add_vec(&mut polygon_containers, &object.polygons_world);
        for polygon in &object.polygons_world {
            add_vec(&mut polygon_points, polygon);
        }
    }
    diagnostics.add_component("renderer.canonical_polygon_containers", polygon_containers);
    diagnostics.add_component("renderer.canonical_polygon_points", polygon_points);

    let mut spatial_index = RetainedMemoryComponent::default();
    add_arc_container(&mut spatial_index, &result.bins);
    add_vec(&mut spatial_index, &result.bins.indices);
    add_vec(&mut spatial_index, &result.bins.offsets);
    add_vec(&mut spatial_index, &result.bins.counts);
    diagnostics.add_component("renderer.object_spatial_index", spatial_index);

    diagnostics.add_component(
        "renderer.outline_lods",
        line_lods_component(&result.render_lods),
    );
    if let Some(lods) = result.object_selection_lods.as_ref() {
        diagnostics.add_component(
            "renderer.selection_outline_lods",
            selection_lods_component(lods),
        );
    }

    if let Some(mesh) = result.object_fill_mesh.as_ref() {
        let mut full_mesh = RetainedMemoryComponent::default();
        add_arc_vec(&mut full_mesh, &mesh.vertices_local);
        diagnostics.add_component("renderer.fill_mesh_full", full_mesh);

        let mut spatial_bins = RetainedMemoryComponent::default();
        add_vec(&mut spatial_bins, &mesh.bin_vertices);
        for vertices in &mesh.bin_vertices {
            add_arc_vec(&mut spatial_bins, vertices);
        }
        diagnostics.add_component("renderer.fill_mesh_spatial_bins", spatial_bins);
    }

    let mut point_payload = RetainedMemoryComponent::default();
    add_arc_vec(&mut point_payload, &result.point_positions_world);
    add_arc_vec(&mut point_payload, &result.point_values);
    add_arc_vec(&mut point_payload, &result.point_lods);
    for lod in result.point_lods.iter() {
        add_arc_vec(&mut point_payload, &lod.positions_world);
        add_arc_vec(&mut point_payload, &lod.values);
    }
    diagnostics.add_component("renderer.point_payload", point_payload);

    diagnostics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_component_uses_capacity_not_only_length() {
        let mut values = Vec::<[f32; 2]>::with_capacity(8);
        values.push([1.0, 2.0]);
        let mut component = RetainedMemoryComponent::default();
        add_vec(&mut component, &values);
        assert_eq!(component.payload_capacity_bytes, 8 * 8);
        assert_eq!(component.logical_element_count, 1);
        assert_eq!(component.allocation_count, 1);
    }

    #[test]
    fn diagnostics_merge_preserves_component_breakdown() {
        let mut left = ControlObjectMemoryDiagnostics::default();
        left.add_component(
            "renderer.outline_lods",
            RetainedMemoryComponent {
                payload_capacity_bytes: 32,
                allocation_count: 1,
                ..Default::default()
            },
        );
        let mut right = ControlObjectMemoryDiagnostics::default();
        right.add_component(
            "renderer.outline_lods",
            RetainedMemoryComponent {
                payload_capacity_bytes: 64,
                allocation_count: 2,
                ..Default::default()
            },
        );
        left.merge(&right);
        assert_eq!(left.total().retained_bytes(), 96);
        assert_eq!(left.total().allocation_count, 3);
    }
}
