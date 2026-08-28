//! Geometry and render-cache regression tests.

use super::*;

fn presentation_test_object(id: &str, x: f32) -> ObjectFeature {
    ObjectFeature {
        id: id.to_string(),
        polygons_world: Vec::new(),
        point_position_world: Some(egui::pos2(x, 0.0)),
        bbox_world: egui::Rect::from_min_max(egui::pos2(x, 0.0), egui::pos2(x, 0.0)),
        area_px: 0.0,
        perimeter_px: 0.0,
        centroid_world: egui::pos2(x, 0.0),
        inline_properties: serde_json::Map::new(),
        source_row_index: None,
    }
}

#[test]
fn outline_mode_resolver_uses_hysteresis_around_texture_threshold() {
    let mut resolver = ObjectOutlineModeRuntime::default();
    assert!(resolver.resolve_preference(0.30, 200_000, true).0);
    assert!(resolver.resolve_preference(0.50, 200_000, true).0);
    assert!(!resolver.resolve_preference(0.60, 200_000, true).0);
    assert!(!resolver.resolve_preference(0.50, 200_000, true).0);
    assert!(resolver.resolve_preference(0.30, 200_000, true).0);
    assert_eq!(resolver.transitions(), 3);
}

#[test]
fn outline_mode_resolver_resets_when_texture_path_is_unavailable() {
    let mut resolver = ObjectOutlineModeRuntime::default();
    assert!(resolver.resolve_preference(0.10, 200_000, true).0);
    assert!(!resolver.resolve_preference(0.10, 200_000, false).0);
    assert_eq!(resolver.transitions(), 2);
}

#[test]
fn outline_frame_aggregation_ignores_layers_without_a_presented_frame() {
    let mut aggregate = ObjectOutlineFrameStats::default();
    let mut texture = ObjectOutlineFrameStats::for_layer(0.1, 4, 400_000, 1);
    texture.set_mode(
        ObjectOutlineMode::Texture,
        ObjectOutlineModeReason::TextureWorkload,
    );
    aggregate.merge(texture);
    aggregate.merge(ObjectOutlineFrameStats::default());

    assert_eq!(aggregate.layer_count, 1);
    assert_eq!(aggregate.mode, ObjectOutlineMode::Texture);
    assert_eq!(aggregate.visible_records, 400_000);
}

#[test]
fn outline_frame_aggregation_keeps_a_shared_mode_when_only_reasons_differ() {
    let mut aggregate = ObjectOutlineFrameStats::default();
    let mut workload = ObjectOutlineFrameStats::for_layer(0.1, 4, 400_000, 1);
    workload.set_mode(
        ObjectOutlineMode::Texture,
        ObjectOutlineModeReason::TextureWorkload,
    );
    let mut hysteresis = ObjectOutlineFrameStats::for_layer(0.2, 3, 200_000, 1);
    hysteresis.set_mode(
        ObjectOutlineMode::Texture,
        ObjectOutlineModeReason::TextureHysteresis,
    );

    aggregate.merge(workload);
    aggregate.merge(hysteresis);

    assert_eq!(aggregate.mode, ObjectOutlineMode::Texture);
    assert_eq!(aggregate.reason, ObjectOutlineModeReason::Mixed);
    assert_eq!(aggregate.texture_layers, 2);
}

#[test]
fn camera_only_frames_reuse_visibility_presentation_state() {
    let mut layer = ObjectsLayer::default();
    layer.objects = Some(Arc::new(vec![
        presentation_test_object("a", 0.0),
        presentation_test_object("b", 1.0),
    ]));

    let first = layer.cached_visibility_state();
    let first_stats = layer.presentation_state_stats();
    let second = layer.cached_visibility_state();
    let second_stats = layer.presentation_state_stats();

    assert!(Arc::ptr_eq(&first.values, &second.values));
    assert_eq!(first.generation, second.generation);
    assert_eq!(first_stats.visibility_rebuilds, 1);
    assert_eq!(second_stats.visibility_rebuilds, 1);

    layer.filter_generation = layer.filter_generation.wrapping_add(1).max(1);
    let filtered = layer.cached_visibility_state();
    assert!(!Arc::ptr_eq(&first.values, &filtered.values));
    assert_ne!(first.generation, filtered.generation);
    assert_eq!(layer.presentation_state_stats().visibility_rebuilds, 2);
}

#[test]
fn selection_only_invalidates_continuous_outline_presentation_state() {
    let mut layer = ObjectsLayer::default();
    layer.objects = Some(Arc::new(vec![
        presentation_test_object("a", 0.0),
        presentation_test_object("b", 1.0),
    ]));
    layer.show_selection_overlay = true;

    let visibility = layer.cached_visibility_state();
    let first = layer.cached_continuous_outline_state();
    let first_stats = layer.presentation_state_stats();
    let again = layer.cached_continuous_outline_state();
    assert!(Arc::ptr_eq(&first.values, &again.values));
    assert_eq!(first_stats.continuous_outline_rebuilds, 1);

    layer.selected_object_indices.insert(1);
    layer.selected_object_index = Some(1);
    layer.selection_generation = layer.selection_generation.wrapping_add(1).max(1);
    let selected = layer.cached_continuous_outline_state();
    let visibility_after_selection = layer.cached_visibility_state();

    assert!(Arc::ptr_eq(
        &visibility.values,
        &visibility_after_selection.values
    ));
    assert!(!Arc::ptr_eq(&first.values, &selected.values));
    assert_eq!(selected.values.as_slice(), &[64, 255]);
    assert_eq!(layer.presentation_state_stats().visibility_rebuilds, 1);
    assert_eq!(
        layer.presentation_state_stats().continuous_outline_rebuilds,
        2
    );
}

#[test]
fn fill_proxy_points_follow_the_low_zoom_geometry_preference() {
    let lods = build_render_lods_from_polylines(&[vec![
        egui::pos2(0.0, 0.0),
        egui::pos2(100.0, 0.0),
        egui::pos2(100.0, 100.0),
        egui::pos2(0.0, 100.0),
        egui::pos2(0.0, 0.0),
    ]])
    .expect("test polygon should produce render LODs");
    let coarse = lods
        .iter()
        .find(|lod| lod.lod >= 2)
        .expect("test polygon should produce a coarse LOD");

    let mut layer = ObjectsLayer {
        fill_cells: true,
        fill_opacity: 0.3,
        fast_rendering: true,
        ..ObjectsLayer::default()
    };
    assert!(layer.should_use_fill_proxy_points(coarse));

    layer.fast_rendering = false;
    assert!(!layer.should_use_fill_proxy_points(coarse));
}

#[test]
fn outline_geometry_cache_generation_ignores_presentation_changes() {
    let mut layer = ObjectsLayer::default();
    layer.geometry_generation = 41;
    layer.generation = 41;
    let geometry_cache_generation = layer.outline_geometry_cache_generation();

    // Marker, palette, filter, and other presentation changes advance this general generation.
    layer.generation = layer.generation.wrapping_add(1);

    assert_eq!(
        layer.outline_geometry_cache_generation(),
        geometry_cache_generation,
        "presentation changes must reuse uploaded outline geometry"
    );
    layer.geometry_generation = layer.geometry_generation.wrapping_add(1);
    assert_ne!(
        layer.outline_geometry_cache_generation(),
        geometry_cache_generation,
        "actual geometry changes must invalidate uploaded outline geometry"
    );
}

#[test]
fn outline_gpu_stats_expose_empty_cache_capacities() {
    let renderer = ObjectLineBinsGlRenderer::new(32, 3);

    let stats = renderer.stats();

    assert_eq!(stats.bin_entries, 0);
    assert_eq!(stats.bin_capacity_entries, 32);
    assert_eq!(stats.bin_bytes, 0);
    assert_eq!(stats.state_entries, 0);
    assert_eq!(stats.state_capacity_entries, 3);
    assert_eq!(stats.color_entries, 0);
    assert_eq!(stats.color_capacity_entries, 3);
}

#[cfg(test)]
mod rectangle_selection_tests {
    use super::*;

    fn object_with_polygons(polygons_world: Vec<Vec<egui::Pos2>>) -> ObjectFeature {
        let bbox_world = polygons_world
            .iter()
            .flat_map(|poly| poly.iter().copied())
            .fold(None, |acc: Option<egui::Rect>, point| {
                let rect = egui::Rect::from_min_max(point, point);
                Some(acc.map_or(rect, |acc| acc.union(rect)))
            })
            .unwrap_or_else(|| {
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(0.0, 0.0))
            });
        ObjectFeature {
            id: "test".to_string(),
            polygons_world,
            point_position_world: None,
            bbox_world,
            area_px: 0.0,
            perimeter_px: 0.0,
            centroid_world: egui::pos2(1000.0, 1000.0),
            inline_properties: serde_json::Map::new(),
            source_row_index: None,
        }
    }

    fn point_object(point: egui::Pos2, centroid: egui::Pos2) -> ObjectFeature {
        ObjectFeature {
            id: "point".to_string(),
            polygons_world: Vec::new(),
            point_position_world: Some(point),
            bbox_world: egui::Rect::from_min_max(point, point),
            area_px: 0.0,
            perimeter_px: 0.0,
            centroid_world: centroid,
            inline_properties: serde_json::Map::new(),
            source_row_index: None,
        }
    }

    #[test]
    fn fill_spatial_query_returns_only_intersecting_non_empty_bins() {
        let objects = vec![
            object_with_polygons(vec![vec![
                egui::pos2(10.0, 10.0),
                egui::pos2(110.0, 10.0),
                egui::pos2(110.0, 110.0),
                egui::pos2(10.0, 110.0),
                egui::pos2(10.0, 10.0),
            ]]),
            object_with_polygons(vec![vec![
                egui::pos2(5000.0, 5000.0),
                egui::pos2(5100.0, 5000.0),
                egui::pos2(5100.0, 5100.0),
                egui::pos2(5000.0, 5100.0),
                egui::pos2(5000.0, 5000.0),
            ]]),
        ];
        let mesh = build_object_fill_mesh(&objects).expect("test polygons should tessellate");
        let visible = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(200.0, 200.0));
        let slices = mesh.spatial_slices_for_local_rect(visible);

        assert!(!slices.is_empty());
        assert!(slices.len() < mesh.bin_vertices.len());
        assert!(
            slices
                .iter()
                .all(|slice| slice.bounds_local.intersects(visible))
        );
        assert!(
            slices
                .iter()
                .flat_map(|slice| slice.vertices_local.iter())
                .all(|vertex| vertex[2] == 0.0,)
        );
        assert!(
            mesh.spatial_slices_for_local_rect(egui::Rect::from_min_max(
                egui::pos2(-500.0, -500.0),
                egui::pos2(-100.0, -100.0),
            ))
            .is_empty()
        );
    }

    #[test]
    fn rect_contains_point_inclusive_accepts_edges() {
        let rect = egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(30.0, 40.0));

        assert!(rect_contains_point_inclusive(rect, egui::pos2(10.0, 20.0)));
        assert!(rect_contains_point_inclusive(rect, egui::pos2(30.0, 40.0)));
        assert!(rect_contains_point_inclusive(rect, egui::pos2(20.0, 30.0)));
        assert!(!rect_contains_point_inclusive(rect, egui::pos2(30.1, 30.0)));
        assert!(!rect_contains_point_inclusive(rect, egui::pos2(20.0, 40.1)));
    }

    #[test]
    fn object_render_cache_ids_keep_namespace_and_index_separate() {
        let old_bin_98 = 0x5345474f424a80u64 | 98;
        let old_bin_226 = 0x5345474f424a80u64 | 226;
        assert_eq!(old_bin_98, old_bin_226);

        let new_bin_98 = object_render_cache_id(0x4a80, 98);
        let new_bin_226 = object_render_cache_id(0x4a80, 226);
        assert_ne!(new_bin_98, new_bin_226);
        assert_ne!(
            object_property_render_cache_id(0x4a20, "marker_a", 0),
            object_property_render_cache_id(0x4a20, "marker_b", 0),
            "simultaneous viewport styles need distinct GPU state caches"
        );
    }

    #[test]
    fn polygon_selection_uses_geometry_not_centroid() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(2.0, 2.0),
            egui::pos2(8.0, 2.0),
            egui::pos2(8.0, 8.0),
            egui::pos2(2.0, 8.0),
            egui::pos2(2.0, 2.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_rejects_bbox_outside_even_when_centroid_inside() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let mut object = object_with_polygons(vec![vec![
            egui::pos2(20.0, 20.0),
            egui::pos2(30.0, 20.0),
            egui::pos2(30.0, 30.0),
            egui::pos2(20.0, 30.0),
            egui::pos2(20.0, 20.0),
        ]]);
        object.centroid_world = egui::pos2(5.0, 5.0);

        assert!(!object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_detects_rect_inside_polygon() {
        let rect = egui::Rect::from_min_max(egui::pos2(4.0, 4.0), egui::pos2(6.0, 6.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(0.0, 0.0),
            egui::pos2(10.0, 0.0),
            egui::pos2(10.0, 10.0),
            egui::pos2(0.0, 10.0),
            egui::pos2(0.0, 0.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn polygon_selection_detects_edge_crossing() {
        let rect = egui::Rect::from_min_max(egui::pos2(4.0, 4.0), egui::pos2(6.0, 6.0));
        let object = object_with_polygons(vec![vec![
            egui::pos2(2.0, 5.0),
            egui::pos2(8.0, 5.0),
            egui::pos2(8.0, 8.0),
            egui::pos2(2.0, 8.0),
            egui::pos2(2.0, 5.0),
        ]]);

        assert!(object_intersects_rect_for_selection(&object, rect));
    }

    #[test]
    fn point_only_selection_uses_point_position_before_centroid() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(10.0, 10.0));
        let object = point_object(egui::pos2(20.0, 20.0), egui::pos2(5.0, 5.0));

        assert!(!object_intersects_rect_for_selection(&object, rect));
    }
}

#[cfg(test)]
mod object_fill_tile_tests {
    use super::tiles::{
        MAX_VISIBLE_OBJECT_FILL_TILES, ObjectFillTileSpec, choose_object_fill_tile_level,
        object_fill_tile_key, object_fill_tile_object_count_supported,
        object_fill_tile_planning_scales, object_fill_tile_raster_bounds, plan_object_fill_tiles,
    };
    use super::*;

    #[test]
    fn level_tracks_screen_resolution_in_powers_of_two() {
        assert_eq!(choose_object_fill_tile_level(4.0), 0);
        assert_eq!(choose_object_fill_tile_level(1.0), 0);
        assert_eq!(choose_object_fill_tile_level(0.5), 1);
        assert_eq!(choose_object_fill_tile_level(0.25), 2);
        assert_eq!(choose_object_fill_tile_level(0.01), 7);
    }

    #[test]
    fn texture_outlines_provide_two_levels_for_coverage_resolving() {
        let local_screen_per_pixel = 0.25;
        let (target, fallback) = object_fill_tile_planning_scales(local_screen_per_pixel, true);

        assert_eq!(choose_object_fill_tile_level(target), 0);
        assert_eq!(choose_object_fill_tile_level(fallback), 1);
    }

    #[test]
    fn fill_only_frames_keep_the_existing_coarse_fallback() {
        let local_screen_per_pixel = 0.25;
        let (target, fallback) = object_fill_tile_planning_scales(local_screen_per_pixel, false);

        assert_eq!(choose_object_fill_tile_level(target), 2);
        assert_eq!(choose_object_fill_tile_level(fallback), 4);
    }

    #[test]
    fn tile_keys_are_world_aligned_and_camera_independent() {
        let bounds =
            egui::Rect::from_min_max(egui::pos2(-1000.0, -1000.0), egui::pos2(3000.0, 3000.0));
        let first = plan_object_fill_tiles(
            egui::Rect::from_min_max(egui::pos2(-10.0, -10.0), egui::pos2(600.0, 600.0)),
            bounds,
            1.0,
        );
        assert!(
            first
                .iter()
                .any(|tile| tile.tile_x == -1 && tile.tile_y == -1)
        );
        assert!(
            first
                .iter()
                .any(|tile| tile.tile_x == 0 && tile.tile_y == 0)
        );

        let shifted = plan_object_fill_tiles(
            egui::Rect::from_min_max(egui::pos2(20.0, 20.0), egui::pos2(620.0, 620.0)),
            bounds,
            1.0,
        );
        assert!(
            shifted
                .iter()
                .any(|tile| tile.tile_x == 0 && tile.tile_y == 0)
        );
        assert_eq!(first[0].level, shifted[0].level);
    }

    #[test]
    fn zooming_out_requests_fewer_coarser_tiles() {
        let bounds = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(8192.0, 8192.0));
        let visible = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(4096.0, 4096.0));
        let fine = plan_object_fill_tiles(visible, bounds, 1.0);
        let coarse = plan_object_fill_tiles(visible, bounds, 0.125);

        assert!(coarse.len() < fine.len());
        assert!(coarse[0].level > fine[0].level);
    }

    #[test]
    fn visible_tiles_are_bounded_and_prioritize_the_camera_center() {
        let bounds = egui::Rect::from_min_max(egui::pos2(-1.0e7, -1.0e7), egui::pos2(1.0e7, 1.0e7));
        let visible =
            egui::Rect::from_min_max(egui::pos2(-5000.0, -5000.0), egui::pos2(5000.0, 5000.0));
        let tiles = plan_object_fill_tiles(visible, bounds, 8.0);

        assert!(tiles.len() <= MAX_VISIBLE_OBJECT_FILL_TILES as usize);
        let first_distance = tiles[0].bounds_local.center().distance_sq(visible.center());
        assert!(tiles.iter().all(|tile| {
            tile.bounds_local.center().distance_sq(visible.center()) >= first_distance
        }));
    }

    #[test]
    fn repeated_camera_trace_has_stable_world_keys_and_no_planner_queue() {
        let bounds =
            egui::Rect::from_min_max(egui::pos2(-8192.0, -8192.0), egui::pos2(8192.0, 8192.0));
        let cameras = [
            (egui::pos2(0.0, 0.0), 1.0),
            (egui::pos2(64.0, -32.0), 0.5),
            (egui::pos2(2048.0, 1024.0), 0.125),
            (egui::pos2(0.0, 0.0), 1.0),
        ];
        let trace = || {
            cameras
                .iter()
                .map(|(center, scale)| {
                    let half = egui::vec2(1024.0 / scale, 768.0 / scale) * 0.5;
                    plan_object_fill_tiles(
                        egui::Rect::from_min_max(*center - half, *center + half),
                        bounds,
                        *scale,
                    )
                    .into_iter()
                    .map(|tile| (tile.level, tile.tile_x, tile.tile_y))
                    .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(trace(), trace());
        assert_eq!(trace().first(), trace().last());
    }

    #[test]
    fn integer_id_path_supports_more_than_65535_objects_without_truncation() {
        assert!(object_fill_tile_object_count_supported(65_536));
        assert!(object_fill_tile_object_count_supported(16_777_215));
        assert!(!object_fill_tile_object_count_supported(16_777_216));
    }

    #[test]
    fn style_edits_reuse_id_tiles_while_geometry_reload_changes_the_key() {
        let spec = ObjectFillTileSpec {
            level: 3,
            tile_x: -4,
            tile_y: 9,
            bounds_local: egui::Rect::from_min_size(
                egui::pos2(-16384.0, 36864.0),
                egui::vec2(4096.0, 4096.0),
            ),
        };
        let before_style_edit = object_fill_tile_key(11, 7, spec);
        let after_property_palette_domain_filter_selection_and_opacity_edits =
            object_fill_tile_key(11, 7, spec);
        let after_geometry_reload = object_fill_tile_key(11, 8, spec);
        let different_resource = object_fill_tile_key(12, 7, spec);

        assert_eq!(
            before_style_edit,
            after_property_palette_domain_filter_selection_and_opacity_edits
        );
        assert_ne!(before_style_edit, after_geometry_reload);
        assert_ne!(before_style_edit, different_resource);
    }

    #[test]
    fn raster_bounds_add_two_logical_texels_on_every_side() {
        let logical =
            egui::Rect::from_min_size(egui::pos2(-4096.0, 8192.0), egui::vec2(4096.0, 4096.0));
        let raster = object_fill_tile_raster_bounds(logical);
        assert_eq!(raster.min, logical.min - egui::vec2(16.0, 16.0));
        assert_eq!(raster.max, logical.max + egui::vec2(16.0, 16.0));
    }

    #[test]
    fn selection_fill_is_a_state_lookup_attached_to_existing_id_tiles() {
        let mut layer = ObjectsLayer {
            show_selection_overlay: true,
            selected_fill_opacity: 0.4,
            ..ObjectsLayer::default()
        };
        layer.selected_object_indices.extend([1, 2]);
        layer.selected_object_index = Some(1);
        layer.rebuild_selection_fill_state(3);

        let style = layer
            .object_fill_selection_tile_style(3, false)
            .expect("visible non-empty selection should provide a tile state lookup");

        assert_eq!(style.state_generation, layer.selection_generation);
        assert!(Arc::ptr_eq(
            &style.object_state,
            &layer.selection_fill_state
        ));
        assert_eq!(style.object_state.as_slice(), &[0, 255, 128]);
        assert_eq!(style.selected_color.a(), 102);
        assert_eq!(style.primary_color, style.selected_color);
    }

    #[test]
    fn selection_overlay_tile_style_requires_visible_matching_state() {
        let mut layer = ObjectsLayer::default();
        layer.selected_object_indices.insert(0);
        layer.selected_object_index = Some(0);
        layer.rebuild_selection_fill_state(1);

        layer.show_selection_overlay = false;
        assert!(layer.object_fill_selection_tile_style(1, false).is_none());

        layer.show_selection_overlay = true;
        layer.selected_fill_opacity = 0.0;
        assert!(layer.object_fill_selection_tile_style(1, false).is_none());
        assert!(layer.object_fill_selection_tile_style(1, true).is_some());

        layer.selected_fill_opacity = 0.5;
        assert!(layer.object_fill_selection_tile_style(2, false).is_none());
    }

    #[test]
    fn polygon_selection_updates_state_without_rebuilding_selected_geometry() {
        let object = ObjectFeature {
            id: "cell-1".to_string(),
            polygons_world: vec![vec![
                egui::pos2(0.0, 0.0),
                egui::pos2(1.0, 0.0),
                egui::pos2(1.0, 1.0),
                egui::pos2(0.0, 1.0),
                egui::pos2(0.0, 0.0),
            ]],
            point_position_world: None,
            bbox_world: egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            area_px: 1.0,
            perimeter_px: 4.0,
            centroid_world: egui::pos2(0.5, 0.5),
            inline_properties: serde_json::Map::new(),
            source_row_index: Some(0),
        };
        let mut layer = ObjectsLayer::default();
        layer.objects = Some(Arc::new(vec![object]));
        layer.object_selection_lods = Some(Vec::new());
        layer.selected_render_lods = Some(Vec::new());
        layer.primary_selected_render_lods = Some(Vec::new());
        layer.selected_object_indices.insert(0);
        layer.selected_object_index = Some(0);
        let before_generation = layer.selection_generation;

        layer.rebuild_selection_render_lods();

        assert!(layer.object_selection_lods.is_some());
        assert!(layer.selected_render_lods.is_none());
        assert!(layer.primary_selected_render_lods.is_none());
        assert!(layer.selected_fill_mesh.is_none());
        assert!(layer.selection_cpu_overlay_dirty);
        assert_eq!(layer.selection_fill_state.as_slice(), &[255]);
        assert!(layer.selection_generation > before_generation);
    }
}
