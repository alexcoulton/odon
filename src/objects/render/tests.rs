//! Geometry and render-cache regression tests.

use super::*;

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
        object_fill_tile_key, object_fill_tile_object_count_supported, plan_object_fill_tiles,
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
        let before_style_edit = object_fill_tile_key(7, spec);
        let after_property_palette_domain_filter_selection_and_opacity_edits =
            object_fill_tile_key(7, spec);
        let after_geometry_reload = object_fill_tile_key(8, spec);

        assert_eq!(
            before_style_edit,
            after_property_palette_domain_filter_selection_and_opacity_edits
        );
        assert_ne!(before_style_edit, after_geometry_reload);
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
            .object_fill_selection_tile_style(3)
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
        assert!(layer.object_fill_selection_tile_style(1).is_none());

        layer.show_selection_overlay = true;
        layer.selected_fill_opacity = 0.0;
        assert!(layer.object_fill_selection_tile_style(1).is_none());

        layer.selected_fill_opacity = 0.5;
        assert!(layer.object_fill_selection_tile_style(2).is_none());
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
