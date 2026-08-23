//! Geometry and render-cache regression tests.

use super::*;

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
