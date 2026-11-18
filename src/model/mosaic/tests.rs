//! Mosaic-model layout regression tests.

use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_layout_accumulates_rows_without_overlapping() {
        let mut items = (0..3)
            .map(|id| MosaicItemModel {
                id,
                roi_id: format!("roi-{id}"),
                metadata: HashMap::new(),
                source: format!("source-{id}"),
                level0_size: [100.0 + id as f32 * 10.0, 50.0 + id as f32 * 5.0],
                offset: [0.0, 0.0],
                scale: 1.0,
                placed_size: [1.0, 1.0],
                segmentation_path: None,
            })
            .collect::<Vec<_>>();
        let size = layout_block(
            &mut items,
            48.0,
            2,
            [1.0, 1.0],
            10.0,
            MosaicLayoutMode::NativePixels,
        );
        assert!(items[2].offset[1] > items[0].bounds()[1][1]);
        assert!(size[0] >= 220.0);
        assert!(size[1] >= 115.0);
    }

    #[test]
    fn group_heading_bounds_follow_layout_and_clear_when_ungrouped() {
        let mut mosaic = MosaicModel::default();
        mosaic.group_by = "patient".to_string();
        mosaic.grid_cell_size = [100.0, 50.0];
        mosaic.columns = 2;
        mosaic.items = (0..3)
            .map(|id| MosaicItemModel {
                id,
                roi_id: format!("ROI-{id}"),
                metadata: HashMap::from([(
                    "patient".to_string(),
                    if id < 2 { "Patient 1" } else { "Patient 2" }.to_string(),
                )]),
                source: format!("source-{id}"),
                level0_size: [100.0, 50.0],
                offset: [0.0, 0.0],
                scale: 1.0,
                placed_size: [1.0, 1.0],
                segmentation_path: None,
            })
            .collect();
        mosaic.apply_layout();
        assert_eq!(mosaic.group_blocks.len(), 2);
        assert_eq!(mosaic.group_blocks[0].name, "Patient 1");
        assert_eq!(mosaic.group_blocks[1].name, "Patient 2");
        assert_eq!(mosaic.group_blocks[0].bounds, [[0.0, 0.0], [264.0, 98.0]]);
        assert_eq!(
            mosaic.group_blocks[1].bounds,
            [[0.0, 194.0], [264.0, 292.0]]
        );
        mosaic.group_by.clear();
        mosaic.apply_layout();
        assert!(mosaic.group_blocks.is_empty());
    }

    #[test]
    fn renderer_object_observation_is_exposed_in_mosaic_object_state() {
        let mut mosaic = MosaicModel::default();
        let observation = json!({
            "resident_property_bytes": 1234,
            "in_flight_property_loads": 2,
            "loads_cancelled": 7,
        });

        mosaic.observe_renderer_object_state(&observation);

        assert_eq!(mosaic.object_state()["renderer"], observation);
    }
}
