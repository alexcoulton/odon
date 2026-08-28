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
