use super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn rebuild_layer_orders(&mut self) {
        // Channels: retain valid indices, then append missing.
        let n = self.channels.len();
        self.channel_layer_order.retain(|&i| i < n);
        let mut seen = HashSet::new();
        self.channel_layer_order.retain(|i| seen.insert(*i));
        if self.channel_layer_order.len() != n {
            self.channel_layer_order = (0..n).collect();
        }

        let mut want: Vec<LayerId> = Vec::new();
        for layer in &self.spatial_image_layers.images {
            want.push(LayerId::SpatialImage(layer.id));
        }
        for l in &self.mask_layers {
            want.push(LayerId::Mask(l.id));
        }
        for l in &self.annotation_layers {
            want.push(LayerId::Annotation(l.id));
        }
        if self.seg_geojson.loaded_geojson.is_some() {
            want.push(LayerId::SegmentationGeoJson);
        }
        if self.seg_objects.has_data() {
            want.push(LayerId::SegmentationObjects);
        }
        if self.label_cells.is_some() {
            want.push(LayerId::SegmentationLabels);
        }
        if !self.cell_points.points.is_empty() {
            want.push(LayerId::Points);
        }
        for layer in &self.spatial_layers.shapes {
            want.push(LayerId::SpatialShape(layer.id));
        }
        if self.spatial_layers.points.is_some() {
            want.push(LayerId::SpatialPoints);
        }
        if self.xenium_layers.cells.is_some() {
            want.push(LayerId::XeniumCells);
        }
        if self.xenium_layers.transcripts.is_some() {
            want.push(LayerId::XeniumTranscripts);
        }

        let mut seen2 = HashSet::new();
        self.overlay_layer_order
            .retain(|id| want.contains(id) && seen2.insert(*id));
        for id in want {
            if !self.overlay_layer_order.contains(&id) {
                self.overlay_layer_order.push(id);
            }
        }

        if let LayerId::Channel(idx) = self.active_layer {
            if idx >= n {
                self.active_layer = if n > 0 {
                    LayerId::Channel(0)
                } else {
                    LayerId::Points
                };
            }
        }
        if matches!(
            self.active_layer,
            LayerId::SpatialImage(_)
                | LayerId::Mask(_)
                | LayerId::SegmentationGeoJson
                | LayerId::SegmentationObjects
                | LayerId::SegmentationLabels
                | LayerId::Points
                | LayerId::Annotation(_)
                | LayerId::SpatialShape(_)
                | LayerId::SpatialPoints
                | LayerId::XeniumCells
                | LayerId::XeniumTranscripts
        ) && !self.overlay_layer_order.contains(&self.active_layer)
        {
            self.active_layer = if n > 0 {
                LayerId::Channel(self.selected_channel.min(n - 1))
            } else {
                LayerId::Points
            };
        }
    }
}
