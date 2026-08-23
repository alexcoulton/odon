use super::super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn control_native_layer_kind(id: LayerId) -> &'static str {
        match id {
            LayerId::Channel(_) => "channel",
            LayerId::SpatialImage(_) => "spatial_image",
            LayerId::SegmentationLabels => "segmentation_labels",
            LayerId::SegmentationGeoJson => "segmentation_geojson",
            LayerId::SegmentationObjects => "segmentation_objects",
            LayerId::Mask(_) => "mask",
            LayerId::Points => "points",
            LayerId::Annotation(_) => "annotation",
            LayerId::SpatialShape(_) => "spatial_shape",
            LayerId::SpatialPoints => "spatial_points",
            LayerId::XeniumCells => "xenium_cells",
            LayerId::XeniumTranscripts => "xenium_transcripts",
        }
    }

    pub(in crate::app) fn control_native_layer_snapshot(
        &self,
        id: LayerId,
        stack: &str,
        order: usize,
    ) -> serde_json::Value {
        let offset = self.layer_offset_world(id);
        let loaded_offset = self
            .loaded_layer_offsets_world
            .get(&id)
            .copied()
            .unwrap_or(offset);
        serde_json::json!({
            "layer_id": Self::layer_id_storage_key(id),
            "kind": Self::control_native_layer_kind(id),
            "name": self.layer_display_name(id),
            "stack": stack,
            "order": order,
            "active": self.active_layer == id,
            "visible": self.layer_visible_value(id).unwrap_or(false),
            "available": self.layer_is_available(id),
            "offset_world": [offset.x, offset.y],
            "loaded_offset_world": [loaded_offset.x, loaded_offset.y],
            "presentation": self.control_native_layer_presentation(id),
        })
    }

    pub(in crate::app) fn control_native_layer_presentation(
        &self,
        id: LayerId,
    ) -> serde_json::Value {
        match id {
            LayerId::Channel(index) => self
                .channels
                .get(index)
                .map(|channel| {
                    serde_json::json!({
                        "visible": channel.visible,
                        "color_rgb": channel.color_rgb,
                        "window": channel.window.map(|(min, max)| serde_json::json!({
                            "min": min,
                            "max": max,
                        })),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "current_z_level0": layer.current_z_level0,
                        "channels": layer.channels.iter().map(|channel| serde_json::json!({
                            "index": channel.index,
                            "name": channel.name,
                            "visible": channel.visible,
                            "color_rgb": channel.color_rgb,
                            "window": channel.window.map(|(min, max)| [min, max]),
                        })).collect::<Vec<_>>(),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SegmentationLabels => serde_json::json!({
                "visible": self.cells_outlines_visible,
                "opacity": self.cells_outlines_opacity,
                "width_screen_px": self.cells_outlines_width_px,
                "color_rgb": self.cells_outlines_color_rgb,
            }),
            LayerId::SegmentationGeoJson => serde_json::json!({
                "visible": self.seg_geojson.visible,
                "opacity": self.seg_geojson.opacity,
                "width_screen_px": self.seg_geojson.width_screen_px,
                "color_rgb": self.seg_geojson.color_rgb,
            }),
            LayerId::SegmentationObjects => ViewerViewportState::object_layer_presentation_json(
                &ObjectLayerViewportPresentation::capture(&self.seg_objects),
            ),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "display_mode": layer.display_mode.storage_key(),
                        "color_rgb": layer.color_rgb,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::Points => serde_json::json!({
                "visible": self.cell_points.visible,
                "style": ViewerViewportState::points_style_json(&self.cell_points.style),
            }),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": {
                            "radius_screen_px": layer.style.radius_screen_px,
                            "opacity": layer.style.opacity,
                            "stroke_width": layer.style.stroke.width,
                            "stroke_color_rgba": ViewerViewportState::color_json(layer.style.stroke.color),
                        },
                        "category_styles": layer.category_styles.iter().map(|category| serde_json::json!({
                            "name": category.name,
                            "visible": category.visible,
                            "color_rgba": ViewerViewportState::color_json(category.color),
                            "shape": category.shape.storage_key(),
                        })).collect::<Vec<_>>(),
                        "continuous_shape": layer.continuous_shape.storage_key(),
                        "continuous_range": layer.continuous_range.map(|(min, max)| [min, max]),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|layer| layer.id == id)
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "color_rgb": layer.color_rgb,
                        "objects": layer.object_layer().map(|objects| {
                            ViewerViewportState::object_layer_presentation_json(
                                &ObjectLayerViewportPresentation::capture(objects),
                            )
                        }),
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::SpatialPoints => self
                .spatial_layers
                .points
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": ViewerViewportState::points_style_json(&layer.style),
                        "threshold": layer.threshold,
                        "max_render_points_total": layer.max_render_points_total,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::XeniumCells => self
                .xenium_layers
                .cells
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "opacity": layer.opacity,
                        "width_screen_px": layer.width_screen_px,
                        "color_rgb": layer.color_rgb,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_ref()
                .map(|layer| {
                    serde_json::json!({
                        "visible": layer.visible,
                        "style": ViewerViewportState::points_style_json(&layer.style),
                        "gene_query": layer.gene_query,
                        "max_render_points_total": layer.max_render_points_total,
                    })
                })
                .unwrap_or(serde_json::Value::Null),
        }
    }

    pub fn control_native_layer_snapshot_list(&self) -> serde_json::Value {
        let mut layers = self
            .channel_layer_order
            .iter()
            .copied()
            .enumerate()
            .map(|(order, idx)| {
                self.control_native_layer_snapshot(LayerId::Channel(idx), "channels", order)
            })
            .collect::<Vec<_>>();
        layers.extend(
            self.overlay_layer_order
                .iter()
                .copied()
                .enumerate()
                .map(|(order, id)| self.control_native_layer_snapshot(id, "overlays", order)),
        );
        serde_json::Value::Array(layers)
    }
}
