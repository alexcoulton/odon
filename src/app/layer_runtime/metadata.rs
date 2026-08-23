use super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn add_annotation_layer(&mut self) {
        self.native_control_intents.push(NativeControlIntent {
            method: "viewer.annotations.layers.create",
            params: serde_json::json!({}),
        });
    }

    pub fn add_annotation_layer_from_menu(&mut self) {
        self.add_annotation_layer();
    }

    pub(in crate::app) fn queue_object_ui_action(
        &mut self,
        action: crate::objects::ObjectUiAction,
    ) {
        let (method, params) = match action {
            crate::objects::ObjectUiAction::Load { path, options } => {
                let path = if options.is_some() {
                    path
                } else {
                    let Some(path) = self.seg_objects.prepare_source_path(path) else {
                        return;
                    };
                    path
                };
                let mut params = serde_json::json!({
                    "path": path,
                    "downsample_factor": self.seg_objects.downsample_factor,
                });
                if let Some(options) = options {
                    params
                        .as_object_mut()
                        .expect("object source params are an object")
                        .insert("loader_options".to_string(), options);
                }
                ("viewer.objects.source.load", params)
            }
            crate::objects::ObjectUiAction::Reload => {
                ("viewer.objects.source.reload", serde_json::json!({}))
            }
            crate::objects::ObjectUiAction::Clear => {
                ("viewer.objects.source.clear", serde_json::json!({}))
            }
            crate::objects::ObjectUiAction::ClearSelection => {
                let target = self.active_or_spatial_selection_layer();
                (
                    "viewer.objects.clear_selection",
                    serde_json::Value::Object(self.object_selection_target_params(Some(target))),
                )
            }
            crate::objects::ObjectUiAction::SelectFiltered => {
                let target = self.active_or_spatial_selection_layer();
                let mut params = self.object_selection_target_params(Some(target));
                params.insert("mode".to_string(), serde_json::json!("replace"));
                (
                    "viewer.objects.selection.select_filtered",
                    serde_json::Value::Object(params),
                )
            }
            crate::objects::ObjectUiAction::SelectIds { ids } => {
                let target = self.active_or_spatial_selection_layer();
                let mut params = self.object_selection_target_params(Some(target));
                params.insert("ids".to_string(), serde_json::json!(ids));
                params.insert("mode".to_string(), serde_json::json!("replace"));
                (
                    "viewer.objects.selection.select_ids",
                    serde_json::Value::Object(params),
                )
            }
        };
        self.native_control_intents
            .push(NativeControlIntent { method, params });
    }

    pub fn open_seg_geojson_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        self.seg_geojson.open_dialog(&default_dir);
    }

    pub fn open_seg_objects_dialog(&mut self) {
        let default_dir = self
            .dataset
            .source
            .local_path()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        if let Some(path) = self.seg_objects.choose_source_dialog(&default_dir) {
            self.queue_object_ui_action(crate::objects::ObjectUiAction::Load {
                path,
                options: None,
            });
        }
    }

    pub(in crate::app) fn layer_is_available(&self, id: LayerId) -> bool {
        match id {
            LayerId::SegmentationLabels => self.tiles_gl.is_some(),
            _ => true,
        }
    }

    pub(in crate::app) fn layer_display_name(&self, id: LayerId) -> String {
        match id {
            LayerId::Channel(idx) => self
                .channels
                .get(idx)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("Channel {idx}")),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Image {id}")),
            LayerId::SegmentationLabels => {
                let name = self.seg_label_selected.trim();
                if name.is_empty() {
                    "Segmentation labels".to_string()
                } else {
                    format!("Segmentation ({name})")
                }
            }
            LayerId::SegmentationGeoJson => "Segmentation (GeoJSON)".to_string(),
            LayerId::SegmentationObjects => "Segmentation (Objects)".to_string(),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Mask {id}")),
            LayerId::Points => "Points".to_string(),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.name.clone())
                .unwrap_or_else(|| format!("Annotations {id}")),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.name.clone())
                .unwrap_or_else(|| format!("Shapes {id}")),
            LayerId::SpatialPoints => self
                .spatial_layers
                .points
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or_else(|| "Points (SpatialData)".to_string()),
            LayerId::XeniumCells => self
                .xenium_layers
                .cells
                .as_ref()
                .map(|c| c.name.clone())
                .unwrap_or_else(|| "Cells (Xenium)".to_string()),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_ref()
                .map(|t| t.name.clone())
                .unwrap_or_else(|| "Transcripts (Xenium)".to_string()),
        }
    }

    pub(in crate::app) fn layer_icon(&self, id: LayerId) -> Icon {
        match id {
            LayerId::Channel(_) => Icon::Image,
            LayerId::SpatialImage(_) => Icon::Image,
            LayerId::Points => Icon::Points,
            LayerId::Annotation(_) => Icon::Points,
            LayerId::SpatialPoints => Icon::Points,
            LayerId::XeniumTranscripts => Icon::Points,
            LayerId::SegmentationLabels
            | LayerId::SegmentationGeoJson
            | LayerId::SegmentationObjects
            | LayerId::Mask(_)
            | LayerId::SpatialShape(_)
            | LayerId::XeniumCells => Icon::Polygon,
        }
    }

    pub(in crate::app) fn layer_visible_mut(&mut self, id: LayerId) -> Option<&mut bool> {
        match id {
            LayerId::Channel(idx) => self.channels.get_mut(idx).map(|c| &mut c.visible),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SegmentationLabels => Some(&mut self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(&mut self.seg_geojson.visible),
            LayerId::SegmentationObjects => Some(&mut self.seg_objects.visible),
            LayerId::Mask(id) => self
                .mask_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::Points => Some(&mut self.cell_points.visible),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter_mut()
                .find(|l| l.id == id)
                .map(|l| &mut l.visible),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter_mut()
                .find(|s| s.id == id)
                .map(|s| s.visible_mut()),
            LayerId::SpatialPoints => self.spatial_layers.points.as_mut().map(|p| &mut p.visible),
            LayerId::XeniumCells => self.xenium_layers.cells.as_mut().map(|c| &mut c.visible),
            LayerId::XeniumTranscripts => self
                .xenium_layers
                .transcripts
                .as_mut()
                .map(|t| &mut t.visible),
        }
    }

    pub(in crate::app) fn layer_visible_value(&self, id: LayerId) -> Option<bool> {
        match id {
            LayerId::Channel(idx) => self.channels.get(idx).map(|c| c.visible),
            LayerId::SpatialImage(id) => self
                .spatial_image_layers
                .images
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::SegmentationLabels => Some(self.cells_outlines_visible),
            LayerId::SegmentationGeoJson => Some(self.seg_geojson.visible),
            LayerId::SegmentationObjects => Some(self.seg_objects.visible),
            LayerId::Mask(id) => self
                .mask_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::Points => Some(self.cell_points.visible),
            LayerId::Annotation(id) => self
                .annotation_layers
                .iter()
                .find(|l| l.id == id)
                .map(|l| l.visible),
            LayerId::SpatialShape(id) => self
                .spatial_layers
                .shapes
                .iter()
                .find(|s| s.id == id)
                .map(|s| s.visible),
            LayerId::SpatialPoints => self.spatial_layers.points.as_ref().map(|p| p.visible),
            LayerId::XeniumCells => self.xenium_layers.cells.as_ref().map(|c| c.visible),
            LayerId::XeniumTranscripts => {
                self.xenium_layers.transcripts.as_ref().map(|t| t.visible)
            }
        }
    }
}
