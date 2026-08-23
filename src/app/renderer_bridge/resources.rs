use super::super::*;

impl OmeZarrViewerApp {
    pub fn control_labels_json(&self) -> serde_json::Value {
        let mut available = self.seg_label_names.clone();
        if self.dataset.is_root_label_mask() {
            let name = LabelZarrDataset::root_label_name(&self.dataset);
            if !available.contains(&name) {
                available.push(name);
            }
        }
        serde_json::json!({
            "available": available,
            "selected": self.seg_label_selected,
            "loaded": self.label_cells.as_ref().map(|labels| labels.label_name.clone()),
            "visible": self.cells_outlines_visible,
            "busy": self.labels_gl.as_ref().is_some_and(|labels| labels.is_busy()),
            "gpu_available": self.tiles_gl.is_some(),
            "status": self.seg_label_status,
            "offset_world": [self.seg_labels_offset_world.x, self.seg_labels_offset_world.y],
            "generation": self.control_actor_label_generation.max(1),
            "actor_owned": self.control_actor_label_generation > 0,
        })
    }
}

#[cfg(test)]
impl OmeZarrViewerApp {
    pub fn control_load_labels(&mut self, params: &serde_json::Value) -> serde_json::Value {
        let name = params
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .unwrap_or(self.seg_label_selected.as_str())
            .to_string();
        if name.is_empty() {
            return serde_json::json!({"error": "label name is required because this dataset has no default label group"});
        }
        match self.load_segmentation_labels(&name) {
            Ok(()) => {
                self.cells_outlines_visible = true;
                self.seg_label_status = format!("Loaded labels/{name}.");
                self.control_labels_json()
            }
            Err(error) => {
                serde_json::json!({"error": format!("load labels/{name} failed: {error}")})
            }
        }
    }

    pub fn control_unload_labels(&mut self) -> serde_json::Value {
        let unloaded = self.label_cells.take().map(|labels| labels.label_name);
        self.label_loader = None;
        self.label_cells_xform = None;
        self.cells_outlines_visible = false;
        if let Some(labels) = self.labels_gl.as_ref() {
            labels.reset();
        }
        self.seg_label_status = "Unloaded segmentation labels.".to_string();
        serde_json::json!({"unloaded": unloaded, "labels": self.control_labels_json()})
    }
}
