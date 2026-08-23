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
