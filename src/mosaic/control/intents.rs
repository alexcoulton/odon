use super::super::*;

impl MosaicViewerApp {
    pub fn set_control_actor_owned(&mut self, owned: bool) {
        self.control_actor_owned = owned;
        if !owned {
            self.native_control_intents.clear();
        }
    }

    pub fn take_native_control_intents(&mut self) -> Vec<NativeControlIntent> {
        std::mem::take(&mut self.native_control_intents)
    }

    pub(in crate::mosaic) fn submit_native_control_intent(
        &mut self,
        method: &'static str,
        params: serde_json::Value,
    ) -> bool {
        if !self.control_actor_owned {
            return false;
        }
        self.native_control_intents
            .push(NativeControlIntent { method, params });
        true
    }

    pub(in crate::mosaic) fn layout_command_params(&self) -> serde_json::Value {
        serde_json::json!({
            "group_by":self.group_by,
            "sort_by":self.sort_by,
            "sort_secondary_enabled":self.sort_secondary_enabled,
            "sort_by_secondary":self.sort_by_secondary,
            "layout":self.layout_mode.storage_key(),
            "columns":self.grid_cols,
            "group_gap":self.group_gap,
            "show_group_labels":self.show_group_labels,
            "show_text_labels":self.show_text_labels,
            "label_columns":self.label_columns,
            "fit":false,
        })
    }

    pub(in crate::mosaic) fn submit_layout_value(
        &mut self,
        key: &str,
        value: serde_json::Value,
    ) -> bool {
        if !self.control_actor_owned {
            return false;
        }
        let mut params = self.layout_command_params();
        params[key] = value;
        self.submit_native_control_intent("mosaic.layout.configure", params)
    }

    pub(in crate::mosaic) fn submit_camera_preview_if_changed(
        &mut self,
        before: &serde_json::Value,
    ) {
        if !self.control_actor_owned {
            return;
        }
        let after = self.control_camera_snapshot();
        if &after != before {
            self.submit_native_control_intent("viewer.camera.set", after);
        }
    }
}
