use super::super::*;

impl MosaicViewerApp {
    #[cfg(test)]
    pub fn take_native_control_intents(&mut self) -> Vec<NativeControlIntent> {
        self.native_command_ingress.take_recorded()
    }

    pub fn set_native_command_ingress(&mut self, ingress: NativeControlIngress) {
        self.native_command_ingress = ingress;
    }

    pub(in crate::mosaic) fn submit_native_control_intent(
        &mut self,
        method: &'static str,
        params: serde_json::Value,
    ) -> bool {
        self.native_command_ingress
            .push(NativeControlIntent { method, params })
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
            "label_font_size":self.label_font_size,
            "group_label_font_size":self.group_label_font_size,
            "fit":false,
        })
    }

    pub(in crate::mosaic) fn submit_layout_value(
        &mut self,
        key: &str,
        value: serde_json::Value,
    ) -> bool {
        let mut params = self.layout_command_params();
        params[key] = value;
        self.submit_native_control_intent("mosaic.layout.configure", params)
    }

    pub(in crate::mosaic) fn submit_camera_preview_if_changed(
        &mut self,
        before: &serde_json::Value,
    ) {
        let after = self.control_camera_snapshot();
        if &after == before {
            return;
        }
        // The camera was already moved locally. Keep it authoritative until the actor catches up
        // so a projection of an earlier step cannot rewind the gesture, and keep at most one
        // camera command in flight.
        let camera = self.camera.clone();
        match self.native_camera_hold.as_mut() {
            Some(hold) => hold.record_local(camera),
            None => self.native_camera_hold = Some(crate::camera::CameraIntentHold::new(camera)),
        }
        self.flush_native_camera_hold();
    }

    /// Drops a hold that settled while the renderer was idle, so the next projection applies.
    pub(in crate::mosaic) fn release_settled_native_camera_hold(&mut self) {
        let in_flight = self
            .native_command_ingress
            .contains_pending("viewer.camera.set");
        if self
            .native_camera_hold
            .as_ref()
            .is_some_and(|hold| hold.settled(in_flight) && !hold.needs_wake())
        {
            self.native_camera_hold = None;
        }
    }

    fn flush_native_camera_hold(&mut self) {
        let in_flight = self
            .native_command_ingress
            .contains_pending("viewer.camera.set");
        let Some(camera) = self
            .native_camera_hold
            .as_mut()
            .and_then(|hold| hold.take_unsent(in_flight))
        else {
            return;
        };
        let params = serde_json::json!({
            "center_world_lvl0": [camera.center_world_lvl0.x, camera.center_world_lvl0.y],
            "zoom_screen_per_lvl0_px": camera.zoom_screen_per_lvl0_px,
        });
        if !self.submit_native_control_intent("viewer.camera.set", params)
            && let Some(hold) = self.native_camera_hold.as_mut()
        {
            hold.mark_unsent();
        }
    }

    /// Sends the newest unsent gesture camera and releases the hold once the actor caught up.
    pub(in crate::mosaic) fn tick_native_camera_hold(&mut self, ctx: &egui::Context) {
        self.flush_native_camera_hold();
        let in_flight = self
            .native_command_ingress
            .contains_pending("viewer.camera.set");
        let Some(hold) = self.native_camera_hold.as_ref() else {
            return;
        };
        if !hold.settled(in_flight) {
            if hold.needs_wake() {
                ctx.request_repaint_after(crate::camera::CameraIntentHold::POLL);
            }
            return;
        }
        if let Some(camera) = self
            .native_camera_hold
            .take()
            .and_then(crate::camera::CameraIntentHold::finish)
        {
            // The actor published a different camera while the gesture owned it; adopt it now.
            self.camera = camera;
            ctx.request_repaint();
        }
    }
}
