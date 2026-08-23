use super::super::*;

impl MosaicViewerApp {
    pub fn take_request(&mut self) -> Option<MosaicRequest> {
        self.pending_request.take()
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn set_fast_object_rendering(&mut self, enabled: bool) {
        if !self.submit_native_control_intent(
            "viewer.objects.rendering.set_fast",
            serde_json::json!({"enabled":enabled}),
        ) {
            self.seg_geojson.set_fast_object_rendering(enabled);
        }
    }

    pub fn confirm_or_request_close_dialog(&mut self) -> bool {
        if self.close_dialog_open {
            self.close_dialog_open = false;
            return true;
        }
        self.close_dialog_open = true;
        false
    }
}
