use super::super::*;

impl MosaicViewerApp {
    pub fn set_return_dataset_root(&mut self, root: Option<PathBuf>) {
        self.return_dataset_root = root;
    }

    pub fn take_platform_effect(&mut self) -> Option<MosaicPlatformEffect> {
        self.pending_platform_effect.take()
    }

    pub fn set_fast_object_rendering(&mut self, enabled: bool) {
        self.submit_native_control_intent(
            "viewer.objects.rendering.set_fast",
            serde_json::json!({"enabled":enabled}),
        );
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
