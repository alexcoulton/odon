use super::*;

impl OmeZarrViewerApp {
    pub fn confirm_or_request_close_dialog(&mut self) -> bool {
        if self.close_dialog_open {
            self.close_dialog_open = false;
            return true;
        }
        self.close_dialog_open = true;
        false
    }
}
