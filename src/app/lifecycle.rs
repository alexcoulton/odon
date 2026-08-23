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

    pub(super) fn maybe_apply_auto_contrast_on_open(&mut self) {
        if self.auto_contrast_settings.enabled_on_open {
            self.request_auto_contrast(false);
        }
    }

    pub(super) fn request_auto_contrast(&mut self, overwrite_manual: bool) {
        if self.channels.is_empty() {
            return;
        }

        self.chanmax_request_id = self.chanmax_request_id.wrapping_add(1).max(1);
        if overwrite_manual {
            self.channel_window_overrides.clear();
            self.chanmax_pending = vec![true; self.channels.len()];
        } else {
            self.chanmax_pending = self
                .channels
                .iter()
                .map(|c| !self.channel_window_overrides.contains_key(&c.name))
                .collect();
        }

        if !self.chanmax_pending.iter().any(|pending| *pending) {
            return;
        }

        // One epoch for all channels; ignore stale responses on ROI switches.
        let request_id = self.chanmax_request_id;
        let level = self
            .chanmax_level
            .min(self.dataset.levels.len().saturating_sub(1));
        self.chanmax_snapshot = self.channels.iter().map(|c| c.window).collect();
        for (i, ch) in self.channels.iter().enumerate() {
            if !self.chanmax_pending.get(i).copied().unwrap_or(false) {
                continue;
            }
            let _ = self.chanmax_loader.tx.send(ChannelMaxRequest {
                request_id,
                view: self.active_view_selection(),
                level,
                channel: ch.index as u64,
                settings: self.auto_contrast_settings,
            });
        }
    }
}
