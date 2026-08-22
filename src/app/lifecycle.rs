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

    pub(super) fn configure_root_label_dataset_if_needed(&mut self) {
        if !self.dataset.is_root_label_mask() {
            return;
        }

        self.channels.clear();
        self.channel_offsets_world.clear();
        self.channel_scales.clear();
        self.channel_rotations_rad.clear();
        self.channel_layer_order.clear();
        self.selected_channel_layers.clear();
        self.channel_select_anchor_idx = None;
        self.selected_channel = 0;
        self.chanmax_pending.clear();
        self.chanmax_snapshot.clear();
        self.seg_label_names.clear();
        self.seg_label_prompt_open = false;
        self.active_layer = LayerId::SegmentationLabels;

        match self.load_root_segmentation_labels() {
            Ok(()) => {
                self.seg_label_status =
                    format!("Opened top-level label mask '{}'.", self.seg_label_selected);
            }
            Err(err) => {
                self.label_cells = None;
                self.label_loader = None;
                self.label_cells_xform = None;
                self.cells_outlines_visible = false;
                self.seg_label_selected = LabelZarrDataset::root_label_name(&self.dataset);
                self.seg_label_input = self.seg_label_selected.clone();
                self.seg_label_status = format!("Open top-level label mask failed: {err}");
            }
        }
    }

    pub(super) fn load_root_segmentation_labels(&mut self) -> anyhow::Result<()> {
        if self.tiles_gl.is_none() {
            anyhow::bail!("top-level label masks require the GPU renderer");
        }
        self.labels_gl
            .get_or_insert_with(|| LabelsGl::new(1024))
            .reset();

        let lbl = LabelZarrDataset::from_root_dataset(&self.dataset);
        let label_loader =
            spawn_label_tile_loader(self.store.clone(), lbl.levels.clone(), lbl.dims.clone())?;

        self.label_loader = Some(label_loader);
        self.spatial_label_transform = SpatialDataTransform2::default();
        self.label_cells_xform = Some(compute_label_to_world_xforms(
            &self.dataset,
            &lbl,
            self.spatial_label_transform,
        ));
        self.seg_label_names.clear();
        self.seg_label_selected = lbl.label_name.clone();
        self.seg_label_input = self.seg_label_selected.clone();
        self.label_cells = Some(lbl);
        self.cells_outlines_visible = true;
        self.seg_label_prompt_open = false;
        Ok(())
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
