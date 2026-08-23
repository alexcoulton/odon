//! Native remote-dataset dialog and actor-backed S3/HTTP operations.

use super::*;

impl RootApp {
    pub(super) fn start_remote_s3_connect(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let browse_prefix = if self.remote_s3_prefix.trim().ends_with(".ome.zarr")
            || self.remote_s3_prefix.trim().ends_with(".zarr")
        {
            self.remote_s3_prefix
                .trim()
                .trim_matches('/')
                .rsplit_once('/')
                .map(|(parent, _)| parent.to_string())
                .unwrap_or_default()
        } else {
            self.remote_s3_prefix.trim().trim_matches('/').to_string()
        };
        let reply = self
            .control_runtime
            .submit_native_command_with_reply(
                ctx,
                "datasets.s3.configure_session",
                serde_json::json!({
                    "endpoint":self.remote_s3_endpoint,
                    "region":self.remote_s3_region,
                    "bucket":self.remote_s3_bucket,
                    "access_key":self.remote_s3_access_key,
                    "secret_key":self.remote_s3_secret_key,
                }),
            )
            .ok_or_else(|| "Could not submit S3 session configuration".to_string())?;
        self.remote_control_pending = Some(RootRemoteControlPending::Configure {
            reply,
            browse_prefix,
        });
        self.remote_status = "Connecting to S3...".to_string();
        Ok(())
    }

    pub(super) fn start_remote_s3_list(
        &mut self,
        ctx: &egui::Context,
        prefix: String,
    ) -> Result<(), String> {
        if let Some(listing) = self
            .remote_s3_browser
            .as_ref()
            .and_then(|state| state.listing_cache.get(&prefix))
            .cloned()
        {
            self.apply_actor_remote_s3_listing(listing);
            self.remote_status.clear();
            return Ok(());
        }
        let reply = self
            .control_runtime
            .submit_native_command_with_reply(
                ctx,
                "datasets.s3.list",
                serde_json::json!({"prefix":prefix}),
            )
            .ok_or_else(|| "Could not submit S3 listing request".to_string())?;
        self.remote_control_pending = Some(RootRemoteControlPending::List { reply });
        self.remote_status = "Listing S3 prefix...".to_string();
        Ok(())
    }

    pub(super) fn start_remote_single_open(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let (method, params) = match self.remote_mode {
            RemoteMode::Http => {
                let mut url = self
                    .remote_http_url
                    .trim()
                    .trim_end_matches('/')
                    .to_string();
                if url.is_empty() {
                    return Err("URL is empty".to_string());
                }
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{url}");
                }
                ("datasets.open_http", serde_json::json!({"url":url}))
            }
            RemoteMode::S3 => (
                "datasets.open_s3",
                serde_json::json!({
                    "prefix":self.remote_s3_prefix.trim().trim_matches('/'),
                }),
            ),
        };
        let reply = self
            .control_runtime
            .submit_native_command_with_reply(ctx, method, params)
            .ok_or_else(|| "Could not submit remote dataset open".to_string())?;
        self.remote_control_pending = Some(RootRemoteControlPending::Open { reply });
        self.remote_status = "Opening remote OME-Zarr...".to_string();
        Ok(())
    }

    pub(super) fn apply_actor_remote_s3_listing(&mut self, listing: S3BrowseListing) {
        let (selected_dataset_prefixes, mut listing_cache) = self
            .remote_s3_browser
            .take()
            .map(|state| (state.selected_dataset_prefixes, state.listing_cache))
            .unwrap_or_default();
        listing_cache.insert(listing.prefix.clone(), listing.clone());
        self.remote_s3_browser = Some(RootRemoteS3BrowserState {
            current_prefix: listing.prefix,
            parent_prefix: listing.parent_prefix,
            entries: listing.entries,
            current_is_dataset: listing.current_is_dataset,
            selected_dataset_prefixes,
            listing_cache,
        });
    }

    pub(super) fn poll_remote_control_pending(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.remote_control_pending.as_ref() else {
            return;
        };
        let result = match pending {
            RootRemoteControlPending::Configure { reply, .. }
            | RootRemoteControlPending::List { reply }
            | RootRemoteControlPending::Open { reply } => reply.try_recv(),
        };
        let result = match result {
            Ok(result) => result,
            Err(crossbeam_channel::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(16));
                return;
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.remote_control_pending = None;
                self.remote_status = "Remote control request disconnected".to_string();
                return;
            }
        };
        let pending = self
            .remote_control_pending
            .take()
            .expect("pending remote operation was inspected above");
        match (pending, result) {
            (RootRemoteControlPending::Configure { browse_prefix, .. }, Ok(_)) => {
                if let Err(error) = self.start_remote_s3_list(ctx, browse_prefix) {
                    self.remote_status = error;
                }
            }
            (RootRemoteControlPending::List { .. }, Ok(response)) => {
                match serde_json::from_value::<S3BrowseListing>(response) {
                    Ok(listing) => {
                        self.apply_actor_remote_s3_listing(listing);
                        self.remote_status.clear();
                    }
                    Err(error) => {
                        self.remote_status = format!("Invalid S3 listing response: {error}");
                    }
                }
            }
            (RootRemoteControlPending::Open { .. }, Ok(_)) => {
                self.remote_dialog_open = false;
                self.remote_status.clear();
            }
            (_, Err(error)) => self.remote_status = error.to_string(),
        }
    }

    pub(super) fn selected_remote_s3_datasets(&self) -> Vec<S3DatasetSelection> {
        let Some(state) = self.remote_s3_browser.as_ref() else {
            return Vec::new();
        };
        let endpoint = self.remote_s3_endpoint.trim().to_string();
        let region = self.remote_s3_region.trim().to_string();
        let bucket = self.remote_s3_bucket.trim().to_string();
        let mut prefixes = state
            .selected_dataset_prefixes
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        prefixes.sort();
        prefixes
            .into_iter()
            .map(|prefix| S3DatasetSelection {
                endpoint: endpoint.clone(),
                region: region.clone(),
                bucket: bucket.clone(),
                prefix,
            })
            .collect()
    }

    pub(super) fn ui_remote_dialog(&mut self, ctx: &egui::Context) -> Option<RootRemoteAction> {
        self.poll_remote_control_pending(ctx);
        if !self.remote_dialog_open {
            return None;
        }
        let remote_busy = self.remote_control_pending.is_some();
        let mut open = self.remote_dialog_open;
        let mut s3_inputs_changed = false;
        let mut connect_s3 = false;
        let mut refresh_s3 = false;
        let mut browse_to: Option<String> = None;
        let mut open_single = false;
        let mut open_mosaic = false;
        let mut add_to_project = false;
        let mut action = None;
        egui::Window::new("Open Remote OME-Zarr")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::Http, "HTTP(S)");
                    ui.selectable_value(&mut self.remote_mode, RemoteMode::S3, "S3 / R2");
                });
                ui.separator();
                match self.remote_mode {
                    RemoteMode::Http => {
                        ui.label("Dataset URL (points to the OME-Zarr directory):");
                        ui.text_edit_singleline(&mut self.remote_http_url);
                    }
                    RemoteMode::S3 => {
                        ui.label("Endpoint (R2): https://<accountid>.r2.cloudflarestorage.com");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_endpoint)
                            .changed();
                        ui.horizontal(|ui| {
                            ui.label("Region:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_region)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Bucket:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_bucket)
                                .changed();
                        });
                        ui.label("Prefix (path to the OME-Zarr directory within the bucket):");
                        s3_inputs_changed |= ui
                            .text_edit_singleline(&mut self.remote_s3_prefix)
                            .changed();
                        ui.separator();
                        ui.label("Credentials (static):");
                        ui.horizontal(|ui| {
                            ui.label("Access key:");
                            s3_inputs_changed |= ui
                                .text_edit_singleline(&mut self.remote_s3_access_key)
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Secret key:");
                            s3_inputs_changed |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.remote_s3_secret_key)
                                        .password(true),
                                )
                                .changed();
                        });
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            let connect_label = if self.remote_s3_browser.is_some() {
                                "Reconnect"
                            } else {
                                "Connect"
                            };
                            if ui
                                .add_enabled(!remote_busy, egui::Button::new(connect_label))
                                .clicked()
                            {
                                connect_s3 = true;
                            }
                            if ui
                                .add_enabled(
                                    !remote_busy && self.remote_s3_browser.is_some(),
                                    egui::Button::new("Refresh"),
                                )
                                .clicked()
                            {
                                refresh_s3 = true;
                            }
                        });
                        let browser_view = self.remote_s3_browser.as_ref().map(|state| {
                            (
                                state.current_prefix.clone(),
                                state.parent_prefix.clone(),
                                state.current_is_dataset,
                                state.entries.clone(),
                                state.selected_dataset_prefixes.clone(),
                            )
                        });
                        if let Some((
                            current_prefix,
                            parent_prefix,
                            current_is_dataset,
                            entries,
                            mut selected_prefixes,
                        )) = browser_view
                        {
                            ui.add_space(6.0);
                            ui.separator();
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_min_width(620.0);
                                ui.horizontal(|ui| {
                                    ui.label("Browser");
                                    ui.label(if current_prefix.is_empty() {
                                        "<bucket root>".to_string()
                                    } else {
                                        current_prefix.clone()
                                    });
                                    if ui
                                        .add_enabled(
                                            parent_prefix.is_some() || !current_prefix.is_empty(),
                                            egui::Button::new("Up"),
                                        )
                                        .clicked()
                                    {
                                        browse_to = Some(parent_prefix.unwrap_or_default());
                                    }
                                });
                                if current_is_dataset {
                                    ui.horizontal(|ui| {
                                        let mut selected =
                                            selected_prefixes.contains(&current_prefix);
                                        if ui.checkbox(&mut selected, "Select current").changed() {
                                            if selected {
                                                selected_prefixes.insert(current_prefix.clone());
                                            } else {
                                                selected_prefixes.remove(&current_prefix);
                                            }
                                        }
                                        ui.label("This prefix looks like an OME-Zarr dataset.");
                                        if ui.button("Use this prefix").clicked() {
                                            self.remote_s3_prefix = current_prefix.clone();
                                        }
                                    });
                                }
                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.add_sized([28.0, 18.0], egui::Label::new("Sel"));
                                    ui.small("Name");
                                });
                                egui::ScrollArea::vertical()
                                    .auto_shrink([false, false])
                                    .max_height(260.0)
                                    .show(ui, |ui| {
                                        ui.set_min_width(ui.available_width());
                                        for entry in &entries {
                                            ui.horizontal(|ui| {
                                                if entry.is_dataset {
                                                    let mut selected =
                                                        selected_prefixes.contains(&entry.prefix);
                                                    if ui
                                                        .add_sized(
                                                            [28.0, 20.0],
                                                            egui::Checkbox::without_text(
                                                                &mut selected,
                                                            ),
                                                        )
                                                        .on_hover_text("Select this OME-Zarr")
                                                        .changed()
                                                    {
                                                        if selected {
                                                            selected_prefixes
                                                                .insert(entry.prefix.clone());
                                                        } else {
                                                            selected_prefixes.remove(&entry.prefix);
                                                        }
                                                    }
                                                    if ui
                                                        .selectable_label(
                                                            self.remote_s3_prefix.trim()
                                                                == entry.prefix,
                                                            format!(
                                                                "[{}] {}",
                                                                if entry
                                                                    .prefix
                                                                    .ends_with(".ome.zarr")
                                                                {
                                                                    "OME-Zarr"
                                                                } else {
                                                                    "Zarr"
                                                                },
                                                                entry.name
                                                            ),
                                                        )
                                                        .clicked()
                                                    {
                                                        self.remote_s3_prefix =
                                                            entry.prefix.clone();
                                                    }
                                                    if ui.small_button("Browse").clicked() {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                } else {
                                                    ui.add_space(28.0);
                                                    if ui
                                                        .button(format!("[dir] {}", entry.name))
                                                        .clicked()
                                                    {
                                                        browse_to = Some(entry.prefix.clone());
                                                    }
                                                }
                                            });
                                        }
                                    });
                            });
                            if let Some(state) = self.remote_s3_browser.as_mut() {
                                state.selected_dataset_prefixes = selected_prefixes;
                            }
                        }
                    }
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        self.remote_dialog_open = false;
                        self.remote_status.clear();
                    }
                    if ui
                        .add_enabled(!remote_busy, egui::Button::new("Open"))
                        .clicked()
                    {
                        open_single = true;
                    }
                    let selected_remote = self.selected_remote_s3_datasets();
                    if ui
                        .add_enabled(
                            self.remote_mode == RemoteMode::S3 && selected_remote.len() >= 2,
                            egui::Button::new(format!("Open Mosaic ({})", selected_remote.len())),
                        )
                        .clicked()
                    {
                        open_mosaic = true;
                    }
                    if ui
                        .add_enabled(
                            self.remote_mode == RemoteMode::S3 && !selected_remote.is_empty(),
                            egui::Button::new(format!(
                                "Add to Project ({})",
                                selected_remote.len()
                            )),
                        )
                        .clicked()
                    {
                        add_to_project = true;
                    }
                });
                if !self.remote_status.is_empty() {
                    ui.add_space(6.0);
                    ui.label(self.remote_status.clone());
                }
            });

        if s3_inputs_changed {
            self.clear_remote_s3_browser();
            self.remote_control_pending = None;
            let _ = self.control_runtime.submit_native_command(
                ctx,
                "datasets.s3.clear_session",
                serde_json::json!({}),
            );
        }
        if connect_s3 {
            if let Err(err) = self.start_remote_s3_connect(ctx) {
                self.remote_status = err;
            }
        } else if refresh_s3 {
            let prefix = self
                .remote_s3_browser
                .as_ref()
                .map(|state| state.current_prefix.clone())
                .unwrap_or_default();
            if let Err(err) = self.start_remote_s3_list(ctx, prefix) {
                self.remote_status = err;
            }
        } else if let Some(prefix) = browse_to {
            if let Err(err) = self.start_remote_s3_list(ctx, prefix) {
                self.remote_status = err;
            }
        } else if open_single {
            if let Err(err) = self.start_remote_single_open(ctx) {
                self.remote_status = err;
            }
        } else if open_mosaic {
            let selected = self.selected_remote_s3_datasets();
            if selected.len() >= 2 {
                self.remote_dialog_open = false;
                self.remote_status.clear();
                action = Some(RootRemoteAction::OpenS3Mosaic(selected));
            } else {
                self.remote_status = "Select at least 2 S3 OME-Zarr datasets.".to_string();
            }
        } else if add_to_project {
            let sources = self
                .selected_remote_s3_datasets()
                .into_iter()
                .map(|dataset| DatasetSource::S3 {
                    endpoint: dataset.endpoint,
                    region: dataset.region,
                    bucket: dataset.bucket,
                    prefix: dataset.prefix,
                })
                .collect::<Vec<_>>();
            if sources.is_empty() {
                self.remote_status = "Select at least 1 S3 OME-Zarr dataset.".to_string();
            } else {
                self.remote_dialog_open = false;
                self.remote_status.clear();
                action = Some(RootRemoteAction::AddToProject(sources));
            }
        }

        self.remote_dialog_open = open && self.remote_dialog_open;
        if !open {
            self.remote_dialog_open = false;
        }
        action
    }
}
