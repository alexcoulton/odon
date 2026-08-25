//! Native realization of actor-resolved application commands.

use super::*;

impl RootApp {
    pub(super) fn execute_native_command(
        &mut self,
        command_id: &str,
        action: &str,
        checked: Option<bool>,
    ) {
        match action {
            "about" => {
                let _ = rfd::MessageDialog::new()
                    .set_title("About odon")
                    .set_description(format!(
                        "Odon {}\nOME-Zarr viewer with a Python-controlled application shell.",
                        env!("CARGO_PKG_VERSION")
                    ))
                    .set_level(rfd::MessageLevel::Info)
                    .show();
            }
            "settings" => self.settings_open = true,
            "open_ome_zarr" => {
                if let Some(root) = FileDialog::new().set_title("Open OME-Zarr").pick_folder() {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "datasets.open_ome_zarr",
                        params: serde_json::json!({"path":root}),
                    });
                }
            }
            "open_tiff" => {
                if let Some(root) = FileDialog::new()
                    .add_filter("TIFF / OME-TIFF", &["tif", "tiff"])
                    .set_title("Open TIFF / OME-TIFF")
                    .pick_file()
                {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "datasets.open_tiff",
                        params: serde_json::json!({"path":root}),
                    });
                }
            }
            "open_project" => {
                if let Some(path) = FileDialog::new()
                    .add_filter("Project JSON", &["json"])
                    .set_title("Load Project")
                    .pick_file()
                {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "project.open",
                        params: serde_json::json!({"path":path}),
                    });
                }
            }
            "save_project" => {
                let save_target = match &self.mode {
                    Mode::Project { project_space } => project_space.saved_project_path(),
                    Mode::Single(app) => app.project_space().saved_project_path(),
                    Mode::Mosaic { mosaic, .. } => mosaic.project_space().saved_project_path(),
                    Mode::Transition => None,
                }
                .or_else(|| {
                    FileDialog::new()
                        .add_filter("Project JSON", &["json"])
                        .set_file_name("odon.project.json")
                        .set_title("Save Project")
                        .save_file()
                });
                if let Some(path) = save_target {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "project.save_as",
                        params: serde_json::json!({"path":path}),
                    });
                }
            }
            "save_new_project" => {
                if let Some(path) = FileDialog::new()
                    .add_filter("Project JSON", &["json"])
                    .set_file_name("odon.project.json")
                    .set_title("Save Project As")
                    .save_file()
                {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: "project.save_as",
                        params: serde_json::json!({"path":path}),
                    });
                }
            }
            "save_screenshot" => self.save_screenshot_via_dialog(),
            "quick_screenshot" => self.quick_screenshot(),
            "screenshot_settings" => match &mut self.mode {
                Mode::Single(app) => app.open_screenshot_settings(),
                Mode::Project { project_space } => project_space
                    .set_status("Screenshot Settings: open a dataset first.".to_string()),
                Mode::Mosaic { mosaic, .. } => mosaic.open_screenshot_settings(),
                Mode::Transition => {}
            },
            "roi_info" => match &mut self.mode {
                Mode::Single(app) => app.open_roi_info_window(),
                Mode::Project { project_space } => {
                    project_space.set_status("ROI Info: open a dataset first.".to_string())
                }
                Mode::Mosaic { mosaic, .. } => mosaic
                    .project_space_mut()
                    .set_status("ROI Info: open a single ROI first.".to_string()),
                Mode::Transition => {}
            },
            "add_annotations" => match &mut self.mode {
                Mode::Single(app) => app.add_annotation_layer_from_menu(),
                Mode::Project { project_space } => {
                    project_space.set_status("Add annotations: open a dataset first.".to_string())
                }
                Mode::Mosaic { mosaic, .. } => mosaic
                    .project_space_mut()
                    .set_status("Add annotations: open a single ROI first.".to_string()),
                Mode::Transition => {}
            },
            "load_seg_geojson" => match &mut self.mode {
                Mode::Single(app) => app.open_seg_geojson_dialog(),
                Mode::Project { project_space } => {
                    project_space.set_status("Load Seg GeoJSON: open a dataset first.".to_string())
                }
                Mode::Mosaic { mosaic, .. } => mosaic
                    .project_space_mut()
                    .set_status("Load Seg GeoJSON: open a single ROI first.".to_string()),
                Mode::Transition => {}
            },
            "load_seg_objects" => match &mut self.mode {
                Mode::Single(app) => app.open_seg_objects_dialog(),
                Mode::Project { project_space } => {
                    project_space.set_status("Load Seg Objects: open a dataset first.".to_string())
                }
                Mode::Mosaic { mosaic, .. } => mosaic
                    .project_space_mut()
                    .set_status("Load Seg Objects: open a single ROI first.".to_string()),
                Mode::Transition => {}
            },
            "export_masks_geojson" => {
                if let Some(path) = FileDialog::new()
                    .add_filter("GeoJSON", &["geojson", "json"])
                    .set_file_name("masks.geojson")
                    .set_title("Export Masks GeoJSON")
                    .save_file()
                {
                    match &mut self.mode {
                        Mode::Single(app) => app.request_mask_export(&path, None),
                        Mode::Project { project_space } => project_space
                            .set_status("Export masks failed: open a dataset first.".to_string()),
                        Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                            "Export masks failed: open a single ROI first.".to_string(),
                        ),
                        Mode::Transition => {}
                    }
                }
            }
            "toggle_scale_bar" => {
                if let Mode::Single(app) = &mut self.mode {
                    let visible = checked.unwrap_or_else(|| !app.scale_bar_visible());
                    app.submit_native_scale_bar_visibility(visible);
                }
            }
            "close_window" | "quit" => {
                let quit = action == "quit";
                let should_close = match &mut self.mode {
                    Mode::Project { .. } => {
                        if self.close_dialog_open {
                            self.close_dialog_open = false;
                            true
                        } else {
                            self.close_dialog_open = true;
                            false
                        }
                    }
                    Mode::Single(app) => app.confirm_or_request_close_dialog(),
                    Mode::Mosaic { mosaic, .. } => mosaic.confirm_or_request_close_dialog(),
                    Mode::Transition => false,
                };
                if should_close {
                    self.native_command_ingress.push(NativeControlIntent {
                        method: if quit {
                            "app.lifecycle.request_quit"
                        } else {
                            "app.lifecycle.request_close"
                        },
                        params: serde_json::json!({"save":"discard"}),
                    });
                }
            }
            _ => log_warn!(
                "actor resolved command '{command_id}' to unsupported native action '{action}'"
            ),
        }
    }
}
