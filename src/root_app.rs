use std::path::PathBuf;
use std::sync::Arc;
use std::{collections::HashMap, collections::HashSet};

use eframe::egui;

use crate::app::{
    LabelPromptSessionPreference, OmeZarrViewerApp, S3DatasetSelection, ViewerRequest,
};
use crate::app_support::menu::{NativeMenu, NativeMenuAction};
use crate::data::dataset_kind::{
    LocalDatasetKind, classify_local_dataset_path, normalize_local_dataset_path,
};
use crate::data::dataset_source::DatasetSource;
use crate::data::ome::OmeZarrDataset;
use crate::data::project_config::ProjectRoi;
use crate::data::remote_store::{
    S3BrowseEntry, S3BrowseListing, S3Browser, S3Store, build_http_store, build_s3_browser,
    build_s3_store, list_s3_prefix,
};
use crate::mosaic::{MosaicRequest, MosaicViewerApp};
use crate::project::{ProjectSpace, ProjectSpaceAction};
use crate::spatialdata::{SpatialDataDiscovery, discover_spatialdata};
use crate::ui::top_bar;
use crate::xenium::discover_xenium_explorer;
use crate::{log_debug, log_info, log_warn};
use rfd::FileDialog;

#[derive(Debug, Clone)]
struct SpatialOpenDialog {
    discovery: SpatialDataDiscovery,
    selected_image: usize,
    selected_labels: Option<usize>,
    selected_shapes: Vec<usize>,
    selected_points: Option<usize>,
    points_max: usize,
    status: String,
}

struct ReturnToSingleState {
    dataset_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteMode {
    Http,
    S3,
}

struct RootRemoteS3BrowserState {
    session: S3Browser,
    signature: String,
    current_prefix: String,
    parent_prefix: Option<String>,
    entries: Vec<S3BrowseEntry>,
    current_is_dataset: bool,
    selected_dataset_prefixes: HashSet<String>,
    listing_cache: HashMap<String, S3BrowseListing>,
}

enum RootRemoteAction {
    OpenSingle {
        dataset: OmeZarrDataset,
        store: Arc<dyn zarrs::storage::ReadableStorageTraits>,
        runtime: Option<Arc<tokio::runtime::Runtime>>,
    },
    OpenS3Mosaic(Vec<S3DatasetSelection>),
    AddToProject(Vec<DatasetSource>),
}

enum Mode {
    Project {
        project_space: ProjectSpace,
    },
    Single(OmeZarrViewerApp),
    Mosaic {
        mosaic: MosaicViewerApp,
        ret: ReturnToSingleState,
    },
    Transition,
}

pub struct RootApp {
    mode: Mode,
    gpu_available: bool,
    close_dialog_open: bool,
    spatial_open: Option<SpatialOpenDialog>,
    pending_open_root: Option<PathBuf>,
    view_show_scale_bar: bool,
    remote_dialog_open: bool,
    remote_mode: RemoteMode,
    remote_http_url: String,
    remote_s3_endpoint: String,
    remote_s3_region: String,
    remote_s3_bucket: String,
    remote_s3_prefix: String,
    remote_s3_access_key: String,
    remote_s3_secret_key: String,
    remote_status: String,
    remote_s3_browser: Option<RootRemoteS3BrowserState>,
    label_prompt_preference: LabelPromptSessionPreference,
    #[cfg(target_os = "macos")]
    native_menu: Option<NativeMenu>,
}

impl RootApp {
    fn remote_s3_signature(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}",
            self.remote_s3_endpoint.trim(),
            self.remote_s3_region.trim(),
            self.remote_s3_bucket.trim(),
            self.remote_s3_access_key.trim(),
            self.remote_s3_secret_key.trim()
        )
    }

    fn clear_remote_s3_browser(&mut self) {
        self.remote_s3_browser = None;
    }

    fn save_screenshot_via_dialog(&mut self) {
        let default_name = match &self.mode {
            Mode::Single(app) => app.default_screenshot_filename(),
            Mode::Mosaic { mosaic, .. } => mosaic.default_screenshot_filename(),
            _ => "odon.screenshot.png".to_string(),
        };
        if let Some(path) = FileDialog::new()
            .add_filter("PNG", &["png"])
            .set_file_name(&default_name)
            .set_title("Save Screenshot (Canvas PNG)")
            .save_file()
        {
            match &mut self.mode {
                Mode::Single(app) => {
                    app.request_screenshot_png(path);
                }
                Mode::Project { project_space } => {
                    project_space.set_status("Save Screenshot: open a dataset first.".to_string());
                }
                Mode::Mosaic { mosaic, .. } => {
                    mosaic.request_screenshot_png(path);
                }
                Mode::Transition => {}
            }
        }
    }

    fn quick_screenshot(&mut self) {
        let mut fallback_to_dialog = false;
        match &mut self.mode {
            Mode::Single(app) => {
                if app.screenshot_output_dir().is_some() {
                    if let Err(err) = app.request_quick_screenshot_png() {
                        app.set_status(format!("Quick screenshot failed: {err}"));
                    }
                } else {
                    fallback_to_dialog = true;
                }
            }
            Mode::Project { project_space } => {
                project_space.set_status("Save Screenshot: open a dataset first.".to_string());
            }
            Mode::Mosaic { mosaic, .. } => {
                if mosaic.screenshot_output_dir().is_some() {
                    if let Err(err) = mosaic.request_quick_screenshot_png() {
                        mosaic.set_status(format!("Quick screenshot failed: {err}"));
                    }
                } else {
                    fallback_to_dialog = true;
                }
            }
            Mode::Transition => {}
        }
        if fallback_to_dialog {
            self.save_screenshot_via_dialog();
        }
    }

    fn apply_remote_s3_listing(
        &mut self,
        session: S3Browser,
        signature: String,
        listing: S3BrowseListing,
        selected_dataset_prefixes: HashSet<String>,
    ) {
        let current_prefix = listing.prefix.clone();
        let parent_prefix = listing.parent_prefix.clone();
        let entries = listing.entries.clone();
        let current_is_dataset = listing.current_is_dataset;
        self.remote_s3_browser = Some(RootRemoteS3BrowserState {
            session,
            signature,
            current_prefix: current_prefix.clone(),
            parent_prefix,
            entries,
            current_is_dataset,
            selected_dataset_prefixes,
            listing_cache: HashMap::new(),
        });
        if let Some(state) = self.remote_s3_browser.as_mut() {
            state.listing_cache.insert(current_prefix, listing);
        }
    }

    fn connect_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let browser = build_s3_browser(
            &self.remote_s3_endpoint,
            &self.remote_s3_region,
            &self.remote_s3_bucket,
            &self.remote_s3_access_key,
            &self.remote_s3_secret_key,
        )?;
        let signature = self.remote_s3_signature();
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
        let listing = list_s3_prefix(&browser, &browse_prefix)?;
        self.apply_remote_s3_listing(browser, signature, listing, Default::default());
        Ok(())
    }

    fn refresh_remote_s3_browser(&mut self) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let listing = list_s3_prefix(&state.session, &state.current_prefix)?;
        let mut selected = state.selected_dataset_prefixes;
        let mut cache = state.listing_cache;
        cache.insert(listing.prefix.clone(), listing.clone());
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected.clone());
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.selected_dataset_prefixes = std::mem::take(&mut selected);
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn browse_remote_s3_prefix(&mut self, prefix: String) -> anyhow::Result<()> {
        let Some(state) = self.remote_s3_browser.take() else {
            anyhow::bail!("not connected to S3");
        };
        let mut selected = state.selected_dataset_prefixes;
        let mut cache = state.listing_cache;
        let listing = if let Some(cached) = cache.get(&prefix).cloned() {
            cached
        } else {
            let listing = list_s3_prefix(&state.session, &prefix)?;
            cache.insert(prefix.clone(), listing.clone());
            listing
        };
        self.apply_remote_s3_listing(state.session, state.signature, listing, selected.clone());
        if let Some(next) = self.remote_s3_browser.as_mut() {
            next.selected_dataset_prefixes = std::mem::take(&mut selected);
            next.listing_cache = cache;
        }
        Ok(())
    }

    fn selected_remote_s3_datasets(&self) -> Vec<S3DatasetSelection> {
        let Some(state) = self.remote_s3_browser.as_ref() else {
            return Vec::new();
        };
        let endpoint = self.remote_s3_endpoint.trim().to_string();
        let region = self.remote_s3_region.trim().to_string();
        let bucket = self.remote_s3_bucket.trim().to_string();
        let access_key = self.remote_s3_access_key.trim().to_string();
        let secret_key = self.remote_s3_secret_key.trim().to_string();
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
                access_key: access_key.clone(),
                secret_key: secret_key.clone(),
            })
            .collect()
    }

    fn open_remote_dataset_from_dialog(&mut self) -> anyhow::Result<RootRemoteAction> {
        match self.remote_mode {
            RemoteMode::Http => {
                let mut url = self
                    .remote_http_url
                    .trim()
                    .trim_end_matches('/')
                    .to_string();
                if url.is_empty() {
                    anyhow::bail!("URL is empty");
                }
                if !url.starts_with("http://") && !url.starts_with("https://") {
                    url = format!("https://{url}");
                }
                let store = build_http_store(&url)?;
                let source = crate::data::dataset_source::DatasetSource::Http { base_url: url };
                let dataset = OmeZarrDataset::open_with_store(source, store.clone())?;
                Ok(RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime: None,
                })
            }
            RemoteMode::S3 => {
                let mut endpoint = self.remote_s3_endpoint.trim().to_string();
                let region = self.remote_s3_region.trim().to_string();
                let bucket = self.remote_s3_bucket.trim().to_string();
                let prefix = self.remote_s3_prefix.trim().trim_matches('/').to_string();
                let access_key = self.remote_s3_access_key.trim().to_string();
                let secret_key = self.remote_s3_secret_key.trim().to_string();
                if endpoint.is_empty() || bucket.is_empty() {
                    anyhow::bail!("endpoint and bucket are required");
                }
                if access_key.is_empty() || secret_key.is_empty() {
                    anyhow::bail!("access key / secret key are required");
                }
                if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
                    endpoint = format!("https://{endpoint}");
                }
                let S3Store { store, runtime } = build_s3_store(
                    &endpoint,
                    &region,
                    &bucket,
                    &prefix,
                    &access_key,
                    &secret_key,
                )?;
                let source = crate::data::dataset_source::DatasetSource::S3 {
                    endpoint,
                    region: if region.is_empty() {
                        "auto".to_string()
                    } else {
                        region
                    },
                    bucket,
                    prefix,
                };
                let dataset = OmeZarrDataset::open_with_store(source, store.clone())?;
                Ok(RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime: Some(runtime),
                })
            }
        }
    }

    fn ui_remote_dialog(&mut self, ctx: &egui::Context) -> Option<RootRemoteAction> {
        if !self.remote_dialog_open {
            return None;
        }
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
                            if ui.button(connect_label).clicked() {
                                connect_s3 = true;
                            }
                            if ui
                                .add_enabled(
                                    self.remote_s3_browser.is_some(),
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
                    if ui.button("Open").clicked() {
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
        }
        if connect_s3 {
            match self.connect_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if refresh_s3 {
            match self.refresh_remote_s3_browser() {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if let Some(prefix) = browse_to {
            match self.browse_remote_s3_prefix(prefix) {
                Ok(()) => self.remote_status.clear(),
                Err(err) => self.remote_status = format!("{err}"),
            }
        } else if open_single {
            match self.open_remote_dataset_from_dialog() {
                Ok(req) => {
                    self.remote_dialog_open = false;
                    self.remote_status.clear();
                    action = Some(req);
                }
                Err(err) => self.remote_status = format!("{err}"),
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

    pub fn new_project(
        cc: &eframe::CreationContext<'_>,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let mut ps = ProjectSpace::default();
        if let Some(path) = project_path.as_deref() {
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
        }
        Ok(Self {
            mode: Mode::Project { project_space: ps },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
    }

    pub fn new_single(
        cc: &eframe::CreationContext<'_>,
        dataset: OmeZarrDataset,
        store: std::sync::Arc<dyn zarrs::storage::ReadableStorageTraits>,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let mut app = OmeZarrViewerApp::new(cc, dataset, store);
        app.set_show_scale_bar(true);
        if let Some(path) = project_path.as_deref() {
            let mut ps = ProjectSpace::default();
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
            app.set_project_space(ps);
        }
        Ok(Self {
            mode: Mode::Single(app),
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
    }

    pub fn new_mosaic(
        cc: &eframe::CreationContext<'_>,
        mut mosaic: MosaicViewerApp,
        project_path: Option<PathBuf>,
    ) -> anyhow::Result<Self> {
        let mut ps = ProjectSpace::default();
        if let Some(path) = project_path.as_deref() {
            if let Err(err) = ps.load_from_file(path) {
                ps.set_status(format!("Load project failed: {err}"));
            }
        }
        mosaic.set_layer_groups(ps.layer_groups().clone());
        Ok(Self {
            mode: Mode::Mosaic {
                mosaic,
                ret: ReturnToSingleState { dataset_root: None },
            },
            gpu_available: cc.gl.is_some(),
            close_dialog_open: false,
            spatial_open: None,
            pending_open_root: None,
            view_show_scale_bar: true,
            remote_dialog_open: false,
            remote_mode: RemoteMode::Http,
            remote_http_url: String::new(),
            remote_s3_endpoint: String::new(),
            remote_s3_region: "auto".to_string(),
            remote_s3_bucket: String::new(),
            remote_s3_prefix: String::new(),
            remote_s3_access_key: String::new(),
            remote_s3_secret_key: String::new(),
            remote_status: String::new(),
            remote_s3_browser: None,
            label_prompt_preference: LabelPromptSessionPreference::Ask,
            #[cfg(target_os = "macos")]
            native_menu: None,
        })
    }

    pub fn queue_open_root(&mut self, root: PathBuf) {
        self.pending_open_root = Some(root);
    }

    pub fn add_paths_to_project(&mut self, paths: Vec<PathBuf>) {
        match &mut self.mode {
            Mode::Project { project_space } => project_space.handle_dropped_paths(paths),
            Mode::Single(app) => {
                let mut ps = app.take_project_space();
                ps.handle_dropped_paths(paths);
                app.set_project_space(ps);
            }
            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().handle_dropped_paths(paths),
            Mode::Transition => {}
        }
    }

    fn open_single(&mut self, ctx: &egui::Context, root: &PathBuf, project_space: ProjectSpace) {
        let root = normalize_local_dataset_path(root).unwrap_or_else(|| root.clone());
        log_info!("open_single: {}", root.to_string_lossy());
        if matches!(
            classify_local_dataset_path(&root),
            Some(LocalDatasetKind::Tiff)
        ) {
            match OmeZarrViewerApp::new_tiff_runtime(ctx, self.gpu_available, root.clone()) {
                Ok(mut app) => {
                    log_debug!("open_single: detected TIFF");
                    app.set_show_scale_bar(self.view_show_scale_bar);
                    app.set_label_prompt_preference(self.label_prompt_preference);
                    app.set_project_space(project_space);
                    self.mode = Mode::Single(app);
                }
                Err(err) => {
                    let mut ps = project_space;
                    log_warn!("open_single: open_tiff failed: {err:?}");
                    ps.set_status(format!("Open TIFF failed: {err}"));
                    self.mode = Mode::Project { project_space: ps };
                }
            }
            return;
        }
        match OmeZarrDataset::open_local(&root) {
            Ok((dataset, store)) => {
                log_debug!("open_single: detected OME-Zarr");
                let mut app =
                    OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
                app.set_show_scale_bar(self.view_show_scale_bar);
                app.set_label_prompt_preference(self.label_prompt_preference);
                app.set_project_space(project_space);
                self.mode = Mode::Single(app);
            }
            Err(err) => {
                // If the root looks like SpatialData, show a chooser for which image group to open
                // (and optional points/shapes overlays).
                match discover_spatialdata(&root) {
                    Ok(discovery) if !discovery.images.is_empty() => {
                        log_debug!("open_single: detected SpatialData");
                        let mut dlg = SpatialOpenDialog {
                            discovery,
                            selected_image: 0,
                            selected_labels: None,
                            selected_shapes: Vec::new(),
                            selected_points: None,
                            points_max: 200_000,
                            status: String::new(),
                        };
                        if let Some(i) = dlg
                            .discovery
                            .labels
                            .iter()
                            .position(|s| s.name == "cells" || s.name == "point8_labels")
                        {
                            dlg.selected_labels = Some(i);
                        }
                        // Default: turn on cell boundaries if present; keep points off by default.
                        if let Some(i) = dlg
                            .discovery
                            .shapes
                            .iter()
                            .position(|s| s.name == "cell_boundaries")
                        {
                            dlg.selected_shapes.push(i);
                        }
                        // Restore the project state so the UI stays intact while the dialog is open.
                        self.mode = Mode::Project { project_space };
                        self.spatial_open = Some(dlg);
                    }
                    _ => {
                        // Xenium Explorer bundle (experiment.xenium + morphology OME-TIFF + zarr.zip overlays).
                        if let Ok(x) = discover_xenium_explorer(&root) {
                            log_debug!("open_single: detected Xenium Explorer");
                            if let Some(img_root) = x.morphology_mip_omezarr.clone() {
                                match OmeZarrDataset::open_local(&img_root) {
                                    Ok((dataset, store)) => {
                                        let mut app = OmeZarrViewerApp::new_runtime(
                                            ctx,
                                            self.gpu_available,
                                            dataset,
                                            store,
                                        );
                                        app.set_show_scale_bar(self.view_show_scale_bar);
                                        app.set_label_prompt_preference(
                                            self.label_prompt_preference,
                                        );
                                        app.attach_xenium_layers(
                                            x.root.clone(),
                                            x.cells_zarr_zip.clone(),
                                            x.transcripts_zarr_zip.clone(),
                                            x.pixel_size_um,
                                        );
                                        app.set_project_space(project_space);
                                        self.mode = Mode::Single(app);
                                        return;
                                    }
                                    Err(e) => {
                                        let mut ps = project_space;
                                        ps.set_status(format!(
                                            "Open Xenium failed: could not open morphology OME-Zarr: {e}"
                                        ));
                                        self.mode = Mode::Project { project_space: ps };
                                        return;
                                    }
                                }
                            } else {
                                if let Some(morph_tiff) = x.morphology_mip_tiff.clone() {
                                    match OmeZarrViewerApp::new_xenium_runtime(
                                        ctx,
                                        self.gpu_available,
                                        x.root.clone(),
                                        morph_tiff,
                                        x.cells_zarr_zip.clone(),
                                        x.transcripts_zarr_zip.clone(),
                                        x.pixel_size_um,
                                    ) {
                                        Ok(mut app) => {
                                            app.set_show_scale_bar(self.view_show_scale_bar);
                                            app.set_project_space(project_space);
                                            self.mode = Mode::Single(app);
                                            return;
                                        }
                                        Err(e) => {
                                            let mut ps = project_space;
                                            ps.set_status(format!(
                                                "Open Xenium failed: could not open morphology OME-TIFF: {e}"
                                            ));
                                            self.mode = Mode::Project { project_space: ps };
                                            return;
                                        }
                                    }
                                }
                                let mut ps = project_space;
                                ps.set_status(
                                    "Open Xenium failed: morphology base image was not found as OME-Zarr or OME-TIFF."
                                        .to_string(),
                                );
                                self.mode = Mode::Project { project_space: ps };
                                return;
                            }
                        }

                        let mut ps = project_space;
                        log_warn!("open_single: open_local failed: {err:?}");
                        ps.set_status(format!("Open failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
        }
    }

    fn open_dataset_source(
        &mut self,
        ctx: &egui::Context,
        source: DatasetSource,
        project_space: ProjectSpace,
    ) {
        match source {
            DatasetSource::Local(path) => self.open_single(ctx, &path, project_space),
            DatasetSource::Http { base_url } => {
                match build_http_store(&base_url).and_then(|store| {
                    OmeZarrDataset::open_with_store(
                        DatasetSource::Http {
                            base_url: base_url.clone(),
                        },
                        store.clone(),
                    )
                    .map(|dataset| (dataset, store))
                }) {
                    Ok((dataset, store)) => {
                        let mut app =
                            OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
                        app.set_show_scale_bar(self.view_show_scale_bar);
                        app.set_label_prompt_preference(self.label_prompt_preference);
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    }
                    Err(err) => {
                        let mut ps = project_space;
                        ps.set_status(format!("Open remote dataset failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
            DatasetSource::S3 {
                endpoint,
                region,
                bucket,
                prefix,
            } => {
                if self.remote_s3_access_key.trim().is_empty()
                    || self.remote_s3_secret_key.trim().is_empty()
                {
                    let mut ps = project_space;
                    ps.set_status(
                        "S3 credentials are not available in this session. Use Open Remote... and reconnect first."
                            .to_string(),
                    );
                    self.mode = Mode::Project { project_space: ps };
                    return;
                }
                match build_s3_store(
                    &endpoint,
                    &region,
                    &bucket,
                    &prefix,
                    &self.remote_s3_access_key,
                    &self.remote_s3_secret_key,
                )
                .and_then(|S3Store { store, runtime }| {
                    OmeZarrDataset::open_with_store(
                        DatasetSource::S3 {
                            endpoint: endpoint.clone(),
                            region: region.clone(),
                            bucket: bucket.clone(),
                            prefix: prefix.clone(),
                        },
                        store.clone(),
                    )
                    .map(|dataset| (dataset, store, runtime))
                }) {
                    Ok((dataset, store, runtime)) => {
                        let mut app =
                            OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
                        app.set_remote_runtime(Some(runtime));
                        app.set_show_scale_bar(self.view_show_scale_bar);
                        app.set_label_prompt_preference(self.label_prompt_preference);
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    }
                    Err(err) => {
                        let mut ps = project_space;
                        ps.set_status(format!("Open remote dataset failed: {err}"));
                        self.mode = Mode::Project { project_space: ps };
                    }
                }
            }
        }
    }

    fn open_project_roi(
        &mut self,
        ctx: &egui::Context,
        roi: ProjectRoi,
        project_space: ProjectSpace,
    ) {
        let Some(source) = roi.dataset_source() else {
            let mut ps = project_space;
            ps.set_status("Project ROI has no dataset source configured.".to_string());
            self.mode = Mode::Project { project_space: ps };
            return;
        };
        self.open_dataset_source(ctx, source, project_space);
    }

    fn open_mosaic_from_project(
        &mut self,
        ctx: &egui::Context,
        rois: Vec<ProjectRoi>,
        project_space: ProjectSpace,
    ) {
        let ret = ReturnToSingleState { dataset_root: None };
        let project_dir = project_space.project_dir();
        if rois.len() < 2 {
            let mut ps = project_space;
            ps.set_status("Need at least 2 ROIs to open mosaic.".to_string());
            self.mode = Mode::Project { project_space: ps };
            return;
        }
        let mosaic_result =
            MosaicViewerApp::from_project_rois(ctx, self.gpu_available, rois, project_dir, None);
        match mosaic_result {
            Ok(mut mosaic) => {
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
            Err(err) => {
                let mut ps = project_space;
                ps.set_status(format!("Open mosaic failed: {err}"));
                self.mode = Mode::Project { project_space: ps };
            }
        }
    }

    fn switch_single_to_mosaic(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        let Mode::Single(mut single) = prev else {
            self.mode = prev;
            return;
        };

        let ret = ReturnToSingleState {
            dataset_root: single.current_local_dataset_root(),
        };
        let project_space = single.take_project_space();
        let project_rois = project_space.rois_for_local_paths(&paths);
        let project_dir = project_space.project_dir();
        let mosaic_result = if project_rois.len() >= 2 {
            MosaicViewerApp::from_project_rois(
                ctx,
                self.gpu_available,
                project_rois,
                project_dir,
                None,
            )
        } else {
            MosaicViewerApp::from_local_paths(ctx, self.gpu_available, paths, None)
        };

        match mosaic_result {
            Ok(mut mosaic) => {
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
            Err(err) => {
                let mut single = single;
                let mut ps = project_space;
                ps.set_status(format!("Open mosaic failed: {err}"));
                single.set_project_space(ps);
                self.mode = Mode::Single(single);
            }
        }
    }

    fn switch_mosaic_to_single(&mut self, ctx: &egui::Context) {
        let prev = std::mem::replace(&mut self.mode, Mode::Transition);
        let Mode::Mosaic { mosaic, ret } = prev else {
            self.mode = prev;
            return;
        };

        let mut mosaic = mosaic;
        let project_space = mosaic.take_project_space();

        let Some(root) = ret.dataset_root.clone() else {
            // No known return target; return to project landing.
            self.mode = Mode::Project { project_space };
            return;
        };

        match OmeZarrDataset::open_local(&root) {
            Ok((dataset, store)) => {
                let mut app =
                    OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
                app.set_show_scale_bar(self.view_show_scale_bar);
                app.set_label_prompt_preference(self.label_prompt_preference);
                app.set_project_space(project_space);
                self.mode = Mode::Single(app);
            }
            Err(err) => {
                // If reopen fails, fall back to staying in mosaic.
                eprintln!("Back failed: {err}");
                mosaic.set_project_space(project_space);
                self.mode = Mode::Mosaic { mosaic, ret };
            }
        }
    }

    fn ui_spatial_open_dialog(&mut self, ctx: &egui::Context) {
        let mut open_clicked = false;
        let mut cancel_clicked = false;

        {
            let Some(dlg) = self.spatial_open.as_mut() else {
                return;
            };

            egui::Window::new("Open SpatialData")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(dlg.discovery.root.to_string_lossy().to_string());
                    ui.add_space(6.0);

                    ui.label("Image");
                    egui::ComboBox::from_id_salt("spatial_image")
                        .selected_text(
                            dlg.discovery
                                .images
                                .get(dlg.selected_image)
                                .map(|e| e.name.clone())
                                .unwrap_or_else(|| "(none)".to_string()),
                        )
                        .show_ui(ui, |ui| {
                            for (i, e) in dlg.discovery.images.iter().enumerate() {
                                ui.selectable_value(&mut dlg.selected_image, i, e.name.clone());
                            }
                        });

                    ui.separator();
                    ui.label("Overlays");

                    ui.horizontal(|ui| {
                        ui.label("Labels");
                        let selected_text = dlg
                            .selected_labels
                            .and_then(|i| dlg.discovery.labels.get(i))
                            .map(|e| e.name.clone())
                            .unwrap_or_else(|| "None".to_string());
                        egui::ComboBox::from_id_salt("spatial_labels")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut dlg.selected_labels, None, "None");
                                for (i, e) in dlg.discovery.labels.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut dlg.selected_labels,
                                        Some(i),
                                        e.name.clone(),
                                    );
                                }
                            });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Shapes");
                        if dlg.selected_shapes.is_empty() {
                            ui.label("None");
                        } else {
                            ui.label(format!("{} selected", dlg.selected_shapes.len()));
                        }
                    });
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_min_width(240.0);
                        for (i, e) in dlg.discovery.shapes.iter().enumerate() {
                            let mut selected = dlg.selected_shapes.contains(&i);
                            if ui.checkbox(&mut selected, e.name.as_str()).changed() {
                                if selected {
                                    if !dlg.selected_shapes.contains(&i) {
                                        dlg.selected_shapes.push(i);
                                        dlg.selected_shapes.sort_unstable();
                                    }
                                } else {
                                    dlg.selected_shapes.retain(|&idx| idx != i);
                                }
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("Points");
                        let selected_text = dlg
                            .selected_points
                            .and_then(|i| dlg.discovery.points.get(i))
                            .map(|e| e.name.clone())
                            .unwrap_or_else(|| "None".to_string());
                        egui::ComboBox::from_id_salt("spatial_points")
                            .selected_text(selected_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut dlg.selected_points, None, "None");
                                for (i, e) in dlg.discovery.points.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut dlg.selected_points,
                                        Some(i),
                                        e.name.clone(),
                                    );
                                }
                            });
                        let mut all = dlg.points_max == 0;
                        if ui
                            .checkbox(&mut all, "All")
                            .on_hover_text("Load all points (may be slow / memory-heavy).")
                            .changed()
                        {
                            dlg.points_max = if all { 0 } else { 200_000 };
                        }
                        ui.add(
                            egui::DragValue::new(&mut dlg.points_max)
                                .speed(1)
                                .range(0..=200_000_000)
                                .prefix("Max "),
                        )
                        .on_hover_text("0 means no cap (load all).");
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        open_clicked = ui.button("Open").clicked();
                        cancel_clicked = ui.button("Cancel").clicked();
                    });

                    if !dlg.status.is_empty() {
                        ui.add_space(6.0);
                        ui.label(dlg.status.clone());
                    }
                });
        }

        if cancel_clicked {
            self.spatial_open = None;
            return;
        }
        if !open_clicked {
            return;
        }

        let (root, img, labels, tables, shapes, points, points_max) = {
            let Some(dlg) = self.spatial_open.as_ref() else {
                return;
            };
            let root = dlg.discovery.root.clone();
            let img = dlg.discovery.images.get(dlg.selected_image).cloned();
            let labels = dlg
                .selected_labels
                .and_then(|i| dlg.discovery.labels.get(i))
                .cloned();
            let tables = dlg.discovery.tables.clone();
            let shapes = dlg
                .selected_shapes
                .iter()
                .filter_map(|&i| dlg.discovery.shapes.get(i).cloned())
                .collect::<Vec<_>>();
            let points = dlg
                .selected_points
                .and_then(|i| dlg.discovery.points.get(i))
                .cloned();
            (root, img, labels, tables, shapes, points, dlg.points_max)
        };

        let Some(img) = img else {
            if let Some(dlg) = self.spatial_open.as_mut() {
                dlg.status = "No image selected.".to_string();
            }
            return;
        };

        // Take the project space from the current mode (the dialog always runs in Project mode).
        let project_space = match std::mem::replace(&mut self.mode, Mode::Transition) {
            Mode::Project { project_space } => project_space,
            other => {
                self.mode = other;
                if let Some(dlg) = self.spatial_open.as_mut() {
                    dlg.status = "Internal error: not in Project mode.".to_string();
                }
                return;
            }
        };

        let img_root = root.join(&img.rel_group);
        match OmeZarrDataset::open_local(&img_root) {
            Ok((dataset, store)) => {
                let mut app =
                    OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
                app.set_show_scale_bar(self.view_show_scale_bar);
                app.set_label_prompt_preference(self.label_prompt_preference);
                app.set_project_space(project_space);
                app.attach_spatialdata_layers(
                    root,
                    img.transform,
                    Vec::new(),
                    labels,
                    tables,
                    shapes,
                    points.map(|e| (e, points_max)),
                );
                if let Some(viewport) = ctx.input(|i| i.viewport().inner_rect) {
                    app.fit_to_viewport(viewport);
                }
                self.mode = Mode::Single(app);
                self.spatial_open = None;
            }
            Err(err) => {
                self.mode = Mode::Project { project_space };
                if let Some(dlg) = self.spatial_open.as_mut() {
                    dlg.status = format!("Open image failed: {err}");
                }
            }
        }
    }
}

impl eframe::App for RootApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let open_mosaic: Option<Vec<PathBuf>> = None;
        let mut open_remote_single: Option<(
            OmeZarrDataset,
            Arc<dyn zarrs::storage::ReadableStorageTraits>,
            Option<Arc<tokio::runtime::Runtime>>,
            ProjectSpace,
        )> = None;
        let mut open_remote_s3_mosaic: Option<(Vec<crate::app::S3DatasetSelection>, ProjectSpace)> =
            None;
        let mut back_to_single = false;
        let mut open_single: Option<(PathBuf, ProjectSpace)> = None;
        let mut open_project_roi: Option<(ProjectRoi, ProjectSpace)> = None;
        let mut open_mosaic_from_project: Option<(Vec<ProjectRoi>, ProjectSpace)> = None;

        #[cfg(target_os = "macos")]
        {
            if self.native_menu.is_none() {
                if let Ok(m) = NativeMenu::init("odon", self.view_show_scale_bar) {
                    self.native_menu = Some(m);
                }
            }
            if let Some(menu) = self.native_menu.as_ref() {
                for action in menu.drain_actions() {
                    match action {
                        NativeMenuAction::OpenOmeZarr => {
                            if let Some(root) =
                                FileDialog::new().set_title("Open OME-Zarr").pick_folder()
                            {
                                let ps = match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        let mut ps = std::mem::take(project_space);
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        ps.handle_dropped_paths([root.clone()]);
                                        ps
                                    }
                                    Mode::Transition => ProjectSpace::default(),
                                };
                                open_single = Some((root, ps));
                            }
                        }
                        NativeMenuAction::OpenProject => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("Project JSON", &["json"])
                                .set_title("Load Project")
                                .pick_file()
                            {
                                match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        if let Err(err) = project_space.load_from_file(&path) {
                                            project_space
                                                .set_status(format!("Load project failed: {err}"));
                                        }
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        if let Err(err) = ps.load_from_file(&path) {
                                            ps.set_status(format!("Load project failed: {err}"));
                                        }
                                        app.set_project_space(ps);
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        if let Err(err) = ps.load_from_file(&path) {
                                            ps.set_status(format!("Load project failed: {err}"));
                                        } else {
                                            mosaic.set_layer_groups(ps.layer_groups().clone());
                                        }
                                        mosaic.set_project_space(ps);
                                    }
                                    Mode::Transition => {}
                                }
                            }
                        }
                        NativeMenuAction::SaveProject => {
                            let save_target = match &self.mode {
                                Mode::Project { project_space } => {
                                    project_space.current_project_path()
                                }
                                Mode::Single(app) => app.project_space().current_project_path(),
                                Mode::Mosaic { mosaic, .. } => {
                                    mosaic.project_space().current_project_path()
                                }
                                Mode::Transition => None,
                            };
                            let path = if let Some(path) = save_target {
                                Some(path)
                            } else {
                                FileDialog::new()
                                    .add_filter("Project JSON", &["json"])
                                    .set_file_name("odon.project.json")
                                    .set_title("Save Project")
                                    .save_file()
                            };
                            if let Some(path) = path {
                                match &mut self.mode {
                                    Mode::Project { project_space } => {
                                        if let Err(err) = project_space.save_to_file(&path) {
                                            project_space
                                                .set_status(format!("Save project failed: {err}"));
                                        }
                                    }
                                    Mode::Single(app) => {
                                        let mut ps = app.take_project_space();
                                        if let Err(err) = ps.save_to_file(&path) {
                                            ps.set_status(format!("Save project failed: {err}"));
                                        }
                                        app.set_project_space(ps);
                                    }
                                    Mode::Mosaic { mosaic, .. } => {
                                        let mut ps = mosaic.take_project_space();
                                        if let Err(err) = ps.save_to_file(&path) {
                                            ps.set_status(format!("Save project failed: {err}"));
                                        }
                                        mosaic.set_project_space(ps);
                                    }
                                    Mode::Transition => {}
                                }
                            }
                        }
                        NativeMenuAction::SaveScreenshot => {
                            self.save_screenshot_via_dialog();
                        }
                        NativeMenuAction::QuickScreenshot => {
                            self.quick_screenshot();
                        }
                        NativeMenuAction::ScreenshotSettings => match &mut self.mode {
                            Mode::Single(app) => app.open_screenshot_settings(),
                            Mode::Project { project_space } => project_space.set_status(
                                "Screenshot Settings: open a dataset first.".to_string(),
                            ),
                            Mode::Mosaic { mosaic, .. } => mosaic.open_screenshot_settings(),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::AddAnnotations => match &mut self.mode {
                            Mode::Single(app) => app.add_annotation_layer_from_menu(),
                            Mode::Project { project_space } => project_space
                                .set_status("Add annotations: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Add annotations: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::LoadSegGeoJson => match &mut self.mode {
                            Mode::Single(app) => app.open_seg_geojson_dialog(),
                            Mode::Project { project_space } => project_space
                                .set_status("Load Seg GeoJSON: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Load Seg GeoJSON: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::LoadSegObjects => match &mut self.mode {
                            Mode::Single(app) => app.open_seg_objects_dialog(),
                            Mode::Project { project_space } => project_space
                                .set_status("Load Seg Objects: open a dataset first.".to_string()),
                            Mode::Mosaic { mosaic, .. } => mosaic.project_space_mut().set_status(
                                "Load Seg Objects: open a single ROI first.".to_string(),
                            ),
                            Mode::Transition => {}
                        },
                        NativeMenuAction::ExportMasksGeoJson => {
                            if let Some(path) = FileDialog::new()
                                .add_filter("GeoJSON", &["geojson", "json"])
                                .set_file_name("masks.geojson")
                                .set_title("Export Masks GeoJSON")
                                .save_file()
                            {
                                match &mut self.mode {
                                    Mode::Single(app) => match app.export_masks_geojson(&path) {
                                        Ok(()) => app.set_status(format!(
                                            "Exported masks -> {}",
                                            path.to_string_lossy()
                                        )),
                                        Err(err) => {
                                            app.set_status(format!("Export masks failed: {err}"))
                                        }
                                    },
                                    Mode::Project { project_space } => project_space.set_status(
                                        "Export masks failed: open a dataset first.".to_string(),
                                    ),
                                    Mode::Mosaic { mosaic, .. } => {
                                        mosaic.project_space_mut().set_status(
                                            "Export masks failed: open a single ROI first."
                                                .to_string(),
                                        )
                                    }
                                    Mode::Transition => {}
                                }
                            }
                        }
                        NativeMenuAction::SetScaleBarVisible(visible) => {
                            self.view_show_scale_bar = visible;
                            if let Mode::Single(app) = &mut self.mode {
                                app.set_show_scale_bar(visible);
                            }
                        }
                        NativeMenuAction::CloseWindow | NativeMenuAction::Quit => {
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
                                Mode::Mosaic { mosaic, .. } => {
                                    mosaic.confirm_or_request_close_dialog()
                                }
                                Mode::Transition => false,
                            };
                            if should_close {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                        }
                    }
                }
            }
        }

        match &mut self.mode {
            Mode::Project { project_space } => {
                let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                if !dropped.is_empty() {
                    project_space.handle_dropped_paths(
                        dropped
                            .into_iter()
                            .filter_map(|f| f.path)
                            .collect::<Vec<_>>(),
                    );
                }

                // Napari-like "close window" prompt:
                // - Cmd/Ctrl+W opens confirmation
                // - Cmd/Ctrl+W again confirms close
                if top_bar::handle_cmd_w_close(ctx, &mut self.close_dialog_open)
                    || top_bar::ui_close_dialog(ctx, &mut self.close_dialog_open)
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }

                // Minimal "landing" UI: show the project panel and let users open datasets.
                egui::TopBottomPanel::top("top").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("odon");
                        ui.add_space(8.0);
                        ui.label("Project");
                    });
                });
                egui::SidePanel::left("project-left")
                    .default_width(420.0)
                    .resizable(true)
                    .show(ctx, |ui| {
                        let action = project_space.ui(ui, None);
                        if let Some(action) = action {
                            match action {
                                ProjectSpaceAction::Open(roi) => {
                                    let ps = std::mem::take(project_space);
                                    open_project_roi = Some((roi, ps));
                                }
                                ProjectSpaceAction::OpenMosaic(rois) => {
                                    let ps = std::mem::take(project_space);
                                    open_mosaic_from_project = Some((rois, ps));
                                }
                                ProjectSpaceAction::OpenRemoteDialog => {
                                    self.remote_dialog_open = true;
                                    self.remote_status.clear();
                                }
                            }
                        }
                    });
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.heading("Drop OME-Zarr folders or TIFF files to start");
                        ui.label(
                            "Use the Project panel on the left to open a dataset or a mosaic.",
                        );
                    });
                });

                // Startup open (e.g. when launched with a dataset path that isn't a direct OME image root).
                if open_single.is_none() && self.spatial_open.is_none() {
                    if let Some(root) = self.pending_open_root.take() {
                        let ps = std::mem::take(project_space);
                        open_single = Some((root, ps));
                    }
                }
            }
            Mode::Single(app) => {
                app.update(ctx, frame);
                self.label_prompt_preference = app.label_prompt_preference();
                if let Some(req) = app.take_request() {
                    match req {
                        ViewerRequest::OpenProjectRoi(roi) => {
                            let ps = app.take_project_space();
                            open_project_roi = Some((roi, ps));
                        }
                        ViewerRequest::OpenProjectMosaic(rois) => {
                            let ps = app.take_project_space();
                            open_mosaic_from_project = Some((rois, ps));
                        }
                        ViewerRequest::OpenRemoteS3Mosaic(datasets) => {
                            let ps = app.take_project_space();
                            open_remote_s3_mosaic = Some((datasets, ps));
                        }
                    }
                }
            }
            Mode::Mosaic { mosaic, .. } => {
                let dropped = ctx.input(|i| i.raw.dropped_files.clone());
                if !dropped.is_empty() {
                    mosaic.project_space_mut().handle_dropped_paths(
                        dropped
                            .into_iter()
                            .filter_map(|f| f.path)
                            .collect::<Vec<_>>(),
                    );
                }
                mosaic.update(ctx, frame);
                if let Some(req) = mosaic.take_request() {
                    match req {
                        MosaicRequest::BackToSingle => {
                            back_to_single = true;
                        }
                        MosaicRequest::OpenProjectRoi(roi) => {
                            let ps = mosaic.take_project_space();
                            open_project_roi = Some((roi, ps));
                        }
                        MosaicRequest::OpenProjectMosaic(rois) => {
                            let ps = mosaic.take_project_space();
                            open_mosaic_from_project = Some((rois, ps));
                        }
                        MosaicRequest::OpenRemoteDialog => {
                            self.remote_dialog_open = true;
                            self.remote_status.clear();
                        }
                    }
                }
            }
            Mode::Transition => {}
        }

        if matches!(self.mode, Mode::Project { .. }) {
            self.ui_spatial_open_dialog(ctx);
        }

        if let Some(action) = self.ui_remote_dialog(ctx) {
            let previous_mode = std::mem::replace(&mut self.mode, Mode::Transition);
            let (project_space, single_restore, mosaic_restore) = match previous_mode {
                Mode::Project { project_space } => (project_space, None, None),
                Mode::Single(mut app) => (app.take_project_space(), Some(app), None),
                Mode::Mosaic { mut mosaic, ret } => {
                    (mosaic.take_project_space(), None, Some((mosaic, ret)))
                }
                Mode::Transition => (ProjectSpace::default(), None, None),
            };
            match action {
                RootRemoteAction::OpenSingle {
                    dataset,
                    store,
                    runtime,
                } => {
                    open_remote_single = Some((dataset, store, runtime, project_space));
                }
                RootRemoteAction::OpenS3Mosaic(datasets) => {
                    open_remote_s3_mosaic = Some((datasets, project_space));
                }
                RootRemoteAction::AddToProject(sources) => {
                    let mut project_space = project_space;
                    let count = sources.len();
                    for source in sources {
                        project_space.add_roi_source(source);
                    }
                    project_space
                        .set_status(format!("Added {count} remote ROI(s) to the project."));
                    if let Some(mut app) = single_restore {
                        app.set_project_space(project_space);
                        self.mode = Mode::Single(app);
                    } else if let Some((mut mosaic, ret)) = mosaic_restore {
                        mosaic.set_project_space(project_space);
                        self.mode = Mode::Mosaic { mosaic, ret };
                    } else {
                        self.mode = Mode::Project { project_space };
                    }
                }
            }
        }

        if let Some((root, ps)) = open_single {
            self.open_single(ctx, &root, ps);
        }
        if let Some((roi, ps)) = open_project_roi {
            self.open_project_roi(ctx, roi, ps);
        }
        if let Some((dataset, store, runtime, project_space)) = open_remote_single {
            let mut app = OmeZarrViewerApp::new_runtime(ctx, self.gpu_available, dataset, store);
            app.set_remote_runtime(runtime);
            app.set_show_scale_bar(self.view_show_scale_bar);
            app.set_label_prompt_preference(self.label_prompt_preference);
            app.set_project_space(project_space);
            self.mode = Mode::Single(app);
        }
        if let Some((paths, ps)) = open_mosaic_from_project {
            self.open_mosaic_from_project(ctx, paths, ps);
        }
        if let Some((datasets, project_space)) = open_remote_s3_mosaic {
            let ret = ReturnToSingleState { dataset_root: None };
            match MosaicViewerApp::from_remote_s3_sources(ctx, self.gpu_available, datasets, None) {
                Ok(mut mosaic) => {
                    mosaic.set_project_space(project_space);
                    self.mode = Mode::Mosaic { mosaic, ret };
                }
                Err(err) => {
                    let mut ps = project_space;
                    ps.set_status(format!("Open remote mosaic failed: {err}"));
                    self.mode = Mode::Project { project_space: ps };
                }
            }
        }
        if let Some(paths) = open_mosaic {
            self.switch_single_to_mosaic(ctx, paths);
        }
        if back_to_single {
            self.switch_mosaic_to_single(ctx);
        }
    }
}
