use anyhow::Context;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeMenuAction {
    OpenOmeZarr,
    OpenProject,
    SaveProject,
    SaveScreenshot,
    QuickScreenshot,
    ScreenshotSettings,
    AddAnnotations,
    LoadSegGeoJson,
    LoadSegObjects,
    ExportMasksGeoJson,
    SetScaleBarVisible(bool),
    CloseWindow,
    Quit,
}

#[cfg(target_os = "macos")]
pub struct NativeMenu {
    _menu: muda::Menu,
    id_open_omezarr: muda::MenuId,
    id_open_project: muda::MenuId,
    id_save_project: muda::MenuId,
    id_save_screenshot: muda::MenuId,
    id_quick_screenshot: muda::MenuId,
    id_screenshot_settings: muda::MenuId,
    id_add_annotations: muda::MenuId,
    id_load_seg_geojson: muda::MenuId,
    id_load_seg_objects: muda::MenuId,
    id_export_masks_geojson: muda::MenuId,
    view_scale_bar: muda::CheckMenuItem,
    id_view_scale_bar: muda::MenuId,
    id_close_window: muda::MenuId,
    id_quit: muda::MenuId,
}

#[cfg(not(target_os = "macos"))]
pub struct NativeMenu;

impl NativeMenu {
    #[cfg(target_os = "macos")]
    pub fn init(app_name: &str, show_scale_bar: bool) -> anyhow::Result<Self> {
        use muda::accelerator::{Accelerator, Code, Modifiers};
        use muda::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};

        let menu = Menu::new();

        // macOS expects an "app" menu first. We keep it minimal for now.
        let about = PredefinedMenuItem::about(Some(&format!("About {app_name}")), None);
        let quit = MenuItem::new(
            format!("Quit {app_name}"),
            true,
            Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyQ)),
        );
        let app_menu = Submenu::with_items(
            app_name,
            true,
            &[&about, &PredefinedMenuItem::separator(), &quit],
        )
        .context("failed to build app menu")?;

        let open_omezarr = MenuItem::new(
            "Open OME-Zarr...",
            true,
            Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyO)),
        );
        let open_project = MenuItem::new(
            "Open Project...",
            true,
            Some(Accelerator::new(
                Some(Modifiers::SUPER | Modifiers::SHIFT),
                Code::KeyO,
            )),
        );
        let save_project = MenuItem::new(
            "Save Project...",
            true,
            Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyS)),
        );
        let save_screenshot = MenuItem::new("Save Screenshot...", true, None);
        let quick_screenshot = MenuItem::new(
            "Quick Screenshot",
            true,
            Some(Accelerator::new(
                Some(Modifiers::SUPER | Modifiers::SHIFT),
                Code::KeyS,
            )),
        );
        let screenshot_settings = MenuItem::new("Screenshot Settings...", true, None);
        let add_annotations = MenuItem::new("Annotations", true, None);
        let load_seg_geojson = MenuItem::new("Load Seg GeoJSON...", true, None);
        let load_seg_objects = MenuItem::new("Load Seg Objects...", true, None);
        let export_masks_geojson = MenuItem::new("Export Masks GeoJSON...", true, None);
        let close_window = MenuItem::new(
            "Close Window",
            true,
            Some(Accelerator::new(Some(Modifiers::SUPER), Code::KeyW)),
        );
        let file_menu = Submenu::with_items(
            "File",
            true,
            &[
                &open_omezarr,
                &PredefinedMenuItem::separator(),
                &open_project,
                &save_project,
                &PredefinedMenuItem::separator(),
                &export_masks_geojson,
                &PredefinedMenuItem::separator(),
                &save_screenshot,
                &quick_screenshot,
                &screenshot_settings,
                &PredefinedMenuItem::separator(),
                &close_window,
            ],
        )
        .context("failed to build file menu")?;
        let add_menu = Submenu::with_items(
            "Add",
            true,
            &[
                &add_annotations,
                &PredefinedMenuItem::separator(),
                &load_seg_geojson,
                &load_seg_objects,
            ],
        )
        .context("failed to build add menu")?;

        let view_scale_bar = CheckMenuItem::new("Scale Bar", true, show_scale_bar, None);
        let view_menu = Submenu::with_items("View", true, &[&view_scale_bar])
            .context("failed to build view menu")?;

        menu.append(&app_menu)
            .context("failed to append app menu")?;
        menu.append(&file_menu)
            .context("failed to append file menu")?;
        menu.append(&add_menu)
            .context("failed to append add menu")?;
        menu.append(&view_menu)
            .context("failed to append view menu")?;

        menu.init_for_nsapp();

        Ok(Self {
            _menu: menu,
            id_open_omezarr: open_omezarr.id().clone(),
            id_open_project: open_project.id().clone(),
            id_save_project: save_project.id().clone(),
            id_save_screenshot: save_screenshot.id().clone(),
            id_quick_screenshot: quick_screenshot.id().clone(),
            id_screenshot_settings: screenshot_settings.id().clone(),
            id_add_annotations: add_annotations.id().clone(),
            id_load_seg_geojson: load_seg_geojson.id().clone(),
            id_load_seg_objects: load_seg_objects.id().clone(),
            id_export_masks_geojson: export_masks_geojson.id().clone(),
            view_scale_bar: view_scale_bar.clone(),
            id_view_scale_bar: view_scale_bar.id().clone(),
            id_close_window: close_window.id().clone(),
            id_quit: quit.id().clone(),
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn init(_app_name: &str, _show_scale_bar: bool) -> anyhow::Result<Self> {
        Ok(Self)
    }

    #[cfg(target_os = "macos")]
    pub fn drain_actions(&self) -> Vec<NativeMenuAction> {
        let mut out = Vec::new();
        while let Ok(ev) = muda::MenuEvent::receiver().try_recv() {
            let id = ev.id();
            if id == &self.id_open_omezarr {
                out.push(NativeMenuAction::OpenOmeZarr);
            } else if id == &self.id_open_project {
                out.push(NativeMenuAction::OpenProject);
            } else if id == &self.id_save_project {
                out.push(NativeMenuAction::SaveProject);
            } else if id == &self.id_save_screenshot {
                out.push(NativeMenuAction::SaveScreenshot);
            } else if id == &self.id_quick_screenshot {
                out.push(NativeMenuAction::QuickScreenshot);
            } else if id == &self.id_screenshot_settings {
                out.push(NativeMenuAction::ScreenshotSettings);
            } else if id == &self.id_add_annotations {
                out.push(NativeMenuAction::AddAnnotations);
            } else if id == &self.id_load_seg_geojson {
                out.push(NativeMenuAction::LoadSegGeoJson);
            } else if id == &self.id_load_seg_objects {
                out.push(NativeMenuAction::LoadSegObjects);
            } else if id == &self.id_export_masks_geojson {
                out.push(NativeMenuAction::ExportMasksGeoJson);
            } else if id == &self.id_view_scale_bar {
                out.push(NativeMenuAction::SetScaleBarVisible(
                    self.view_scale_bar.is_checked(),
                ));
            } else if id == &self.id_close_window {
                out.push(NativeMenuAction::CloseWindow);
            } else if id == &self.id_quit {
                out.push(NativeMenuAction::Quit);
            }
        }
        out
    }

    #[cfg(not(target_os = "macos"))]
    pub fn drain_actions(&self) -> Vec<NativeMenuAction> {
        Vec::new()
    }
}
