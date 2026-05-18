use eframe::egui;

use crate::imaging::view_plane::ViewPlaneMode;

pub const SMOOTH_TOOLTIP: &str = "When enabled, pixels are linearly filtered while zooming.\nDisable for crisp (nearest-neighbor) pixels.";

pub fn ui_title(ui: &mut egui::Ui, title: impl Into<egui::WidgetText>) -> egui::Response {
    ui.label(title)
}

pub fn ui_back(ui: &mut egui::Ui, enabled: bool) -> bool {
    ui.add_enabled(enabled, egui::Button::new("Back")).clicked()
}

pub fn ui_fit(ui: &mut egui::Ui, label: &str) -> bool {
    ui.button(label).clicked()
}

pub fn ui_status(ui: &mut egui::Ui, status: &str) {
    ui.label(status);
}

pub fn ui_auto_level(
    ui: &mut egui::Ui,
    auto_level: &mut bool,
    manual_level: &mut usize,
    max_level: usize,
) -> bool {
    let mut changed = ui.checkbox(auto_level, "Auto level").changed();
    if !*auto_level {
        changed |= ui
            .add(egui::Slider::new(manual_level, 0..=max_level).text("Level"))
            .changed();
    }
    changed
}

pub fn ui_view_plane_mode(
    ui: &mut egui::Ui,
    mode: &mut ViewPlaneMode,
    supported: &[ViewPlaneMode],
) -> bool {
    let before = *mode;
    egui::ComboBox::from_id_salt("view-plane-mode")
        .selected_text(mode.label())
        .show_ui(ui, |ui| {
            for candidate in supported {
                ui.selectable_value(mode, *candidate, candidate.label());
            }
        });
    *mode != before
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SliceSliderResponse {
    pub changed: bool,
    pub dragging: bool,
    pub released: bool,
}

pub fn ui_view_plane_slice(
    ui: &mut egui::Ui,
    label: &str,
    slice_level0: &mut u64,
    max_slice_level0: u64,
) -> SliceSliderResponse {
    let mut changed = false;
    let mut dragging = false;
    let mut released = false;

    ui.horizontal(|ui| {
        let response = ui.add_sized(
            [180.0, 18.0],
            egui::Slider::new(slice_level0, 0..=max_slice_level0)
                .text(label)
                .clamping(egui::SliderClamping::Always),
        );
        changed |= response.changed();
        dragging |= response.dragged();
        released |= response.drag_stopped();

        let prev = ui
            .add_enabled(*slice_level0 > 0, egui::Button::new("◀").small())
            .on_hover_text(format!("Previous {label} slice"));
        if prev.clicked() {
            *slice_level0 = slice_level0.saturating_sub(1);
            changed = true;
        }

        let next = ui
            .add_enabled(
                *slice_level0 < max_slice_level0,
                egui::Button::new("▶").small(),
            )
            .on_hover_text(format!("Next {label} slice"));
        if next.clicked() {
            *slice_level0 = (*slice_level0 + 1).min(max_slice_level0);
            changed = true;
        }
    });

    SliceSliderResponse {
        changed,
        dragging,
        released,
    }
}

pub fn ui_prev_next_core(ui: &mut egui::Ui, have_items: bool) -> Option<i32> {
    if ui
        .add_enabled(have_items, egui::Button::new("Prev. Core"))
        .clicked()
    {
        return Some(-1);
    }
    if ui
        .add_enabled(have_items, egui::Button::new("Next. Core"))
        .clicked()
    {
        return Some(1);
    }
    None
}

pub fn ui_core_index(ui: &mut egui::Ui, summary: Option<(usize, usize, String)>) {
    let Some((idx, n, name)) = summary else {
        return;
    };
    ui.label(format!("{idx}/{n}: {name}"));
}

/// Napari-like "close window" prompt:
/// - Cmd/Ctrl+W opens confirmation
/// - Cmd/Ctrl+W again confirms close
pub fn handle_cmd_w_close(ctx: &egui::Context, close_dialog_open: &mut bool) -> bool {
    let cmd_w = ctx.input(|i| i.key_pressed(egui::Key::W) && i.modifiers.command);
    // Unlike plain `W`, Cmd/Ctrl+W is a window-level shortcut and should work even when a
    // text field has focus (napari-like behavior).
    if cmd_w {
        if *close_dialog_open {
            *close_dialog_open = false;
            return true;
        }
        *close_dialog_open = true;
    }
    false
}

/// Draws the close confirmation dialog if `close_dialog_open` is true.
///
/// Returns `true` if the app should close now.
pub fn ui_close_dialog(ctx: &egui::Context, close_dialog_open: &mut bool) -> bool {
    if !*close_dialog_open {
        return false;
    }

    if ctx.input(|i| i.key_pressed(egui::Key::W) && i.modifiers.command) {
        *close_dialog_open = false;
        return true;
    }

    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        *close_dialog_open = false;
        return false;
    }

    if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
        *close_dialog_open = false;
        return true;
    }

    let mut open = *close_dialog_open;
    let mut close_now = false;
    let mut cancel_now = false;
    egui::Window::new("Close?")
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label("Close the viewer?");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancel_now = true;
                }
                if ui.button("Close (Enter)").clicked() {
                    close_now = true;
                }
            });
        });
    if cancel_now || close_now {
        open = false;
    }
    *close_dialog_open = open;
    close_now
}

pub fn ui_prev_next_channel(ui: &mut egui::Ui, have_channels: bool) -> Option<i32> {
    if ui
        .add_enabled(have_channels, egui::Button::new("Prev. Channel"))
        .clicked()
    {
        return Some(-1);
    }
    if ui
        .add_enabled(have_channels, egui::Button::new("Next Channel"))
        .clicked()
    {
        return Some(1);
    }
    None
}

pub fn ui_prev_next_roi(ui: &mut egui::Ui, have_rois: bool) -> Option<i32> {
    if ui
        .add_enabled(have_rois, egui::Button::new("Prev. ROI"))
        .clicked()
    {
        return Some(-1);
    }
    if ui
        .add_enabled(have_rois, egui::Button::new("Next ROI"))
        .clicked()
    {
        return Some(1);
    }
    None
}

pub fn ui_panel_toggles(
    ui: &mut egui::Ui,
    show_left_panel: &mut bool,
    show_right_panel: &mut bool,
) {
    if ui
        .add(egui::Button::new("Both").selected(*show_left_panel && *show_right_panel))
        .clicked()
    {
        let show = !(*show_left_panel && *show_right_panel);
        *show_left_panel = show;
        *show_right_panel = show;
    }
    ui.checkbox(show_left_panel, "Left");
    ui.checkbox(show_right_panel, "Right");
}

pub fn ui_smooth_toggle(ui: &mut egui::Ui, smooth_pixels: &mut bool) -> bool {
    ui.checkbox(smooth_pixels, "Smooth")
        .on_hover_text(SMOOTH_TOOLTIP)
        .changed()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuickContrastTarget {
    Active,
    Visible,
    SelectedGroup,
}

impl QuickContrastTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::Active => "Active channel",
            Self::Visible => "Visible channels",
            Self::SelectedGroup => "Selected group",
        }
    }
}

pub struct QuickContrastTargetOption {
    pub target: QuickContrastTarget,
    pub label: String,
    pub enabled: bool,
}

pub struct QuickContrastParams<'a> {
    pub abs_max: f32,
    pub target: &'a mut QuickContrastTarget,
    pub target_options: &'a [QuickContrastTargetOption],
    pub target_count: usize,
    pub reference_channel_name: &'a str,
    pub window: (f32, f32),
    pub mixed: bool,
    pub step: f32,
    pub id_salt: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuickContrastResponse {
    pub window: (f32, f32),
    pub changed: bool,
}

pub fn ui_quick_contrast(
    ui: &mut egui::Ui,
    params: QuickContrastParams<'_>,
) -> QuickContrastResponse {
    let mut response = QuickContrastResponse {
        window: params.window,
        changed: false,
    };

    ui.menu_button("Contrast", |ui| {
        ui.set_min_width(340.0);

        let current_target = *params.target;
        let selected_label = params
            .target_options
            .iter()
            .find(|option| option.target == current_target)
            .map(|option| option.label.as_str())
            .unwrap_or_else(|| current_target.label());
        ui.horizontal(|ui| {
            ui.label("Target");
            egui::ComboBox::from_id_salt(ui.id().with(params.id_salt).with("target"))
                .selected_text(selected_label)
                .show_ui(ui, |ui| {
                    for option in params.target_options {
                        ui.add_enabled_ui(option.enabled, |ui| {
                            ui.selectable_value(params.target, option.target, &option.label);
                        });
                    }
                });
        });

        ui.small(format!(
            "Editing {} channel(s), based on {}.",
            params.target_count.max(1),
            params.reference_channel_name
        ));
        if params.mixed {
            ui.small("Target channels currently have mixed limits; editing will overwrite them.");
        }
        ui.separator();

        let abs_max = params.abs_max.max(1.0);
        let (mut lo, mut hi) = params.window;
        lo = lo.clamp(0.0, abs_max);
        hi = hi.clamp(0.0, abs_max);
        if hi <= lo {
            hi = (lo + 1.0).min(abs_max);
        }

        ui.horizontal(|ui| {
            if ui
                .button("Reset")
                .on_hover_text("Set target to 0-max")
                .clicked()
            {
                lo = 0.0;
                hi = abs_max;
                response.changed = true;
            }
            if ui
                .button("Set max")
                .on_hover_text("Keep the current minimum and set maximum to the dataset maximum")
                .clicked()
            {
                hi = abs_max;
                response.changed = true;
            }
        });

        let id = ui.id().with(params.id_salt).with("range");
        let slider = crate::ui::range_slider::range_slider(
            ui,
            id,
            &mut lo,
            &mut hi,
            0.0,
            abs_max,
            params.step,
        );
        response.changed |= slider.changed();

        ui.horizontal(|ui| {
            response.changed |= ui
                .add(egui::DragValue::new(&mut lo).speed(10.0).prefix("min "))
                .changed();
            response.changed |= ui
                .add(egui::DragValue::new(&mut hi).speed(10.0).prefix("max "))
                .changed();
        });

        lo = lo.clamp(0.0, abs_max);
        hi = hi.clamp(0.0, abs_max);
        if hi <= lo {
            hi = (lo + 1.0).min(abs_max);
        }
        response.window = (lo, hi);
    });

    response
}
