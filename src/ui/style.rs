use eframe::egui;

use crate::ui::icons::try_install_fontawesome;

pub fn apply_napari_like_dark(ctx: &egui::Context) {
    let mut style: egui::Style = (*ctx.style()).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.menu_margin = egui::Margin::symmetric(10, 8);
    style.spacing.indent = 18.0;
    style.spacing.slider_width = 220.0;

    let mut visuals = egui::Visuals::dark();

    // Backgrounds
    visuals.panel_fill = egui::Color32::from_rgb(18, 18, 20);
    visuals.window_fill = egui::Color32::from_rgb(22, 22, 26);
    visuals.faint_bg_color = egui::Color32::from_rgb(28, 28, 32);
    visuals.extreme_bg_color = egui::Color32::from_rgb(10, 10, 12);

    // Rounded corners & subtle strokes
    let corner = egui::CornerRadius::same(8);
    visuals.window_corner_radius = corner;
    visuals.menu_corner_radius = corner;
    visuals.widgets.noninteractive.corner_radius = corner;
    visuals.widgets.inactive.corner_radius = corner;
    visuals.widgets.hovered.corner_radius = corner;
    visuals.widgets.active.corner_radius = corner;
    visuals.widgets.open.corner_radius = corner;

    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(22, 22, 26);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(26, 26, 30);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(34, 34, 40);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(42, 42, 50);

    visuals.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 45, 55));
    visuals.widgets.inactive.bg_stroke =
        egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 50, 62));
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(72, 72, 92));
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(95, 95, 120));

    // Accent (selection / links)
    let accent = egui::Color32::from_rgb(70, 130, 255);
    visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(70, 130, 255, 80);
    visuals.selection.stroke = egui::Stroke::new(1.0, accent);
    visuals.hyperlink_color = accent;

    // Slightly softer shadows
    visuals.window_shadow = egui::Shadow {
        offset: [0, 8],
        blur: 18,
        spread: 0,
        color: egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
    };

    style.visuals = visuals;
    ctx.set_style(style);

    // Optional: icon font (Font Awesome). If not present, the app falls back to vector icons.
    let _ = try_install_fontawesome(ctx);
}
