use eframe::egui;

pub fn paint_hud(ui: &egui::Ui, rect: egui::Rect, text: impl Into<String>) {
    ui.painter().text(
        rect.left_top() + egui::vec2(8.0, 8.0),
        egui::Align2::LEFT_TOP,
        text.into(),
        egui::FontId::monospace(12.0),
        egui::Color32::from_gray(220),
    );
}

#[derive(Debug, Clone, Copy)]
pub struct ScaleBarParams {
    /// Camera zoom in screen points per lvl0 pixel.
    pub zoom_screen_per_lvl0_px: f32,
    /// Micrometers per lvl0 pixel if known.
    pub um_per_lvl0_px: Option<f32>,
    /// Visual size multiplier for screenshot/export use.
    pub scale: f32,
}

fn nice_125_leq(target: f32) -> f32 {
    if !target.is_finite() || target <= 0.0 {
        return 0.0;
    }
    let e = target.log10().floor();
    let pow10 = 10.0f32.powf(e);
    let m = target / pow10;
    let base = if m >= 5.0 {
        5.0
    } else if m >= 2.0 {
        2.0
    } else if m >= 1.0 {
        1.0
    } else {
        // Go down one decade.
        return nice_125_leq(target * 10.0) / 10.0;
    };
    base * pow10
}

fn format_um_or_mm(um: f32) -> String {
    if um >= 1000.0 {
        let mm = um / 1000.0;
        if (mm - mm.round()).abs() < 1e-3 {
            format!("{:.0} mm", mm)
        } else if mm >= 10.0 {
            format!("{:.1} mm", mm)
        } else {
            format!("{:.2} mm", mm)
        }
    } else if um >= 10.0 {
        format!("{:.0} µm", um)
    } else if um >= 1.0 {
        format!("{:.1} µm", um)
    } else {
        format!("{:.2} µm", um)
    }
}

pub fn paint_scale_bar(ui: &egui::Ui, rect: egui::Rect, params: ScaleBarParams) {
    let zoom = params.zoom_screen_per_lvl0_px.max(1e-6);
    let scale = params.scale.clamp(0.25, 4.0);
    let target_bar_px = 120.0f32 * scale;

    let (bar_px, label) = if let Some(um_per_px) = params.um_per_lvl0_px.filter(|v| *v > 0.0) {
        let um_per_screen_px = um_per_px / zoom;
        let target_um = um_per_screen_px * target_bar_px;
        let nice_um = nice_125_leq(target_um).max(um_per_screen_px);
        let bar_px = (nice_um / um_per_screen_px).clamp(24.0, 220.0);
        (bar_px, format_um_or_mm(nice_um))
    } else {
        // Unknown physical scale: use pixels.
        let world_px_per_screen_px = 1.0 / zoom;
        let target_world_px = world_px_per_screen_px * target_bar_px;
        let nice_world_px = nice_125_leq(target_world_px).max(1.0);
        let bar_px = (nice_world_px * zoom).clamp(24.0, 220.0);
        (bar_px, format!("{:.0} px", nice_world_px))
    };

    let pad = 10.0 * scale;
    let x0 = rect.left() + pad;
    let y0 = rect.bottom() - pad;

    let bar_h = 4.0 * scale;
    let text_h = ui.text_style_height(&egui::TextStyle::Small) * scale;
    let bg_w = bar_px + 18.0 * scale;
    let bg_h = bar_h + text_h + 14.0 * scale;

    let bg = egui::Rect::from_min_size(egui::pos2(x0, y0 - bg_h), egui::vec2(bg_w, bg_h));
    ui.painter()
        .rect_filled(bg, 4.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160));

    let bar = egui::Rect::from_min_size(
        egui::pos2(x0 + 9.0 * scale, y0 - 9.0 * scale - bar_h),
        egui::vec2(bar_px, bar_h),
    );
    ui.painter()
        .rect_filled(bar, 1.0, egui::Color32::from_gray(230));

    ui.painter().text(
        egui::pos2(bar.left(), bar.top() - 6.0),
        egui::Align2::LEFT_BOTTOM,
        label,
        egui::FontId::proportional(11.0 * scale),
        egui::Color32::from_gray(230),
    );
}

pub fn paint_spinner(ui: &egui::Ui, rect: egui::Rect, visible: bool, label: Option<&str>) {
    if !visible {
        return;
    }
    let size = 18.0;
    let pad = 10.0;
    let spinner_rect = egui::Rect::from_min_size(
        egui::pos2(rect.right() - pad - size, rect.top() + pad),
        egui::vec2(size, size),
    );
    egui::Spinner::new()
        .size(size)
        .color(egui::Color32::from_rgb(120, 200, 255))
        .paint_at(ui, spinner_rect);

    if let Some(label) = label.filter(|s| !s.trim().is_empty()) {
        ui.painter().text(
            egui::pos2(spinner_rect.left() - 6.0, spinner_rect.center().y),
            egui::Align2::RIGHT_CENTER,
            label,
            egui::FontId::monospace(12.0),
            egui::Color32::from_gray(220),
        );
    }
}

pub fn paint_loading_badge(ui: &egui::Ui, rect: egui::Rect, text: &str) {
    if text.trim().is_empty() {
        return;
    }

    let font = egui::FontId::proportional(11.0);
    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        font.clone(),
        egui::Color32::from_gray(235),
    );
    let pad = egui::vec2(8.0, 6.0);
    let size = galley.size() + pad * 2.0;
    let badge = egui::Rect::from_min_size(
        egui::pos2(rect.right() - 10.0 - size.x, rect.top() + 34.0),
        size,
    );

    ui.painter().rect_filled(
        badge,
        6.0,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 170),
    );
    ui.painter()
        .galley(badge.min + pad, galley, egui::Color32::from_gray(235));
}

pub fn paint_marker_legend(
    ui: &egui::Ui,
    rect: egui::Rect,
    entries: &[(egui::Color32, String)],
    scale: f32,
) {
    if entries.is_empty() {
        return;
    }
    let scale = scale.clamp(0.25, 4.0);
    let max_items = 24usize;
    let mut shown: Vec<(egui::Color32, String)> = Vec::new();
    shown.extend(entries.iter().take(max_items).cloned());
    let extra = entries.len().saturating_sub(shown.len());
    if extra > 0 {
        shown.push((egui::Color32::from_gray(220), format!("+{extra} more")));
    }

    let pad = egui::vec2(10.0 * scale, 10.0 * scale);
    let swatch = 10.0 * scale;
    let gap = 6.0 * scale;
    let font = egui::FontId::proportional(11.0 * scale);
    let row_h = ui.text_style_height(&egui::TextStyle::Small).max(swatch);

    let mut max_text_w = 0.0f32;
    for (_, name) in &shown {
        let galley =
            ui.painter()
                .layout_no_wrap(name.clone(), font.clone(), egui::Color32::from_gray(230));
        max_text_w = max_text_w.max(galley.size().x);
    }
    let bg_w = pad.x * 2.0 + swatch + gap + max_text_w;
    let bg_h = pad.y * 2.0 + row_h * (shown.len() as f32);

    let bg_min = egui::pos2(rect.right() - 10.0 - bg_w, rect.bottom() - 10.0 - bg_h);
    let bg = egui::Rect::from_min_size(bg_min, egui::vec2(bg_w, bg_h));
    ui.painter()
        .rect_filled(bg, 6.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 160));

    let mut y = bg.top() + pad.y + row_h * 0.5;
    for (color, name) in &shown {
        let cy = y;
        let sw = egui::Rect::from_center_size(
            egui::pos2(bg.left() + pad.x + swatch * 0.5, cy),
            egui::vec2(swatch, swatch),
        );
        ui.painter().rect_filled(sw, 2.0, *color);
        ui.painter().text(
            egui::pos2(sw.right() + gap, cy),
            egui::Align2::LEFT_CENTER,
            name,
            font.clone(),
            egui::Color32::from_gray(230),
        );
        y += row_h;
    }
}
