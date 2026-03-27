use eframe::egui;

#[derive(Debug, Clone, Copy)]
pub struct ContrastUiOptions {
    pub show_nudge_buttons: bool,
    pub set_max_button_label: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct ContrastUiResult {
    pub window: (f32, f32),
    pub limits_touched: bool,
    pub set_max_all_clicked: bool,
}

pub fn ui_contrast_window(
    ui: &mut egui::Ui,
    abs_max: f32,
    window: (f32, f32),
    opts: ContrastUiOptions,
) -> ContrastUiResult {
    let abs_max = abs_max.max(1.0);

    let (mut lo, mut hi) = window;
    lo = lo.clamp(0.0, abs_max);
    hi = hi.clamp(0.0, abs_max);
    if hi <= lo {
        hi = (lo + 1.0).min(abs_max);
    }

    let mut reset_clicked = false;
    let mut set_max_all_clicked = false;
    ui.horizontal(|ui| {
        reset_clicked |= ui.button("Reset (0-max)").clicked();
        set_max_all_clicked |= ui.button(opts.set_max_button_label).clicked();
    });
    if reset_clicked {
        lo = 0.0;
        hi = abs_max;
    }

    let mut limits_touched = reset_clicked;
    if opts.show_nudge_buttons {
        ui.horizontal(|ui| {
            let deltas = [
                (-1000.0, "-1000"),
                (-100.0, "-100"),
                (100.0, "+100"),
                (1000.0, "+1000"),
            ];
            for (delta, label) in deltas {
                if ui.button(label).clicked() {
                    hi = (hi + delta).clamp(0.0, abs_max);
                    if hi <= lo {
                        hi = (lo + 1.0).min(abs_max);
                    }
                    limits_touched = true;
                }
            }
        });
    }

    limits_touched |= ui
        .add(egui::Slider::new(&mut lo, 0.0..=abs_max).text("Min"))
        .changed();
    limits_touched |= ui
        .add(egui::Slider::new(&mut hi, 0.0..=abs_max).text("Max"))
        .changed();

    ui.horizontal(|ui| {
        limits_touched |= ui
            .add(egui::DragValue::new(&mut lo).speed(10.0).prefix("min "))
            .changed();
        limits_touched |= ui
            .add(egui::DragValue::new(&mut hi).speed(10.0).prefix("max "))
            .changed();
    });

    lo = lo.clamp(0.0, abs_max);
    hi = hi.clamp(0.0, abs_max);
    if hi <= lo {
        hi = (lo + 1.0).min(abs_max);
    }

    ContrastUiResult {
        window: (lo, hi),
        limits_touched,
        set_max_all_clicked,
    }
}
