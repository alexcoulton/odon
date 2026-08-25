//! Protected recovery control for layouts that mount recovery directly.

use eframe::egui;

pub(crate) fn render(ui: &mut egui::Ui) -> bool {
    ui.heading("Application layout recovery");
    ui.label(
        "Replace the active layout with Odon's protected minimal workspace. This keeps the required project or canvas mount reachable.",
    );
    ui.add_space(8.0);
    ui.button("Recover active layout").clicked()
}
