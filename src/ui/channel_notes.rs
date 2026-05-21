use eframe::egui;

pub fn ui_channel_notes(ui: &mut egui::Ui, channel_name: &str, note: &mut String) -> bool {
    ui.separator();
    ui.heading("Notes");
    ui.label(format!("Channel: {channel_name}"));
    ui.add(
        egui::TextEdit::multiline(note)
            .desired_rows(6)
            .desired_width(f32::INFINITY),
    )
    .changed()
}
