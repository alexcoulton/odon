use eframe::egui;

pub fn show_tooltip_at_pointer(
    ctx: &egui::Context,
    id_salt: egui::Id,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let parent_layer = egui::LayerId::new(egui::Order::Tooltip, id_salt);
    let _ = egui::Tooltip::always_open(
        ctx.clone(),
        parent_layer,
        id_salt,
        egui::PopupAnchor::Pointer,
    )
    .gap(12.0)
    .show(add_contents);
}
