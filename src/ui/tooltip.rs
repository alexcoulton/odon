use eframe::egui;

pub fn show_tooltip_at_pointer(
    ctx: &egui::Context,
    id_salt: egui::Id,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    let layer_id = egui::LayerId::new(egui::Order::Tooltip, id_salt);

    // TODO: `egui::show_tooltip_at_pointer` is deprecated in egui 0.33.
    // Keep it wrapped here so we can swap implementations in one place.
    egui::show_tooltip_at_pointer(ctx, layer_id, id_salt, add_contents);
}
