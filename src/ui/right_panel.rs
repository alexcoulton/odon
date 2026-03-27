use eframe::egui;

#[derive(Debug, Clone, Copy)]
pub struct TabSpec<T: Copy + Eq> {
    pub tab: T,
    pub label: &'static str,
    /// If true, wraps the tab body in a vertical scroll area.
    pub scroll: bool,
}

pub fn show<T: Copy + Eq>(
    ctx: &egui::Context,
    panel_id: &'static str,
    default_width: f32,
    tab: &mut T,
    tabs: &[TabSpec<T>],
    mut body: impl FnMut(&mut egui::Ui, T),
) {
    egui::SidePanel::right(panel_id)
        .default_width(default_width)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                for spec in tabs {
                    ui.selectable_value(tab, spec.tab, spec.label);
                }
            });
            ui.separator();

            let scroll = tabs
                .iter()
                .find(|s| s.tab == *tab)
                .map(|s| s.scroll)
                .unwrap_or(false);

            if scroll {
                egui::ScrollArea::vertical()
                    .id_salt((panel_id, "scroll"))
                    .auto_shrink([false, false])
                    .show(ui, |ui| body(ui, *tab));
            } else {
                body(ui, *tab);
            }
        });
}
