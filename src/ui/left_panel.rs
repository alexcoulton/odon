use eframe::egui;

#[derive(Debug, Clone, Copy)]
pub struct TabSpec<T: Copy + Eq> {
    pub tab: T,
    pub label: &'static str,
    /// Distinguishes persisted panel state between tabs.
    pub panel_key: &'static str,
    /// Default width to use for this tab's side panel.
    pub default_width: f32,
    /// If true, wraps the tab body in a vertical scroll area.
    pub scroll: bool,
}

pub fn show<T: Copy + Eq>(
    ctx: &egui::Context,
    panel_id: &'static str,
    tab: &mut T,
    tabs: &[TabSpec<T>],
    mut body: impl FnMut(&mut egui::Ui, T),
) {
    let active = tabs.iter().find(|s| s.tab == *tab).unwrap_or_else(|| {
        tabs.first()
            .expect("ui_left_panel::show requires at least one tab spec")
    });
    let panel_state_id = egui::Id::new((panel_id, active.panel_key));
    let previous_panel_key_id = egui::Id::new((panel_id, "previous-panel-key"));
    let previous_panel_key = ctx.data_mut(|d| d.get_temp::<&'static str>(previous_panel_key_id));
    let tab_changed = previous_panel_key.is_some_and(|prev| prev != active.panel_key);

    let mut panel = egui::SidePanel::left(panel_state_id).resizable(true);
    panel = if tab_changed {
        panel.exact_width(active.default_width)
    } else {
        panel.default_width(active.default_width)
    };

    panel.show(ctx, |ui| {
        if tabs.len() > 1 {
            ui.horizontal(|ui| {
                for spec in tabs {
                    ui.selectable_value(tab, spec.tab, spec.label);
                }
            });
            ui.separator();
        }

        if active.scroll {
            egui::ScrollArea::vertical()
                .id_salt((panel_state_id, "scroll"))
                .auto_shrink([false, false])
                .show(ui, |ui| body(ui, *tab));
        } else {
            body(ui, *tab);
        }
    });

    ctx.data_mut(|d| d.insert_temp(previous_panel_key_id, active.panel_key));
}
