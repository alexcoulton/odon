use super::*;

impl ObjectsLayer {
    pub(super) fn ui_continuous_color_legend(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        let mapping = self.color_mapping().clone();
        let payload = self.ensure_continuous_color_payload().cloned();
        let ObjectColorMapping::Continuous {
            property,
            palette,
            reverse,
            ..
        } = mapping
        else {
            return;
        };
        ui.label(format!("Legend: {property}"));
        let width = ui.available_width().max(80.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 16.0), egui::Sense::hover());
        let steps = 64usize;
        for index in 0..steps {
            let start = index as f64 / steps as f64;
            let end = (index + 1) as f64 / steps as f64;
            let position = if reverse { 1.0 - start } else { start };
            let rgb = palette.color_rgb(position);
            let segment = egui::Rect::from_min_max(
                egui::pos2(rect.left() + rect.width() * start as f32, rect.top()),
                egui::pos2(rect.left() + rect.width() * end as f32, rect.bottom()),
            );
            ui.painter().rect_filled(
                segment,
                0.0,
                egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
            );
        }
        if let Some([minimum, maximum]) = self.resolved_continuous_domain() {
            ui.columns(3, |columns| {
                columns[0].label(format!("{minimum:.4}"));
                columns[1].vertical_centered(|ui| {
                    ui.label(format!("{:.4}", 0.5 * (minimum + maximum)));
                });
                columns[2].with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{maximum:.4}"));
                });
            });
        } else {
            ui.colored_label(
                egui::Color32::from_rgb(220, 150, 60),
                "No finite numeric values are available for this property.",
            );
        }
        if let Some(payload) = payload {
            ui.label(format!(
                "Numeric values: {}; missing/non-numeric: {}",
                payload.numeric_count, payload.missing_count
            ));
        }
    }
}
