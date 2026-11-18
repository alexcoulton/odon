use super::*;

impl ObjectsLayer {
    pub(in crate::objects) fn apply_legend_actions(
        &mut self,
        actions: Vec<legend_ui::LegendAction>,
    ) {
        for action in actions {
            match action {
                legend_ui::LegendAction::SetVisible { label, visible } => {
                    self.color_level_overrides.entry(label).or_default().visible = visible;
                }
                legend_ui::LegendAction::SetColor { label, rgb } => {
                    self.color_level_overrides
                        .entry(label.clone())
                        .or_default()
                        .color_rgb = None;
                    self.category_colors.set(&label, rgb);
                }
                legend_ui::LegendAction::SetUntickedCategories(mode) => {
                    self.unticked_categories = mode;
                }
                legend_ui::LegendAction::ResetColor { label } => {
                    self.color_level_overrides
                        .entry(label.clone())
                        .or_default()
                        .color_rgb = None;
                    self.category_colors.remove(&label);
                }
            }
        }
    }

    pub(crate) fn sync_category_colors(
        &mut self,
        project: &std::collections::BTreeMap<String, [u8; 3]>,
    ) {
        self.category_colors.sync(project);
    }

    pub(crate) fn sync_property_preferences(
        &mut self,
        project: &crate::data::project_config::ObjectPropertyPreferences,
    ) {
        self.property_preferences.sync(project);
    }

    pub(crate) fn take_property_preferences_commit(&mut self) -> Option<serde_json::Value> {
        self.property_preferences.take_commit()
    }

    /// The project-metadata params for any released category colour edit.
    pub(crate) fn take_category_color_commit(
        &mut self,
        pointer_down: bool,
    ) -> Option<serde_json::Value> {
        self.category_colors
            .take_commit(pointer_down)
            .map(|colors| legend_ui::category_colors_params(&colors))
    }
}
