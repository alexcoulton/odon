//! The mosaic view saved with a project and read back by `restore_project_state`.

use super::*;

impl MosaicModel {
    /// Group colours, membership, and inheritance. Restored from the project config, which is
    /// shared with the project page, rather than from the saved mosaic view.
    pub(crate) fn project_layer_groups(&self) -> Option<&ProjectLayerGroups> {
        self.resource.as_ref()?;
        Some(&self.layer_groups)
    }

    /// Channels are saved by name, because positions differ when the project's mosaic is later
    /// opened with a different set of ROIs.
    pub(crate) fn project_view_state(&self) -> Option<Value> {
        self.resource.as_ref()?;
        let layers = self.native_layers.snapshots();
        let overlays = layers
            .iter()
            .filter(|layer| layer["stack"] == "overlays")
            .collect::<Vec<_>>();
        Some(json!({
            "channels":self.channels.iter().map(|channel| json!({
                "name":channel.name,
                "visible":channel.visible,
                "color_rgb":channel.color_rgb,
                "window":channel.window.map(|(minimum, maximum)| [minimum, maximum]),
                "note":channel.note,
            })).collect::<Vec<_>>(),
            "channel_order_names":self
                .channel_order
                .iter()
                .filter_map(|index| self.channels.get(*index))
                .map(|channel| channel.name.as_str())
                .collect::<Vec<_>>(),
            "active_channel_name":self
                .channels
                .get(self.active_channel)
                .map(|channel| channel.name.as_str()),
            "group_by":self.group_by,
            "sort_by":self.sort_by,
            "sort_secondary_enabled":self.sort_secondary_enabled,
            "sort_by_secondary":self.sort_by_secondary,
            "layout_mode":self.layout_mode.as_str(),
            "columns":self.columns,
            "group_gap":self.group_gap,
            "show_group_labels":self.show_group_labels,
            "show_text_labels":self.show_text_labels,
            "label_columns":self.label_columns,
            "label_font_size":self.label_font_size,
            "group_label_font_size":self.group_label_font_size,
            "ui":{
                "show_left_panel":self.show_left_panel,
                "show_right_panel":self.show_right_panel,
                "left_tab":self.left_tab,
                "right_tab":self.right_tab,
                "channel_sort":self.channel_sort,
                "smooth_pixels":self.smooth_pixels,
                "show_tile_debug":self.show_tile_debug,
            },
            "overlay_visibility":overlays
                .iter()
                .filter_map(|layer| {
                    Some((layer["layer_id"].as_str()?.to_string(), layer["visible"].clone()))
                })
                .collect::<serde_json::Map<_, _>>(),
            "overlay_order":overlays
                .iter()
                .filter_map(|layer| layer["layer_id"].as_str())
                .collect::<Vec<_>>(),
            "active_layer":layers
                .iter()
                .find(|layer| layer["active"] == true)
                .and_then(|layer| layer["layer_id"].as_str()),
            "camera":self.camera_snapshot(),
            "object_style":self.object_style,
        }))
    }

    /// Names missing from this mosaic are dropped, and its other channels keep their order after
    /// the saved ones. Returns `None` for older projects, which saved positions instead.
    pub(super) fn saved_channel_order(&self, state: &Value) -> Option<Vec<usize>> {
        let names = state.get("channel_order_names")?.as_array()?;
        let mut order = Vec::with_capacity(self.channels.len());
        for name in names.iter().filter_map(Value::as_str) {
            if let Some(index) = self
                .channels
                .iter()
                .position(|channel| channel.name == name)
                && !order.contains(&index)
            {
                order.push(index);
            }
        }
        for index in &self.channel_order {
            if !order.contains(index) {
                order.push(*index);
            }
        }
        Some(order)
    }

    pub(super) fn camera_center_is_on_mosaic(&self) -> bool {
        let [x, y] = self.camera_center;
        let [minimum, maximum] = self.bounds;
        (minimum[0]..=maximum[0]).contains(&x) && (minimum[1]..=maximum[1]).contains(&y)
    }
}
