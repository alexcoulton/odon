use super::*;

impl DeepLinkRequest {
    /// Serialize this request into Odon's canonical `odon://open` URL form.
    ///
    /// The URL form intentionally contains only public deep-link fields. Internal
    /// channel-alternative hints are collapsed to their first usable name because
    /// the installed URL protocol has no separate representation for them.
    pub fn to_url(&self) -> String {
        let mut query = url::form_urlencoded::Serializer::new(String::new());
        append_option(&mut query, "example", self.example.as_deref());
        if let Some(path) = self.project_path.as_deref() {
            query.append_pair("project", &path.to_string_lossy());
        }
        append_option(&mut query, "roi", self.roi.as_deref());
        append_option(&mut query, "sample", self.sample.as_deref());
        append_option(
            &mut query,
            "channel",
            self.channel
                .as_deref()
                .or_else(|| self.channel_alternatives.first().map(String::as_str)),
        );

        let visible =
            canonical_channel_names(&self.visible_channels, &self.visible_channel_alternatives);
        append_list(&mut query, "visible_channels", &visible);
        if self.group_visible_channels {
            query.append_pair("group_visible_channels", "1");
        }
        append_option(
            &mut query,
            "visible_channel_group",
            self.visible_channel_group.as_deref(),
        );
        if let Some(color) = self.visible_channel_group_color {
            query.append_pair("visible_channel_group_color", &format_color(color));
        }
        if matches!(self.channel_order, Some(DeepLinkChannelOrder::Listed)) {
            query.append_pair("channel_order", "listed");
        }
        let hidden =
            canonical_channel_names(&self.hidden_channels, &self.hidden_channel_alternatives);
        append_list(&mut query, "hidden_channels", &hidden);
        append_f32(&mut query, "contrast_min", self.contrast_min);
        append_f32(&mut query, "contrast_max", self.contrast_max);
        if !self.channel_contrasts.is_empty() {
            let value = self
                .channel_contrasts
                .iter()
                .map(|item| format!("{}:{}:{}", item.channel, item.min, item.max))
                .collect::<Vec<_>>()
                .join("|");
            query.append_pair("channel_contrasts", &value);
        }
        if !self.channel_colors.is_empty() {
            let value = self
                .channel_colors
                .iter()
                .map(|item| format!("{}:{}", item.channel, format_color(item.color_rgb)))
                .collect::<Vec<_>>()
                .join("|");
            query.append_pair("channel_colors", &value);
        }
        append_option(&mut query, "segmentation", self.segmentation.as_deref());
        append_option(
            &mut query,
            "segmentation_source",
            self.segmentation_source.as_deref(),
        );
        append_bool(
            &mut query,
            "load_segmentation_labels",
            self.load_segmentation_labels,
        );
        append_option(&mut query, "cell_color_by", self.cell_color_by.as_deref());
        append_bool(&mut query, "fill_cells", self.fill_cells);
        append_bool(
            &mut query,
            "show_selection_overlay",
            self.show_selection_overlay,
        );
        append_bool(
            &mut query,
            "fast_object_rendering",
            self.fast_object_rendering,
        );
        append_list(&mut query, "visible_cell_types", &self.visible_cell_types);
        append_list(&mut query, "hidden_cell_types", &self.hidden_cell_types);
        if !self.object_level_colors.is_empty() {
            let value = self
                .object_level_colors
                .iter()
                .map(|item| format!("{}:{}", item.value, format_color(item.color_rgb)))
                .collect::<Vec<_>>()
                .join("|");
            query.append_pair("object_level_colors", &value);
        }
        if !self.object_filters.is_empty() {
            let value = self
                .object_filters
                .iter()
                .map(|item| format!("{}:{}", item.property_key, item.query))
                .collect::<Vec<_>>()
                .join("|");
            query.append_pair("object_filters", &value);
        }
        if let Some(logic) = self.object_filter_logic {
            query.append_pair(
                "object_filter_logic",
                match logic {
                    DeepLinkObjectFilterLogic::All => "all",
                    DeepLinkObjectFilterLogic::Any => "any",
                },
            );
        }
        append_option(&mut query, "object_query", self.object_query.as_deref());
        if let Some([x, y]) = self.center_world {
            query.append_pair("center", &format!("{x},{y}"));
        }
        append_f32(&mut query, "zoom", self.zoom);

        let query = query.finish();
        if query.is_empty() {
            "odon://open".to_string()
        } else {
            format!("odon://open?{query}")
        }
    }
}

fn append_option(
    query: &mut url::form_urlencoded::Serializer<'_, String>,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        query.append_pair(key, value);
    }
}

fn append_list(
    query: &mut url::form_urlencoded::Serializer<'_, String>,
    key: &str,
    values: &[String],
) {
    if !values.is_empty() {
        query.append_pair(key, &values.join("|"));
    }
}

fn append_bool(
    query: &mut url::form_urlencoded::Serializer<'_, String>,
    key: &str,
    value: Option<bool>,
) {
    if let Some(value) = value {
        query.append_pair(key, if value { "1" } else { "0" });
    }
}

fn append_f32(
    query: &mut url::form_urlencoded::Serializer<'_, String>,
    key: &str,
    value: Option<f32>,
) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        query.append_pair(key, &value.to_string());
    }
}

fn canonical_channel_names(names: &[String], alternatives: &[Vec<String>]) -> Vec<String> {
    if !names.is_empty() {
        return names.to_vec();
    }
    alternatives
        .iter()
        .filter_map(|terms| terms.first().cloned())
        .collect()
}

fn format_color([r, g, b]: [u8; 3]) -> String {
    format!("#{r:02x}{g:02x}{b:02x}")
}
