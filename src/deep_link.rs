use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::data::project_config::ProjectRoi;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DeepLinkChannelContrast {
    pub channel: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkChannelColor {
    pub channel: String,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepLinkChannelOrder {
    Listed,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkObjectLevelColor {
    pub value: String,
    pub color_rgb: [u8; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct DeepLinkObjectFilterClause {
    pub property_key: String,
    pub query: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepLinkObjectFilterLogic {
    All,
    Any,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeepLinkRequest {
    pub example: Option<String>,
    pub project_path: Option<PathBuf>,
    pub roi: Option<String>,
    pub sample: Option<String>,
    pub channel: Option<String>,
    pub channel_alternatives: Vec<String>,
    pub visible_channels: Vec<String>,
    pub visible_channel_alternatives: Vec<Vec<String>>,
    pub group_visible_channels: bool,
    pub visible_channel_group: Option<String>,
    pub visible_channel_group_color: Option<[u8; 3]>,
    pub channel_order: Option<DeepLinkChannelOrder>,
    pub hidden_channels: Vec<String>,
    pub hidden_channel_alternatives: Vec<Vec<String>>,
    pub contrast_min: Option<f32>,
    pub contrast_max: Option<f32>,
    pub channel_contrasts: Vec<DeepLinkChannelContrast>,
    pub channel_colors: Vec<DeepLinkChannelColor>,
    pub segmentation: Option<String>,
    pub segmentation_source: Option<String>,
    pub load_segmentation_labels: Option<bool>,
    pub cell_color_by: Option<String>,
    pub fill_cells: Option<bool>,
    pub show_selection_overlay: Option<bool>,
    pub fast_object_rendering: Option<bool>,
    pub visible_cell_types: Vec<String>,
    pub hidden_cell_types: Vec<String>,
    pub object_level_colors: Vec<DeepLinkObjectLevelColor>,
    pub object_filters: Vec<DeepLinkObjectFilterClause>,
    pub object_filter_logic: Option<DeepLinkObjectFilterLogic>,
    pub object_query: Option<String>,
    pub center_world: Option<[f32; 2]>,
    pub zoom: Option<f32>,
}

impl DeepLinkRequest {
    pub fn parse_arg(arg: &str) -> anyhow::Result<Option<Self>> {
        if !is_deep_link(arg) {
            return Ok(None);
        }
        Ok(Some(parse_deep_link(arg)?))
    }

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

impl Default for DeepLinkRequest {
    fn default() -> Self {
        Self {
            example: None,
            project_path: None,
            roi: None,
            sample: None,
            channel: None,
            channel_alternatives: Vec::new(),
            visible_channels: Vec::new(),
            visible_channel_alternatives: Vec::new(),
            group_visible_channels: false,
            visible_channel_group: None,
            visible_channel_group_color: None,
            channel_order: None,
            hidden_channels: Vec::new(),
            hidden_channel_alternatives: Vec::new(),
            contrast_min: None,
            contrast_max: None,
            channel_contrasts: Vec::new(),
            channel_colors: Vec::new(),
            segmentation: None,
            segmentation_source: None,
            load_segmentation_labels: None,
            cell_color_by: None,
            fill_cells: None,
            show_selection_overlay: None,
            fast_object_rendering: None,
            visible_cell_types: Vec::new(),
            hidden_cell_types: Vec::new(),
            object_level_colors: Vec::new(),
            object_filters: Vec::new(),
            object_filter_logic: None,
            object_query: None,
            center_world: None,
            zoom: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[allow(dead_code)]
pub struct DeepLinkResolution {
    pub project_source: String,
    pub project_path: Option<PathBuf>,
    pub roi: ProjectRoi,
}

pub fn apply_example_defaults(request: &mut DeepLinkRequest, example: &str) {
    let normalized = normalize_example_name(example);
    if !matches!(normalized.as_str(), "synthetic5ch" | "synthetic" | "demo") {
        return;
    }
    if request.roi.is_none() {
        request.roi = Some("synthetic_5ch.ome.zarr".to_string());
    }
    if request.channel.is_none() {
        request.channel = Some("DAPI".to_string());
    }
    if request.visible_channels.is_empty() {
        request.visible_channels = vec!["DAPI".to_string(), "CD3".to_string(), "PanCK".to_string()];
    }
    if request.visible_channel_group.is_none() {
        request.visible_channel_group = Some("Synthetic example".to_string());
    }
    if request.channel_order.is_none() {
        request.channel_order = Some(DeepLinkChannelOrder::Listed);
    }
}

pub fn resolve_example_project_path(example: &str) -> Option<PathBuf> {
    let normalized = normalize_example_name(example);
    let project_name = match normalized.as_str() {
        "synthetic5ch" | "synthetic" | "demo" => "synthetic_5ch.project.json",
        _ => return None,
    };
    example_dirs()
        .into_iter()
        .map(|directory| directory.join(project_name))
        .find(|path| path.is_file())
}

/// Resolve a public deep-link ROI target against normalized project records.
///
/// Both the native project UI and the control actor use this function so matching, ambiguity, and
/// error semantics cannot drift.
pub fn resolve_roi_target(
    rois: &[ProjectRoi],
    roi_query: Option<&str>,
    sample_query: Option<&str>,
) -> Result<ProjectRoi, String> {
    let Some(roi_query) = roi_query.map(str::trim).filter(|value| !value.is_empty()) else {
        return Err("Deep link is missing a roi=... parameter.".to_string());
    };

    let roi_norm = normalize_link_match_text(roi_query);
    let sample_norm = sample_query
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_link_match_text);
    let mut matches = rois
        .iter()
        .filter(|roi| {
            sample_norm.as_ref().is_none_or(|sample| {
                roi_link_match_texts(roi)
                    .iter()
                    .any(|candidate| candidate.contains(sample))
            })
        })
        .filter(|roi| {
            roi_link_match_texts(roi)
                .iter()
                .any(|candidate| candidate == &roi_norm || candidate.contains(&roi_norm))
        })
        .cloned()
        .collect::<Vec<_>>();

    if matches.is_empty() {
        return Err(format!("No project ROI matches '{roi_query}'."));
    }
    let exact = matches
        .iter()
        .filter(|roi| {
            roi_link_match_texts(roi)
                .iter()
                .any(|candidate| candidate == &roi_norm)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !exact.is_empty() {
        matches = exact;
    }
    if matches.len() == 1 {
        return Ok(matches.remove(0));
    }
    let examples = matches
        .iter()
        .take(4)
        .map(ProjectRoi::source_display)
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "Deep link ROI '{roi_query}' matches {} project ROIs. Add sample=... or use a more specific roi path. Examples: {examples}",
        matches.len()
    ))
}

fn roi_link_match_texts(roi: &ProjectRoi) -> Vec<String> {
    let mut texts = Vec::new();
    let mut push = |value: String| {
        let normalized = normalize_link_match_text(&value);
        if !normalized.is_empty() && !texts.iter().any(|existing| existing == &normalized) {
            texts.push(normalized);
        }
    };
    push(roi.id.clone());
    if let Some(display_name) = roi.display_name.as_ref() {
        push(display_name.clone());
    }
    if let Some(dataset) = roi.dataset.as_ref() {
        push(dataset.clone());
    }
    if let Some(source_key) = roi.source_key() {
        push(source_key);
    }
    push(roi.source_display());
    if let Some(path) = roi.local_path() {
        push(path.to_string_lossy().to_string());
        if let Some(file_name) = path.file_name().and_then(|value| value.to_str()) {
            push(file_name.to_string());
        }
        let components = path
            .components()
            .filter_map(|component| component.as_os_str().to_str())
            .collect::<Vec<_>>();
        for pair in components.windows(2) {
            push(pair.join("/"));
        }
        for triple in components.windows(3) {
            push(triple.join("/"));
        }
    }
    for (key, value) in &roi.meta {
        push(key.clone());
        push(value.clone());
        push(format!("{key}:{value}"));
    }
    texts
}

fn normalize_link_match_text(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .replace('\\', "/")
        .to_ascii_lowercase()
}

fn normalize_example_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

fn example_dirs() -> Vec<PathBuf> {
    let mut directories = Vec::new();
    if let Ok(executable) = std::env::current_exe()
        && let Some(binary_directory) = executable.parent()
    {
        directories.push(binary_directory.join("examples"));
        directories.push(binary_directory.join("../Resources/examples"));
        directories.push(binary_directory.join("../../Resources/examples"));
    }
    directories.push(PathBuf::from("/usr/share/odon/examples"));
    if let Ok(current_directory) = std::env::current_dir() {
        directories.push(current_directory.join("fixtures"));
    }
    directories
}

pub fn is_deep_link(value: &str) -> bool {
    value
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("odon://"))
        || value
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("odon:"))
}

fn parse_deep_link(raw: &str) -> anyhow::Result<DeepLinkRequest> {
    let rest = raw
        .strip_prefix("odon://")
        .or_else(|| raw.strip_prefix("ODON://"))
        .or_else(|| raw.strip_prefix("odon:"))
        .or_else(|| raw.strip_prefix("ODON:"))
        .unwrap_or(raw);
    let (action, query) = match rest.split_once('?') {
        Some((action, query)) => (action.trim_matches('/'), query),
        None => (rest.trim_matches('/'), ""),
    };
    if !action.is_empty() && !action.eq_ignore_ascii_case("open") {
        anyhow::bail!("unsupported odon deep-link action '{action}'");
    }

    let mut req = DeepLinkRequest {
        example: None,
        project_path: None,
        roi: None,
        sample: None,
        channel: None,
        channel_alternatives: Vec::new(),
        visible_channels: Vec::new(),
        visible_channel_alternatives: Vec::new(),
        group_visible_channels: false,
        visible_channel_group: None,
        visible_channel_group_color: None,
        channel_order: None,
        hidden_channels: Vec::new(),
        hidden_channel_alternatives: Vec::new(),
        contrast_min: None,
        contrast_max: None,
        channel_contrasts: Vec::new(),
        channel_colors: Vec::new(),
        segmentation: None,
        segmentation_source: None,
        load_segmentation_labels: None,
        cell_color_by: None,
        fill_cells: None,
        show_selection_overlay: None,
        fast_object_rendering: None,
        visible_cell_types: Vec::new(),
        hidden_cell_types: Vec::new(),
        object_level_colors: Vec::new(),
        object_filters: Vec::new(),
        object_filter_logic: None,
        object_query: None,
        center_world: None,
        zoom: None,
    };
    let mut filter_property: Option<String> = None;
    let mut filter_query: Option<String> = None;

    for (key, value) in query_pairs(query) {
        match key.as_str() {
            "example" | "demo" | "example_dataset" => req.example = non_empty(value),
            "project" | "project_path" => req.project_path = Some(path_from_link_value(&value)?),
            "roi" | "roi_id" => req.roi = non_empty(value),
            "sample" | "case" | "dataset_id" => req.sample = non_empty(value),
            "marker" | "channel" => req.channel = non_empty(value),
            "visible_channels" | "show_channels" | "only_channels" => {
                req.visible_channels = parse_list(&value)
            }
            "group_visible_channels" | "group_channels" | "group_visible" => {
                req.group_visible_channels = parse_bool(&value).unwrap_or(false)
            }
            "visible_channel_group" | "channel_group" | "group_name" => {
                req.visible_channel_group = non_empty(value)
            }
            "visible_channel_group_color" | "channel_group_color" | "group_color" => {
                req.visible_channel_group_color = parse_color_rgb(&value)
            }
            "channel_order" | "channels_order" | "channel_sort" => {
                req.channel_order = parse_channel_order(&value)
            }
            "order_visible_channels" | "order_listed_channels" => {
                if parse_bool(&value).unwrap_or(false) {
                    req.channel_order = Some(DeepLinkChannelOrder::Listed);
                }
            }
            "hidden_channels" | "hide_channels" => req.hidden_channels = parse_list(&value),
            "contrast_min" | "channel_min" | "window_min" => {
                req.contrast_min = parse_finite_f32(&value)
            }
            "contrast_max" | "channel_max" | "window_max" => {
                req.contrast_max = parse_finite_f32(&value)
            }
            "channel_contrast" | "channel_contrasts" | "channel_window" | "channel_windows" => {
                req.channel_contrasts = parse_channel_contrasts(&value)
            }
            "channel_color" | "channel_colors" | "channel_colour" | "channel_colours" => {
                req.channel_colors = parse_channel_colors(&value)
            }
            "segmentation" | "label" | "labels" => req.segmentation = non_empty(value),
            "segmentation_source" | "segmentation_layer" | "segmentation_kind" => {
                req.segmentation_source = non_empty(value)
            }
            "load_labels"
            | "load_segmentation_labels"
            | "load_ome_zarr_labels"
            | "load_bundled_labels" => req.load_segmentation_labels = parse_bool(&value),
            "cell_color_by" | "color_by" | "object_color_by" => {
                req.cell_color_by = non_empty(value)
            }
            "fill_cells" => req.fill_cells = parse_bool(&value),
            "show_selection_overlay" | "selection_overlay" => {
                req.show_selection_overlay = parse_bool(&value)
            }
            "fast_rendering" | "fast_object_rendering" | "object_fast_rendering" => {
                req.fast_object_rendering = parse_bool(&value)
            }
            "visible_cell_types" | "show_cell_types" | "only_cell_types" | "cell_types" => {
                req.visible_cell_types = parse_list(&value)
            }
            "hidden_cell_types" | "hide_cell_types" => req.hidden_cell_types = parse_list(&value),
            "object_level_colors"
            | "object_level_colours"
            | "level_colors"
            | "level_colours"
            | "cell_type_colors"
            | "cell_type_colours"
            | "category_colors"
            | "category_colours" => req.object_level_colors = parse_object_level_colors(&value),
            "filter" | "filters" | "object_filter" | "object_filters" => {
                req.object_filters.extend(parse_object_filters(&value));
            }
            "object_query"
            | "filter_query_expr"
            | "filter_expression"
            | "object_filter_query_expr" => {
                req.object_query = non_empty(value);
            }
            "filter_logic"
            | "filter_mode"
            | "object_filter_logic"
            | "object_filter_mode"
            | "object_filters_logic"
            | "object_filters_mode" => {
                req.object_filter_logic = parse_object_filter_logic(&value);
            }
            "filter_property" | "filter_key" | "object_filter_property" | "object_filter_key" => {
                filter_property = non_empty(value);
            }
            "filter_query" | "filter_value" | "object_filter_query" | "object_filter_value" => {
                filter_query = non_empty(value);
            }
            "center" | "center_world" => req.center_world = parse_pair_f32(&value),
            "zoom" => {
                req.zoom = value
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && *v > 0.0)
            }
            "v" | "version" => {}
            _ => {}
        }
    }
    if let (Some(property_key), Some(query)) = (filter_property, filter_query)
        && !req
            .object_filters
            .iter()
            .any(|clause| clause.property_key == property_key && clause.query == query)
    {
        req.object_filters.push(DeepLinkObjectFilterClause {
            property_key,
            query,
        });
    }

    Ok(req)
}

fn query_pairs(query: &str) -> impl Iterator<Item = (String, String)> + '_ {
    query
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (key, value) = part.split_once('=').unwrap_or((part, ""));
            (
                percent_decode(key).to_ascii_lowercase(),
                percent_decode(value),
            )
        })
}

fn path_from_link_value(value: &str) -> anyhow::Result<PathBuf> {
    if let Some(rest) = value.strip_prefix("file://localhost/") {
        return Ok(PathBuf::from(format!("/{rest}")));
    }
    if let Some(rest) = value.strip_prefix("file:///") {
        return Ok(PathBuf::from(format!("/{rest}")));
    }
    if let Some(rest) = value.strip_prefix("file://") {
        return Ok(PathBuf::from(rest));
    }
    Ok(PathBuf::from(value))
}

fn non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_list(value: &str) -> Vec<String> {
    value
        .split([',', '|', ';'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_finite_f32(value: &str) -> Option<f32> {
    value.trim().parse::<f32>().ok().filter(|v| v.is_finite())
}

fn parse_channel_order(value: &str) -> Option<DeepLinkChannelOrder> {
    match value.trim().to_ascii_lowercase().as_str() {
        "listed" | "link" | "deep_link" | "deeplink" | "visible" | "visible_channels" => {
            Some(DeepLinkChannelOrder::Listed)
        }
        _ => None,
    }
}

fn parse_channel_contrasts(value: &str) -> Vec<DeepLinkChannelContrast> {
    value
        .split(['|', ';'])
        .filter_map(|item| {
            let mut parts = item.rsplitn(3, ':');
            let max = parse_finite_f32(parts.next()?.trim())?;
            let min = parse_finite_f32(parts.next()?.trim())?;
            let channel = parts.next()?.trim();
            if channel.is_empty() || max <= min {
                return None;
            }
            Some(DeepLinkChannelContrast {
                channel: channel.to_string(),
                min,
                max,
            })
        })
        .collect()
}

fn parse_channel_colors(value: &str) -> Vec<DeepLinkChannelColor> {
    value
        .split(['|', ';'])
        .filter_map(|item| {
            let item = item.trim();
            let (channel, color) = item.rsplit_once(':').or_else(|| item.rsplit_once('='))?;
            let channel = channel.trim();
            if channel.is_empty() {
                return None;
            }
            Some(DeepLinkChannelColor {
                channel: channel.to_string(),
                color_rgb: parse_color_rgb(color.trim())?,
            })
        })
        .collect()
}

fn parse_object_level_colors(value: &str) -> Vec<DeepLinkObjectLevelColor> {
    value
        .split(['|', ';'])
        .filter_map(|item| {
            let item = item.trim();
            let (value, color) = item.rsplit_once(':').or_else(|| item.rsplit_once('='))?;
            let value = value.trim();
            if value.is_empty() {
                return None;
            }
            Some(DeepLinkObjectLevelColor {
                value: value.to_string(),
                color_rgb: parse_color_rgb(color.trim())?,
            })
        })
        .collect()
}

fn parse_object_filters(value: &str) -> Vec<DeepLinkObjectFilterClause> {
    value
        .split(['|', ';'])
        .filter_map(parse_object_filter_clause)
        .collect()
}

fn parse_object_filter_logic(value: &str) -> Option<DeepLinkObjectFilterLogic> {
    match value.trim().to_ascii_lowercase().as_str() {
        "all" | "and" => Some(DeepLinkObjectFilterLogic::All),
        "any" | "or" => Some(DeepLinkObjectFilterLogic::Any),
        _ => None,
    }
}

fn parse_object_filter_clause(item: &str) -> Option<DeepLinkObjectFilterClause> {
    let item = item.trim();
    let (property_key, query) = item
        .split_once("==")
        .or_else(|| item.split_once('='))
        .or_else(|| item.split_once(':'))
        .or_else(|| item.split_once('~'))?;
    let property_key = property_key.trim();
    let query = query.trim();
    if property_key.is_empty() || query.is_empty() {
        return None;
    }
    Some(DeepLinkObjectFilterClause {
        property_key: property_key.to_string(),
        query: query.to_string(),
    })
}

fn parse_color_rgb(value: &str) -> Option<[u8; 3]> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "white" => Some([255, 255, 255]),
        "black" => Some([0, 0, 0]),
        "red" => Some([230, 57, 70]),
        "green" => Some([42, 157, 143]),
        "blue" => Some([69, 123, 157]),
        "cyan" => Some([0, 188, 212]),
        "magenta" => Some([216, 27, 96]),
        "yellow" => Some([255, 202, 40]),
        "orange" => Some([251, 133, 0]),
        "purple" => Some([126, 87, 194]),
        "pink" => Some([244, 143, 177]),
        "lime" => Some([139, 195, 74]),
        "teal" => Some([0, 150, 136]),
        "amber" => Some([255, 193, 7]),
        "gray" | "grey" => Some([158, 158, 158]),
        _ => parse_hex_color_rgb(trimmed),
    }
}

fn parse_hex_color_rgb(value: &str) -> Option<[u8; 3]> {
    let hex = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        return Some([r, g, b]);
    }
    if hex.len() == 3 {
        let r = u8::from_str_radix(&hex[0..1], 16).ok()?;
        let g = u8::from_str_radix(&hex[1..2], 16).ok()?;
        let b = u8::from_str_radix(&hex[2..3], 16).ok()?;
        return Some([r * 17, g * 17, b * 17]);
    }
    None
}

fn parse_pair_f32(value: &str) -> Option<[f32; 2]> {
    let (x, y) = value.split_once(',')?;
    let x = x.trim().parse::<f32>().ok()?;
    let y = y.trim().parse::<f32>().ok()?;
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = from_hex(bytes[i + 1]);
                let lo = from_hex(bytes[i + 2]);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn from_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_open_link() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?v=1&project=file:///tmp/my%20project.json&roi=18S1746%2FROI2&marker=CD68&segmentation=cells&segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type&fill_cells=1&show_selection_overlay=0&fast_rendering=0&center=10.5,20&zoom=0.25",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            req.project_path,
            Some(PathBuf::from("/tmp/my project.json"))
        );
        assert_eq!(req.roi.as_deref(), Some("18S1746/ROI2"));
        assert_eq!(req.channel.as_deref(), Some("CD68"));
        assert!(req.visible_channels.is_empty());
        assert_eq!(req.channel_order, None);
        assert!(req.hidden_channels.is_empty());
        assert_eq!(req.contrast_min, None);
        assert_eq!(req.contrast_max, None);
        assert!(req.channel_contrasts.is_empty());
        assert_eq!(req.segmentation.as_deref(), Some("cells"));
        assert_eq!(req.segmentation_source.as_deref(), Some("geoparquet"));
        assert_eq!(req.load_segmentation_labels, Some(false));
        assert_eq!(req.cell_color_by.as_deref(), Some("broad_cell_type"));
        assert_eq!(req.fill_cells, Some(true));
        assert_eq!(req.show_selection_overlay, Some(false));
        assert_eq!(req.fast_object_rendering, Some(false));
        assert!(req.visible_cell_types.is_empty());
        assert!(req.hidden_cell_types.is_empty());
        assert!(req.object_level_colors.is_empty());
        assert_eq!(req.center_world, Some([10.5, 20.0]));
        assert_eq!(req.zoom, Some(0.25));
    }

    #[test]
    fn ignores_non_odon_args() {
        assert!(DeepLinkRequest::parse_arg("--project").unwrap().is_none());
    }

    #[test]
    fn parses_example_alias() {
        let req = DeepLinkRequest::parse_arg("odon://open?example=synthetic_5ch")
            .unwrap()
            .unwrap();
        assert_eq!(req.example.as_deref(), Some("synthetic_5ch"));
    }

    #[test]
    fn parses_load_label_aliases() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?segmentation_source=geoparquet&load_ome_zarr_labels=false",
        )
        .unwrap()
        .unwrap();

        assert_eq!(req.segmentation_source.as_deref(), Some("geoparquet"));
        assert_eq!(req.load_segmentation_labels, Some(false));

        let req = DeepLinkRequest::parse_arg("odon://open?load_bundled_labels=1")
            .unwrap()
            .unwrap();
        assert_eq!(req.load_segmentation_labels, Some(true));
    }

    #[test]
    fn parses_cell_type_visibility_lists() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?cell_color_by=broad_cell_type&visible_cell_types=tumor_myogenic%7Cimmune_myeloid&hide_cell_types=unknown,ambiguous_mixed",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            req.visible_cell_types,
            vec!["tumor_myogenic".to_string(), "immune_myeloid".to_string()]
        );
        assert_eq!(
            req.hidden_cell_types,
            vec!["unknown".to_string(), "ambiguous_mixed".to_string()]
        );
    }

    #[test]
    fn parses_object_filter_clauses() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?filter=broad_cell_type:immune_myeloid%7Czz_mask_galectin_3%3D%3DTRUE&filter_property=sample_id&filter_query=18S1746&filter_logic=or",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            req.object_filters,
            vec![
                DeepLinkObjectFilterClause {
                    property_key: "broad_cell_type".to_string(),
                    query: "immune_myeloid".to_string(),
                },
                DeepLinkObjectFilterClause {
                    property_key: "zz_mask_galectin_3".to_string(),
                    query: "TRUE".to_string(),
                },
                DeepLinkObjectFilterClause {
                    property_key: "sample_id".to_string(),
                    query: "18S1746".to_string(),
                },
            ]
        );
        assert_eq!(
            req.object_filter_logic,
            Some(DeepLinkObjectFilterLogic::Any)
        );
    }

    #[test]
    fn parses_object_filter_logic_aliases() {
        let req = DeepLinkRequest::parse_arg("odon://open?object_filter_mode=all")
            .unwrap()
            .unwrap();
        assert_eq!(
            req.object_filter_logic,
            Some(DeepLinkObjectFilterLogic::All)
        );

        let req = DeepLinkRequest::parse_arg("odon://open?object_filters_logic=any")
            .unwrap()
            .unwrap();
        assert_eq!(
            req.object_filter_logic,
            Some(DeepLinkObjectFilterLogic::Any)
        );

        let req = DeepLinkRequest::parse_arg("odon://open?filter_logic=unexpected")
            .unwrap()
            .unwrap();
        assert_eq!(req.object_filter_logic, None);
    }

    #[test]
    fn parses_object_query() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?object_query=(broad_cell_type%20%3D%3D%20%22immune_lymphoid%22)%20or%20zz_mask_hla_dr",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            req.object_query.as_deref(),
            Some("(broad_cell_type == \"immune_lymphoid\") or zz_mask_hla_dr")
        );
    }

    #[test]
    fn parses_object_level_colours() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?cell_color_by=broad_cell_type&object_level_colors=tumor_myogenic:%23ff4f8b%7Cimmune_myeloid:cyan%7Cendothelial=00aa66",
        )
        .unwrap()
        .unwrap();

        assert_eq!(
            req.object_level_colors,
            vec![
                DeepLinkObjectLevelColor {
                    value: "tumor_myogenic".to_string(),
                    color_rgb: [255, 79, 139],
                },
                DeepLinkObjectLevelColor {
                    value: "immune_myeloid".to_string(),
                    color_rgb: [0, 188, 212],
                },
                DeepLinkObjectLevelColor {
                    value: "endothelial".to_string(),
                    color_rgb: [0, 170, 102],
                },
            ]
        );
    }

    #[test]
    fn parses_channel_visibility_and_contrast() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?channel=CD3&visible_channels=CD3%7CCD8&channel_order=listed&group_visible_channels=1&visible_channel_group=T%20cell%20markers&visible_channel_group_color=%23ffffff&channel_color=CD3:red%7CCD8:%2300ccff&hidden_channels=DAPI&contrast_min=120&contrast_max=4500&channel_contrast=CD3:120:4500%7CCD8:80:3000",
        )
        .unwrap()
        .unwrap();

        assert_eq!(req.channel.as_deref(), Some("CD3"));
        assert_eq!(
            req.visible_channels,
            vec!["CD3".to_string(), "CD8".to_string()]
        );
        assert_eq!(req.channel_order, Some(DeepLinkChannelOrder::Listed));
        assert!(req.group_visible_channels);
        assert_eq!(
            req.visible_channel_group,
            Some("T cell markers".to_string())
        );
        assert_eq!(req.visible_channel_group_color, Some([255, 255, 255]));
        assert_eq!(
            req.channel_colors,
            vec![
                DeepLinkChannelColor {
                    channel: "CD3".to_string(),
                    color_rgb: [230, 57, 70],
                },
                DeepLinkChannelColor {
                    channel: "CD8".to_string(),
                    color_rgb: [0, 204, 255],
                },
            ]
        );
        assert_eq!(req.hidden_channels, vec!["DAPI".to_string()]);
        assert_eq!(req.contrast_min, Some(120.0));
        assert_eq!(req.contrast_max, Some(4500.0));
        assert_eq!(
            req.channel_contrasts,
            vec![
                DeepLinkChannelContrast {
                    channel: "CD3".to_string(),
                    min: 120.0,
                    max: 4500.0,
                },
                DeepLinkChannelContrast {
                    channel: "CD8".to_string(),
                    min: 80.0,
                    max: 3000.0,
                },
            ]
        );
    }

    #[test]
    fn parses_hex_and_named_colours() {
        assert_eq!(parse_color_rgb("#abc"), Some([170, 187, 204]));
        assert_eq!(parse_color_rgb("00ff80"), Some([0, 255, 128]));
        assert_eq!(parse_color_rgb("cyan"), Some([0, 188, 212]));
        assert_eq!(parse_color_rgb("not-a-colour"), None);
    }

    #[test]
    fn canonical_url_round_trips_public_state() {
        let request = DeepLinkRequest {
            project_path: Some(PathBuf::from("/tmp/My project.json")),
            roi: Some("ROI/2".to_string()),
            visible_channels: vec!["CD3".to_string(), "CD8".to_string()],
            group_visible_channels: true,
            visible_channel_group_color: Some([255, 79, 139]),
            channel_order: Some(DeepLinkChannelOrder::Listed),
            channel_contrasts: vec![DeepLinkChannelContrast {
                channel: "CD3".to_string(),
                min: 12.5,
                max: 4500.0,
            }],
            channel_colors: vec![DeepLinkChannelColor {
                channel: "CD3".to_string(),
                color_rgb: [0, 204, 255],
            }],
            fill_cells: Some(true),
            object_filter_logic: Some(DeepLinkObjectFilterLogic::Any),
            object_filters: vec![DeepLinkObjectFilterClause {
                property_key: "cell_type".to_string(),
                query: "T cell".to_string(),
            }],
            center_world: Some([10.25, 20.5]),
            zoom: Some(0.5),
            ..DeepLinkRequest::default()
        };

        let url = request.to_url();
        let parsed = DeepLinkRequest::parse_arg(&url).unwrap().unwrap();
        assert_eq!(parsed.project_path, request.project_path);
        assert_eq!(parsed.roi, request.roi);
        assert_eq!(parsed.visible_channels, request.visible_channels);
        assert_eq!(
            parsed.group_visible_channels,
            request.group_visible_channels
        );
        assert_eq!(
            parsed.visible_channel_group_color,
            request.visible_channel_group_color
        );
        assert_eq!(parsed.channel_order, request.channel_order);
        assert_eq!(parsed.channel_contrasts, request.channel_contrasts);
        assert_eq!(parsed.channel_colors, request.channel_colors);
        assert_eq!(parsed.fill_cells, request.fill_cells);
        assert_eq!(parsed.object_filter_logic, request.object_filter_logic);
        assert_eq!(parsed.object_filters, request.object_filters);
        assert_eq!(parsed.center_world, request.center_world);
        assert_eq!(parsed.zoom, request.zoom);
    }

    #[test]
    fn structured_request_defaults_missing_fields() {
        let request: DeepLinkRequest = serde_json::from_value(serde_json::json!({
            "roi": "ROI-1",
            "fill_cells": false
        }))
        .unwrap();
        assert_eq!(request.roi.as_deref(), Some("ROI-1"));
        assert_eq!(request.fill_cells, Some(false));
        assert!(request.visible_channels.is_empty());
    }
}
