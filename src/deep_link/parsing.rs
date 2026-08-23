use std::path::PathBuf;

use super::*;

pub fn is_deep_link(value: &str) -> bool {
    value
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("odon://"))
        || value
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("odon:"))
}

pub(super) fn parse_deep_link(raw: &str) -> anyhow::Result<DeepLinkRequest> {
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

pub(super) fn parse_color_rgb(value: &str) -> Option<[u8; 3]> {
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
