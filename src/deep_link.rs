use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub struct DeepLinkChannelContrast {
    pub channel: String,
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DeepLinkRequest {
    pub project_path: Option<PathBuf>,
    pub roi: Option<String>,
    pub sample: Option<String>,
    pub channel: Option<String>,
    pub channel_alternatives: Vec<String>,
    pub visible_channels: Vec<String>,
    pub visible_channel_alternatives: Vec<Vec<String>>,
    pub hidden_channels: Vec<String>,
    pub hidden_channel_alternatives: Vec<Vec<String>>,
    pub contrast_min: Option<f32>,
    pub contrast_max: Option<f32>,
    pub channel_contrasts: Vec<DeepLinkChannelContrast>,
    pub segmentation: Option<String>,
    pub segmentation_source: Option<String>,
    pub load_segmentation_labels: Option<bool>,
    pub cell_color_by: Option<String>,
    pub fill_cells: Option<bool>,
    pub show_selection_overlay: Option<bool>,
    pub visible_cell_types: Vec<String>,
    pub hidden_cell_types: Vec<String>,
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
        project_path: None,
        roi: None,
        sample: None,
        channel: None,
        channel_alternatives: Vec::new(),
        visible_channels: Vec::new(),
        visible_channel_alternatives: Vec::new(),
        hidden_channels: Vec::new(),
        hidden_channel_alternatives: Vec::new(),
        contrast_min: None,
        contrast_max: None,
        channel_contrasts: Vec::new(),
        segmentation: None,
        segmentation_source: None,
        load_segmentation_labels: None,
        cell_color_by: None,
        fill_cells: None,
        show_selection_overlay: None,
        visible_cell_types: Vec::new(),
        hidden_cell_types: Vec::new(),
        center_world: None,
        zoom: None,
    };

    for (key, value) in query_pairs(query) {
        match key.as_str() {
            "project" | "project_path" => req.project_path = Some(path_from_link_value(&value)?),
            "roi" | "roi_id" => req.roi = non_empty(value),
            "sample" | "case" | "dataset_id" => req.sample = non_empty(value),
            "marker" | "channel" => req.channel = non_empty(value),
            "visible_channels" | "show_channels" | "only_channels" => {
                req.visible_channels = parse_list(&value)
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
            "visible_cell_types" | "show_cell_types" | "only_cell_types" | "cell_types" => {
                req.visible_cell_types = parse_list(&value)
            }
            "hidden_cell_types" | "hide_cell_types" => req.hidden_cell_types = parse_list(&value),
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
            "odon://open?v=1&project=file:///tmp/my%20project.json&roi=18S1746%2FROI2&marker=CD68&segmentation=cells&segmentation_source=geoparquet&load_labels=0&cell_color_by=broad_cell_type&fill_cells=1&show_selection_overlay=0&center=10.5,20&zoom=0.25",
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
        assert!(req.visible_cell_types.is_empty());
        assert!(req.hidden_cell_types.is_empty());
        assert_eq!(req.center_world, Some([10.5, 20.0]));
        assert_eq!(req.zoom, Some(0.25));
    }

    #[test]
    fn ignores_non_odon_args() {
        assert!(DeepLinkRequest::parse_arg("--project").unwrap().is_none());
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
    fn parses_channel_visibility_and_contrast() {
        let req = DeepLinkRequest::parse_arg(
            "odon://open?channel=CD3&visible_channels=CD3%7CCD8&hidden_channels=DAPI&contrast_min=120&contrast_max=4500&channel_contrast=CD3:120:4500%7CCD8:80:3000",
        )
        .unwrap()
        .unwrap();

        assert_eq!(req.channel.as_deref(), Some("CD3"));
        assert_eq!(
            req.visible_channels,
            vec!["CD3".to_string(), "CD8".to_string()]
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
}
