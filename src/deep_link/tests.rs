use super::parsing::parse_color_rgb;
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
        object_color_mapping: Some(crate::model::ObjectColorMapping::Continuous {
            property: "mean_channel_1".to_string(),
            palette: crate::model::ContinuousPalette::Named("viridis".to_string()),
            domain: crate::model::ContinuousDomain::Fixed([4_000.0, 42_000.0]),
            scale: crate::model::ContinuousScale::Linear,
            reverse: false,
            out_of_range: crate::model::OutOfRangeMode::Clamp,
            missing_color_rgb: None,
        }),
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
    assert_eq!(parsed.object_color_mapping, request.object_color_mapping);
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
