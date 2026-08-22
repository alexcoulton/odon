use super::*;
use crate::data::ome::LevelInfo;

fn test_level(index: usize, width: u64, height: u64) -> LevelInfo {
    LevelInfo {
        index,
        path: index.to_string(),
        shape: vec![1, 1, height, width],
        chunks: vec![1, 1, height.min(1024), width.min(1024)],
        downsample: 1.0,
        dtype: "uint16".to_string(),
        scale: Vec::new(),
        translation: Vec::new(),
    }
}

#[test]
fn extracts_marker_name_from_channel_label() {
    assert_eq!(
        marker_name_from_channel_label("C013 - Desmin (FITC) [S]"),
        "Desmin"
    );
    assert_eq!(
        marker_name_from_channel_label("C028 - CD8a (APC) [S]"),
        "CD8a"
    );
    assert_eq!(marker_name_from_channel_label("DAPI [S]"), "DAPI");
}

#[test]
fn matches_marker_aliases_without_cd_prefix_collisions() {
    assert!(marker_alias_matches("desmin", "Desmin"));
    assert!(marker_alias_matches("myogenin", "Myogenin"));
    assert!(marker_alias_matches("cd68", "CD68"));
    assert!(marker_alias_matches("cd8", "CD8a"));
    assert!(!marker_alias_matches("cd8", "CD88"));
    assert!(!marker_alias_matches("cd3", "CD31"));
    assert!(!marker_alias_matches("cd4", "CD45"));
    assert!(!marker_alias_matches("cd1", "CD163"));
}

#[test]
fn suggests_channel_aliases_from_common_labels() {
    assert_eq!(suggest_channel_alias("C013 - Desmin (FITC) [S]"), "desmin");
    assert_eq!(suggest_channel_alias("Desmin_Opal520"), "desmin");
    assert_eq!(suggest_channel_alias("CD8a-AF647"), "cd8a");
    assert_eq!(suggest_channel_alias("C019_MYOG"), "myog");
    assert_eq!(suggest_channel_alias("DAPI"), "dapi");
}

#[test]
fn default_threshold_full_level_uses_highest_resolution_safe_level() {
    let levels = vec![
        test_level(0, 100_000, 100_000),
        test_level(1, 20_000, 20_000),
        test_level(2, 5_000, 5_000),
        test_level(3, 3_000, 3_000),
        test_level(4, 1_000, 1_000),
    ];

    assert_eq!(
        default_threshold_full_level(&levels, 2, 3, THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS,),
        Some(3)
    );
}

#[test]
fn default_threshold_full_level_returns_none_when_all_levels_are_too_large() {
    let levels = vec![
        test_level(0, 100_000, 100_000),
        test_level(1, 10_000, 10_000),
    ];

    assert_eq!(
        default_threshold_full_level(&levels, 2, 3, THRESHOLD_REGION_MAX_INTERACTIVE_PIXELS),
        None
    );
}
