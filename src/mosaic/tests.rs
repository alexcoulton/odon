use super::*;
use crate::data::dataset_source::DatasetSource;
use crate::data::ome::{DatasetRenderKind, Dims, LevelInfo, Multiscale};
use odon::control::ControlCommand;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

struct TestMosaicDir(PathBuf);

impl TestMosaicDir {
    fn new() -> Self {
        let unique = format!(
            "odon-mosaic-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TestMosaicDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn item(id: usize, sample_id: &str, width: u64, height: u64) -> MosaicItem {
    MosaicItem {
        id,
        sample_id: sample_id.to_string(),
        meta: HashMap::new(),
        dataset: OmeZarrDataset {
            source: DatasetSource::Local(PathBuf::from(format!("roi-{id}.ome.zarr"))),
            multiscale: Multiscale {
                name: Some(sample_id.to_string()),
                axes: Vec::new(),
                datasets: Vec::new(),
            },
            levels: vec![LevelInfo {
                index: 0,
                path: "0".to_string(),
                shape: vec![height, width],
                chunks: vec![height, width],
                downsample: 1.0,
                dtype: "|u1".to_string(),
                scale: vec![1.0, 1.0],
                translation: vec![0.0, 0.0],
            }],
            channels: Vec::new(),
            dims: Dims {
                c: None,
                z: None,
                y: 0,
                x: 1,
                ndim: 2,
            },
            abs_max: 255.0,
            render_kind: DatasetRenderKind::Image,
        },
        offset: egui::Vec2::ZERO,
        scale: 1.0,
        placed_size: egui::Vec2::ZERO,
    }
}

#[test]
fn fit_cell_layout_preserves_aspect_ratio_and_grid_bounds() {
    let mut items = vec![
        item(0, "wide", 200, 100),
        item(1, "tall", 50, 200),
        item(2, "square", 100, 100),
    ];

    let bounds = layout_items(&mut items, 2, 100.0, 100.0, 10.0);

    assert_eq!(bounds.size(), egui::vec2(210.0, 210.0));
    assert_eq!(items[0].placed_size, egui::vec2(100.0, 50.0));
    assert_eq!(items[0].offset, egui::vec2(0.0, 25.0));
    assert_eq!(items[1].placed_size, egui::vec2(25.0, 100.0));
    assert_eq!(items[1].offset, egui::vec2(147.5, 0.0));
    assert_eq!(items[2].offset, egui::vec2(0.0, 110.0));
}

#[test]
fn native_layout_keeps_level_zero_pixels_and_centers_shorter_items() {
    let mut items = vec![
        item(0, "short", 80, 40),
        item(1, "tall", 50, 100),
        item(2, "next-row", 120, 30),
    ];

    let bounds = layout_items_native(&mut items, 2, 5.0);

    assert_eq!(bounds.size(), egui::vec2(135.0, 135.0));
    assert_eq!(items[0].offset, egui::vec2(0.0, 30.0));
    assert_eq!(items[1].offset, egui::vec2(85.0, 0.0));
    assert_eq!(items[2].offset, egui::vec2(0.0, 105.0));
    assert!(items.iter().all(|item| item.scale == 1.0));
}

#[test]
fn grouped_layout_separates_case_insensitive_groups_and_missing_values() {
    let mut items = vec![
        item(0, "A1", 100, 100),
        item(1, "A2", 100, 100),
        item(2, "B1", 100, 100),
        item(3, "missing", 100, 100),
    ];
    items[0]
        .meta
        .insert("cohort".to_string(), "Alpha".to_string());
    items[1]
        .meta
        .insert("cohort".to_string(), "alpha".to_string());
    items[2]
        .meta
        .insert("cohort".to_string(), "Beta".to_string());

    let (bounds, blocks) = layout_items_grouped(
        &mut items,
        2,
        100.0,
        100.0,
        10.0,
        Some("cohort"),
        20.0,
        MosaicLayoutMode::FitCells,
    );

    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].name, "Alpha");
    assert_eq!(blocks[1].name, "Beta");
    assert_eq!(blocks[2].name, "(missing)");
    assert_eq!(
        blocks[1].world_rect.top() - blocks[0].world_rect.bottom(),
        20.0
    );
    assert_eq!(
        blocks[2].world_rect.top() - blocks[1].world_rect.bottom(),
        20.0
    );
    assert_eq!(bounds.width(), 210.0);
    assert_eq!(bounds.bottom(), blocks[2].world_rect.bottom());
}

#[test]
fn samplesheet_mosaic_constructs_shared_channels_and_metadata() {
    let dir = TestMosaicDir::new();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let sheet = dir.0.join("mosaic.csv");
    fs::write(
        &sheet,
        format!(
            "id,path,cohort,site\nROI-B,{},B,2\nROI-A,{},A,1\n",
            fixture.display(),
            fixture.display()
        ),
    )
    .expect("write samplesheet");
    let ctx = egui::Context::default();
    let mosaic = MosaicViewerApp::from_samplesheet_runtime(&ctx, true, &sheet, Some(2))
        .expect("construct samplesheet mosaic");

    assert_eq!(mosaic.items.len(), 2);
    assert_eq!(mosaic.focused_core_id, Some(mosaic.items[0].id));
    assert_eq!(mosaic.metadata_columns, vec!["cohort", "site"]);
    assert_eq!(mosaic.channels.len(), 5, "channels are shared across ROIs");
}

#[test]
fn actor_owned_mosaic_interactions_emit_commands_without_semantic_mutation() {
    let dir = TestMosaicDir::new();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let sheet = dir.0.join("actor-mosaic.csv");
    fs::write(
        &sheet,
        format!(
            "id,path,cohort\nROI-A,{},A\nROI-B,{},B\n",
            fixture.display(),
            fixture.display()
        ),
    )
    .expect("write actor mosaic samplesheet");
    let ctx = egui::Context::default();
    let mut mosaic = MosaicViewerApp::from_samplesheet_runtime(&ctx, true, &sheet, Some(2))
        .expect("construct actor mosaic");
    mosaic.consumed_mosaic_resource_generation = 1;
    let semantic_snapshot = |mosaic: &MosaicViewerApp| {
        serde_json::json!({
            "channels":mosaic.control_channel_snapshot(),
            "channel_sort":mosaic.channel_sort_mode.storage_key(),
            "groups":mosaic.layer_groups,
            "active_layer":MosaicViewerApp::layer_id_storage_key(mosaic.active_layer),
            "objects_visible":mosaic.seg_geojson.visible,
            "camera":mosaic.control_camera_snapshot(),
            "focused":mosaic.focused_core_id,
            "fast_objects":mosaic.seg_geojson.fast_rendering,
            "show_group_labels":mosaic.show_group_labels,
        })
    };
    let before = semantic_snapshot(&mosaic);
    ChannelListHost::set_channel_visible(&mut mosaic, 1, true);
    ChannelListHost::set_channel_sort_mode(&mut mosaic, ChannelSortMode::NameAsc);
    let mut groups = mosaic.layer_groups.clone();
    groups
        .channel_groups
        .push(crate::data::project_config::ProjectChannelGroup {
            id: 7,
            name: "Review".to_string(),
            expanded: true,
            color_rgb: [255, 255, 255],
        });
    ChannelListHost::set_layer_groups(&mut mosaic, groups);
    mosaic.set_active_layer(MosaicLayerId::TextLabels);
    mosaic.set_layer_visible(MosaicLayerId::SegmentationGeoJson, true);
    mosaic.apply_channel_window_to_indices(&[1, 2], 5.0, 100.0);
    mosaic.commit_channel_color(1, [1, 2, 3]);
    mosaic.commit_channel_note(1, "review".to_string());
    mosaic.fit_mosaic();
    mosaic.step_focused_core(&ctx, 1);
    mosaic.set_fast_object_rendering(false);
    mosaic.submit_layout_value("show_group_labels", serde_json::json!(false));

    let intents = mosaic.take_native_control_intents();
    assert_eq!(
        intents
            .iter()
            .map(|intent| intent.method)
            .collect::<Vec<_>>(),
        vec![
            "viewer.channels.set_visible",
            "viewer.channels.presentation.set",
            "viewer.channels.set_group",
            "viewer.native_layers.set_active",
            "viewer.objects.set_visibility",
            "viewer.channels.set_contrast",
            "viewer.channels.set_color",
            "viewer.channels.set_note",
            "mosaic.fit_all",
            "mosaic.focus.next",
            "viewer.objects.rendering.set_fast",
            "mosaic.layout.configure",
        ]
    );
    for intent in &intents {
        ControlCommand::decode(intent.method, intent.params.clone()).unwrap_or_else(|error| {
            panic!(
                "native mosaic intent {} did not pass typed decoding: {error}",
                intent.method
            )
        });
    }
    let contrast = intents
        .iter()
        .find(|intent| intent.method == "viewer.channels.set_contrast")
        .expect("contrast intent");
    assert_eq!(contrast.params["channels"], serde_json::json!([1, 2]));
    assert_eq!(contrast.params["min"], 5.0);
    assert_eq!(contrast.params["max"], 100.0);
    assert_eq!(semantic_snapshot(&mosaic), before);

    let projection = serde_json::json!({
        "generation":1,
        "native_layers":[{
            "layer_id":"text_labels",
            "stack":"overlays",
            "visible":false,
            "active":true,
        }],
    });
    mosaic
        .apply_control_actor_state(&projection, &serde_json::json!({}), &[], &[])
        .expect("apply actor projection");
    assert!(!mosaic.show_text_labels);
    assert_eq!(mosaic.active_layer, MosaicLayerId::TextLabels);
    assert!(
        mosaic.take_native_control_intents().is_empty(),
        "applying an actor projection must not feed commands back to the actor"
    );
}

#[test]
fn mosaic_contrast_coalesces_drag_updates_and_batches_distinct_windows() {
    let dir = TestMosaicDir::new();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let sheet = dir.0.join("contrast-mosaic.csv");
    fs::write(
        &sheet,
        format!("id,path,cohort\nROI-A,{},A\n", fixture.display()),
    )
    .expect("write contrast mosaic samplesheet");
    let ctx = egui::Context::default();
    let mut mosaic = MosaicViewerApp::from_samplesheet_runtime(&ctx, true, &sheet, Some(1))
        .expect("construct contrast mosaic");
    let channels_before = mosaic.control_channel_snapshot();

    mosaic.apply_channel_window_to_indices(&[0, 1], 5.0, 100.0);
    mosaic.apply_channel_window_to_indices(&[0, 1], 15.0, 120.0);
    assert_eq!(mosaic.preview_channel_window(0), Some((15.0, 120.0)));
    let first = mosaic.take_native_control_intents();
    assert_eq!(first.len(), 1, "one actor command may be in flight");
    assert_eq!(first[0].params["channels"], serde_json::json!([0, 1]));
    assert_eq!(first[0].params["min"], 5.0);

    mosaic.flush_pending_channel_contrast();
    let latest = mosaic.take_native_control_intents();
    assert_eq!(
        latest.len(),
        1,
        "only the newest drag position is queued next"
    );
    assert_eq!(latest[0].params["channels"], serde_json::json!([0, 1]));
    assert_eq!(latest[0].params["min"], 15.0);
    assert_eq!(latest[0].params["max"], 120.0);

    mosaic.apply_channel_windows(&[(0, 8.0, 140.0), (1, 12.0, 140.0)]);
    let batch = mosaic.take_native_control_intents();
    assert_eq!(batch.len(), 1);
    assert_eq!(
        batch[0].params["windows"],
        serde_json::json!([
            {"index":0,"min":8.0,"max":140.0},
            {"index":1,"min":12.0,"max":140.0},
        ])
    );
    assert_eq!(
        mosaic.control_channel_snapshot(),
        channels_before,
        "optimistic contrast is render-only until projected by the actor"
    );
}
