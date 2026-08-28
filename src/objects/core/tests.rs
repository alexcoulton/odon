use super::*;
use std::sync::atomic::AtomicU64;

struct TestObjectDir(PathBuf);

impl TestObjectDir {
    fn new() -> Self {
        static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
        let sequence = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "odon-object-tests-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create object test directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestObjectDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn write_object_fixture(path: &Path) {
    let fixture = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "cell-a",
                "properties": {"class": "tumor", "score": 1.5, "positive": true},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"cell_id": "cell-b", "class": "immune", "score": 2.5, "positive": false},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]
                }
            },
            {
                "type": "Feature",
                "id": "cell-c",
                "properties": {"class": "tumor", "score": 3, "positive": true},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 20], [5, 20], [5, 25], [0, 25], [0, 20]]],
                        [[[6, 20], [10, 20], [10, 24], [6, 24], [6, 20]]]
                    ]
                }
            }
        ]
    });
    std::fs::write(
        path,
        serde_json::to_vec_pretty(&fixture).expect("serialize object fixture"),
    )
    .expect("write object fixture");
}

fn object_with_geometry_and_bad_point() -> GeoJsonObjectFeature {
    GeoJsonObjectFeature {
        id: "cell".to_string(),
        polygons_world: vec![vec![
            egui::pos2(10.0, 20.0),
            egui::pos2(30.0, 20.0),
            egui::pos2(30.0, 40.0),
            egui::pos2(10.0, 40.0),
            egui::pos2(10.0, 20.0),
        ]],
        point_position_world: Some(egui::pos2(1000.0, 1000.0)),
        bbox_world: egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(30.0, 40.0)),
        area_px: 0.0,
        perimeter_px: 0.0,
        centroid_world: egui::pos2(25.0, 35.0),
        inline_properties: serde_json::Map::new(),
        source_row_index: None,
    }
}

#[test]
fn polygon_proxy_position_uses_rendered_geometry_bounds() {
    let obj = object_with_geometry_and_bad_point();

    assert_eq!(object_proxy_position_world(&obj), egui::pos2(20.0, 30.0));
}

#[test]
fn point_proxy_position_uses_point_position() {
    let mut obj = object_with_geometry_and_bad_point();
    obj.polygons_world.clear();

    assert_eq!(
        object_proxy_position_world(&obj),
        egui::pos2(1000.0, 1000.0)
    );
}

fn id_clause(needle: &str) -> PreparedObjectFilterClause<'static> {
    PreparedObjectFilterClause {
        property_key: "id",
        needle: needle.to_string(),
        column: None,
        column_matcher: None,
    }
}

#[test]
fn object_filter_logic_all_requires_every_clause() {
    let obj = object_with_geometry_and_bad_point();
    let clauses = vec![id_clause("ce"), id_clause("ll")];

    assert!(ObjectsLayer::object_matches_prepared_filter(
        0,
        &obj,
        &clauses,
        ObjectFilterLogic::All
    ));

    let clauses = vec![id_clause("ce"), id_clause("missing")];
    assert!(!ObjectsLayer::object_matches_prepared_filter(
        0,
        &obj,
        &clauses,
        ObjectFilterLogic::All
    ));
}

#[test]
fn object_filter_logic_any_accepts_any_clause() {
    let obj = object_with_geometry_and_bad_point();
    let clauses = vec![id_clause("missing"), id_clause("ll")];

    assert!(ObjectsLayer::object_matches_prepared_filter(
        0,
        &obj,
        &clauses,
        ObjectFilterLogic::Any
    ));

    let clauses = vec![id_clause("missing"), id_clause("absent")];
    assert!(!ObjectsLayer::object_matches_prepared_filter(
        0,
        &obj,
        &clauses,
        ObjectFilterLogic::Any
    ));
}

#[test]
fn display_restore_preserves_runtime_color_overrides_for_same_property() {
    let mut layer = ObjectsLayer::default();
    layer.color_mode = ObjectColorMode::ByProperty;
    layer.color_property_key = "broad_cell_type".to_string();
    layer.color_level_overrides_property_key = "broad_cell_type".to_string();
    layer.color_level_overrides.insert(
        "immune_lymphoid".to_string(),
        ObjectColorLevelOverride {
            visible: true,
            color_rgb: Some([216, 70, 104]),
        },
    );

    let saved_state = ObjectProjectDisplayState {
        color_property_key: Some("broad_cell_type".to_string()),
        color_mapping: None,
        color_level_overrides: BTreeMap::new(),
        fill_cells: true,
        fill_opacity: 0.3,
        selected_fill_opacity: 0.7,
        fast_rendering: true,
    };

    layer.apply_project_display_state_preserving_color_visibility(&saved_state);

    assert_eq!(
        layer
            .color_level_overrides
            .get("immune_lymphoid")
            .and_then(|style| style.color_rgb),
        Some([216, 70, 104])
    );
}

#[test]
fn continuous_display_state_round_trips_and_legacy_state_migrates() {
    let mapping = ObjectColorMapping::Continuous {
        property: "score".to_string(),
        palette: ContinuousPalette::Named("plasma".to_string()),
        domain: ContinuousDomain::Fixed([2.0, 8.0]),
        scale: ContinuousScale::Linear,
        reverse: true,
        out_of_range: OutOfRangeMode::Hide,
        missing_color_rgb: Some([12, 34, 56]),
    };
    let state = ObjectProjectDisplayState {
        color_property_key: Some("score".to_string()),
        color_mapping: Some(mapping.clone()),
        ..ObjectProjectDisplayState::default()
    };
    let encoded = serde_json::to_value(&state).expect("serialize continuous display state");
    let decoded: ObjectProjectDisplayState =
        serde_json::from_value(encoded).expect("deserialize continuous display state");
    let mut layer = ObjectsLayer::default();
    layer.apply_project_display_state(&decoded);
    assert_eq!(layer.color_mapping(), &mapping);

    let legacy: ObjectProjectDisplayState = serde_json::from_value(serde_json::json!({
        "color_property_key":"phenotype",
        "fill_cells":true
    }))
    .expect("deserialize legacy categorical display state");
    layer.apply_project_display_state(&legacy);
    assert_eq!(
        layer.color_mapping(),
        &ObjectColorMapping::categorical("phenotype")
    );
}

#[test]
fn continuous_payload_uses_exact_values_and_is_independent_of_filter_visibility() {
    let mut low = object_with_geometry_and_bad_point();
    low.inline_properties
        .insert("score".to_string(), serde_json::json!(0.0));
    let mut high = object_with_geometry_and_bad_point();
    high.id = "high".to_string();
    high.inline_properties
        .insert("score".to_string(), serde_json::json!(10.0));
    let missing = object_with_geometry_and_bad_point();
    let mut layer = ObjectsLayer::default();
    layer.objects = Some(Arc::new(vec![low, high, missing]));
    let mapping = ObjectColorMapping::Continuous {
        property: "score".to_string(),
        palette: ContinuousPalette::Named("gray".to_string()),
        domain: ContinuousDomain::Fixed([0.0, 10.0]),
        scale: ContinuousScale::Linear,
        reverse: false,
        out_of_range: OutOfRangeMode::Clamp,
        missing_color_rgb: None,
    };
    layer
        .set_color_mapping(mapping)
        .expect("set continuous mapping");
    let payload = layer
        .ensure_continuous_color_payload()
        .expect("build continuous payload")
        .clone();
    assert_eq!(payload.colors_rgba[0], [0, 0, 0, 255]);
    assert_eq!(payload.colors_rgba[1], [255, 255, 255, 255]);
    assert_eq!(payload.colors_rgba[2], [0, 0, 0, 0]);
    assert_eq!(payload.numeric_count, 2);
    assert_eq!(payload.missing_count, 1);

    layer.filtered_mask = Some(Arc::new(vec![false, true, false]));
    let after_filter = layer
        .ensure_continuous_color_payload()
        .expect("reuse payload after filter");
    assert_eq!(after_filter.generation, payload.generation);
    assert!(Arc::ptr_eq(&after_filter.colors_rgba, &payload.colors_rgba));
}

#[test]
#[ignore = "diagnostic benchmark; run explicitly with --ignored --nocapture"]
fn benchmark_continuous_payload_for_45_000_objects() {
    let objects = (0..45_000)
        .map(|index| {
            let mut object = object_with_geometry_and_bad_point();
            object.id = format!("cell-{index}");
            object.inline_properties.insert(
                "score".to_string(),
                serde_json::json!(index as f64 / 44_999.0),
            );
            object
        })
        .collect::<Vec<_>>();
    let mut layer = ObjectsLayer::default();
    layer.objects = Some(Arc::new(objects));
    layer
        .set_color_mapping(ObjectColorMapping::Continuous {
            property: "score".to_string(),
            palette: ContinuousPalette::Named("viridis".to_string()),
            domain: ContinuousDomain::Fixed([0.0, 1.0]),
            scale: ContinuousScale::Linear,
            reverse: false,
            out_of_range: OutOfRangeMode::Clamp,
            missing_color_rgb: None,
        })
        .expect("set continuous mapping");

    let started = std::time::Instant::now();
    let payload = layer
        .ensure_continuous_color_payload()
        .expect("build continuous payload")
        .clone();
    let build_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let colors = Arc::clone(&payload.colors_rgba);

    let reuse_started = std::time::Instant::now();
    let reused = layer
        .ensure_continuous_color_payload()
        .expect("reuse continuous payload");
    let reuse_ms = reuse_started.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "continuous payload benchmark: objects={} build_ms={build_ms:.3} reuse_ms={reuse_ms:.3} bytes={}",
        reused.colors_rgba.len(),
        reused.colors_rgba.len() * std::mem::size_of::<[u8; 4]>()
    );

    assert_eq!(reused.numeric_count, 45_000);
    assert_eq!(reused.missing_count, 0);
    assert!(Arc::ptr_eq(&colors, &reused.colors_rgba));
}

#[test]
fn color_value_colors_are_staged_before_objects_load() {
    let mut layer = ObjectsLayer::default();
    layer.color_mode = ObjectColorMode::ByProperty;
    layer.color_property_key = "broad_cell_type".to_string();

    layer.set_color_value_colors(
        Some("broad_cell_type"),
        &[("immune_lymphoid".to_string(), [216, 70, 104])],
    );

    let display = layer.project_display_state();
    assert_eq!(
        display
            .color_level_overrides
            .get("immune_lymphoid")
            .and_then(|style| style.color_rgb),
        Some([216, 70, 104])
    );
    assert!(layer.pending_color_value_colors.is_some());

    let dir = TestObjectDir::new();
    let path = dir.path("staged-colours.geojson");
    write_object_fixture(&path);
    layer.color_property_key = "class".to_string();
    layer.set_color_value_colors(Some("class"), &[("immune".to_string(), [10, 20, 30])]);
    let cancel = AtomicBool::new(false);
    let result = load_in_thread(path, 1.0, None, 1, &cancel).expect("load object fixture");
    layer.install_load_result(result);
    assert!(layer.pending_color_value_colors.is_none());
    assert_eq!(
        layer
            .color_level_overrides
            .get("immune")
            .and_then(|style| style.color_rgb),
        Some([10, 20, 30])
    );
}

#[test]
fn reinstalling_shared_property_resource_preserves_geometry_generation() {
    let dir = TestObjectDir::new();
    let path = dir.path("shared-geometry.geojson");
    write_object_fixture(&path);
    let cancel = AtomicBool::new(false);
    let first = load_in_thread(path.clone(), 1.0, None, 1, &cancel).expect("first load");
    let shared_property_update = first.clone();

    let mut layer = ObjectsLayer::default();
    layer.install_load_result(first);
    let shared_generation = layer.geometry_generation;
    layer.ensure_default_object_property_threshold("score", 2.0, Some("Marker"));
    layer.selected_object_indices.insert(1);
    layer.selected_object_index = Some(1);
    let analysis_before = layer.project_analysis_state();
    let renderer_generation = layer.generation;
    let live_selection_generation = layer.analysis_live_selection_generation;
    for _ in 0..100 {
        layer.install_load_result(shared_property_update.clone());
    }
    assert_eq!(layer.geometry_generation, shared_generation);
    assert_eq!(layer.generation, renderer_generation);
    assert_eq!(
        layer.analysis_live_selection_generation,
        live_selection_generation
    );
    assert_eq!(layer.project_analysis_state(), analysis_before);
    assert_eq!(layer.selected_object_indices, HashSet::from([1]));
    assert_eq!(layer.selected_object_index, Some(1));

    let mut property_update = shared_property_update;
    property_update.property_store.insert_column(
        "new_score".to_string(),
        ObjectPropertyColumn::F32(Arc::new(NullableF32Column::from_optional_values([
            Some(1.0),
            Some(2.0),
            Some(3.0),
        ]))),
    );
    property_update
        .scalar_property_keys
        .push("new_score".to_string());
    layer.install_load_result(property_update);
    assert_eq!(layer.geometry_generation, shared_generation);
    assert_eq!(layer.project_analysis_state(), analysis_before);
    assert_eq!(layer.selected_object_indices, HashSet::from([1]));
    assert_eq!(layer.selected_object_index, Some(1));
    assert_eq!(
        layer.analysis_live_selection_generation,
        live_selection_generation + 1
    );

    let replacement = load_in_thread(path, 1.0, None, 2, &cancel).expect("replacement load");
    layer.install_load_result(replacement);
    assert_ne!(layer.geometry_generation, shared_generation);
    assert!(layer.project_analysis_state().threshold_elements.is_empty());
    assert!(layer.selected_object_indices.is_empty());
}

#[test]
fn reinstalling_identical_control_payload_is_a_complete_no_op() {
    let dir = TestObjectDir::new();
    let path = dir.path("identical-control-payload.geojson");
    write_object_fixture(&path);
    let resource = load_control_object_resource(path, 1.0).expect("load control object resource");

    let mut layer = ObjectsLayer::default();
    assert!(layer.install_control_resource(&resource));
    layer.ensure_default_object_property_threshold("score", 2.0, Some("Marker"));
    layer.selected_object_indices.insert(1);
    layer.selected_object_index = Some(1);
    let analysis_before = layer.project_analysis_state();
    let renderer_generation = layer.generation;
    let geometry_generation = layer.geometry_generation;
    let live_selection_generation = layer.analysis_live_selection_generation;

    for _ in 0..100 {
        assert!(layer.install_control_resource(&resource));
    }

    assert_eq!(layer.generation, renderer_generation);
    assert_eq!(layer.geometry_generation, geometry_generation);
    assert_eq!(
        layer.analysis_live_selection_generation,
        live_selection_generation
    );
    assert_eq!(layer.project_analysis_state(), analysis_before);
    assert_eq!(layer.selected_object_indices, HashSet::from([1]));
    assert_eq!(layer.selected_object_index, Some(1));
}

#[test]
fn lazy_property_lru_evicts_old_columns_but_pins_active_references() {
    let dir = TestObjectDir::new();
    let path = dir.path("lazy-property-cache.geojson");
    write_object_fixture(&path);
    let cancel = AtomicBool::new(false);
    let result = load_in_thread(path, 1.0, None, 1, &cancel).expect("load object fixture");
    let mut layer = ObjectsLayer::default();
    layer.install_load_result(result);
    layer.lazy_parquet_source = Some(LazyParquetSource {
        available_property_columns: vec!["a", "b", "c", "d"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        numeric_property_columns: vec!["a", "b", "c", "d"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        loaded_property_columns: HashSet::new(),
    });
    layer.set_lazy_property_cache_capacity(Some(2));

    let values = HashMap::from([
        (0usize, serde_json::json!(1.0)),
        (1usize, serde_json::json!(2.0)),
        (2usize, serde_json::json!(3.0)),
    ]);
    let activate = |layer: &mut ObjectsLayer, property: &str| {
        layer.color_mode = ObjectColorMode::Continuous;
        layer.color_property_key = property.to_string();
        layer.color_mapping = ObjectColorMapping::Continuous {
            property: property.to_string(),
            palette: ContinuousPalette::Named("viridis".to_string()),
            domain: ContinuousDomain::Fixed([0.0, 4.0]),
            scale: ContinuousScale::Linear,
            reverse: false,
            out_of_range: OutOfRangeMode::Clamp,
            missing_color_rgb: None,
        };
    };

    activate(&mut layer, "a");
    layer.apply_loaded_property_values("a", &values);
    activate(&mut layer, "b");
    layer.apply_loaded_property_values("b", &values);
    activate(&mut layer, "c");
    layer.apply_loaded_property_values("c", &values);

    assert!(!layer.property_store.has_loaded("a"));
    assert!(layer.property_store.has_loaded("b"));
    assert!(layer.property_store.has_loaded("c"));
    assert_eq!(
        layer.lazy_property_lru,
        VecDeque::from(["b".into(), "c".into()])
    );
    assert_eq!(layer.lazy_property_cache_evictions, 1);
    assert!(layer.property_column_available_but_unloaded("a"));

    layer.filter_clauses = vec![ObjectFilterClause {
        enabled: true,
        property_key: "b".to_string(),
        query: "2".to_string(),
    }];
    activate(&mut layer, "d");
    layer.apply_loaded_property_values("d", &values);

    assert!(layer.property_store.has_loaded("b"));
    assert!(!layer.property_store.has_loaded("c"));
    assert!(layer.property_store.has_loaded("d"));
    assert_eq!(layer.lazy_property_cache_evictions, 2);
}

#[test]
fn automatic_analysis_default_does_not_retarget_an_existing_rule() {
    let mut layer = ObjectsLayer::default();
    layer
        .analysis_property_thresholds
        .push(ObjectPropertyThresholdRule {
            column_key: "median_marker".to_string(),
            channel_name: Some("Marker".to_string()),
            op: AnalysisThresholdOp::GreaterEqual,
            value: 12.0,
            value_transform: HistogramValueTransform::None,
        });

    layer.ensure_default_object_property_threshold("label", 3.0, Some("Marker"));

    let rule = &layer.analysis_property_thresholds[0];
    assert_eq!(rule.column_key, "median_marker");
    assert_eq!(rule.value, 12.0);
}

#[test]
fn explicit_analysis_column_change_retargets_the_existing_rule() {
    let mut layer = ObjectsLayer::default();
    layer
        .analysis_property_thresholds
        .push(ObjectPropertyThresholdRule {
            column_key: "median_marker".to_string(),
            channel_name: Some("Marker".to_string()),
            op: AnalysisThresholdOp::GreaterEqual,
            value: 12.0,
            value_transform: HistogramValueTransform::None,
        });

    layer.retarget_object_property_threshold("nimbus_marker", 7.0, Some("Marker"));

    let rule = &layer.analysis_property_thresholds[0];
    assert_eq!(rule.column_key, "nimbus_marker");
    assert_eq!(rule.value, 7.0);
}

#[test]
fn histogram_threshold_drag_previews_locally_and_commits_once_on_release() {
    let mut layer = ObjectsLayer::default();
    layer.ensure_default_object_property_threshold("score", 2.0, Some("Marker"));
    let committed_before = layer.project_analysis_state();
    let live_generation_before = layer.analysis_live_selection_generation;

    layer.analysis_hist_drag_rule = Some(0);
    for value in [2.5, 3.0, 3.5, 4.0] {
        layer.preview_histogram_threshold_drag(0, "score", value);
        assert_eq!(
            layer.project_analysis_state(),
            committed_before,
            "pointer motion must not mutate actor-bound Analysis state"
        );
    }
    assert_eq!(layer.analysis_property_thresholds[0].value, 4.0);
    assert_eq!(
        layer.analysis_live_selection_generation,
        live_generation_before + 4,
        "each distinct preview remains available to live selection"
    );

    assert!(layer.commit_histogram_threshold_drag());
    assert!(layer.analysis_hist_drag_rule.is_none());
    let committed_after = layer.project_analysis_state();
    assert_ne!(committed_after, committed_before);
    assert_eq!(committed_after.threshold_elements[0].rules[0].value, 4.0);
    assert!(!layer.commit_histogram_threshold_drag());
}

#[test]
fn geojson_lifecycle_filter_selection_and_exports_round_trip() {
    let dir = TestObjectDir::new();
    let geojson_path = dir.path("objects.geojson");
    write_object_fixture(&geojson_path);
    let cancel = AtomicBool::new(false);
    let result = load_in_thread(geojson_path.clone(), 1.0, None, 7, &cancel)
        .expect("load GeoJSON object fixture");

    let mut layer = ObjectsLayer::default();
    layer.install_load_result(result);

    assert_eq!(layer.object_count(), 3);
    assert!(layer.visible);
    assert_eq!(
        layer.loaded_geojson.as_deref(),
        Some(geojson_path.as_path())
    );
    assert_eq!(layer.display_mode, ObjectDisplayMode::Polygons);
    assert_eq!(
        layer.available_property_columns(),
        &["cell_id", "class", "positive", "score"]
    );
    assert!(
        layer
            .available_numeric_object_property_keys()
            .contains(&"score".to_string())
    );
    assert_eq!(
        layer.objects.as_ref().expect("loaded objects")[1].id,
        "cell-b"
    );

    layer.set_filter_clauses_from_pairs(&[("class".to_string(), "tumor".to_string())]);
    let filter = layer.filter_snapshot_json();
    assert_eq!(filter["active"], true);
    assert_eq!(filter["visible_count"], 2);
    assert_eq!(filter["hidden_count"], 1);
    layer.bulk_measurement_filtered_only = true;
    assert_eq!(layer.bulk_measurement_target_indices(), vec![0, 2]);

    layer.apply_bulk_measurement_result(BulkMeasurementResult {
        metric: BulkMeasurementMetric::Mean,
        scope_label: "test cells".to_string(),
        level_index: 0,
        level_downsample: 1.0,
        object_count: 3,
        measured_count: 3,
        failed_count: 0,
        column_values: vec![(
            "mean_dapi".to_string(),
            vec![Some(10.0), Some(20.0), Some(30.0)],
        )],
    });
    assert!(
        layer
            .available_numeric_object_property_keys()
            .contains(&"mean_dapi".to_string())
    );
    layer.set_filter_clauses_from_pairs(&[("mean_dapi".to_string(), "20".to_string())]);
    assert_eq!(layer.filter_snapshot_json()["visible_count"], 1);
    layer.clear_filter();

    let first_rect = egui::Rect::from_min_max(egui::pos2(1.0, 1.0), egui::pos2(9.0, 9.0));
    let query = layer.query_world_rect_snapshot_json(first_rect, egui::Vec2::ZERO, 10);
    assert_eq!(query["match_count"], 1);
    assert_eq!(query["matches"][0]["id"], "cell-a");
    let selection =
        layer.select_in_world_rect_snapshot_json(first_rect, egui::Vec2::ZERO, false, 10);
    assert_eq!(selection["selection"]["selection_count"], 1);
    assert_eq!(selection["selection"]["primary"]["id"], "cell-a");

    layer.set_filter_clauses_from_pairs(&[("class".to_string(), "immune".to_string())]);
    let hidden_selection = layer.selection_snapshot_json(egui::Vec2::ZERO, 10);
    assert_eq!(
        hidden_selection["selection_count"], 1,
        "a viewport filter must not delete shared selection identity"
    );

    layer.clear_filter();
    let second_rect = egui::Rect::from_min_max(egui::pos2(21.0, 1.0), egui::pos2(29.0, 9.0));
    layer.select_in_world_rect_snapshot_json(second_rect, egui::Vec2::ZERO, true, 10);
    let selection = layer.selection_snapshot_json(egui::Vec2::ZERO, 10);
    assert_eq!(selection["selection_count"], 2);
    assert_eq!(selection["selected"][0]["id"], "cell-a");
    assert_eq!(selection["selected"][1]["id"], "cell-b");

    let lasso = [
        egui::pos2(-1.0, 19.0),
        egui::pos2(12.0, 19.0),
        egui::pos2(12.0, 27.0),
        egui::pos2(-1.0, 27.0),
    ];
    assert_eq!(
        layer.select_in_world_lasso(&lasso, egui::Vec2::ZERO, false),
        1
    );
    let selection = layer.selection_snapshot_json(egui::Vec2::ZERO, 10);
    assert_eq!(selection["selection_count"], 1);
    assert_eq!(selection["primary"]["id"], "cell-c");

    layer.set_color_by_property(Some("class".to_string()));
    let legend = layer
        .active_color_legend_entries()
        .expect("categorical class legend");
    assert_eq!(legend.len(), 2);
    assert!(legend.iter().any(|entry| entry.value_label == "immune"));
    assert!(legend.iter().any(|entry| entry.value_label == "tumor"));
    layer.set_color_by_property(Some("positive".to_string()));
    layer.ensure_color_groups();
    assert!(layer.color_groups_cache.contains_key("class"));
    layer.set_color_by_property(Some("class".to_string()));
    assert_eq!(
        layer
            .color_groups
            .as_ref()
            .map(|groups| groups.property_key.as_str()),
        Some("class"),
        "switching viewport styles should restore cached property groups"
    );
    assert!(layer.color_groups_cache.contains_key("positive"));

    let export_columns = layer
        .build_object_export_column_names()
        .expect("build export columns");
    assert!(export_columns.iter().any(|column| column == "class"));
    assert!(export_columns.iter().any(|column| column == "score"));
    assert!(
        export_columns
            .iter()
            .any(|column| column == "_odon_selected")
    );
    let selected_columns = export_columns.into_iter().collect::<HashSet<_>>();
    let snapshot = layer
        .object_export_snapshot()
        .expect("object export snapshot");

    let csv_path = dir.path("objects.csv");
    ObjectsLayer::export_objects_csv(&snapshot, &csv_path, &selected_columns)
        .expect("export objects CSV");
    let mut csv = csv::Reader::from_path(&csv_path).expect("open exported CSV");
    let headers = csv.headers().expect("CSV headers").clone();
    assert!(headers.iter().any(|header| header == "class"));
    assert!(headers.iter().any(|header| header == "_odon_selected"));
    let rows = csv
        .records()
        .collect::<Result<Vec<_>, _>>()
        .expect("read exported CSV rows");
    assert_eq!(rows.len(), 3);
    let selected_idx = headers
        .iter()
        .position(|header| header == "_odon_selected")
        .expect("selection export column");
    assert_eq!(
        rows.iter()
            .filter(|row| row.get(selected_idx) == Some("true"))
            .count(),
        1
    );

    let parquet_path = dir.path("objects.geoparquet");
    ObjectsLayer::export_objects_geoparquet(&snapshot, &parquet_path, &selected_columns)
        .expect("export objects GeoParquet");
    let reloaded =
        parse_geoparquet_objects(&parquet_path, None, &cancel).expect("reload exported GeoParquet");
    assert_eq!(reloaded.len(), 3);
    assert_eq!(reloaded[0].id, "cell-a");
    assert!(reloaded[1].inline_properties.is_empty());
    assert_eq!(reloaded[2].polygons_world.len(), 2);

    let resource = load_control_object_resource_with_options(
        parquet_path.clone(),
        1.0,
        Some(&serde_json::json!({
            "format":"geoparquet",
            "source":"geometry",
            "geometry_column":"geometry",
            "property_columns":["id", "class"],
        })),
    )
    .expect("load shared GeoParquet resource");
    assert!(
        resource
            .features
            .iter()
            .all(|feature| feature.inline_properties.is_empty()),
        "GeoParquet rows must not retain JSON property maps"
    );
    assert_eq!(
        resource.property_value(1, "class"),
        Some(serde_json::json!("immune"))
    );
    let filtered = crate::objects::control_service::evaluate_control_object_filter(
        &resource,
        &serde_json::json!({"mode":"query", "query":"class == 'immune'"}),
    )
    .expect("filter shared GeoParquet columns");
    assert_eq!(filtered.matching_indices.as_ref(), &[1]);
    let renderer = resource
        .renderer_payload::<PreloadedObjectLayer>()
        .expect("renderer preload");
    assert!(Arc::ptr_eq(&resource.features, &renderer.result.objects));
    assert_eq!(
        resource.memory_diagnostics.components["renderer.inline_property_maps"]
            .opaque_allocation_count,
        0
    );

    layer.clear();
    assert_eq!(layer.object_count(), 0);
    assert!(!layer.visible);
    assert!(layer.loaded_geojson.is_none());
    assert_eq!(layer.selection_count(), 0);
}

#[test]
fn csv_point_objects_infer_coordinates_properties_and_reject_bad_inputs() {
    let dir = TestObjectDir::new();
    let path = dir.path("points.csv");
    std::fs::write(
        &path,
        "cell_id,x_centroid,y_centroid,class,score,positive\n\
         p-1,10.5,20.25,immune,3.5,true\n\
         p-2,30,40,tumor,7,false\n\
         skipped,not-a-number,50,invalid,9,true\n",
    )
    .expect("write CSV object fixture");
    let cancel = AtomicBool::new(false);
    let objects = parse_csv_objects(&path, None, &cancel).expect("parse inferred CSV points");
    assert_eq!(objects.len(), 2);
    assert_eq!(objects[0].id, "p-1");
    assert_eq!(
        objects[0].point_position_world,
        Some(egui::pos2(10.5, 20.25))
    );
    assert_eq!(objects[0].inline_properties["class"], "immune");
    assert_eq!(objects[0].inline_properties["score"], 3.5);
    assert_eq!(objects[0].inline_properties["positive"], true);
    assert_eq!(objects[0].source_row_index, Some(0));
    assert_eq!(objects[1].id, "p-2");

    let selected = ObjectCsvLoadOptions {
        x_column: "x_centroid".to_string(),
        y_column: "y_centroid".to_string(),
        property_columns: Some(vec!["class".to_string()]),
    };
    let objects =
        parse_csv_objects(&path, Some(&selected), &cancel).expect("parse selected CSV properties");
    assert!(objects[0].inline_properties.contains_key("class"));
    assert!(!objects[0].inline_properties.contains_key("score"));

    let invalid = dir.path("invalid.csv");
    std::fs::write(&invalid, "id,label\na,cell\n").expect("write invalid CSV");
    assert!(
        parse_csv_objects(&invalid, None, &cancel)
            .expect_err("missing coordinate columns")
            .to_string()
            .contains("usable X column")
    );
}
