//! Object control-service and property-column regression tests.

use super::control_service::evaluate_control_object_filter;
use super::*;

fn control_filter_fixture() -> ControlObjectResource {
    ControlObjectResource {
        source: PathBuf::from("objects.geojson"),
        downsample_factor: 1.0,
        features: Arc::new(vec![
            odon::model::ControlObjectFeature {
                id: "cell-a".to_string(),
                bbox_world: egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                centroid_world: egui::pos2(0.5, 0.5),
                polygons_world: Vec::new(),
                point_position_world: Some(egui::pos2(0.5, 0.5)),
                area_px: 0.0,
                perimeter_px: 0.0,
                inline_properties: serde_json::json!({"kind":"tumour","score":0.9})
                    .as_object()
                    .unwrap()
                    .clone(),
                source_row_index: None,
            },
            odon::model::ControlObjectFeature {
                id: "cell-b".to_string(),
                bbox_world: egui::Rect::from_min_max(egui::pos2(1.0, 1.0), egui::pos2(2.0, 2.0)),
                centroid_world: egui::pos2(1.5, 1.5),
                polygons_world: Vec::new(),
                point_position_world: Some(egui::pos2(1.5, 1.5)),
                area_px: 0.0,
                perimeter_px: 0.0,
                inline_properties: serde_json::json!({"kind":"immune","score":0.2})
                    .as_object()
                    .unwrap()
                    .clone(),
                source_row_index: None,
            },
        ]),
        property_names: Arc::new(vec![
            "id".to_string(),
            "kind".to_string(),
            "score".to_string(),
        ]),
        property_source: Arc::new(odon::model::EmptyControlObjectPropertySource),
        numeric_summaries: Arc::new(Default::default()),
        memory_diagnostics: Arc::new(Default::default()),
        renderer_payload: None,
    }
}

#[test]
fn loaded_resource_shares_renderer_features_and_reports_memory_once() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_objects.geojson");
    let resource = load_control_object_resource(path, 1.0).expect("load synthetic objects");
    let components = &resource.memory_diagnostics.components;

    let renderer = resource
        .renderer_payload::<PreloadedObjectLayer>()
        .expect("renderer preload");
    assert!(Arc::ptr_eq(&resource.features, &renderer.result.objects));
    assert!(
        components
            .keys()
            .all(|name| !name.starts_with("actor_model.")),
        "the control model must not retain a second feature or geometry representation"
    );
    assert!(
        components["renderer.canonical_polygon_points"].retained_bytes() > 0,
        "canonical renderer polygons must be accounted"
    );
    assert!(
        components["renderer.outline_lods"].retained_bytes() > 0,
        "derived outline LODs must be accounted"
    );
    assert!(
        components["renderer.fill_mesh_full"].retained_bytes() > 0,
        "the full tessellated fill mesh must be accounted"
    );
    assert_eq!(
        components["renderer.inline_property_maps"].opaque_allocation_count,
        resource.features.len() as u64,
        "GeoJSON fallback maps must be counted once"
    );
    assert!(
        resource.memory_diagnostics.total().retained_bytes()
            >= components["renderer.fill_mesh_full"].retained_bytes()
    );
    assert_eq!(
        resource.descriptor_json(1)["cpu_geometry_memory"]["measurement"],
        "retained_cpu_object_capacity"
    );
}

#[test]
#[ignore = "diagnostic benchmark over the checked-out Amy ten-ROI object tables"]
fn benchmark_amy_ten_roi_shared_object_memory() {
    let samplesheet = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("amy_nimbus_comparison/environment/mosaic_all_roi/samplesheet.top-10.csv");
    if !samplesheet.exists() {
        eprintln!("skipping: {} is not available", samplesheet.display());
        return;
    }

    let started = std::time::Instant::now();
    let mut resources = Vec::new();
    let mut diagnostics = odon::model::ControlObjectMemoryDiagnostics::default();
    let mut cell_count = 0usize;
    let mut reader = csv::Reader::from_path(&samplesheet).expect("open Amy ten-ROI samplesheet");
    for record in reader.deserialize::<HashMap<String, String>>() {
        let record = record.expect("read Amy samplesheet row");
        let recorded_path = PathBuf::from(record.get("segpath").expect("segpath column"));
        let path = samplesheet
            .parent()
            .expect("samplesheet parent")
            .join("objects")
            .join(recorded_path.file_name().expect("segpath filename"));
        let resource = load_control_object_resource_with_options(
            path.clone(),
            1.0,
            Some(&serde_json::json!({
                "project_preload": {
                    "mode": "full_geometry",
                    "lazy_properties": true,
                }
            })),
        )
        .unwrap_or_else(|error| panic!("load {}: {error:#}", path.display()));
        let renderer = resource
            .renderer_payload::<PreloadedObjectLayer>()
            .expect("renderer preload");
        assert!(Arc::ptr_eq(&resource.features, &renderer.result.objects));
        assert!(
            resource
                .features
                .iter()
                .all(|feature| feature.inline_properties.is_empty()),
            "GeoParquet rows must not retain fallback JSON maps"
        );
        cell_count += resource.features.len();
        diagnostics.merge(resource.memory_diagnostics.as_ref());
        eprintln!(
            "loaded {}: {} cells, {:.3} GB accounted retained CPU object data",
            path.file_name().unwrap_or_default().to_string_lossy(),
            resource.features.len(),
            resource.memory_diagnostics.total().retained_bytes() as f64 / 1_000_000_000.0,
        );
        resources.push(resource);
    }

    let total = diagnostics.total();
    eprintln!(
        "Amy ten-ROI shared-object benchmark: {} ROIs, {} cells, {:.3} GB accounted retained CPU object data, {} opaque row maps, {:.3}s load time",
        resources.len(),
        cell_count,
        total.retained_bytes() as f64 / 1_000_000_000.0,
        diagnostics
            .components
            .get("renderer.inline_property_maps")
            .map_or(0, |component| component.opaque_allocation_count),
        started.elapsed().as_secs_f64(),
    );
    assert_eq!(resources.len(), 10);
    assert_eq!(cell_count, 1_480_967);
    assert_eq!(
        diagnostics
            .components
            .get("renderer.inline_property_maps")
            .map_or(0, |component| component.opaque_allocation_count),
        0
    );
}

#[test]
fn actor_query_filter_uses_the_native_typed_expression_engine() {
    let result = evaluate_control_object_filter(
        &control_filter_fixture(),
        &serde_json::json!({
            "mode":"query",
            "query":"kind == 'tumour' and score >= 0.5",
        }),
    )
    .unwrap();

    assert!(result.active);
    assert_eq!(result.matching_indices.as_ref(), &[0]);
    assert_eq!(result.model["mode"], "query");
}

#[test]
fn actor_simple_filter_matches_renderer_contains_semantics() {
    let result = evaluate_control_object_filter(
        &control_filter_fixture(),
        &serde_json::json!({
            "mode":"simple",
            "logic":"any",
            "clauses":[
                {"property":"kind","query":"IMM"},
                {"property":"id","query":"missing"},
            ],
        }),
    )
    .unwrap();

    assert!(result.active);
    assert_eq!(result.matching_indices.as_ref(), &[1]);
    assert_eq!(result.model["logic"], "any");
}

#[test]
fn actor_configured_csv_load_preserves_native_column_choices() {
    let path = std::env::temp_dir().join(format!(
        "odon-control-configured-{}-{}.csv",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(
        &path,
        "cx,cy,kind,ignored\n1,2,tumour,nope\n3,4,immune,nope\n",
    )
    .unwrap();
    let resource = load_control_object_resource_with_options(
        path.clone(),
        1.0,
        Some(&serde_json::json!({
            "format":"csv",
            "x_column":"cx",
            "y_column":"cy",
            "property_columns":["kind"],
        })),
    )
    .unwrap();

    assert_eq!(resource.features.len(), 2);
    assert!((resource.features[0].centroid_world.x - 1.0).abs() < 1e-5);
    assert!((resource.features[0].centroid_world.y - 2.0).abs() < 1e-5);
    assert_eq!(
        resource.property_value(1, "kind"),
        Some(serde_json::json!("immune"))
    );
    assert!(!resource.property_names.iter().any(|name| name == "ignored"));
    assert!(resource.renderer_payload.is_some());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn dictionary_contains_filter_matches_codes_without_decoding_rows() {
    let column = ObjectPropertyColumn::from_json_values(vec![
        Some(serde_json::Value::String("immune_myeloid".to_string())),
        Some(serde_json::Value::String("tumor_myogenic".to_string())),
        Some(serde_json::Value::String("immune_lymphoid".to_string())),
        None,
    ]);

    let matcher = column.contains_matcher("MYELOID");

    assert!(matches!(column, ObjectPropertyColumn::Dictionary { .. }));
    assert!(column.matches_contains(0, &matcher));
    assert!(!column.matches_contains(1, &matcher));
    assert!(!column.matches_contains(2, &matcher));
    assert!(!column.matches_contains(3, &matcher));
}

#[test]
fn bool_contains_filter_uses_typed_values() {
    let column = ObjectPropertyColumn::from_json_values(vec![
        Some(serde_json::Value::Bool(true)),
        Some(serde_json::Value::Bool(false)),
        None,
    ]);

    let matcher = column.contains_matcher("TRUE");

    assert!(column.matches_contains(0, &matcher));
    assert!(!column.matches_contains(1, &matcher));
    assert!(!column.matches_contains(2, &matcher));
}

#[test]
fn numeric_contains_filter_preserves_text_matching_behavior() {
    let column = ObjectPropertyColumn::from_json_values(vec![
        Some(serde_json::Value::Number(serde_json::Number::from(1234))),
        Some(serde_json::Value::Number(serde_json::Number::from(56))),
    ]);

    let matcher = column.contains_matcher("23");

    assert!(column.matches_contains(0, &matcher));
    assert!(!column.matches_contains(1, &matcher));
}

#[test]
fn nullable_f32_column_preserves_values_and_nulls_across_bitmap_words() {
    let expected = (0..130)
        .map(|index| (index % 3 != 0).then_some(index as f32 + 0.25))
        .collect::<Vec<_>>();
    let column = NullableF32Column::from_optional_values(expected.iter().copied());

    assert_eq!(column.len(), expected.len());
    for (index, expected) in expected.into_iter().enumerate() {
        assert_eq!(column.get(index), expected);
    }
    assert_eq!(column.get(130), None);
}

#[test]
fn floating_property_columns_use_compact_f32_storage() {
    let row_count = 1_000;
    let column = ObjectPropertyColumn::from_json_values(
        (0..row_count)
            .map(|index| (index % 5 != 0).then(|| serde_json::json!(index as f64 + 0.125)))
            .collect(),
    );
    let ObjectPropertyColumn::F32(values) = column else {
        panic!("floating values should use compact f32 storage");
    };

    assert_eq!(values.len(), row_count);
    assert_eq!(values.get(0), None);
    assert_eq!(values.get(1), Some(1.125));
    assert!(values.heap_bytes() < row_count * std::mem::size_of::<Option<f64>>() / 2);
}

#[test]
fn categorical_filter_options_are_available_for_dictionary_columns() {
    let column = ObjectPropertyColumn::from_json_values(vec![
        Some(serde_json::Value::String("tumor_myogenic".to_string())),
        Some(serde_json::Value::String("immune_myeloid".to_string())),
        Some(serde_json::Value::String("tumor_myogenic".to_string())),
    ]);

    assert_eq!(
        column.filter_value_options(8),
        Some(vec![
            "immune_myeloid".to_string(),
            "tumor_myogenic".to_string()
        ])
    );
    assert_eq!(column.filter_value_options(1), None);
}

#[test]
fn categorical_filter_options_are_available_for_bool_columns() {
    let column = ObjectPropertyColumn::from_json_values(vec![
        Some(serde_json::Value::Bool(true)),
        Some(serde_json::Value::Bool(false)),
    ]);

    assert_eq!(
        column.filter_value_options(8),
        Some(vec!["true".to_string(), "false".to_string()])
    );
}

#[test]
fn filtered_mask_contains_uses_dense_membership() {
    let mut layer = ObjectsLayer::default();
    layer.filtered_mask = Some(Arc::new(vec![false, true, false, true]));

    assert!(!layer.filtered_mask_contains(0));
    assert!(layer.filtered_mask_contains(1));
    assert!(!layer.filtered_mask_contains(2));
    assert!(layer.filtered_mask_contains(3));
    assert!(!layer.filtered_mask_contains(4));

    layer.filtered_mask = None;
    assert!(layer.filtered_mask_contains(4));
}

#[test]
fn object_layers_can_share_one_gpu_pool_without_sharing_resource_keys() {
    let pool = ObjectFillGlRenderer::application_pool();
    let mut first = ObjectsLayer::default();
    let mut second = ObjectsLayer::default();

    assert_ne!(
        first.render_resource_cache_id,
        second.render_resource_cache_id
    );
    assert_ne!(first.render_style_cache_id, second.render_style_cache_id);

    first.set_object_fill_renderer(pool.clone());
    second.set_object_fill_renderer(pool.clone());

    assert!(first.gl_object_fill.shares_pool_with(&pool));
    assert!(second.gl_object_fill.shares_pool_with(&pool));
    assert!(
        first
            .gl_object_fill
            .shares_pool_with(&second.gl_object_fill)
    );
}
