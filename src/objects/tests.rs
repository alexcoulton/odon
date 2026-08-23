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
                bbox_world: [0.0, 0.0, 1.0, 1.0],
                centroid_world: [0.5, 0.5],
                polygons_world: Arc::new(Vec::new()),
                point_position_world: Some([0.5, 0.5]),
                area_px: 0.0,
                perimeter_px: 0.0,
                properties: serde_json::json!({"kind":"tumour","score":0.9})
                    .as_object()
                    .unwrap()
                    .clone(),
            },
            odon::model::ControlObjectFeature {
                id: "cell-b".to_string(),
                bbox_world: [1.0, 1.0, 2.0, 2.0],
                centroid_world: [1.5, 1.5],
                polygons_world: Arc::new(Vec::new()),
                point_position_world: Some([1.5, 1.5]),
                area_px: 0.0,
                perimeter_px: 0.0,
                properties: serde_json::json!({"kind":"immune","score":0.2})
                    .as_object()
                    .unwrap()
                    .clone(),
            },
        ]),
        property_names: Arc::new(vec![
            "id".to_string(),
            "kind".to_string(),
            "score".to_string(),
        ]),
        renderer_payload: None,
    }
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
    assert!((resource.features[0].centroid_world[0] - 1.0).abs() < 1e-5);
    assert!((resource.features[0].centroid_world[1] - 2.0).abs() < 1e-5);
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
