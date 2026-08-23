use super::*;
use crate::control::ControlErrorKind;
use serde_json::json;

#[test]
fn mask_crud_selection_and_undo_are_renderer_independent() {
    let mut masks = MaskModel::default();
    let layer = masks
        .dispatch("viewer.masks.layers.create", &json!({"name":"Cells"}))
        .unwrap();
    let id = layer["id"].as_u64().unwrap();
    masks
        .dispatch(
            "viewer.masks.polygons.add",
            &json!({"id":id,"vertices":[[1,2],[4,2],[4,5]]}),
        )
        .unwrap();
    let polygons = masks
        .dispatch("viewer.masks.polygons.list", &json!({"id":id}))
        .unwrap();
    assert_eq!(
        polygons["polygons"][0]["vertices_local"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    let selected = masks
        .dispatch(
            "viewer.masks.selection.set",
            &json!({"id":id,"index":0,"vertex_index":1}),
        )
        .unwrap();
    assert_eq!(selected["selection"]["vertex_index"], 1);
    assert_eq!(
        masks.dispatch("viewer.masks.undo", &json!({})).unwrap()["undone"],
        true
    );
    assert_eq!(
        masks
            .dispatch("viewer.masks.polygons.list", &json!({"id":id}))
            .unwrap()["total"],
        0
    );
}

#[test]
fn atomic_replacement_rejects_a_stale_native_generation() {
    let mut masks = MaskModel::default();
    masks
        .dispatch("viewer.masks.layers.create", &json!({"name":"Python"}))
        .unwrap();
    let error = masks
        .dispatch(
            "viewer.masks.state.replace",
            &json!({
                "expected_generation":1,
                "state":{"layers":[],"active_layer_id":null,"selection":null},
            }),
        )
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Conflict);
    assert_eq!(masks.projection_json()["layers"][0]["name"], "Python");
}

#[test]
fn granular_native_edit_rejects_a_stale_generation() {
    let mut masks = MaskModel::default();
    let layer = masks
        .dispatch("viewer.masks.layers.create", &json!({"name":"Python"}))
        .unwrap();
    let error = masks
        .dispatch(
            "viewer.masks.layers.update",
            &json!({
                "id":layer["id"],
                "visible":false,
                "expected_generation":1,
            }),
        )
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Conflict);
    assert_eq!(masks.projection_json()["layers"][0]["visible"], true);
}

#[test]
fn append_reconciliation_preserves_edits_made_after_the_worker_snapshot() {
    let mut masks = MaskModel::default();
    let layer = masks
        .dispatch("viewer.masks.layers.create", &json!({"name":"Drawn"}))
        .unwrap();
    let id = layer["id"].as_u64().unwrap();
    masks
        .dispatch(
            "viewer.masks.polygons.add",
            &json!({"id":id,"vertices":[[1,1],[4,1],[4,4]]}),
        )
        .unwrap();
    let saved = masks.appendable_layers();
    masks
        .dispatch(
            "viewer.masks.polygons.add",
            &json!({"id":id,"vertices":[[10,10],[14,10],[14,14]]}),
        )
        .unwrap();

    let response = masks.reconcile_appended_file(
        &saved,
        "Exclusion masks".to_string(),
        vec![vec![[1.0, 1.0], [4.0, 1.0], [4.0, 4.0], [1.0, 1.0]]],
        std::path::PathBuf::from("project-masks.geojson"),
    );
    assert_eq!(response["cleared_polygon_count"], 1);
    assert_eq!(masks.layers.len(), 2);
    assert_eq!(masks.layers[0].polygons_world.len(), 1);
    assert_eq!(masks.layers[0].polygons_world[0][0], [10.0, 10.0]);
    assert_eq!(masks.layers[1].polygons_world.len(), 1);
    assert!(!masks.layers[1].editable);
}
