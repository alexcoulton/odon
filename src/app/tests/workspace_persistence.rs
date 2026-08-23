use super::*;
#[test]
fn multi_viewport_workspace_roundtrips_through_versioned_project_state() {
    let mut app = fixture_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.control_create_viewport(&serde_json::json!({
        "title": "Marker B",
        "layout": "vertical",
        "ratio": 0.65,
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(app.set_viewport_links(ViewportLinks {
        camera: false,
        plane: true,
        selection: true,
    }));
    app.control_set_viewport_channels(&serde_json::json!({
        "viewport_id": left,
        "channels": ["CD3"],
        "mode": "only",
    }));
    app.control_set_viewport_channels(&serde_json::json!({
        "viewport_id": right,
        "channels": ["PanCK"],
        "mode": "only",
    }));
    app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": left,
        "fill_cells": true,
        "fill_opacity": 0.25,
    }));
    app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": right,
        "fill_cells": true,
        "fill_opacity": 0.75,
    }));
    app.control_set_viewport_object_filter(&serde_json::json!({
        "viewport_id": right,
        "mode": "simple",
        "logic": "all",
        "clauses": [{"property": "id", "query": "1"}],
    }));
    app.control_set_viewport_rendering(&serde_json::json!({
        "viewport_id": left,
        "smooth_pixels": false,
        "show_scale_bar": false,
        "show_hud": false,
        "show_tile_debug": true,
    }));
    app.control_set_viewport_camera(&serde_json::json!({
        "viewport_id": left,
        "center_world_lvl0": [10.0, 20.0],
        "zoom": 2.0,
    }));
    app.control_set_viewport_camera(&serde_json::json!({
        "viewport_id": right,
        "center_world_lvl0": [30.0, 40.0],
        "zoom": 3.0,
    }));

    app.sync_current_view_state_into_project_space();
    let saved = app
        .project_space
        .roi_view_state(&app.dataset.source)
        .cloned()
        .expect("saved view state");
    assert_eq!(saved.workspace.as_ref().unwrap().version, 1);
    let encoded = serde_json::to_value(&saved).expect("serialize workspace");
    let decoded: ProjectRoiViewState =
        serde_json::from_value(encoded).expect("deserialize workspace");

    let mut restored = fixture_app();
    restored
        .project_space
        .set_roi_view_state(&restored.dataset.source, decoded);
    restored.apply_view_state_from_project_space();
    let workspace = restored.control_viewport_workspace_snapshot();
    assert_eq!(workspace["layout"], "vertical");
    assert!((workspace["ratio"].as_f64().unwrap() - 0.65).abs() < 1.0e-6);
    assert_eq!(workspace["active_viewport_id"], right);
    assert_eq!(workspace["links"]["camera"], false);
    let viewports = workspace["viewports"].as_array().unwrap();
    let left = viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == left)
        .unwrap();
    let right = viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == right)
        .unwrap();
    assert_eq!(
        left["camera"]["center_world_lvl0"],
        serde_json::json!([10.0, 20.0])
    );
    assert_eq!(
        right["camera"]["center_world_lvl0"],
        serde_json::json!([30.0, 40.0])
    );
    assert!((left["objects"]["fill_opacity"].as_f64().unwrap() - 0.25).abs() < 1e-6);
    assert!((right["objects"]["fill_opacity"].as_f64().unwrap() - 0.75).abs() < 1e-6);
    assert_eq!(left["rendering"]["smooth_pixels"], false);
    assert_eq!(left["rendering"]["show_scale_bar"], false);
    assert_eq!(left["rendering"]["show_hud"], false);
    assert_eq!(left["rendering"]["show_tile_debug"], true);
    assert_eq!(right["rendering"]["smooth_pixels"], true);
    let restored_filter = restored.control_get_viewport_object_filter(&serde_json::json!({
        "viewport_id": right["viewport_id"],
    }));
    assert_eq!(restored_filter["result"]["mode"], "simple");
    assert_eq!(restored_filter["result"]["active"], true);
}
