use super::*;
#[test]
fn two_viewport_controls_keep_presentation_independent_and_navigation_linked() {
    let mut app = fixture_actor_app();
    let initial = app.control_viewport_workspace_snapshot();
    let left = initial["active_viewport_id"]
        .as_str()
        .expect("initial viewport ID")
        .to_string();

    let created = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "source_viewport_id": left,
            "title": "Property B",
            "layout": "horizontal",
        }),
    );
    let right = created["viewport_id"]
        .as_str()
        .expect("created viewport ID")
        .to_string();
    assert_ne!(left, right);
    assert_eq!(created["workspace"]["layout"], "horizontal");
    assert_eq!(
        created["workspace"]["viewports"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        created["workspace"]["shared_resources"]["document_instances"],
        1
    );
    assert_eq!(
        created["workspace"]["shared_resources"]["dataset_instances"],
        1
    );
    assert_eq!(
        created["workspace"]["shared_resources"]["cpu_tile_cache_instances"],
        1
    );
    assert_eq!(
        created["workspace"]["shared_resources"]["primary_object_geometry_instances"],
        1
    );

    let left_style = app.actor_command(
        "viewer.viewports.objects.style.set",
        serde_json::json!({
            "viewport_id": left,
            "fill_cells": true,
            "fill_opacity": 0.2,
        }),
    );
    assert!(
        (left_style["result"]["style"]["fill_opacity"]
            .as_f64()
            .unwrap()
            - 0.2)
            .abs()
            < 1.0e-6
    );
    let right_style = app.actor_command(
        "viewer.viewports.objects.style.set",
        serde_json::json!({
            "viewport_id": right,
            "fill_cells": true,
            "fill_opacity": 0.8,
        }),
    );
    assert!(
        (right_style["result"]["style"]["fill_opacity"]
            .as_f64()
            .unwrap()
            - 0.8)
            .abs()
            < 1.0e-6
    );

    app.actor_command(
        "viewer.viewports.channels.set_visible",
        serde_json::json!({
            "viewport_id": left,
            "channels": ["CD3"],
            "mode": "only",
        }),
    );
    app.actor_command(
        "viewer.viewports.channels.set_visible",
        serde_json::json!({
            "viewport_id": right,
            "channels": ["PanCK"],
            "mode": "only",
        }),
    );

    app.actor_command(
        "viewer.viewports.camera.set",
        serde_json::json!({
            "viewport_id": left,
            "center_world_lvl0": [123.0, 456.0],
            "zoom": 3.0,
        }),
    );
    let linked_right = app.actor_query(
        "viewer.viewports.camera.get",
        serde_json::json!({
            "viewport_id": right,
        }),
    );
    assert_eq!(
        linked_right["result"]["center_world_lvl0"],
        serde_json::json!([123.0, 456.0])
    );
    assert_eq!(linked_right["result"]["zoom_screen_per_lvl0_px"], 3.0);

    app.actor_command(
        "viewer.viewport_links.set",
        serde_json::json!({
            "camera": false,
            "plane": true,
            "selection": true,
        }),
    );
    app.actor_command(
        "viewer.viewports.camera.set",
        serde_json::json!({
            "viewport_id": right,
            "center_world_lvl0": [10.0, 20.0],
            "zoom": 1.5,
        }),
    );

    let workspace = app.control_viewport_workspace_snapshot();
    let viewports = workspace["viewports"].as_array().unwrap();
    let left_snapshot = viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == left)
        .unwrap();
    let right_snapshot = viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == right)
        .unwrap();
    assert!((left_snapshot["objects"]["fill_opacity"].as_f64().unwrap() - 0.2).abs() < 1.0e-6);
    assert!((right_snapshot["objects"]["fill_opacity"].as_f64().unwrap() - 0.8).abs() < 1.0e-6);
    assert_eq!(
        left_snapshot["camera"]["center_world_lvl0"],
        serde_json::json!([123.0, 456.0])
    );
    assert_eq!(
        right_snapshot["camera"]["center_world_lvl0"],
        serde_json::json!([10.0, 20.0])
    );
    assert_eq!(
        left_snapshot["channels"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|channel| channel["visible"] == true)
            .map(|channel| channel["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["CD3"]
    );
    assert_eq!(
        right_snapshot["channels"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|channel| channel["visible"] == true)
            .map(|channel| channel["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["PanCK"]
    );
}
