use super::*;
#[test]
fn viewport_lifecycle_rejects_invalid_layouts_and_preserves_final_view() {
    let mut app = fixture_app();
    assert!(
        app.control_set_viewport_layout(&serde_json::json!({
            "layout": "single",
            "ratio": 0.05,
        }))
        .get("error")
        .is_some()
    );
    assert!(
        app.control_set_viewport_layout(&serde_json::json!({
            "layout": "single",
            "ratio": "half",
        }))
        .get("error")
        .is_some()
    );
    assert!(
        app.control_set_viewport_layout(&serde_json::json!({"layout": "horizontal"}))
            .get("error")
            .is_some()
    );
    let initial_id = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = app.control_create_viewport(&serde_json::json!({
        "layout": "vertical",
        "activate": false,
    }));
    let second_id = created["viewport_id"].as_str().unwrap().to_string();
    assert_eq!(
        app.control_viewport_workspace_snapshot()["active_viewport_id"],
        initial_id
    );
    assert_eq!(app.control_get_viewport_layout()["layout"], "vertical");
    let wrong_order = app.control_set_viewport_layout(&serde_json::json!({
        "layout": "horizontal",
        "viewports": [second_id.clone(), initial_id.clone()],
    }));
    assert!(
        wrong_order["error"]
            .as_str()
            .unwrap()
            .contains("workspace order")
    );
    let explicit_layout = app.control_set_viewport_layout(&serde_json::json!({
        "layout": "horizontal",
        "viewports": [initial_id.clone(), second_id.clone()],
        "ratio": 0.6,
    }));
    assert!(
        explicit_layout.get("error").is_none(),
        "{explicit_layout:#}"
    );
    let resized = app.control_set_viewport_layout(&serde_json::json!({
        "layout": "vertical",
        "ratio": 0.7,
    }));
    assert!((resized["ratio"].as_f64().unwrap() - 0.7).abs() < 1.0e-6);
    assert!(
        app.control_set_viewport_layout(&serde_json::json!({
            "layout": "vertical",
            "ratio": 0.95,
        }))
        .get("error")
        .is_some()
    );
    assert_eq!(app.control_get_viewport_links()["links"]["selection"], true);
    let swapped = app.control_swap_viewports();
    assert_eq!(swapped["changed"], true);
    assert_eq!(
        swapped["workspace"]["viewports"][0]["viewport_id"],
        second_id
    );
    assert!(
        app.control_create_viewport(&serde_json::json!({"layout": "horizontal"}))
            .get("error")
            .is_some()
    );
    assert_eq!(
        app.control_remove_viewport(&serde_json::json!({"viewport_id": second_id}))["removed"],
        true
    );
    assert!(
        app.control_get_viewport(&serde_json::json!({"viewport_id": second_id}))["error"]
            .as_str()
            .unwrap()
            .contains("not found")
    );
    assert!(
        app.control_remove_viewport(&serde_json::json!({"viewport_id": initial_id}))
            .get("error")
            .is_some()
    );
}
