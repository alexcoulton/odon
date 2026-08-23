use super::*;
#[test]
fn viewport_lifecycle_rejects_invalid_layouts_and_preserves_final_view() {
    let mut app = fixture_actor_app();
    assert!(
        app.try_actor_command(
            "viewer.workspace.layout.set",
            serde_json::json!({"layout": "single", "ratio": 0.05}),
        )
        .is_err()
    );
    assert!(
        app.try_actor_command(
            "viewer.workspace.layout.set",
            serde_json::json!({"layout": "single", "ratio": "half"}),
        )
        .is_err()
    );
    assert!(
        app.try_actor_command(
            "viewer.workspace.layout.set",
            serde_json::json!({"layout": "horizontal"}),
        )
        .is_err()
    );
    let initial_id = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({"layout": "vertical", "activate": false}),
    );
    let second_id = created["viewport_id"].as_str().unwrap().to_string();
    assert_eq!(
        app.control_viewport_workspace_snapshot()["active_viewport_id"],
        initial_id
    );
    assert_eq!(
        app.actor_query("viewer.workspace.layout.get", serde_json::json!({}))["layout"],
        "vertical"
    );
    let wrong_order = app
        .try_actor_command(
            "viewer.workspace.layout.set",
            serde_json::json!({
                "layout": "horizontal",
                "viewports": [second_id.clone(), initial_id.clone()],
            }),
        )
        .unwrap_err();
    assert!(wrong_order.to_string().contains("workspace order"));
    let explicit_layout = app.actor_command(
        "viewer.workspace.layout.set",
        serde_json::json!({
            "layout": "horizontal",
            "viewports": [initial_id.clone(), second_id.clone()],
            "ratio": 0.6,
        }),
    );
    assert_eq!(explicit_layout["layout"], "horizontal");
    let resized = app.actor_command(
        "viewer.workspace.layout.set",
        serde_json::json!({"layout": "vertical", "ratio": 0.7}),
    );
    assert!((resized["ratio"].as_f64().unwrap() - 0.7).abs() < 1.0e-6);
    assert!(
        app.try_actor_command(
            "viewer.workspace.layout.set",
            serde_json::json!({"layout": "vertical", "ratio": 0.95}),
        )
        .is_err()
    );
    assert_eq!(
        app.actor_query("viewer.viewport_links.get", serde_json::json!({}))["links"]["selection"],
        true
    );
    let swapped = app.actor_command("viewer.workspace.swap", serde_json::json!({}));
    assert_eq!(swapped["changed"], true);
    assert_eq!(
        swapped["workspace"]["viewports"][0]["viewport_id"],
        second_id
    );
    assert!(
        app.try_actor_command(
            "viewer.viewports.clone",
            serde_json::json!({"layout": "horizontal"}),
        )
        .is_err()
    );
    assert_eq!(
        app.actor_command(
            "viewer.viewports.remove",
            serde_json::json!({"viewport_id": second_id}),
        )["removed"],
        true
    );
    let missing = app
        .try_actor_command(
            "viewer.viewports.get",
            serde_json::json!({"viewport_id": second_id}),
        )
        .unwrap_err();
    assert!(missing.to_string().contains("not found"));
    assert!(
        app.try_actor_command(
            "viewer.viewports.remove",
            serde_json::json!({"viewport_id": initial_id}),
        )
        .is_err()
    );
}
