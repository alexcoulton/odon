use super::*;

#[test]
fn background_actor_projects_workspace_topology_and_link_transactions() {
    let mut app = fixture_actor_app();

    let create = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "source_viewport_id": "viewport-1",
            "title": "Comparison",
            "layout": "horizontal",
            "ratio": 0.65,
        }),
    );
    let right = create["viewport_id"].as_str().unwrap().to_string();
    assert_eq!(
        workspace_topology(&create["workspace"]),
        workspace_topology(&app.control_viewport_workspace_snapshot())
    );

    app.actor_command(
        "viewer.viewports.rename",
        serde_json::json!({"viewport_id": right, "title": "Renamed comparison"}),
    );
    assert_eq!(
        app.actor_query(
            "viewer.viewports.get",
            serde_json::json!({"viewport_id": right}),
        )["title"],
        "Renamed comparison"
    );
    assert_eq!(
        app.control_viewport_workspace_snapshot()["viewports"][1]["title"],
        "Renamed comparison"
    );

    let layout = app.actor_command(
        "viewer.workspace.layout.set",
        serde_json::json!({
            "layout": "vertical",
            "ratio": 0.6,
            "viewports": ["viewport-1", right],
        }),
    );
    assert_eq!(layout["layout"], "vertical");
    assert_eq!(
        workspace_topology(&app.control_viewport_workspace_snapshot())["layout"],
        "vertical"
    );

    let unlink = app.actor_command(
        "viewer.viewport_links.set",
        serde_json::json!({"camera": false, "plane": false, "selection": true}),
    );
    assert_eq!(unlink["links"]["camera"], false);
    assert_eq!(unlink["links"]["plane"], false);
    assert_eq!(
        app.control_viewport_workspace_snapshot()["links"],
        unlink["links"]
    );

    let camera = app.actor_command(
        "viewer.viewports.camera.set",
        serde_json::json!({
            "viewport_id": "viewport-1",
            "center_world_lvl0": [123.0, 234.0],
            "zoom": 2.5,
        }),
    );
    assert_eq!(
        camera["result"]["center_world_lvl0"],
        serde_json::json!([123.0, 234.0])
    );

    app.actor_command(
        "viewer.viewport_links.set",
        serde_json::json!({"camera": true, "plane": true, "selection": true}),
    );
    let swapped = app.actor_command("viewer.workspace.swap", serde_json::json!({}));
    assert_eq!(swapped["changed"], true);
    assert_eq!(
        workspace_topology(&swapped["workspace"]),
        workspace_topology(&app.control_viewport_workspace_snapshot())
    );

    let removed = app.actor_command(
        "viewer.viewports.remove",
        serde_json::json!({"viewport_id": "viewport-1"}),
    );
    assert_eq!(removed["removed"], true);
    assert_eq!(removed["viewport_id"], "viewport-1");
    assert_eq!(
        workspace_topology(&removed["workspace"]),
        workspace_topology(&app.control_viewport_workspace_snapshot())
    );
}
