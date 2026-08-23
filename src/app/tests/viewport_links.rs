use super::*;
#[test]
fn canonical_viewport_link_group_validates_members_and_preserves_shared_selection() {
    let mut app = fixture_actor_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.clone",
        serde_json::json!({
            "layout": "horizontal",
        }),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    let configured = app.actor_command(
        "viewer.viewport_links.create",
        serde_json::json!({
            "viewports": [left, right],
            "fields": ["camera", "selection"],
        }),
    );
    assert!(configured.get("error").is_none(), "{configured:#}");
    assert_eq!(
        configured["link_group"]["link_group_id"],
        "comparison-navigation"
    );
    assert_eq!(
        configured["link_group"]["fields"],
        serde_json::json!(["camera", "selection"])
    );
    assert_eq!(
        app.actor_query("viewer.viewport_links.list", serde_json::json!({}))["link_groups"][0],
        configured["link_group"]
    );

    let wrong_members = app
        .try_actor_command(
            "viewer.viewport_links.create",
            serde_json::json!({"viewports": [left], "fields": ["plane"]}),
        )
        .unwrap_err();
    assert!(wrong_members.to_string().contains("exactly the two"));
    let unknown_field = app
        .try_actor_command(
            "viewer.viewport_links.update",
            serde_json::json!({"fields": ["time"]}),
        )
        .unwrap_err();
    assert!(unknown_field.to_string().contains("unknown"));

    let removed = app.actor_command(
        "viewer.viewport_links.remove",
        serde_json::json!({"link_group_id": "comparison-navigation"}),
    );
    assert_eq!(removed["removed"], true);
    assert_eq!(removed["links"]["camera"], false);
    assert_eq!(removed["links"]["plane"], false);
    assert_eq!(removed["links"]["selection"], true);
    assert_eq!(
        removed["link_group"]["fields"],
        serde_json::json!(["selection"])
    );
}
