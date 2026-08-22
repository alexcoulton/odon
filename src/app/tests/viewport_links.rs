use super::*;
#[test]
fn canonical_viewport_link_group_validates_members_and_preserves_shared_selection() {
    let mut app = fixture_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.control_create_viewport(&serde_json::json!({
        "layout": "horizontal",
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    let configured = app.control_create_viewport_link_group(&serde_json::json!({
        "viewports": [left, right],
        "fields": ["camera", "selection"],
    }));
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
        app.control_list_viewport_link_groups()["link_groups"][0],
        configured["link_group"]
    );

    let wrong_members = app.control_create_viewport_link_group(&serde_json::json!({
        "viewports": [left],
        "fields": ["plane"],
    }));
    assert!(
        wrong_members["error"]
            .as_str()
            .unwrap()
            .contains("exactly the two")
    );
    let unknown_field = app.control_update_viewport_link_group(&serde_json::json!({
        "fields": ["time"],
    }));
    assert!(unknown_field["error"].as_str().unwrap().contains("unknown"));

    let removed = app.control_remove_viewport_link_group(&serde_json::json!({
        "link_group_id": "comparison-navigation",
    }));
    assert_eq!(removed["removed"], true);
    assert_eq!(removed["links"]["camera"], false);
    assert_eq!(removed["links"]["plane"], false);
    assert_eq!(removed["links"]["selection"], true);
    assert_eq!(
        removed["link_group"]["fields"],
        serde_json::json!(["selection"])
    );
}
