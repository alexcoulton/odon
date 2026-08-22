use super::*;
#[test]
fn viewport_navigation_and_presentation_revisions_are_scoped_and_guarded() {
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
    let initial = app.control_get_viewport(&serde_json::json!({"viewport_id": left}));
    assert_eq!(initial["navigation_revision"], 1);
    assert_eq!(initial["presentation_revision"], 1);

    let navigation = app.control_set_viewport_camera(&serde_json::json!({
        "viewport_id": left,
        "center_world_lvl0": [42.0, 24.0],
        "if_navigation_revision": 1,
    }));
    assert_eq!(navigation["navigation_revision"], 2);
    assert_eq!(navigation["presentation_revision"], 1);
    assert_eq!(
        navigation["affected_viewport_ids"],
        serde_json::json!([left, right])
    );
    assert!(navigation["link_transaction_id"].is_string());

    let stale_navigation = app.control_set_viewport_camera(&serde_json::json!({
        "viewport_id": left,
        "zoom": 3.0,
        "if_navigation_revision": 1,
    }));
    assert!(
        stale_navigation["error"]
            .as_str()
            .unwrap()
            .contains("revision conflict")
    );
    assert_eq!(stale_navigation["current_revision"], 2);

    let left_style = app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": left,
        "fill_cells": true,
        "if_presentation_revision": 1,
    }));
    assert_eq!(left_style["presentation_revision"], 2);
    assert_eq!(left_style["navigation_revision"], 2);
    let right_style = app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": right,
        "fill_cells": true,
        "if_presentation_revision": 1,
    }));
    assert_eq!(right_style["presentation_revision"], 2);

    let stale_style = app.control_set_viewport_object_style(&serde_json::json!({
        "viewport_id": left,
        "fill_opacity": 0.2,
        "if_presentation_revision": 1,
    }));
    assert_eq!(stale_style["revision_domain"], "presentation");
    assert_eq!(stale_style["current_revision"], 2);
}
