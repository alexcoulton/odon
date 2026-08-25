use super::*;
#[test]
fn viewport_navigation_and_presentation_revisions_are_scoped_and_guarded() {
    let mut app = fixture_actor_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = app.actor_command(
        "viewer.viewports.create",
        serde_json::json!({"layout": "horizontal"}),
    )["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let initial = app.control_get_viewport(&serde_json::json!({"viewport_id": left}));
    assert_eq!(initial["navigation_revision"], 1);
    assert_eq!(initial["presentation_revision"], 1);

    let navigation = app.actor_command(
        "viewer.viewports.camera.set",
        serde_json::json!({
            "viewport_id": left,
            "center_world_lvl0": [42.0, 24.0],
            "if_navigation_revision": 1,
        }),
    );
    assert_eq!(navigation["navigation_revision"], 2);
    assert_eq!(navigation["presentation_revision"], 1);
    assert_eq!(
        navigation["affected_viewport_ids"],
        serde_json::json!([left, right])
    );
    assert!(navigation["link_transaction_id"].is_string());

    let stale_navigation = app
        .try_actor_command(
            "viewer.viewports.camera.set",
            serde_json::json!({
                "viewport_id": left,
                "zoom": 3.0,
                "if_navigation_revision": 1,
            }),
        )
        .expect_err("stale navigation revision must conflict");
    assert!(stale_navigation.message.contains("revision conflict"));
    assert_eq!(stale_navigation.data.unwrap()["current_revision"], 2);

    let left_style = app.actor_command(
        "viewer.viewports.objects.style.set",
        serde_json::json!({
            "viewport_id": left,
            "fill_cells": true,
            "color_mapping":{
                "mode":"continuous",
                "property":"score",
                "palette":"viridis",
                "domain":[0.0,10.0]
            },
            "if_presentation_revision": 1,
        }),
    );
    assert_eq!(left_style["presentation_revision"], 2);
    assert_eq!(left_style["navigation_revision"], 2);
    let right_style = app.actor_command(
        "viewer.viewports.objects.style.set",
        serde_json::json!({
            "viewport_id": right,
            "fill_cells": true,
            "color_mapping":{
                "mode":"continuous",
                "property":"score",
                "palette":"magma",
                "domain":[10.0,20.0]
            },
            "if_presentation_revision": 1,
        }),
    );
    assert_eq!(right_style["presentation_revision"], 2);
    assert_eq!(
        app.control_get_viewport(&serde_json::json!({"viewport_id":left}))["objects"]["color_mapping"]
            ["domain"],
        serde_json::json!([0.0, 10.0])
    );
    assert_eq!(
        app.control_get_viewport(&serde_json::json!({"viewport_id":right}))["objects"]["color_mapping"]
            ["palette"],
        "magma"
    );

    let stale_style = app
        .try_actor_command(
            "viewer.viewports.objects.style.set",
            serde_json::json!({
                "viewport_id": left,
                "fill_opacity": 0.2,
                "if_presentation_revision": 1,
            }),
        )
        .expect_err("stale presentation revision must conflict");
    let data = stale_style.data.unwrap();
    assert_eq!(data["revision_domain"], "presentation");
    assert_eq!(data["current_revision"], 2);
}
