use super::*;
#[test]
fn channel_group_presentation_is_independent_and_persistent_per_viewport() {
    let mut app = fixture_app();
    let left = app.control_viewport_workspace_snapshot()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let grouped = app.control_set_viewport_channel_group(&serde_json::json!({
        "viewport_id": left,
        "channels": ["DAPI"],
        "group": "Nuclei",
        "color": "#102030",
        "inherit_color": true,
    }));
    assert_eq!(grouped["result"]["changed"], true);
    let right = app.control_create_viewport(&serde_json::json!({
        "title": "Override",
        "layout": "horizontal",
    }))["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let changed = app.control_set_viewport_channel_color(&serde_json::json!({
        "viewport_id": right,
        "channel": "DAPI",
        "color_rgb": [200, 100, 50],
    }));
    assert_eq!(changed["result"]["changed"], true);

    app.sync_current_view_state_into_project_space();
    let saved = app
        .project_space
        .roi_view_state(&app.dataset.source)
        .cloned()
        .unwrap();
    let mut restored = fixture_app();
    restored
        .project_space
        .set_roi_view_state(&restored.dataset.source, saved);
    restored.apply_view_state_from_project_space();

    let workspace = restored.viewport_workspace.as_ref().unwrap();
    let left_state = &workspace
        .get(&ViewportId::new(left).unwrap())
        .unwrap()
        .state;
    let right_state = &workspace
        .get(&ViewportId::new(right).unwrap())
        .unwrap()
        .state;
    let left_dapi = left_state
        .channels
        .iter()
        .find(|channel| channel.name == "DAPI")
        .unwrap();
    let right_dapi = right_state
        .channels
        .iter()
        .find(|channel| channel.name == "DAPI")
        .unwrap();
    assert_eq!(
        layer_groups::effective_channel_color_rgb(
            &left_state.layer_groups,
            &left_dapi.name,
            left_dapi.color_rgb,
        ),
        [0x10, 0x20, 0x30]
    );
    assert_eq!(
        layer_groups::effective_channel_color_rgb(
            &right_state.layer_groups,
            &right_dapi.name,
            right_dapi.color_rgb,
        ),
        [200, 100, 50]
    );
}
