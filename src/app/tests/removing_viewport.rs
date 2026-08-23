use super::*;
#[test]
fn removing_viewport_drops_only_its_cpu_generation_during_loading() {
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
    let left_id = ViewportId::new(left).unwrap();
    let right_id = ViewportId::new(right.clone()).unwrap();
    let workspace = app.viewport_workspace.as_mut().unwrap();
    workspace.get_mut(&left_id).unwrap().state.active_render_id = 91_001;
    workspace.get_mut(&right_id).unwrap().state.active_render_id = 91_002;
    app.active_render_id = 91_002;
    app.loader
        .set_active_render_ids(HashSet::from([91_001, 91_002]));

    let removed = app.actor_command(
        "viewer.viewports.remove",
        serde_json::json!({
            "viewport_id": right,
        }),
    );
    assert_eq!(removed["removed"], true);
    let accepted = app.loader.active_render_ids.lock().unwrap().clone();
    assert!(accepted.contains(&91_001));
    assert!(!accepted.contains(&91_002));
}
