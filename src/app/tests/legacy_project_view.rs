use super::*;
#[test]
fn legacy_project_view_migrates_to_one_viewport() {
    let mut app = fixture_app();
    let legacy = ProjectRoiViewState {
        camera: Some(ProjectCameraState {
            center_world_lvl0: [77.0, 88.0],
            zoom_screen_per_lvl0_px: 1.25,
        }),
        ..Default::default()
    };
    app.project_space
        .set_roi_view_state(&app.dataset.source, legacy);
    app.apply_view_state_from_project_space();
    let workspace = app.control_viewport_workspace_snapshot();
    assert_eq!(workspace["layout"], "single");
    assert_eq!(workspace["viewports"].as_array().unwrap().len(), 1);
    assert_eq!(
        workspace["viewports"][0]["camera"]["center_world_lvl0"],
        serde_json::json!([77.0, 88.0])
    );
}
