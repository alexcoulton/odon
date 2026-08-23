use super::*;
use odon::control::ControlCommand;

fn local_roi(id: &str) -> ProjectRoi {
    let mut roi = ProjectRoi {
        id: id.to_string(),
        dataset: Some("default".to_string()),
        ..Default::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        PathBuf::from("/tmp/fixture.ome.zarr"),
    ));
    roi
}

#[test]
fn actor_owned_project_action_bypasses_the_viewer_host_request_relay() {
    let mut app = fixture_app();
    app.project_space_mut().set_control_actor_owned(true);
    let roi = local_roi("ROI-A");

    app.handle_project_space_action(ProjectSpaceAction::Open(roi));

    assert!(app.take_platform_effect().is_none());
    let mut intents = app.project_space_mut().take_control_intents();
    assert_eq!(intents.len(), 1);
    let intent = intents.remove(0);
    assert_eq!(intent.method, "project.rois.open");
    assert_eq!(intent.params["roi"], "ROI-A");
    ControlCommand::decode(intent.method, intent.params)
        .expect("viewer project action emits a typed actor command");
}

#[test]
fn actor_owned_roi_selector_action_bypasses_the_viewer_host_request_relay() {
    let mut app = fixture_app();
    app.project_space_mut().set_control_actor_owned(true);

    app.handle_roi_selector_action(
        &egui::Context::default(),
        RoiSelectorAction::OpenRoi(local_roi("ROI-B")),
    );

    assert!(app.take_platform_effect().is_none());
    let mut intents = app.project_space_mut().take_control_intents();
    assert_eq!(intents.len(), 1);
    let intent = intents.remove(0);
    assert_eq!(intent.method, "project.rois.open");
    assert_eq!(intent.params["roi"], "ROI-B");
    ControlCommand::decode(intent.method, intent.params)
        .expect("ROI selector action emits a typed actor command");
}
