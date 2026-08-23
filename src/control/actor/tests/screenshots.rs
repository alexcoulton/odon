use super::*;
use image::GenericImageView;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

struct ScreenshotTestDir(PathBuf);

impl ScreenshotTestDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "odon-actor-screenshot-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for ScreenshotTestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn open_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
}

fn acknowledge_latest_projection(channels: &ControlActorChannels) -> RenderProjection {
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("actor projection");
    channels
        .model_tx
        .send(ActorModelUpdate::PresentationApplied(projection.revision))
        .unwrap();
    projection
}

#[test]
fn screenshot_preferences_set_and_get_complete_without_a_ui_frame() {
    let channels = spawn_test_actor();
    open_fixture(&channels);
    let output_dir = std::env::temp_dir();
    let (set, set_rx) = request(
        "viewer.screenshot.settings.set",
        json!({
            "output_dir":output_dir,
            "include_scale_bar":false,
            "include_legend":false,
            "scale_bar_scale":1.5,
            "legend_scale":2.25,
        }),
    );
    channels.request_tx.send(set).unwrap();
    let updated = set_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(updated["output_dir"], json!(output_dir.to_string_lossy()));
    assert_eq!(updated["include_scale_bar"], false);
    assert_eq!(updated["include_legend"], false);
    assert_eq!(updated["scale_bar_scale"], 1.5);
    assert_eq!(updated["legend_scale"], 2.25);
    assert_eq!(updated["settings_pending"], false);

    let (get, get_rx) = request("viewer.screenshot.settings.get", json!({}));
    channels.request_tx.send(get).unwrap();
    let current = get_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    for field in [
        "output_dir",
        "include_scale_bar",
        "include_legend",
        "scale_bar_scale",
        "legend_scale",
        "settings_generation",
        "settings_pending",
    ] {
        assert_eq!(current[field], updated[field], "field {field}");
    }

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.screenshot_preferences.output_dir(),
        Some(output_dir.as_path())
    );
    assert!(!projection.screenshot_preferences.include_scale_bar());
}

#[test]
fn invalid_screenshot_output_directory_does_not_replace_existing_preferences() {
    let channels = spawn_test_actor();
    open_fixture(&channels);
    let (set, set_rx) = request(
        "viewer.screenshot.settings.set",
        json!({"include_legend":false}),
    );
    channels.request_tx.send(set).unwrap();
    set_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();

    let missing = std::env::temp_dir().join(format!(
        "odon-missing-screenshot-directory-{}",
        std::process::id()
    ));
    let (invalid, invalid_rx) = request(
        "viewer.screenshot.settings.set",
        json!({"output_dir":missing,"include_legend":true}),
    );
    channels.request_tx.send(invalid).unwrap();
    let error = invalid_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Application);

    let (get, get_rx) = request("viewer.screenshot.settings.get", json!({}));
    channels.request_tx.send(get).unwrap();
    let current = get_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(current["output_dir"], Value::Null);
    assert_eq!(current["include_legend"], false);
}

#[test]
fn viewer_capture_waits_for_presentation_but_does_not_block_the_actor() {
    let directory = ScreenshotTestDir::new("viewer");
    let path = directory.0.join("capture.png");
    let channels = spawn_test_actor();
    open_fixture(&channels);
    acknowledge_latest_projection(&channels);

    let (capture, capture_rx) = request(
        "viewer.screenshot.capture",
        json!({"path":path,"overwrite":false}),
    );
    channels.request_tx.send(capture).unwrap();
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("capture projection");
    assert!(channels.presentation_capture_rx.try_recv().is_err());
    assert!(capture_rx.try_recv().is_err());

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    assert_eq!(
        state_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("actor remains responsive")
            .unwrap()["mode"],
        "single"
    );
    let (mutation, mutation_rx) = request(
        "viewer.channels.set_visible",
        json!({"channels":[0],"mode":"only"}),
    );
    channels.request_tx.send(mutation).unwrap();
    std::thread::sleep(Duration::from_millis(25));
    assert!(
        mutation_rx.try_recv().is_err(),
        "presentation-affecting mutations must not overtake the capture revision"
    );

    channels
        .model_tx
        .send(ActorModelUpdate::PresentationApplied(projection.revision))
        .unwrap();
    let render_request = channels
        .presentation_capture_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("renderer capture request");
    assert_eq!(
        render_request.desired_projection_revision,
        projection.revision
    );
    assert_eq!(render_request.mode, ModelMode::Single);
    assert!(matches!(
        render_request.scope,
        PresentationCaptureScope::Viewer { .. }
    ));
    channels
        .presentation_completion_tx
        .send(PresentationCaptureCompletion {
            capture_id: render_request.capture_id,
            result: Ok(PresentationPixels {
                width: 2,
                height: 2,
                rgba: vec![
                    0, 0, 255, 255, 255, 255, 255, 128, // bottom row
                    255, 0, 0, 255, 0, 255, 0, 64, // top row
                ],
                bottom_up: true,
            }),
        })
        .unwrap();
    let response = capture_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("capture completion")
        .unwrap();
    assert_eq!(response["screenshot"]["completed"], true);
    assert_eq!(
        response["screenshot"]["presented_projection_revision"],
        projection.revision
    );
    assert!(response["screenshot"]["bytes"].as_u64().unwrap() > 0);
    mutation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("deferred mutation settles after pixel readback")
        .unwrap();

    let image = image::open(&path).expect("open actor-written PNG");
    assert_eq!(image.dimensions(), (2, 2));
    let rgba = image.to_rgba8();
    assert_eq!(rgba.get_pixel(0, 0).0, [255, 0, 0, 255]);
    assert_eq!(rgba.get_pixel(1, 0).0, [0, 255, 0, 64]);
    assert_eq!(rgba.get_pixel(0, 1).0, [0, 0, 255, 255]);
    assert_eq!(rgba.get_pixel(1, 1).0, [255, 255, 255, 128]);
}

#[test]
fn cancelled_capture_is_removed_while_waiting_for_presentation() {
    let directory = ScreenshotTestDir::new("cancel");
    let path = directory.0.join("cancelled.png");
    let channels = spawn_test_actor();
    open_fixture(&channels);
    acknowledge_latest_projection(&channels);

    let (mut capture, capture_rx) = request(
        "viewer.screenshot.capture",
        json!({"path":path,"overwrite":false}),
    );
    let task = capture
        .task_registry
        .create("capture", "test", true)
        .unwrap();
    capture.task_id = Some(task.task_id.clone());
    let registry = Arc::clone(&capture.task_registry);
    channels.request_tx.send(capture).unwrap();
    channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("capture projection");
    registry.cancel(&task.task_id).unwrap();

    let error = capture_rx
        .recv_timeout(Duration::from_secs(3))
        .expect("cancelled capture settles")
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Cancelled);
    assert!(!path.exists());
    assert!(channels.presentation_capture_rx.try_recv().is_err());
}

fn receive_capture_effect(
    channels: &ControlActorChannels,
    method: &str,
    path: &PathBuf,
) -> (
    PresentationCaptureRequest,
    Receiver<Result<Value, ControlError>>,
    RenderProjection,
) {
    let (capture, response) = request(method, json!({"path":path,"overwrite":false}));
    channels.request_tx.send(capture).unwrap();
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("capture projection");
    channels
        .model_tx
        .send(ActorModelUpdate::PresentationApplied(projection.revision))
        .unwrap();
    let effect = channels
        .presentation_capture_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("presentation capture effect");
    (effect, response, projection)
}

fn reject_test_capture(
    channels: &ControlActorChannels,
    effect: PresentationCaptureRequest,
    response: Receiver<Result<Value, ControlError>>,
) {
    channels
        .presentation_completion_tx
        .send(PresentationCaptureCompletion {
            capture_id: effect.capture_id,
            result: Err("synthetic renderer rejection".to_string()),
        })
        .unwrap();
    assert_eq!(
        response
            .recv_timeout(Duration::from_secs(2))
            .expect("capture rejection")
            .unwrap_err()
            .kind,
        ControlErrorKind::Application
    );
}

#[test]
fn workspace_window_and_project_capture_scopes_are_actor_owned() {
    let directory = ScreenshotTestDir::new("scopes");

    let workspace_actor = spawn_test_actor();
    open_fixture(&workspace_actor);
    acknowledge_latest_projection(&workspace_actor);
    let (effect, response, _) = receive_capture_effect(
        &workspace_actor,
        "viewer.workspace.screenshot.capture",
        &directory.0.join("workspace.png"),
    );
    assert_eq!(effect.mode, ModelMode::Single);
    assert_eq!(effect.scope, PresentationCaptureScope::Workspace);
    reject_test_capture(&workspace_actor, effect, response);

    let window_actor = spawn_test_actor();
    let (effect, response, projection) = receive_capture_effect(
        &window_actor,
        "app.screenshot.capture",
        &directory.0.join("window.png"),
    );
    assert_eq!(projection.mode, ModelMode::Project);
    assert_eq!(effect.mode, ModelMode::Project);
    assert_eq!(effect.scope, PresentationCaptureScope::Window);
    reject_test_capture(&window_actor, effect, response);

    let project_actor = spawn_test_actor();
    open_fixture(&project_actor);
    acknowledge_latest_projection(&project_actor);
    let (effect, response, projection) = receive_capture_effect(
        &project_actor,
        "project.screenshot.capture",
        &directory.0.join("project.png"),
    );
    assert_eq!(projection.mode, ModelMode::Project);
    assert_eq!(effect.mode, ModelMode::Project);
    assert_eq!(effect.scope, PresentationCaptureScope::Project);
    reject_test_capture(&project_actor, effect, response);
}

#[test]
fn invalid_project_capture_does_not_change_actor_mode() {
    let channels = spawn_test_actor();
    open_fixture(&channels);
    acknowledge_latest_projection(&channels);
    let (capture, response) = request("project.screenshot.capture", json!({}));
    channels.request_tx.send(capture).unwrap();
    assert_eq!(
        response
            .recv_timeout(Duration::from_secs(1))
            .expect("invalid capture response")
            .unwrap_err()
            .kind,
        ControlErrorKind::InvalidParams
    );
    let (state, state_response) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    assert_eq!(
        state_response
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["mode"],
        "single"
    );
    assert!(channels.presentation_rx.try_recv().is_err());
}
