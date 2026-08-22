use super::*;

fn open_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
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
    assert_eq!(channels.legacy_rx.len(), 0);
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
    assert_eq!(channels.legacy_rx.len(), 0);
}
