use super::*;

fn open_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, reply) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    reply
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
}

fn bootstrap_auto_contrast(channels: &ControlActorChannels, enabled_on_open: bool) {
    let mut settings = AppSettings::default();
    settings.auto_contrast.enabled_on_open = enabled_on_open;
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapSettings {
            settings,
            path: None,
            recent_project_exists: Vec::new(),
        })
        .unwrap();
}

#[test]
fn image_histogram_completes_and_publishes_without_a_frame() {
    let channels = spawn_test_actor();
    bootstrap_auto_contrast(&channels, false);
    open_fixture(&channels);

    let (request, reply) = request(
        "viewer.channels.intensity_stats",
        json!({"channel":0,"level":0,"bins":32,"request_id":71}),
    );
    channels.request_tx.send(request).unwrap();
    let result = reply
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(result["request_id"], 71);
    assert_eq!(result["histogram"]["bins"].as_array().unwrap().len(), 32);
    assert_eq!(
        result["histogram"]["bins"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_u64)
            .sum::<u64>(),
        result["n"].as_u64().unwrap()
    );

    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert_eq!(
        projection.channel_compute_state["histogram"]["request_id"],
        71
    );
    assert_eq!(
        projection.channel_compute_state["histogram"]["pending"],
        false
    );
}

#[test]
fn explicit_auto_contrast_commits_without_a_frame() {
    let channels = spawn_test_actor();
    bootstrap_auto_contrast(&channels, false);
    open_fixture(&channels);

    let (manual, manual_reply) = request(
        "viewer.channels.set_contrast",
        json!({"channel":0,"min":1000.0,"max":2000.0}),
    );
    channels.request_tx.send(manual).unwrap();
    manual_reply
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();

    let (automatic, automatic_reply) = request(
        "viewer.channels.auto_contrast",
        json!({"channels":[0],"overwrite_manual":true,"method":"zero_to_max"}),
    );
    channels.request_tx.send(automatic).unwrap();
    let result = automatic_reply
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(result["result"]["completed"], true);
    assert_eq!(result["result"]["applied"].as_array().unwrap().len(), 1);

    let (contrast, contrast_reply) = request("viewer.channels.get_contrast", json!({"channel":0}));
    channels.request_tx.send(contrast).unwrap();
    let contrast = contrast_reply
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_ne!(contrast["contrast"]["min"], 1000.0);
    assert_ne!(contrast["contrast"]["max"], 2000.0);
}

#[test]
fn on_open_auto_contrast_finishes_while_projection_is_not_consumed() {
    let channels = spawn_test_actor();
    bootstrap_auto_contrast(&channels, true);
    open_fixture(&channels);

    // Deliberately leave the capacity-one presentation queue full while actor work proceeds.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let (get, reply) = request("app.get_loading_state", json!({}));
        channels.request_tx.send(get).unwrap();
        let value = reply.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        if value["loading"]["operations"]["channel_compute:auto_contrast"]["ready"] == true {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }

    let (get, reply) = request("viewer.channels.get_contrast", json!({"channel":0}));
    channels.request_tx.send(get).unwrap();
    let contrast = reply.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
    assert!(
        contrast["contrast"]["max"].as_f64().unwrap()
            > contrast["contrast"]["min"].as_f64().unwrap()
    );

    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert_eq!(
        projection.channel_compute_state["auto_contrast"]["pending"],
        false
    );
    assert_eq!(
        projection.channel_compute_state["auto_contrast"]["completed"],
        true
    );
}
