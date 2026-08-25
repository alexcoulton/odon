use super::*;
#[test]
fn settings_and_recent_projects_persist_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let settings_path = std::env::temp_dir().join(format!(
        "odon-actor-settings-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut settings = AppSettings::default();
    settings.record_recent_project(&PathBuf::from("/tmp/first.odon.project.json"));
    settings.record_recent_project(&PathBuf::from("/tmp/second.odon.project.json"));
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapSettings {
            settings,
            path: Some(settings_path.clone()),
            recent_project_exists: Vec::new(),
        })
        .unwrap();

    let (set, set_rx) = request(
        "app.settings.set",
        json!({
            "fast_object_rendering":false,
            "auto_contrast":{"method":"p1_to_p99","lower_percentile":2,"upper_percentile":98},
        }),
    );
    channels.request_tx.send(set).unwrap();
    let updated = set_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(updated["fast_object_rendering"], false);
    assert_eq!(updated["auto_contrast"]["method"], "p1_to_p99");
    let persisted = AppSettings::load_from(&settings_path).unwrap();
    assert!(!persisted.fast_object_rendering);

    let (forget, forget_rx) = request(
        "app.recent_projects.forget",
        json!({"path":"/tmp/first.odon.project.json"}),
    );
    channels.request_tx.send(forget).unwrap();
    assert_eq!(
        forget_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()["forgotten"],
        true
    );
    let (list, list_rx) = request("app.recent_projects.list", json!({}));
    channels.request_tx.send(list).unwrap();
    assert_eq!(
        list_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["projects"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let (clear, clear_rx) = request("app.recent_projects.clear", json!({}));
    channels.request_tx.send(clear).unwrap();
    assert_eq!(
        clear_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()["cleared"],
        1
    );
    assert!(
        AppSettings::load_from(&settings_path)
            .unwrap()
            .recent_projects
            .is_empty()
    );
    let _ = fs::remove_file(settings_path);
}

#[test]
fn application_shell_profiles_persist_through_the_settings_worker() {
    let channels = spawn_test_actor();
    let settings_path = std::env::temp_dir().join(format!(
        "odon-shell-profiles-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapSettings {
            settings: AppSettings::default(),
            path: Some(settings_path.clone()),
            recent_project_exists: Vec::new(),
        })
        .unwrap();

    let (save, save_rx) = request(
        "ui.shell.profiles.save",
        json!({"name":"Review","scope":"application","mode":"project"}),
    );
    channels.request_tx.send(save).unwrap();
    let saved = save_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(saved["persisted"], true);
    let persisted = AppSettings::load_from(&settings_path).unwrap();
    assert_eq!(
        persisted.shell_layout_profiles["Review"]["format"],
        "odon.shell-layout"
    );

    let (startup, startup_rx) = request(
        "app.settings.set",
        json!({"shell_layout_startup_profiles":{"project":"Review"}}),
    );
    channels.request_tx.send(startup).unwrap();
    let startup = startup_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(
        startup["shell_layout_startup_profiles"]["project"],
        "Review"
    );
    let persisted = AppSettings::load_from(&settings_path).unwrap();
    assert_eq!(persisted.shell_layout_startup_profiles["project"], "Review");

    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    let restarted = spawn_control_actor(Arc::new(|| {}), resources).unwrap();
    restarted
        .model_tx
        .send(ActorModelUpdate::BootstrapSettings {
            settings: persisted,
            path: Some(settings_path.clone()),
            recent_project_exists: Vec::new(),
        })
        .unwrap();
    let (startup_state, startup_state_rx) = request("app.settings.get", json!({}));
    restarted.request_tx.send(startup_state).unwrap();
    let startup_state = startup_state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        startup_state["shell_layout_startup_restore"]["results"]["project"]["status"],
        "restored"
    );

    let (list, list_rx) = request("ui.shell.profiles.list", json!({"scope":"application"}));
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(listed["profiles"][0]["name"], "Review");
    assert_eq!(listed["profiles"][0]["startup_modes"], json!(["project"]));

    let (remove, remove_rx) = request(
        "ui.shell.profiles.remove",
        json!({"name":"Review","scope":"application"}),
    );
    channels.request_tx.send(remove).unwrap();
    assert_eq!(
        remove_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap()["removed"],
        true
    );
    assert!(
        AppSettings::load_from(&settings_path)
            .unwrap()
            .shell_layout_profiles
            .is_empty()
    );
    assert!(
        AppSettings::load_from(&settings_path)
            .unwrap()
            .shell_layout_startup_profiles
            .is_empty()
    );
    let _ = fs::remove_file(settings_path);
}
