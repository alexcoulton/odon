use super::*;
#[test]
fn lifecycle_decisions_are_actor_owned_and_emit_only_platform_effects() {
    let channels = spawn_test_actor();
    let (add, add_rx) = request(
        "project.rois.add",
        json!({"id":"dirty-roi","path":"/tmp/dirty.ome.zarr"}),
    );
    channels.request_tx.send(add).unwrap();
    add_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (get, get_rx) = request("app.lifecycle.get", json!({}));
    channels.request_tx.send(get).unwrap();
    let lifecycle = get_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(lifecycle["dirty"], true);
    assert_eq!(lifecycle["can_save"], false);

    let (prompt, prompt_rx) = request("app.lifecycle.request_close", json!({"save":"prompt"}));
    channels.request_tx.send(prompt).unwrap();
    let confirmation = prompt_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(confirmation["confirmation_required"], true);
    assert!(channels.platform_effect_rx.try_recv().is_err());

    let (discard, discard_rx) = request("app.lifecycle.request_quit", json!({"save":"discard"}));
    channels.request_tx.send(discard).unwrap();
    let accepted = discard_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(accepted["accepted"], true);
    assert_eq!(accepted["action"], "quit");
    assert_eq!(accepted["discarded"], true);
    assert_eq!(
        channels
            .platform_effect_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        PlatformEffect::CloseWindow { quit: true }
    );

    let output = std::env::temp_dir().join(format!(
        "odon-lifecycle-save-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let (save_as, save_as_rx) = request("project.save_as", json!({"path":output}));
    channels.request_tx.send(save_as).unwrap();
    save_as_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    let (update, update_rx) = request(
        "project.rois.update",
        json!({"target_id":"dirty-roi","changes":{"display_name":"Saved before close"}}),
    );
    channels.request_tx.send(update).unwrap();
    update_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (save_close, save_close_rx) =
        request("app.lifecycle.request_close", json!({"save":"save"}));
    channels.request_tx.send(save_close).unwrap();
    let save_close = save_close_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(save_close["accepted"], true);
    assert_eq!(save_close["saved"], true);
    assert_eq!(
        channels
            .platform_effect_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        PlatformEffect::CloseWindow { quit: false }
    );
    let saved: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(
        saved["config"]["rois"][0]["display_name"],
        "Saved before close"
    );
    let _ = fs::remove_file(output);
    assert_eq!(channels.legacy_rx.len(), 0);
}
