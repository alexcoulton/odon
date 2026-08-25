use super::*;
#[test]
fn actor_publishes_scoped_and_active_compatibility_events_before_replying() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();

    let event_hub = EventHub::shared();
    let (event_tx, event_rx) = crossbeam_channel::bounded(8);
    event_hub.register("observer".to_string(), event_tx);
    event_hub
        .subscribe("observer", vec!["*".to_string()])
        .unwrap();
    let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "viewer.viewports.camera.set",
                json!({
                    "viewport_id": "viewport-1",
                    "center_world_lvl0": [123.0, 234.0],
                    "zoom": 2.0,
                }),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "initiator".to_string(),
            request_id: Some(json!(42)),
            event_hub,
            task_registry,
            task_id: None,
        })
        .unwrap();

    let reply = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("actor reply")
        .unwrap();
    assert_eq!(reply["_control"]["revision"], 1);

    // The actor publishes both events before sending the reply. Therefore neither receive
    // should have to wait once the reply is observable.
    let scoped = event_rx
        .try_recv()
        .expect("scoped viewport event was published before the reply");
    let compatibility = event_rx
        .try_recv()
        .expect("active-view compatibility event was published before the reply");
    assert_eq!(
        scoped["params"]["event"],
        "viewer.viewports.navigation.changed"
    );
    assert_eq!(scoped["params"]["source"], "viewport:viewport-1");
    assert_eq!(scoped["params"]["sequence"], 1);
    assert_eq!(scoped["params"]["revision"], 1);
    assert_eq!(compatibility["params"]["event"], "viewer.camera.changed");
    assert_eq!(compatibility["params"]["source"], "viewer:active");
    assert_eq!(compatibility["params"]["sequence"], 2);
    assert_eq!(compatibility["params"]["revision"], 1);
}

#[test]
fn shell_events_report_property_and_revision_changes_for_remote_and_native_updates() {
    let channels = spawn_test_actor();
    let event_hub = EventHub::shared();
    let (event_tx, event_rx) = crossbeam_channel::bounded(8);
    event_hub.register("shell-observer".to_string(), event_tx);
    event_hub
        .subscribe("shell-observer", vec!["*".to_string()])
        .unwrap();
    let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));

    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "ui.shell.patch",
                json!({"visibility":{"builtin:project.top-bar":false}}),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "shell-client".to_string(),
            request_id: Some(json!("shell-patch")),
            event_hub: Arc::clone(&event_hub),
            task_registry: Arc::clone(&task_registry),
            task_id: None,
        })
        .unwrap();
    let reply = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("shell reply")
        .unwrap();
    assert_eq!(reply["change"]["changed"], true);
    let event = event_rx.try_recv().expect("shell event precedes reply");
    assert_eq!(event["params"]["event"], "ui.shell.changed");
    assert_eq!(event["params"]["source"], "application:shell");
    assert_eq!(
        event["params"]["data"]["result"]["change"]["changes"][0]["property"],
        "visibility"
    );

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();

    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "viewer.panels.set",
                json!({"left":false,"right":false}),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "native-ui".to_string(),
            request_id: Some(json!("native-panels")),
            event_hub,
            task_registry,
            task_id: None,
        })
        .unwrap();
    reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("panel reply")
        .unwrap();
    let panel_event = event_rx.try_recv().expect("primary panel event");
    let shell_event = event_rx.try_recv().expect("shell compatibility event");
    assert_eq!(panel_event["params"]["event"], "viewer.panels.changed");
    assert_eq!(shell_event["params"]["event"], "ui.shell.changed");
    assert_eq!(
        shell_event["params"]["data"]["change"]["operation"],
        "native_sync"
    );
    let changes = shell_event["params"]["data"]["change"]["changes"]
        .as_array()
        .unwrap();
    assert!(changes.len() >= 3);
    assert!(
        changes
            .iter()
            .all(|change| change["property"] == "visibility")
    );
    assert!(changes.iter().any(|change| {
        change["node_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("layout:single."))
    }));
}

#[test]
fn desired_layout_replacement_emits_one_atomic_shell_event() {
    let channels = spawn_test_actor();
    let event_hub = EventHub::shared();
    let (event_tx, event_rx) = crossbeam_channel::bounded(4);
    event_hub.register("layout-observer".to_string(), event_tx);
    event_hub
        .subscribe("layout-observer", vec!["ui.shell.changed".to_string()])
        .unwrap();
    let task_registry = TaskRegistry::shared(Arc::clone(&event_hub));
    let desired_tree = json!({
        "root_id":"layout:event.root",
        "nodes":[
            {"id":"layout:event.root","type":"application","children":["layout:event.workspace"]},
            {"id":"layout:event.workspace","type":"builtin_mount","mount":"builtin:project-workspace"}
        ]
    });
    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "ui.shell.replace_layout",
                json!({
                    "desired_tree":desired_tree,
                    "transaction_id":"layout-event-7"
                }),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "layout-client".to_string(),
            request_id: Some(json!("replace-layout")),
            event_hub: Arc::clone(&event_hub),
            task_registry,
            task_id: None,
        })
        .unwrap();
    let reply = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("layout reply")
        .unwrap();
    assert_eq!(reply["change"]["operation"], "replace_layout");
    assert_eq!(reply["change"]["transaction_id"], "layout-event-7");
    assert_eq!(reply["change"]["changes"][0]["property"], "layout");
    let event = event_rx.try_recv().expect("one layout event");
    assert_eq!(event["params"]["event"], "ui.shell.changed");
    assert_eq!(
        event["params"]["data"]["result"]["change"]["transaction_id"],
        "layout-event-7"
    );
    assert_eq!(
        event["params"]["data"]["result"]["layout"]["root_id"],
        "layout:event.root"
    );
    assert!(event_rx.try_recv().is_err());

    let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "ui.shell.patch_layout",
                json!({
                    "if_shell_revision":reply["revision"],
                    "active_region_id":"layout:event.root",
                    "focused_node_id":"layout:event.root",
                }),
            )
            .unwrap(),
            reply: reply_tx,
            session_id: "layout-client".to_string(),
            request_id: Some(json!("focus-layout")),
            event_hub,
            task_registry: TaskRegistry::shared(EventHub::shared()),
            task_id: None,
        })
        .unwrap();
    let focused = reply_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("focus reply")
        .unwrap();
    assert_eq!(focused["focused_node_id"], "layout:event.root");
    let event = event_rx.try_recv().expect("one focus event");
    let properties = event["params"]["data"]["result"]["change"]["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|change| change["property"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        properties,
        std::collections::BTreeSet::from(["active_region", "focus"])
    );
    assert!(event_rx.try_recv().is_err());
}

#[test]
fn actor_enforces_extension_shell_ownership_for_direct_commands() {
    let event_hub = EventHub::shared();
    let resources = ResourceRegistry::shared(Arc::clone(&event_hub));
    let ui_registry = UiRegistry::shared(Arc::clone(&event_hub));
    let channels = configure_test_actor(
        spawn_control_actor_with_services_and_ui(
            Arc::new(|| {}),
            resources,
            None,
            None,
            None,
            None,
            None,
            Some(Arc::clone(&ui_registry)),
        )
        .unwrap(),
    );
    let contribution = |extension_id: &str, session_id: &str| {
        ui_registry
            .register_extension(
                json!({
                    "id":extension_id,
                    "name":extension_id,
                    "version":"1",
                    "capabilities":["ui.panels"],
                    "disconnect_policy":"retain"
                }),
                session_id,
            )
            .unwrap();
        ui_registry
            .register_contribution(
                json!({
                    "extension_id":extension_id,
                    "contribution_id":format!("{extension_id}.panel"),
                    "root":{"id":"root","type":"panel","children":[]}
                }),
                session_id,
            )
            .unwrap()
    };
    let alpha = contribution("org.example.actor-alpha", "alpha-session");
    let beta = contribution("org.example.actor-beta", "beta-session");
    for session in ["alpha-session", "beta-session"] {
        ui_registry.set_session_capabilities(session, &["ui.shell.extension_place".to_string()]);
    }
    ui_registry.set_session_capabilities(
        "application-controller",
        &["ui.shell.application_control".to_string()],
    );
    let tasks = TaskRegistry::shared(Arc::clone(&event_hub));
    let send = |method: &str, params: Value, session_id: &str| {
        let (reply, result) = crossbeam_channel::bounded(1);
        channels
            .request_tx
            .send(OdonControlRequest {
                command: ControlCommand::decode(method, params).unwrap(),
                reply,
                session_id: session_id.to_string(),
                request_id: None,
                event_hub: Arc::clone(&event_hub),
                task_registry: Arc::clone(&tasks),
                task_id: None,
            })
            .unwrap();
        result.recv_timeout(Duration::from_secs(1)).unwrap()
    };
    let desired_tree = json!({
        "root_id":"layout:owned.root",
        "nodes":[
            {"id":"layout:owned.root","type":"application","children":["layout:owned.body"]},
            {"id":"layout:owned.body","type":"row","children":["layout:owned.workspace","layout:owned.alpha","layout:owned.beta"]},
            {"id":"layout:owned.workspace","type":"builtin_mount","mount":"builtin:project-workspace"},
            {"id":"layout:owned.alpha","type":"extension_mount","mount":alpha.shell_mount},
            {"id":"layout:owned.beta","type":"extension_mount","mount":beta.shell_mount}
        ]
    });
    send(
        "ui.shell.replace_layout",
        json!({"desired_tree":desired_tree}),
        "application-controller",
    )
    .expect("controller composes all registered contributions");

    let error = send(
        "ui.shell.patch_layout",
        json!({"visibility":{"layout:owned.beta":false}}),
        "alpha-session",
    )
    .expect_err("one extension cannot mutate another extension's node");
    assert_eq!(error.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        error.data.as_ref().unwrap()["owner"]["owner_id"],
        "org.example.actor-beta"
    );

    let snapshot = send("ui.shell.get", json!({}), "application-controller").unwrap();
    let beta_node = snapshot["layout"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == "layout:owned.beta")
        .unwrap();
    assert_eq!(beta_node["visible"], true);
    assert_eq!(beta_node["ownership"]["scope"], "extension");
    assert_eq!(beta_node["ownership"]["owner_id"], "org.example.actor-beta");

    send(
        "ui.shell.profiles.save",
        json!({"name":"mixed","scope":"session"}),
        "application-controller",
    )
    .expect("save mixed-owner profile");
    send("ui.shell.reset", json!({}), "native-ui").expect("trusted native reset");
    let profile_error = send(
        "ui.shell.profiles.load",
        json!({"name":"mixed","scope":"session"}),
        "alpha-session",
    )
    .expect_err("profile import cannot smuggle a foreign extension node");
    assert_eq!(profile_error.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        profile_error.data.as_ref().unwrap()["owner"]["owner_id"],
        "org.example.actor-beta"
    );
}

#[test]
fn actor_enforces_platform_menu_capability_and_publishes_correlated_changes() {
    let event_hub = EventHub::shared();
    let resources = ResourceRegistry::shared(Arc::clone(&event_hub));
    let ui_registry = UiRegistry::shared(Arc::clone(&event_hub));
    let channels = configure_test_actor(
        spawn_control_actor_with_services_and_ui(
            Arc::new(|| {}),
            resources,
            None,
            None,
            None,
            None,
            None,
            Some(Arc::clone(&ui_registry)),
        )
        .unwrap(),
    );
    ui_registry.set_session_capabilities("menu-controller", &["ui.shell.chrome".to_string()]);
    let (event_tx, event_rx) = crossbeam_channel::bounded(8);
    event_hub.register("menu-observer".to_string(), event_tx);
    event_hub
        .subscribe(
            "menu-observer",
            vec![
                "ui.menus.changed".to_string(),
                "ui.palette.changed".to_string(),
            ],
        )
        .unwrap();
    let tasks = TaskRegistry::shared(Arc::clone(&event_hub));
    let send = |method: &str, params: Value, session_id: &str| {
        let (reply, result) = crossbeam_channel::bounded(1);
        channels
            .request_tx
            .send(OdonControlRequest {
                command: ControlCommand::decode(method, params).unwrap(),
                reply,
                session_id: session_id.to_string(),
                request_id: Some(json!(format!("{session_id}:{method}"))),
                event_hub: Arc::clone(&event_hub),
                task_registry: Arc::clone(&tasks),
                task_id: None,
            })
            .unwrap();
        result.recv_timeout(Duration::from_secs(1)).unwrap()
    };
    let initial = send("ui.menus.get", json!({}), "observer").unwrap();
    let mut menu = initial["menu"].clone();
    menu["children"].as_array_mut().unwrap().swap(2, 3);

    let denied = send("ui.menus.replace", json!({"menu":menu}), "unprivileged").unwrap_err();
    assert_eq!(denied.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        denied.data.unwrap()["required_capability"],
        "ui.shell.chrome"
    );

    let changed = send(
        "ui.menus.replace",
        json!({
            "if_command_revision":initial["revision"],
            "transaction_id":"menu-change-17",
            "menu":menu,
        }),
        "menu-controller",
    )
    .unwrap();
    assert_eq!(changed["change"]["transaction_id"], "menu-change-17");
    let event = event_rx
        .try_recv()
        .expect("menu change event precedes reply");
    assert_eq!(event["params"]["event"], "ui.menus.changed");
    assert_eq!(event["params"]["source"], "application:shell");
    assert_eq!(
        event["params"]["data"]["result"]["change"]["transaction_id"],
        "menu-change-17"
    );

    let initial_palette = send("ui.palette.get", json!({}), "observer").unwrap();
    let mut palette = initial_palette["palette"].clone();
    palette["title"] = json!("Review commands");
    let denied = send(
        "ui.palette.replace",
        json!({"palette":palette}),
        "unprivileged",
    )
    .unwrap_err();
    assert_eq!(denied.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        denied.data.unwrap()["required_capability"],
        "ui.shell.chrome"
    );
    let changed = send(
        "ui.palette.replace",
        json!({
            "if_command_revision":initial_palette["revision"],
            "transaction_id":"palette-change-18",
            "palette":palette,
        }),
        "menu-controller",
    )
    .unwrap();
    assert_eq!(changed["change"]["transaction_id"], "palette-change-18");
    let event = event_rx
        .try_recv()
        .expect("palette change event precedes reply");
    assert_eq!(event["params"]["event"], "ui.palette.changed");
    assert_eq!(
        event["params"]["data"]["result"]["palette"]["title"],
        "Review commands"
    );
}

#[test]
fn actor_owns_extension_command_registration_invocation_and_disconnect_cleanup() {
    let event_hub = EventHub::shared();
    let resources = ResourceRegistry::shared(Arc::clone(&event_hub));
    let ui_registry = UiRegistry::shared(Arc::clone(&event_hub));
    let channels = configure_test_actor(
        spawn_control_actor_with_services_and_ui(
            Arc::new(|| {}),
            resources,
            None,
            None,
            None,
            None,
            None,
            Some(Arc::clone(&ui_registry)),
        )
        .unwrap(),
    );
    ui_registry
        .register_extension(
            json!({
                "id":"org.example.commands",
                "name":"Command extension",
                "version":"1.0.0",
                "capabilities":["ui.actions"],
                "disconnect_policy":"retain",
            }),
            "extension-session",
        )
        .unwrap();
    ui_registry.set_session_capabilities("extension-session", &["ui.shell.shortcuts".to_string()]);
    let (event_tx, event_rx) = crossbeam_channel::bounded(16);
    event_hub.register("command-observer".to_string(), event_tx);
    event_hub
        .subscribe("command-observer", vec!["*".to_string()])
        .unwrap();
    let tasks = TaskRegistry::shared(Arc::clone(&event_hub));
    let send = |method: &str, params: Value, session_id: &str| {
        let (reply, result) = crossbeam_channel::bounded(1);
        channels
            .request_tx
            .send(OdonControlRequest {
                command: ControlCommand::decode(method, params).unwrap(),
                reply,
                session_id: session_id.to_string(),
                request_id: Some(json!(format!("{session_id}:{method}"))),
                event_hub: Arc::clone(&event_hub),
                task_registry: Arc::clone(&tasks),
                task_id: None,
            })
            .unwrap();
        result.recv_timeout(Duration::from_secs(1)).unwrap()
    };

    let denied = send(
        "ui.commands.register",
        json!({
            "extension_id":"org.example.commands",
            "command":{
                "id":"measure","title":"Measure","description":"Measure cells.","event":"measure"
            }
        }),
        "wrong-session",
    )
    .unwrap_err();
    assert_eq!(denied.kind, ControlErrorKind::PermissionDenied);

    let registered = send(
        "ui.commands.register",
        json!({
            "extension_id":"org.example.commands",
            "transaction_id":"command-register-1",
            "command":{
                "id":"measure",
                "title":"Measure",
                "description":"Measure cells.",
                "event":"measure",
                "modes":["project","single","mosaic"],
                "shortcut":{"key":"m","modifiers":["primary","shift"]},
                "predicates":{"visible":{
                    "type":"capability",
                    "capability":"viewer.read",
                    "reason":"Viewer access is required."
                }}
            }
        }),
        "extension-session",
    )
    .unwrap();
    let command_id = "extension:org.example.commands/measure";
    assert_eq!(registered["command"]["id"], command_id);
    let changed = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(changed["params"]["event"], "ui.commands.changed");

    ui_registry.set_session_capabilities("command-reader", &["ui.shell.read".to_string()]);
    let hidden = send("ui.commands.list", json!({}), "command-reader").unwrap();
    let hidden = hidden["commands"]
        .as_array()
        .unwrap()
        .iter()
        .find(|command| command["id"] == command_id)
        .unwrap();
    assert_eq!(hidden["state"]["visible"], false);
    assert_eq!(
        hidden["state"]["missing_capabilities"],
        json!(["viewer.read"])
    );
    ui_registry.set_session_capabilities(
        "command-reader",
        &["ui.shell.read".to_string(), "viewer.read".to_string()],
    );
    let visible = send("ui.commands.list", json!({}), "command-reader").unwrap();
    let visible = visible["commands"]
        .as_array()
        .unwrap()
        .iter()
        .find(|command| command["id"] == command_id)
        .unwrap();
    assert_eq!(visible["state"]["visible"], true);
    assert_eq!(visible["state"]["enabled"], true);

    let invoked = send(
        "ui.commands.execute",
        json!({"command_id":command_id}),
        "native-ui",
    )
    .unwrap();
    assert_eq!(invoked["dispatched"], true);
    let invocation = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        invocation["params"]["event"],
        "ui.extension:org.example.commands.measure"
    );
    assert_eq!(invocation["params"]["data"]["command_id"], command_id);
    let executed = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(executed["params"]["event"], "ui.commands.executed");
    assert_eq!(executed["params"]["data"]["handler_type"], "event");

    ui_registry.set_session_capabilities("command-client", &["ui.shell.read".to_string()]);
    let denied = send(
        "ui.commands.execute",
        json!({"command_id":"app.settings.open"}),
        "command-client",
    )
    .unwrap_err();
    assert_eq!(denied.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        denied.data.unwrap()["required_capability"],
        "ui.shell.application_control"
    );

    ui_registry.set_session_capabilities(
        "command-client",
        &["ui.shell.application_control".to_string()],
    );
    let native = send(
        "ui.commands.execute",
        json!({"command_id":"app.settings.open"}),
        "command-client",
    )
    .unwrap();
    assert_eq!(native["handler_type"], "native");
    assert_eq!(
        channels
            .platform_effect_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        PlatformEffect::InvokeNativeCommand {
            command_id: "app.settings.open".to_string(),
            action: "settings".to_string(),
            checked: None,
        }
    );
    let executed = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(executed["params"]["event"], "ui.commands.executed");
    assert_eq!(executed["params"]["data"]["handler_type"], "native");

    let control = send(
        "ui.commands.execute",
        json!({"command_id":"app.shell.recover"}),
        "command-client",
    )
    .unwrap();
    assert_eq!(control["handler_type"], "control");
    assert_eq!(
        channels
            .platform_effect_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap(),
        PlatformEffect::InvokeControlCommand {
            command_id: "app.shell.recover".to_string(),
            method: "ui.shell.recover".to_string(),
            params: json!({}),
        }
    );
    let executed = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(executed["params"]["event"], "ui.commands.executed");
    assert_eq!(executed["params"]["data"]["handler_type"], "control");

    for (command_id, expected_effect) in [
        (
            "app.window.close",
            PlatformEffect::InvokeNativeCommand {
                command_id: "app.window.close".to_string(),
                action: "close_window".to_string(),
                checked: None,
            },
        ),
        (
            "app.lifecycle.quit",
            PlatformEffect::InvokeNativeCommand {
                command_id: "app.lifecycle.quit".to_string(),
                action: "quit".to_string(),
                checked: None,
            },
        ),
        (
            "app.shell.recover",
            PlatformEffect::InvokeControlCommand {
                command_id: "app.shell.recover".to_string(),
                method: "ui.shell.recover".to_string(),
                params: json!({}),
            },
        ),
    ] {
        let direct = send(
            "ui.commands.execute",
            json!({"command_id":command_id}),
            "command-client",
        )
        .unwrap();
        assert_eq!(
            channels
                .platform_effect_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            expected_effect
        );
        let direct_event = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(direct_event["params"]["event"], "ui.commands.executed");

        let presentation = send(
            "ui.commands.execute",
            json!({"command_id":command_id}),
            "native-ui",
        )
        .unwrap();
        assert_eq!(presentation, direct);
        assert_eq!(
            channels
                .platform_effect_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap(),
            expected_effect
        );
        let presentation_event = event_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(
            presentation_event["params"]["event"],
            "ui.commands.executed"
        );
        assert_eq!(
            presentation_event["params"]["data"],
            direct_event["params"]["data"]
        );
    }

    let cleaned = send(
        "ui.commands.cleanup_extensions",
        json!({"extensions":[{
            "extension_id":"org.example.commands",
            "disconnect_policy":"retain"
        }]}),
        "native-ui",
    )
    .unwrap();
    assert_eq!(cleaned["changed"], true);
    assert_eq!(
        send(
            "ui.commands.execute",
            json!({"command_id":command_id}),
            "native-ui",
        )
        .unwrap_err()
        .kind,
        ControlErrorKind::NotReady
    );
}
