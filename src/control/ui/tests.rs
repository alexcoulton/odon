use super::*;
use crate::control::Ownership;
use eframe::egui;
use std::collections::BTreeMap;

#[test]
fn extension_trees_validate_and_patch_atomically() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id": "org.example.test", "name": "Test", "version": "1.0",
                "capabilities": ["ui.panels", "unknown"]
            }),
            "session",
        )
        .expect("extension");
    let contribution = registry
        .register_contribution(
            json!({
                "extension_id": "org.example.test", "location": "right.tabs",
                            "root": {"id": "root", "type": "panel", "children": [
                                {"id": "threshold", "type": "slider", "value": 0.5,
                                 "minimum": 0.0, "maximum": 1.0,
                                 "event_policy": {"type": "debounce", "milliseconds": 100}}
                ]}
            }),
            "session",
        )
        .expect("contribution");
    let patched = registry
        .patch_values(
            &contribution.contribution_id,
            &HashMap::from([("threshold".to_string(), json!(0.8))]),
            Some(contribution.revision),
            "session",
        )
        .expect("patch");
    assert_eq!(patched.root.children[0].value, 0.8);
    assert!(
        registry
            .patch_values(
                &patched.contribution_id,
                &HashMap::from([("missing".into(), json!(1))]),
                None,
                "session"
            )
            .is_err()
    );
}

#[test]
fn native_bindings_reflect_viewer_and_layer_state() {
    let mut root: Component = serde_json::from_value(json!({
        "id": "root", "type": "column", "children": [
            {"id": "opacity", "type": "slider", "minimum": 0.0, "maximum": 1.0,
             "action": {"type": "bind", "target": "viewer.layers",
                        "layer_id": "layer:test", "property": "opacity"}},
            {"id": "channel", "type": "select", "options": ["DAPI", "CD3"],
             "action": {"type": "bind", "target": "viewer.channels", "property": "active"}}
        ]
    }))
    .expect("component");
    let layers = vec![super::super::LayerSnapshot {
        layer_id: "layer:test".into(),
        name: "Test".into(),
        kind: "labels".into(),
        data_resource_id: "resource:test".into(),
        visible: true,
        opacity: 0.4,
        ownership: Ownership::Session,
        owner_session_id: "session".into(),
        style: BTreeMap::new(),
        provenance: BTreeMap::new(),
        order: 0,
        revision: 1,
    }];
    sync_component_binding(
        &mut root,
        &json!({"channels": {"channels": [
            {"name": "DAPI", "selected": false},
            {"name": "CD3", "selected": true}
        ]}}),
        &layers,
    );
    assert_eq!(root.children[0].value, 0.4);
    assert_eq!(root.children[1].value, "CD3");
}

#[test]
fn command_state_bindings_reconcile_visibility_and_enablement_without_python() {
    let mut root: Component = serde_json::from_value(json!({
        "id":"root",
        "type":"column",
        "children":[{
            "id":"action",
            "type":"button",
            "label":"Run",
            "state_bindings":{
                "visible":{
                    "type":"command_state",
                    "command_id":"extension:org.example.binding/run",
                    "state":"visible"
                },
                "enabled":{
                    "type":"command_state",
                    "command_id":"extension:org.example.binding/run",
                    "state":"enabled"
                }
            }
        }]
    }))
    .expect("component");
    validate_tree(&root).expect("bounded state bindings are valid");

    let projection = |visible: bool, enabled: bool| {
        json!({"shell":{"_command_surface":{"commands":[{
            "id":"extension:org.example.binding/run",
            "state":{"visible":visible,"enabled":enabled,"checked":null}
        }]}}})
    };
    sync_component_binding(&mut root, &projection(false, false), &[]);
    assert!(!root.children[0].visible);
    assert!(!root.children[0].enabled);

    sync_component_binding(&mut root, &projection(true, true), &[]);
    assert!(root.children[0].visible);
    assert!(root.children[0].enabled);

    sync_component_binding(&mut root, &json!({"shell":{}}), &[]);
    assert!(!root.children[0].visible);
    assert!(!root.children[0].enabled);
}

#[test]
fn component_state_bindings_reject_unbounded_or_unknown_expressions() {
    for state_bindings in [
        json!({"opacity":{
            "type":"command_state","command_id":"project.save","state":"enabled"
        }}),
        json!({"visible":{
            "type":"python","callback":"is_visible"
        }}),
        json!({"enabled":{
            "type":"command_state","command_id":"project.save","state":"private"
        }}),
    ] {
        let component: Component = serde_json::from_value(json!({
            "id":"action",
            "type":"button",
            "state_bindings":state_bindings,
        }))
        .expect("component shape");
        assert!(validate_tree(&component).is_err());
    }
}

#[test]
fn high_frequency_component_interactions_coalesce_per_component() {
    let registry = UiRegistry::shared(EventHub::shared());
    let now = std::time::Instant::now();
    let interactions = (0..1_000)
        .map(|value| Interaction {
            extension_id: "org.example.rate".to_string(),
            owner_session_id: "session".to_string(),
            component_id: "threshold".to_string(),
            kind: "change".to_string(),
            value: json!(value),
            action: Some(json!({"type":"emit","event":"threshold"})),
            event_policy: Some(json!({"type":"debounce","milliseconds":1_000})),
            occurred_at: now,
        })
        .collect();
    let context = egui::Context::default();
    registry.commit_interactions(&context, interactions);

    let deferred = registry
        .deferred_interactions
        .lock()
        .expect("deferred interactions");
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred["org.example.rate:threshold"].value, 999);
    assert!(registry.last_emitted.lock().unwrap().is_empty());
}

#[test]
fn disconnected_extensions_can_be_reclaimed_by_a_new_session() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id": "org.example.reconnect", "name": "Reconnect", "version": "1",
                "capabilities": ["ui.panels"], "disconnect_policy": "disable"
            }),
            "first",
        )
        .expect("first registration");
    let contribution = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.reconnect",
                "contribution_id":"panel",
                "root":{"id":"root","type":"panel","children":[]}
            }),
            "first",
        )
        .expect("retained contribution");
    registry.cleanup_session("first");
    let extension = registry
        .register_extension(
            json!({
                "id": "org.example.reconnect", "name": "Reconnect", "version": "1",
                "capabilities": ["ui.panels"], "disconnect_policy": "disable"
            }),
            "second",
        )
        .expect("replacement registration");
    assert!(extension.connected);
    assert_eq!(extension.owner_session_id, "second");
    let contributions = registry.list_contributions();
    assert_eq!(contributions.len(), 1);
    assert_eq!(
        contributions[0].contribution_id,
        contribution.contribution_id
    );
    assert_eq!(
        contributions[0].shell_mount,
        "extension:org.example.reconnect/panel"
    );
    assert_eq!(contributions[0].ownership["owner_session_id"], "second");
    registry
        .validate_shell_layout_access(
            &json!({
                "root_id":"root",
                "nodes":[
                    {"id":"root","type":"application","children":["body"]},
                    {"id":"body","type":"row","children":["canvas","extension"]},
                    {"id":"canvas","type":"canvas_slot","mount":"builtin:viewer-canvas"},
                    {"id":"extension","type":"extension_mount","mount":contributions[0].shell_mount}
                ]
            }),
            "second",
        )
        .expect("new owner can place retained mount");
}

#[test]
fn retained_mounts_report_not_ready_disconnected_incompatible_and_missing_states() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id":"org.example.fixture",
                "name":"Readiness",
                "version":"1",
                "capabilities":["ui.panels"],
                "disconnect_policy":"retain",
                "ready":false,
                "readiness_reason":"warming model"
            }),
            "first",
        )
        .unwrap();
    let contribution = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.fixture",
                "contribution_id":"panel",
                "root":{"id":"root","type":"panel","children":[]}
            }),
            "first",
        )
        .unwrap();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/shell-layouts/v0-project-missing-extension.json"
    ))
    .expect("checked-in v0 shell fixture is valid JSON");
    let persisted_extension = fixture["desired_tree"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["mount"] == contribution.shell_mount)
        .expect("fixture retains the registered extension mount")
        .clone();
    let snapshot = || {
        json!({
            "layout":{"nodes":[persisted_extension.clone()]}
        })
    };
    let mut not_ready = snapshot();
    registry.annotate_shell_snapshot_ownership(&mut not_ready);
    assert_eq!(
        not_ready["layout"]["nodes"][0]["readiness"]["state"],
        "not_ready"
    );
    assert_eq!(
        not_ready["layout"]["nodes"][0]["readiness"]["reason"],
        "warming model"
    );
    registry
        .set_extension_readiness(
            json!({"extension_id":"org.example.fixture","ready":true}),
            "first",
        )
        .unwrap();
    let mut ready = snapshot();
    registry.annotate_shell_snapshot_ownership(&mut ready);
    assert_eq!(ready["layout"]["nodes"][0]["readiness"]["state"], "ready");

    registry.cleanup_session("first");
    let mut disconnected = snapshot();
    registry.annotate_shell_snapshot_ownership(&mut disconnected);
    assert_eq!(
        disconnected["layout"]["nodes"][0]["readiness"]["state"],
        "disconnected"
    );

    registry
        .register_extension(
            json!({
                "id":"org.example.fixture",
                "name":"Readiness",
                "version":"2",
                "capabilities":["ui.panels"],
                "disconnect_policy":"retain"
            }),
            "second",
        )
        .unwrap();
    let mut incompatible = snapshot();
    registry.annotate_shell_snapshot_ownership(&mut incompatible);
    assert_eq!(
        incompatible["layout"]["nodes"][0]["readiness"]["state"],
        "incompatible"
    );
    assert_eq!(
        incompatible["layout"]["nodes"][0]["readiness"]["expected_extension_version"],
        "1"
    );
    assert_eq!(
        incompatible["layout"]["nodes"][0]["readiness"]["current_extension_version"],
        "2"
    );

    registry
        .remove_contribution(&contribution.contribution_id, "second")
        .unwrap();
    let mut missing = snapshot();
    registry.annotate_shell_snapshot_ownership(&mut missing);
    assert_eq!(
        missing["layout"]["nodes"][0]["readiness"]["state"],
        "missing"
    );
}

#[test]
fn contributions_publish_catalogue_descriptors_and_render_by_shell_mount() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id":"org.example.inline",
                "name":"Inline",
                "version":"1",
                "capabilities":["ui.panels"],
                "disconnect_policy":"retain"
            }),
            "session",
        )
        .expect("extension");
    let contribution = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.inline",
                "contribution_id":"analysis",
                "root":{
                    "id":"root",
                    "type":"panel",
                    "children":[{"id":"message","type":"text","value":"Ready"}]
                }
            }),
            "session",
        )
        .expect("contribution");
    assert_eq!(
        contribution.shell_mount,
        "extension:org.example.inline/analysis"
    );
    let descriptors = registry.shell_component_descriptors(Some("single"));
    assert_eq!(descriptors.len(), 1);
    assert_eq!(descriptors[0]["id"], contribution.shell_mount);
    assert_eq!(descriptors[0]["readiness"], json!(["extension_ready"]));
    assert_eq!(descriptors[0]["ownership"]["scope"], "extension");
    assert_eq!(descriptors[0]["ownership"]["owner_session_id"], "session");

    let ctx = egui::Context::default();
    let mut rendered = false;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            rendered = registry.render_shell_mount(ui, &contribution.shell_mount);
        });
    });
    assert!(rendered);

    registry.cleanup_session("session");
    let descriptors = registry.shell_component_descriptors(Some("single"));
    assert_eq!(
        descriptors[0]["readiness"],
        json!(["extension_disconnected"])
    );
}

#[test]
fn shell_mutations_enforce_extension_node_ownership_and_report_the_owner() {
    let registry = UiRegistry::shared(EventHub::shared());
    let register = |id: &str, session: &str| {
        registry
            .register_extension(
                json!({
                    "id":id,
                    "name":id,
                    "version":"1",
                    "capabilities":["ui.panels"],
                    "disconnect_policy":"retain"
                }),
                session,
            )
            .unwrap();
        registry
            .register_contribution(
                json!({
                    "extension_id":id,
                    "contribution_id":format!("{id}.panel"),
                    "root":{"id":"root","type":"panel","children":[]}
                }),
                session,
            )
            .unwrap()
    };
    let alpha = register("org.example.alpha", "alpha-session");
    let beta = register("org.example.beta", "beta-session");
    for session in ["alpha-session", "beta-session"] {
        registry.set_session_capabilities(session, &["ui.shell.extension_place".to_string()]);
    }
    let current = json!({
        "layout":{
            "root_id":"root",
            "nodes":[
                {"id":"root","type":"application","children":["alpha","beta"]},
                {"id":"alpha","type":"extension_mount","mount":alpha.shell_mount},
                {"id":"beta","type":"extension_mount","mount":beta.shell_mount}
            ]
        }
    });

    registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"alpha":false}}),
            &current,
            None,
            "alpha-session",
        )
        .expect("an extension can mutate its own node");
    let error = registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"beta":false}}),
            &current,
            None,
            "alpha-session",
        )
        .expect_err("foreign extension node must be protected");
    assert_eq!(error.kind, ControlErrorKind::PermissionDenied);
    let data = error.data.expect("ownership error data");
    assert_eq!(data["node_id"], "beta");
    assert_eq!(data["owner"]["owner_id"], "org.example.beta");
    assert_eq!(data["required_capability"], "ui.shell.application_control");
    let application_error = registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"root":false}}),
            &current,
            None,
            "alpha-session",
        )
        .expect_err("an extension cannot mutate application-owned nodes");
    assert_eq!(application_error.kind, ControlErrorKind::PermissionDenied);
    let application_data = application_error.data.expect("application ownership data");
    assert_eq!(application_data["owner"]["scope"], "application");
    assert_eq!(application_data["required_capability"], "ui.shell.compose");
    assert!(
        registry
            .validate_shell_mutation_access(
                "ui.shell.reset",
                &json!({}),
                &current,
                None,
                "alpha-session",
            )
            .is_err()
    );
    let ungranted = registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"root":false}}),
            &current,
            None,
            "application-controller",
        )
        .expect_err("authentication alone does not grant application control");
    assert_eq!(ungranted.kind, ControlErrorKind::PermissionDenied);
    registry.set_session_capabilities(
        "application-controller",
        &["ui.shell.application_control".to_string()],
    );
    registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"beta":false}}),
            &current,
            None,
            "application-controller",
        )
        .expect("an explicitly granted application controller can compose the full shell");

    let mut annotated = current;
    registry.annotate_shell_snapshot_ownership(&mut annotated);
    assert_eq!(
        annotated["layout"]["nodes"][1]["ownership"]["owner_session_id"],
        "alpha-session"
    );
    assert_eq!(
        annotated["layout"]["nodes"][2]["ownership"]["owner_id"],
        "org.example.beta"
    );
}

#[test]
fn extension_layout_templates_are_owned_normalized_and_retained_for_reconnect() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry.set_session_capabilities("first", &["ui.shell.extension_place".to_string()]);
    registry
        .register_extension(
            json!({
                "id":"org.example.layouts",
                "name":"Layouts",
                "version":"1",
                "capabilities":["ui.panels"],
                "disconnect_policy":"retain"
            }),
            "first",
        )
        .expect("extension");
    let contribution = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.layouts",
                "contribution_id":"review",
                "root":{"id":"root","type":"panel","children":[]}
            }),
            "first",
        )
        .expect("contribution");
    let desired_tree = json!({
        "root_id":"root",
        "nodes":[
            {"id":"root","type":"application","children":["body"]},
            {"id":"body","type":"row","parent_id":"root","children":["canvas","review"]},
            {"id":"canvas","type":"canvas_slot","parent_id":"body","mount":"builtin:viewer-canvas"},
            {"id":"review","type":"extension_mount","parent_id":"body","mount":contribution.shell_mount}
        ]
    });
    let registered = registry
        .register_extension_layout(
            json!({
                "extension_id":"org.example.layouts",
                "name":"Review",
                "document":{"schema_version":0,"mode":"single","desired_tree":desired_tree}
            }),
            "first",
        )
        .expect("layout template");
    assert_eq!(registered.document["format"], "odon.shell-layout");
    assert_eq!(registered.document["schema_version"], 1);
    assert_eq!(registered.document["mode"], "single");
    assert_eq!(registered.ownership["owner_session_id"], "first");
    assert!(registered.document.get("desired_tree").is_none());
    assert!(matches!(
        registry.list_extension_layouts("org.example.layouts", "other"),
        Err(error) if error.kind == ControlErrorKind::PermissionDenied
    ));

    registry.cleanup_session("first");
    registry
        .register_extension(
            json!({
                "id":"org.example.layouts",
                "name":"Layouts",
                "version":"1",
                "capabilities":["ui.panels"],
                "disconnect_policy":"retain"
            }),
            "second",
        )
        .expect("reconnected extension");
    let retained = registry
        .list_extension_layouts("org.example.layouts", "second")
        .expect("retained templates");
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].name, "Review");
    assert_eq!(retained[0].ownership["owner_session_id"], "second");
    registry
        .remove_extension_layout("org.example.layouts", "Review", "second")
        .expect("remove template");
    assert!(
        registry
            .list_extension_layouts("org.example.layouts", "second")
            .expect("templates")
            .is_empty()
    );
}

#[test]
fn extension_layout_templates_validate_capability_and_disconnect_policy() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry.set_session_capabilities("session", &["ui.shell.extension_place".to_string()]);
    registry
        .register_extension(
            json!({
                "id":"org.example.readonly",
                "name":"Read only",
                "version":"1",
                "capabilities":[],
                "disconnect_policy":"remove"
            }),
            "session",
        )
        .expect("extension");
    let document = json!({
        "format":"odon.shell-layout",
        "schema_version":1,
        "mode":"single",
        "layout":{
            "root_id":"root",
            "nodes":[
                {"id":"root","type":"application","children":["canvas"]},
                {"id":"canvas","type":"canvas_slot","parent_id":"root","mount":"builtin:viewer-canvas"}
            ]
        }
    });
    assert!(matches!(
        registry.register_extension_layout(
            json!({
                "extension_id":"org.example.readonly",
                "name":"Default",
                "document":document.clone()
            }),
            "session"
        ),
        Err(error) if error.kind == ControlErrorKind::PermissionDenied
    ));

    registry
        .remove_extension("org.example.readonly", "session")
        .unwrap();
    registry
        .register_extension(
            json!({
                "id":"org.example.removed",
                "name":"Removed",
                "version":"1",
                "capabilities":["ui.panels"],
                "disconnect_policy":"remove"
            }),
            "session",
        )
        .expect("extension");
    registry
        .register_extension_layout(
            json!({
                "extension_id":"org.example.removed",
                "name":"Default",
                "document":document
            }),
            "session",
        )
        .expect("template");
    registry.cleanup_session("session");
    assert!(matches!(
        registry.list_extension_layouts("org.example.removed", "session"),
        Err(error) if error.kind == ControlErrorKind::ResourceNotFound
    ));
}

#[test]
fn extension_layout_templates_reject_invalid_documents_and_enforce_quota() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry.set_session_capabilities("session", &["ui.shell.extension_place".to_string()]);
    registry
        .register_extension(
            json!({
                "id":"org.example.quota",
                "name":"Quota",
                "version":"1",
                "capabilities":["ui.panels"]
            }),
            "session",
        )
        .expect("extension");
    assert!(matches!(
        registry.register_extension_layout(
            json!({
                "extension_id":"org.example.quota",
                "name":"Invalid",
                "document":{"format":"odon.shell-layout","schema_version":1}
            }),
            "session"
        ),
        Err(error) if error.kind == ControlErrorKind::InvalidParams
    ));
    assert!(
        registry
            .list_extension_layouts("org.example.quota", "session")
            .expect("layouts")
            .is_empty()
    );

    let document = json!({
        "format":"odon.shell-layout",
        "schema_version":1,
        "mode":"single",
        "layout":{
            "root_id":"root",
            "nodes":[
                {"id":"root","type":"application","children":["canvas"]},
                {"id":"canvas","type":"canvas_slot","parent_id":"root","mount":"builtin:viewer-canvas"}
            ]
        }
    });
    for index in 0..64 {
        registry
            .register_extension_layout(
                json!({
                    "extension_id":"org.example.quota",
                    "name":format!("Layout {index}"),
                    "document":document.clone()
                }),
                "session",
            )
            .expect("template within quota");
    }
    assert!(matches!(
        registry.register_extension_layout(
            json!({
                "extension_id":"org.example.quota",
                "name":"Overflow",
                "document":document
            }),
            "session"
        ),
        Err(error) if error.kind == ControlErrorKind::ResourceLimit
    ));
}

#[test]
fn default_hosts_only_render_contributions_not_explicitly_mounted_in_the_tree() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id":"org.example.hosts",
                "name":"Hosts",
                "version":"1",
                "capabilities":["ui.panels"]
            }),
            "session",
        )
        .unwrap();
    let first = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.hosts",
                "contribution_id":"first",
                "location":"right.tabs",
                "root":{"id":"first","type":"text","value":"First"}
            }),
            "session",
        )
        .unwrap();
    let second = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.hosts",
                "contribution_id":"second",
                "location":"right.tabs",
                "root":{"id":"second","type":"text","value":"Second"}
            }),
            "session",
        )
        .unwrap();
    let shell = |mounts: &[&str]| {
        json!({"layout":{"nodes":mounts.iter().enumerate().map(|(index, mount)| json!({
            "id":format!("extension-{index}"),
            "type":"extension_mount",
            "mount":mount,
        })).collect::<Vec<_>>()}})
    };
    let host = "builtin:extension-host.right-tabs";
    assert!(registry.shell_mount_available(host, &shell(&[&first.shell_mount])));
    assert!(
        !registry.shell_mount_available(host, &shell(&[&first.shell_mount, &second.shell_mount]))
    );

    let ctx = egui::Context::default();
    let mut rendered = false;
    let current = shell(&[&first.shell_mount]);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            rendered = registry.render_shell_mount_in_layout(ui, host, Some(&current));
        });
    });
    assert!(rendered);
}

#[test]
fn contribution_location_contracts_allow_chrome_mounts_only_in_matching_containers() {
    let registry = UiRegistry::shared(EventHub::shared());
    registry
        .register_extension(
            json!({
                "id":"org.example.chrome",
                "name":"Chrome",
                "version":"1",
                "capabilities":["ui.panels"]
            }),
            "session",
        )
        .unwrap();
    let action = registry
        .register_contribution(
            json!({
                "extension_id":"org.example.chrome",
                "contribution_id":"action",
                "location":"top_bar.actions",
                "root":{"id":"action","type":"button","label":"Run"}
            }),
            "session",
        )
        .unwrap();
    let valid = json!({
        "root_id":"root",
        "nodes":[
            {"id":"root","type":"application","children":["top","canvas"]},
            {"id":"top","type":"toolbar","children":["action"]},
            {"id":"action","type":"extension_mount","mount":action.shell_mount},
            {"id":"canvas","type":"canvas_slot","mount":"builtin:viewer-canvas"}
        ]
    });
    registry
        .validate_shell_layout_access(&valid, "session")
        .expect("a top-bar action can be mounted in a toolbar");

    let descriptors = registry.shell_component_descriptors(Some("single"));
    assert_eq!(descriptors[0]["kind"], "toolbar");
    assert_eq!(
        descriptors[0]["legal_parent_types"],
        json!(["toolbar", "row", "column", "panel"])
    );

    let mut invalid = valid;
    invalid["nodes"][1]["type"] = json!("tabs");
    assert!(matches!(
        registry.validate_shell_layout_access(&invalid, "session"),
        Err(error) if error.kind == ControlErrorKind::InvalidParams
    ));
}

#[test]
fn application_owned_extension_chrome_hosts_require_the_chrome_grant() {
    let registry = UiRegistry::shared(EventHub::shared());
    let current = json!({"layout":{"nodes":[
        {"id":"root","type":"application","mount":null},
        {"id":"status","type":"builtin_mount","mount":"builtin:extension-host.status-bar"}
    ]}});
    registry.set_session_capabilities("controller", &["ui.shell.compose".to_string()]);
    let error = registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"status":false}}),
            &current,
            None,
            "controller",
        )
        .expect_err("composition alone must not mutate application chrome");
    assert_eq!(error.kind, ControlErrorKind::PermissionDenied);
    assert_eq!(
        error.data.unwrap()["required_capability"],
        "ui.shell.chrome"
    );

    registry.set_session_capabilities("controller", &["ui.shell.chrome".to_string()]);
    registry
        .validate_shell_mutation_access(
            "ui.shell.patch_layout",
            &json!({"visibility":{"status":false}}),
            &current,
            None,
            "controller",
        )
        .expect("the explicit chrome grant can mutate the host");
}
