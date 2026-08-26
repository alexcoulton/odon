use super::*;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr")
}

fn shell_layout_fixture(name: &str) -> Value {
    let source = match name {
        "v0-project-missing-extension" => {
            include_str!("../../../tests/fixtures/shell-layouts/v0-project-missing-extension.json")
        }
        "v1-project" => {
            include_str!("../../../tests/fixtures/shell-layouts/v1-project.json")
        }
        "v1-single-startup" => {
            include_str!("../../../tests/fixtures/shell-layouts/v1-single-startup.json")
        }
        "v1-corrupt-tree" => {
            include_str!("../../../tests/fixtures/shell-layouts/v1-corrupt-tree.json")
        }
        "v99-future" => {
            include_str!("../../../tests/fixtures/shell-layouts/v99-future.json")
        }
        _ => panic!("unknown shell-layout fixture {name}"),
    };
    serde_json::from_str(source).expect("checked-in shell-layout fixtures must be valid JSON")
}

#[test]
fn shell_commands_compose_the_active_application_without_a_renderer() {
    let mut model = AppModel::project();
    let initial = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(initial["schema_version"], 1);
    assert_eq!(initial["mode"], "project");
    assert_eq!(initial["layout"]["root_id"], "layout:project.root");

    let revision = initial["revision"].as_u64().unwrap();
    let patched = model
        .dispatch(
            "ui.shell.patch",
            &json!({
                "visibility":{"builtin:project.top-bar":false},
                "if_shell_revision":revision,
            }),
        )
        .unwrap()
        .unwrap();
    assert!(patched.present);
    assert_eq!(patched.response["change"]["operation"], "patch");
    assert_eq!(patched.response["change"]["changed"], true);
    assert_eq!(
        patched.response["change"]["changes"][0]["node_id"],
        "builtin:project.top-bar"
    );
    let top_bar = patched.response["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == "builtin:project.top-bar")
        .unwrap();
    assert_eq!(top_bar["visible"], false);
    assert!(patched.response["revision"].as_u64().unwrap() > revision);

    let inactive = model
        .dispatch("ui.shell.get", &json!({"mode":"single"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(inactive["mode"], "single");
    assert!(
        model
            .dispatch(
                "ui.shell.patch",
                &json!({"mode":"single","visibility":{"builtin:single.top-bar":false}}),
            )
            .unwrap()
            .is_err()
    );

    let reset = model
        .dispatch("ui.shell.reset", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let top_bar = reset["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == "builtin:project.top-bar")
        .unwrap();
    assert_eq!(top_bar["visible"], true);

    let revision = reset["revision"].as_u64().unwrap();
    let no_op = model
        .dispatch("ui.shell.patch", &json!({"if_shell_revision":revision}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(no_op["revision"], revision);
    assert_eq!(no_op["change"]["changed"], false);

    let schema = model
        .dispatch("ui.shell.describe_schema", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(schema["mutation_scope"], "active_mode_only");
    assert_eq!(schema["layout_limits"]["max_nodes"], 256);

    let components = model
        .dispatch("ui.shell.components.list", &json!({"mode":"single"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(components["mode"], "single");
    let canvas = components["components"]
        .as_array()
        .unwrap()
        .iter()
        .find(|component| component["id"] == "builtin:viewer-canvas")
        .unwrap();
    assert_eq!(canvas["singleton"], true);
    assert!(
        canvas["legal_parent_types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|parent| parent == "split")
    );
}

#[test]
fn command_descriptors_and_platform_menu_presentations_are_independently_actor_owned() {
    let mut model = AppModel::project();
    let commands = model
        .dispatch("ui.commands.list", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let menu = model
        .dispatch("ui.menus.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(commands["revision"], menu["revision"]);
    assert!(
        commands["commands"]
            .as_array()
            .unwrap()
            .iter()
            .any(|command| {
                command["id"] == "dataset.open.ome_zarr"
                    && command["handler"]["action"] == "open_ome_zarr"
            })
    );
    assert!(menu["menu"].get("commands").is_none());

    let mut reordered = menu["menu"].clone();
    reordered["children"].as_array_mut().unwrap().swap(1, 2);
    let changed = model
        .dispatch(
            "ui.menus.replace",
            &json!({
                "if_command_revision":menu["revision"],
                "transaction_id":"python-menu-layout",
                "menu":reordered,
            }),
        )
        .unwrap()
        .unwrap();
    assert!(changed.present);
    assert_eq!(changed.response["revision"], 2);
    assert_eq!(
        changed.response["change"]["transaction_id"],
        "python-menu-layout"
    );
    assert_eq!(changed.response["menu"]["children"][1]["id"], "menu:add");

    let stale = model
        .dispatch(
            "ui.menus.replace",
            &json!({
                "if_command_revision":menu["revision"],
                "menu":changed.response["menu"],
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(stale.kind, ControlErrorKind::Conflict);
    assert_eq!(stale.data.unwrap()["snapshot_method"], "ui.menus.get");
}

#[test]
fn shell_desired_layout_replacement_is_atomic_and_revision_guarded() {
    let mut model = AppModel::project();
    let before = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let revision = before["revision"].as_u64().unwrap();
    let desired_tree = json!({
        "root_id":"layout:review.root",
        "nodes":[
            {"id":"layout:review.root","type":"application","children":["layout:review.column"]},
            {"id":"layout:review.column","type":"column","children":["layout:review.workspace","layout:review.cards"]},
            {"id":"layout:review.workspace","type":"builtin_mount","mount":"builtin:project-workspace","size":{"flex":1.0}},
            {"id":"layout:review.cards","type":"extension_mount","mount":"extension:review.cards","visible":false}
        ]
    });
    let replaced = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({"desired_tree":desired_tree.clone(),"if_shell_revision":revision}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(replaced["layout"]["root_id"], "layout:review.root");
    assert_eq!(replaced["change"]["operation"], "replace_layout");
    assert_eq!(replaced["change"]["changes"][0]["property"], "layout");
    let layout_nodes = replaced["layout"]["nodes"].as_array().unwrap();
    let root_node = layout_nodes
        .iter()
        .find(|node| node["id"] == "layout:review.root")
        .unwrap();
    let extension_node = layout_nodes
        .iter()
        .find(|node| node["id"] == "layout:review.cards")
        .unwrap();
    assert_eq!(root_node["ownership"]["scope"], "application");
    assert_eq!(root_node["ownership"]["protected"], true);
    assert_eq!(extension_node["ownership"]["owner_id"], "review.cards");
    let replaced_revision = replaced["revision"].as_u64().unwrap();
    assert!(replaced_revision > revision);

    let no_op = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({"desired_tree":desired_tree.clone(),"if_shell_revision":replaced_revision}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(no_op["revision"], replaced_revision);
    assert_eq!(no_op["change"]["changed"], false);

    let conflict = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({"desired_tree":desired_tree,"if_shell_revision":revision}),
        )
        .unwrap()
        .unwrap_err();
    let conflict_data = conflict.data.expect("shell conflict details");
    assert_eq!(conflict_data["current_revision"], replaced_revision);
    assert_eq!(conflict_data["snapshot_method"], "ui.shell.get");
    assert_eq!(conflict_data["retry_strategy"], "refetch_merge_retry");

    let invalid = json!({
        "root_id":"layout:invalid.root",
        "nodes":[
            {"id":"layout:invalid.root","type":"application","children":["layout:invalid.layers"]},
            {"id":"layout:invalid.layers","type":"builtin_mount","mount":"builtin:layers"}
        ]
    });
    assert!(
        model
            .dispatch(
                "ui.shell.replace_layout",
                &json!({"desired_tree":invalid,"if_shell_revision":replaced_revision}),
            )
            .unwrap()
            .is_err()
    );
    let after_error = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(after_error["revision"], replaced_revision);
    assert_eq!(after_error["layout"]["root_id"], "layout:review.root");
}

#[test]
fn shell_visibility_bindings_follow_actor_evaluated_command_state() {
    let mut model = AppModel::project();
    let layout = |command_id: &str| {
        json!({
            "root_id":"layout:binding.root",
            "nodes":[
                {
                    "id":"layout:binding.root",
                    "type":"application",
                    "children":["layout:binding.workspace"]
                },
                {
                    "id":"layout:binding.workspace",
                    "type":"builtin_mount",
                    "mount":"builtin:project-workspace",
                    "state_bindings":{"visible":{
                        "type":"command_state",
                        "command_id":command_id,
                        "state":"enabled"
                    }}
                }
            ]
        })
    };
    let available = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({"desired_tree":layout("app.shell.recover")}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(available["layout"]["nodes"][1]["visible"], true);
    assert_eq!(
        available["layout"]["nodes"][1]["state_bindings"]["visible"]["command_id"],
        "app.shell.recover"
    );
    let exported = model
        .dispatch("ui.shell.export_layout", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        exported["layout"]["nodes"][1]["state_bindings"]["visible"]["command_id"],
        "app.shell.recover"
    );

    let missing = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({
                "desired_tree":layout("extension:org.example.missing/run"),
                "if_shell_revision":available["revision"],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(missing["layout"]["nodes"][1]["visible"], false);

    let invalid = layout("app.shell.recover");
    let mut invalid = invalid;
    invalid["nodes"][1]["state_bindings"] = json!({"enabled":{
        "type":"command_state","command_id":"app.shell.recover","state":"enabled"
    }});
    let error = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({"desired_tree":invalid,"if_shell_revision":missing["revision"]}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::InvalidParams);
    assert_eq!(
        model
            .dispatch("ui.shell.get", &json!({}))
            .unwrap()
            .unwrap()
            .response["revision"],
        missing["revision"]
    );
}

#[test]
fn shell_layout_documents_migrate_atomically_and_recover_safely() {
    let mut model = AppModel::project();
    let exported = model
        .dispatch("ui.shell.export_layout", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(exported["format"], "odon.shell-layout");
    assert_eq!(exported["schema_version"], 1);
    assert_eq!(exported["mode"], "project");

    let initial = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let revision = initial["revision"].as_u64().unwrap();
    let imported = model
        .dispatch(
            "ui.shell.import_layout",
            &json!({
                "if_shell_revision":revision,
                "document":shell_layout_fixture("v0-project-missing-extension")
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(imported["layout"]["root_id"], "layout:fixture.v0.root");
    assert_eq!(imported["import"]["source_schema_version"], 0);
    assert_eq!(imported["import"]["migrated"], true);
    assert_eq!(imported["change"]["operation"], "import_layout");

    let v0_revision = imported["revision"].as_u64().unwrap();
    let current = model
        .dispatch(
            "ui.shell.import_layout",
            &json!({
                "if_shell_revision":v0_revision,
                "document":shell_layout_fixture("v1-project")
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(current["layout"]["root_id"], "layout:fixture.v1.root");
    assert_eq!(current["import"]["source_schema_version"], 1);
    assert_eq!(current["import"]["migrated"], false);
    let stable_revision = current["revision"].as_u64().unwrap();

    let corrupt = model
        .dispatch(
            "ui.shell.import_layout",
            &json!({
                "if_shell_revision":stable_revision,
                "document":shell_layout_fixture("v1-corrupt-tree")
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(corrupt.kind, ControlErrorKind::InvalidParams);
    assert_eq!(
        model
            .dispatch("ui.shell.get", &json!({}))
            .unwrap()
            .unwrap()
            .response["revision"],
        stable_revision
    );

    let unsupported = model
        .dispatch(
            "ui.shell.import_layout",
            &json!({
                "if_shell_revision":stable_revision,
                "document":shell_layout_fixture("v99-future")
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(unsupported.kind, ControlErrorKind::Unsupported);
    assert_eq!(
        unsupported.data.unwrap()["recovery_method"],
        "ui.shell.recover"
    );
    assert_eq!(
        model
            .dispatch("ui.shell.get", &json!({}))
            .unwrap()
            .unwrap()
            .response["revision"],
        stable_revision
    );

    let recovered = model
        .dispatch(
            "ui.shell.recover",
            &json!({"if_shell_revision":stable_revision}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(recovered["recovery"]["protected"], true);
    assert_eq!(recovered["change"]["operation"], "recover");
    assert_eq!(recovered["layout"]["nodes"].as_array().unwrap().len(), 2);
    assert!(
        recovered["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node["mount"] == "builtin:project-workspace")
    );
}

#[test]
fn shell_layout_session_profiles_save_list_load_and_remove() {
    let mut model = AppModel::project();
    let custom = json!({
        "root_id":"layout:profile.root",
        "nodes":[
            {"id":"layout:profile.root","type":"application","children":["layout:profile.workspace"]},
            {"id":"layout:profile.workspace","type":"builtin_mount","mount":"builtin:project-workspace"}
        ]
    });
    model
        .dispatch("ui.shell.replace_layout", &json!({"desired_tree":custom}))
        .unwrap()
        .unwrap();
    let SettingsMutationOutcome::Immediate(saved) = model
        .prepare_shell_profile_save(&json!({"name":"Review","scope":"session"}))
        .unwrap()
    else {
        panic!("session profile save must be immediate");
    };
    assert_eq!(saved["persisted"], false);

    model
        .dispatch("ui.shell.reset", &json!({}))
        .unwrap()
        .unwrap();
    let listed = model
        .dispatch("ui.shell.profiles.list", &json!({"scope":"session"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(listed["profiles"][0]["name"], "Review");
    let loaded = model
        .dispatch(
            "ui.shell.profiles.load",
            &json!({"name":"Review","scope":"session"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(loaded["layout"]["root_id"], "layout:profile.root");
    assert_eq!(loaded["change"]["operation"], "load_profile");
    assert_eq!(loaded["profile"]["name"], "Review");

    let SettingsMutationOutcome::Immediate(removed) = model
        .prepare_shell_profile_remove(&json!({"name":"Review","scope":"session"}))
        .unwrap()
    else {
        panic!("session profile removal must be immediate");
    };
    assert_eq!(removed["removed"], true);
    assert!(
        model
            .dispatch(
                "ui.shell.profiles.load",
                &json!({"name":"Review","scope":"session"}),
            )
            .unwrap()
            .is_err()
    );
}

#[test]
fn shell_profile_list_reports_complete_validation_diagnostics() {
    let mut model = AppModel::project();
    model.session_shell_profiles.insert(
        "Future".to_string(),
        json!({
            "format":"odon.shell-layout",
            "schema_version":99,
            "mode":"project",
            "layout":{}
        }),
    );

    let listed = model
        .dispatch("ui.shell.profiles.list", &json!({"scope":"session"}))
        .unwrap()
        .unwrap()
        .response;

    assert_eq!(listed["profiles"][0]["valid"], false);
    assert_eq!(listed["profiles"][0]["error_kind"], "UNSUPPORTED");
    assert!(
        listed["profiles"][0]["error"]
            .as_str()
            .unwrap()
            .contains("schema version 99")
    );
    assert_eq!(listed["profiles"][0]["recovery_method"], "ui.shell.recover");
}

#[test]
fn shell_layout_project_profiles_roundtrip_through_project_state() {
    let mut model = AppModel::project();
    let custom = json!({
        "root_id":"layout:project-profile.root",
        "nodes":[
            {"id":"layout:project-profile.root","type":"application","children":["layout:project-profile.workspace"]},
            {"id":"layout:project-profile.workspace","type":"builtin_mount","mount":"builtin:project-workspace"}
        ]
    });
    model
        .dispatch("ui.shell.replace_layout", &json!({"desired_tree":custom}))
        .unwrap()
        .unwrap();
    let SettingsMutationOutcome::Immediate(saved) = model
        .prepare_shell_profile_save(&json!({"name":"Team review","scope":"project"}))
        .unwrap()
    else {
        panic!("project profile save belongs to the project transaction");
    };
    assert_eq!(saved["project_dirty"], true);
    assert_eq!(saved["persisted"], false);
    let (payload, _) = model.project_persistence_payload().unwrap();
    assert_eq!(
        payload["state"]["shell_layout_profiles"]["Team review"]["layout"]["root_id"],
        "layout:project-profile.root"
    );

    let config = serde_json::from_value(payload["config"].clone()).unwrap();
    let mut restored = AppModel::project();
    let generation = restored.begin_project_operation("restore project profile");
    assert!(
        restored
            .install_project_for_generation(
                generation,
                PathBuf::from("/tmp/profiles.odon.project.json"),
                config,
                payload["state"].clone(),
            )
            .unwrap()
    );
    let listed = restored
        .dispatch("ui.shell.profiles.list", &json!({"scope":"project"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(listed["profiles"][0]["name"], "Team review");
    let loaded = restored
        .dispatch(
            "ui.shell.profiles.load",
            &json!({"name":"Team review","scope":"project"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(loaded["layout"]["root_id"], "layout:project-profile.root");

    let SettingsMutationOutcome::Immediate(removed) = restored
        .prepare_shell_profile_remove(&json!({"name":"Team review","scope":"project"}))
        .unwrap()
    else {
        panic!("project profile removal belongs to the project transaction");
    };
    assert_eq!(removed["removed"], true);
    assert!(
        restored.project_persistence_payload().unwrap().0["state"]["shell_layout_profiles"]
            .as_object()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn application_startup_shell_profile_restores_once_and_reports_status() {
    let mut settings = AppSettings::default();
    settings
        .shell_layout_profiles
        .insert("Home".to_string(), shell_layout_fixture("v1-project"));
    settings
        .shell_layout_startup_profiles
        .insert("project".to_string(), "Home".to_string());
    let mut model = AppModel::project();

    model.bootstrap_settings(settings, None, Vec::new());

    let shell = model.shell_snapshot(None).unwrap();
    assert_eq!(shell["layout"]["root_id"], "layout:fixture.v1.root");
    let restore = model.startup_shell_restore_snapshot();
    assert_eq!(restore["results"]["project"]["status"], "restored");
    assert_eq!(restore["results"]["project"]["profile"], "Home");
    model
        .dispatch("ui.shell.reset", &json!({}))
        .unwrap()
        .unwrap();
    assert!(!model.apply_startup_shell_layout_if_needed());
    assert_ne!(
        model.shell_snapshot(None).unwrap()["layout"]["root_id"],
        "layout:fixture.v1.root"
    );
}

#[test]
fn invalid_application_startup_shell_profile_installs_protected_recovery() {
    for (profile, fixture_name, error_kind) in [
        ("Corrupt", "v1-corrupt-tree", "INVALID_PARAMS"),
        ("Future", "v99-future", "UNSUPPORTED"),
    ] {
        let mut settings = AppSettings::default();
        settings
            .shell_layout_profiles
            .insert(profile.to_string(), shell_layout_fixture(fixture_name));
        settings
            .shell_layout_startup_profiles
            .insert("project".to_string(), profile.to_string());
        let mut model = AppModel::project();

        model.bootstrap_settings(settings, None, Vec::new());

        let shell = model.shell_snapshot(None).unwrap();
        assert_eq!(shell["layout"]["nodes"].as_array().unwrap().len(), 2);
        assert!(
            shell["layout"]["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|node| node["mount"] == "builtin:project-workspace")
        );
        let settings = model.settings_snapshot();
        assert_eq!(
            settings["shell_layout_startup_restore"]["results"]["project"]["status"],
            "recovered"
        );
        assert_eq!(
            settings["shell_layout_startup_restore"]["results"]["project"]["error"]["kind"],
            error_kind
        );
        assert!(
            settings["status"]
                .as_str()
                .unwrap()
                .contains("protected recovery layout installed")
        );
    }
}

#[test]
fn startup_shell_profiles_restore_each_mode_only_on_first_activation() {
    let mut settings = AppSettings::default();
    settings.shell_layout_profiles.insert(
        "Viewer".to_string(),
        shell_layout_fixture("v1-single-startup"),
    );
    settings
        .shell_layout_startup_profiles
        .insert("single".to_string(), "Viewer".to_string());
    let mut model = AppModel::project();
    model.bootstrap_settings(settings, None, Vec::new());

    model.set_mode(ModelMode::Single);
    assert!(model.apply_startup_shell_layout_if_needed());
    model.set_mode(ModelMode::Project);
    assert_eq!(
        model.shell_snapshot(Some("single")).unwrap()["layout"]["root_id"],
        "layout:fixture.startup.single.root"
    );
    model.set_mode(ModelMode::Single);
    assert!(!model.apply_startup_shell_layout_if_needed());
    assert_eq!(
        model.startup_shell_restore_snapshot()["results"]["single"]["status"],
        "restored"
    );
}

#[test]
fn mode_transitions_restore_each_modes_actor_owned_focus_without_cross_mode_leakage() {
    fn node_for_mount(snapshot: &Value, mount: &str) -> String {
        snapshot["layout"]["nodes"]
            .as_array()
            .expect("shell layout nodes")
            .iter()
            .find(|node| node["mount"] == mount)
            .and_then(|node| node["id"].as_str())
            .unwrap_or_else(|| panic!("missing shell mount {mount}"))
            .to_string()
    }

    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    let project = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let project_top_bar = node_for_mount(&project, "builtin:project-top-bar");
    model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":project["revision"],
                "active_region_id":project_top_bar,
                "focused_node_id":project_top_bar,
            }),
        )
        .unwrap()
        .unwrap();

    model.install_dataset(&dataset);
    let single = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let single_canvas = node_for_mount(&single, "builtin:viewer-canvas");
    assert_eq!(single["active_region_id"], single_canvas);
    assert_eq!(single["focused_node_id"], Value::Null);
    let single_top_bar = node_for_mount(&single, "builtin:viewer-top-bar");
    model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":single["revision"],
                "active_region_id":single_top_bar,
                "focused_node_id":single_top_bar,
            }),
        )
        .unwrap()
        .unwrap();

    model.set_mode(ModelMode::Project);
    let restored_project = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(restored_project["active_region_id"], project_top_bar);
    assert_eq!(restored_project["focused_node_id"], project_top_bar);

    model.install_dataset(&dataset);
    let restored_single = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(restored_single["active_region_id"], single_top_bar);
    assert_eq!(restored_single["focused_node_id"], single_top_bar);
}

#[test]
fn shell_layout_state_patch_updates_geometry_selection_and_collapse_atomically() {
    let mut model = AppModel::project();
    let initial = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let desired_tree = json!({
        "root_id":"layout:state.root",
        "nodes":[
            {"id":"layout:state.root","type":"application","children":["layout:state.split"]},
            {"id":"layout:state.split","type":"split","children":["layout:state.workspace","layout:state.panel"],"split":{"axis":"horizontal","ratio":0.7}},
            {"id":"layout:state.workspace","type":"builtin_mount","mount":"builtin:project-workspace"},
            {"id":"layout:state.panel","type":"panel","children":["layout:state.collapsible"],"size":{"width":300.0}},
            {"id":"layout:state.collapsible","type":"collapsible","children":["layout:state.tabs"]},
            {"id":"layout:state.tabs","type":"tabs","children":["layout:state.first","layout:state.second"],"selected_id":"layout:state.first"},
            {"id":"layout:state.first","type":"extension_mount","mount":"extension:first"},
            {"id":"layout:state.second","type":"extension_mount","mount":"extension:second"}
        ]
    });
    let replaced = model
        .dispatch(
            "ui.shell.replace_layout",
            &json!({
                "desired_tree":desired_tree,
                "if_shell_revision":initial["revision"],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    let patched = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":replaced["revision"],
                "selected":{"layout:state.tabs":"layout:state.second"},
                "sizes":{"layout:state.panel":{"width":420.0,"min_width":240.0}},
                "splits":{"layout:state.split":{"axis":"vertical","ratio":0.6,"resizable":false}},
                "collapsed":{"layout:state.collapsible":true},
                "visibility":{"layout:state.first":false},
                "active_region_id":"layout:state.second",
                "focused_node_id":"layout:state.second",
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(patched["change"]["operation"], "patch_layout");
    let properties = patched["change"]["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|change| change["property"].as_str().unwrap())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        properties,
        BTreeSet::from([
            "visibility",
            "selection",
            "size",
            "split",
            "collapse",
            "active_region",
            "focus",
        ])
    );
    assert_eq!(patched["active_region_id"], "layout:state.second");
    assert_eq!(patched["focused_node_id"], "layout:state.second");
    let nodes = patched["layout"]["nodes"].as_array().unwrap();
    let node = |id: &str| nodes.iter().find(|node| node["id"] == id).unwrap();
    assert_eq!(
        node("layout:state.tabs")["selected_id"],
        "layout:state.second"
    );
    assert_eq!(node("layout:state.panel")["size"]["width"], 420.0);
    assert_eq!(node("layout:state.split")["split"]["axis"], "vertical");
    assert_eq!(node("layout:state.collapsible")["collapsed"], true);
    assert_eq!(node("layout:state.first")["visible"], false);

    let cleared = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":patched["revision"],
                "clear_focus":true,
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(cleared["focused_node_id"], Value::Null);
    assert_eq!(cleared["change"]["changes"][0]["property"], "focus");

    let revision = cleared["revision"].as_u64().unwrap();
    let invalid = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":revision,
                "visibility":{"layout:state.workspace":false},
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);
    let after = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(after["revision"], revision);
    assert_eq!(
        after["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == "layout:state.workspace")
            .unwrap()["visible"],
        true
    );
}

#[test]
fn passive_legacy_tab_sync_does_not_replace_an_extension_host_selection() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let initial = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let nodes = initial["layout"]["nodes"].as_array().unwrap();
    let host = nodes
        .iter()
        .find(|node| node["mount"] == "builtin:extension-host.left-sections")
        .unwrap();
    let host_id = host["id"].as_str().unwrap();
    let tabs_id = host["parent_id"].as_str().unwrap();

    let selected = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":initial["revision"],
                "selected":{tabs_id:host_id},
                "focused_node_id":host_id,
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        selected["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == tabs_id)
            .unwrap()["selected_id"],
        host_id
    );

    model
        .sync_active_shell_domain()
        .expect("passive compatibility projection");
    let projected = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        projected["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == tabs_id)
            .unwrap()["selected_id"],
        host_id,
        "passive legacy projection must not replace a desired extension tab"
    );

    model
        .dispatch("viewer.ui.set_left_tab", &json!({"tab":"project"}))
        .unwrap()
        .unwrap();
    let legacy_mutated = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let selected_id = legacy_mutated["layout"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == tabs_id)
        .unwrap()["selected_id"]
        .as_str()
        .unwrap();
    assert_eq!(
        legacy_mutated["layout"]["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|node| node["id"] == selected_id)
            .unwrap()["mount"],
        "builtin:project",
        "an explicit legacy tab command still deliberately updates the desired layout"
    );
}

#[test]
fn shell_mount_configuration_is_schema_validated_and_revisioned() {
    let mut model = AppModel::project();
    let initial = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    let configured = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":initial["revision"],
                "configurations":{
                    "layout:project.top":{"show_title":false},
                },
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    let top = configured["layout"]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["id"] == "layout:project.top")
        .unwrap();
    assert_eq!(top["configuration"]["show_title"], false);
    assert_eq!(
        configured["change"]["changes"][0]["property"],
        "configuration"
    );

    let error = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":configured["revision"],
                "configurations":{
                    "layout:project.top":{"unknown":true},
                },
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::InvalidParams);
    let after = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(after["revision"], configured["revision"]);
    assert_eq!(after["layout"], configured["layout"]);

    let oversized_configuration = (0..100)
        .map(|index| format!("{index:03}-{}", "x".repeat(180)))
        .collect::<Vec<_>>();
    let quota_error = model
        .dispatch(
            "ui.shell.patch_layout",
            &json!({
                "if_shell_revision":after["revision"],
                "configurations":{
                    "layout:project.extensions":{"values":oversized_configuration},
                },
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(quota_error.kind, ControlErrorKind::InvalidParams);
    assert!(quota_error.message.contains("per-node limit"));
    let quota_after = model
        .dispatch("ui.shell.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(quota_after["revision"], after["revision"]);
    assert_eq!(quota_after["layout"], after["layout"]);
}

#[test]
fn late_auto_contrast_does_not_replace_a_newer_manual_window() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let viewport_id = model.dataset().unwrap().workspace.active_id().clone();
    let spec = model
        .prepare_auto_contrast(
            &dataset,
            &json!({"viewport_id":viewport_id.as_str(),"channels":[0]}),
        )
        .unwrap();
    assert!(model.mark_auto_contrast_started(&spec));
    model
        .set_channel_contrast(&json!({
            "viewport_id":viewport_id.as_str(),
            "channel":0,
            "min":1000.0,
            "max":2000.0,
        }))
        .unwrap();
    let result = model
        .install_auto_contrast(
            &spec,
            &[AutoContrastChannelResult {
                channel_index: 0,
                channel_name: spec.channels[0].intensity.channel_name.clone(),
                min: 0,
                max: 500,
                sample_count: 10,
            }],
        )
        .unwrap();
    assert!(result["result"]["applied"].as_array().unwrap().is_empty());
    assert_eq!(
        result["result"]["skipped"][0]["reason"],
        "contrast_changed_after_request"
    );
    let contrast = model
        .get_channel_contrast_global(&json!({"channel":0}))
        .unwrap();
    assert_eq!(contrast["contrast"]["min"], 1000.0);
    assert_eq!(contrast["contrast"]["max"], 2000.0);
}

#[test]
fn stale_auto_contrast_cannot_replace_a_newer_document_generation() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.bootstrap_dataset(&dataset).unwrap();
    let spec = model
        .prepare_auto_contrast(&dataset, &json!({"channels":[0]}))
        .unwrap();
    assert!(model.mark_auto_contrast_started(&spec));
    model.bootstrap_dataset(&dataset).unwrap();
    assert!(
        model
            .install_auto_contrast(
                &spec,
                &[AutoContrastChannelResult {
                    channel_index: 0,
                    channel_name: spec.channels[0].intensity.channel_name.clone(),
                    min: 0,
                    max: 500,
                    sample_count: 10,
                }],
            )
            .is_none()
    );
}

#[test]
fn comparison_commands_advance_without_a_renderer() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = model.dispatch("viewer.viewports.clone", &json!({"source_viewport_id": left, "layout":"horizontal", "ratio":0.5, "title":"Right"})).unwrap().unwrap().response;
    let right = created["viewport_id"].as_str().unwrap().to_string();
    model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":left,"channels":[0],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":right,"channels":[1],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    let fitted = model
        .dispatch("viewer.viewports.camera.fit", &json!({"viewport_id":right}))
        .unwrap()
        .unwrap()
        .response;
    assert!(
        fitted["result"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    let workspace = model.workspace_snapshot().unwrap();
    assert_eq!(workspace["viewports"].as_array().unwrap().len(), 2);
    assert_eq!(workspace["layout"], "horizontal");
}

#[test]
fn invalid_workspace_ratios_are_rejected_before_any_mutation() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    for ratio in [json!("half"), json!(0.95)] {
        let error = model
            .dispatch(
                "viewer.viewports.clone",
                &json!({"layout":"horizontal", "ratio":ratio}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::InvalidParams);
        assert_eq!(
            model.workspace_snapshot().unwrap()["viewports"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"layout":"horizontal", "ratio":0.6}),
        )
        .unwrap()
        .unwrap();
    let before = model.layout_snapshot().unwrap();
    let error = model
        .dispatch(
            "viewer.workspace.layout.set",
            &json!({"layout":"vertical", "ratio":0.95}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.layout_snapshot().unwrap(), before);
}

#[test]
fn readiness_and_availability_queries_are_background_safe_in_every_mode() {
    let mut model = AppModel::project();
    for mode in [ModelMode::Project, ModelMode::Mosaic, ModelMode::Transition] {
        model.bootstrap_mode_from_renderer(mode);
        let loading = model
            .dispatch("app.get_loading_state", &json!({}))
            .expect("loading state is actor-owned")
            .unwrap()
            .response;
        assert_eq!(loading["mode"], mode.as_str());
        let availability = model
            .dispatch(
                "app.get_method_availability",
                &json!({"methods":["app.get_loading_state","viewer.camera.fit"]}),
            )
            .expect("availability is actor-owned")
            .unwrap()
            .response;
        assert_eq!(availability["mode"], mode.as_str());
        assert_eq!(availability["methods"].as_array().unwrap().len(), 2);
    }
}

#[test]
fn concurrent_resource_operations_have_independent_readiness() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let project_generation = model.begin_project_operation("Saving project");
    let (document_generation, label_generation, _) = model
        .begin_label_load(&json!({"name":"cells"}))
        .expect("label operation starts");
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"]["project_io"]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"]["labels"]["phase"],
        "pending"
    );

    assert!(model.fail_label_load_for_generation(
        document_generation,
        label_generation,
        "label fixture failed",
    ));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(loading["loading"]["resources_ready"], false);
    assert_eq!(loading["loading"]["status"], "Saving project");
    assert_eq!(
        loading["loading"]["operations"]["labels"]["phase"],
        "failed"
    );

    assert!(model.finish_project_operation_for_generation(project_generation));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], false);
    assert_eq!(loading["loading"]["resources_ready"], true);
}

#[test]
fn settings_writes_cannot_commit_out_of_order() {
    let mut model = AppModel::project();
    model.bootstrap_settings(
        AppSettings::default(),
        Some(PathBuf::from("/tmp/odon-settings-ordering.json")),
        Vec::new(),
    );
    let SettingsMutationOutcome::Persist(first) = model
        .prepare_settings_set(&json!({"fast_object_rendering":false}))
        .unwrap()
    else {
        panic!("first settings change should require persistence")
    };
    let error = model
        .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
    assert!(
        model
            .install_settings_for_generation(first.generation, first.settings, first.response,)
            .is_some()
    );
    let SettingsMutationOutcome::Persist(second) = model
        .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
        .unwrap()
    else {
        panic!("second settings change should start after the first commits")
    };
    assert!(second.generation > first.generation);
}

#[test]
fn application_low_zoom_geometry_setting_updates_the_active_workspace() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.bootstrap_settings(
        AppSettings::default(),
        Some(PathBuf::from("/tmp/odon-settings-rendering.json")),
        Vec::new(),
    );
    model.install_dataset(&dataset);
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["objects"]["fast_rendering"],
        true
    );

    let SettingsMutationOutcome::Persist(operation) = model
        .prepare_settings_set(&json!({"fast_object_rendering":false}))
        .unwrap()
    else {
        panic!("low-zoom geometry change should require persistence")
    };
    assert!(
        model
            .install_settings_for_generation(
                operation.generation,
                operation.settings,
                operation.response,
            )
            .is_some()
    );
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["objects"]["fast_rendering"],
        false
    );
}

#[test]
fn viewport_filters_of_the_same_kind_have_independent_readiness() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let (document_generation, resource_generation) =
        model.begin_object_resource_load("objects.geojson");
    assert!(model.install_object_resource_for_generation(
        document_generation,
        resource_generation,
        Arc::new(ControlObjectResource {
            source: PathBuf::from("objects.geojson"),
            downsample_factor: 1.0,
            features: Arc::new(Vec::new()),
            property_names: Arc::new(vec!["id".to_string()]),
            numeric_summaries: Arc::new(Default::default()),
            renderer_payload: None,
        }),
    ));
    let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"horizontal"}),
        )
        .unwrap()
        .unwrap()
        .response["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    let left_work = model
        .begin_object_filter_evaluation(
            &json!({"viewport_id":left,"mode":"query","query":"id == 'a'"}),
        )
        .unwrap();
    let right_work = model
        .begin_object_filter_evaluation(
            &json!({"viewport_id":right,"mode":"query","query":"id == 'b'"}),
        )
        .unwrap();
    let loading = model.loading_state();
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{left}:segmentation_objects")]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{right}:segmentation_objects")]["phase"],
        "pending"
    );

    assert!(
        model
            .install_object_filter_for_generation(
                left_work.0,
                left_work.1,
                left_work.2,
                &left_work.3,
                left_work.4,
                left_work.5,
                ControlObjectFilterResult {
                    model: left_work.7,
                    matching_indices: Arc::new(Vec::new()),
                    active: true,
                },
            )
            .is_some()
    );
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{left}:segmentation_objects")]["phase"],
        "ready"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{right}:segmentation_objects")]["phase"],
        "pending"
    );
    assert!(model.fail_object_filter_for_generation(
        &right_work.3,
        right_work.4,
        right_work.2,
        "Right filter failed",
    ));
    assert_eq!(model.loading_state()["loading"]["busy"], false);
}

#[test]
fn spatial_shape_resources_support_the_complete_object_compute_surface() {
    use crate::model::ControlObjectFeature;

    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let resource = Arc::new(ControlObjectResource {
        source: PathBuf::from("shapes/cells.parquet"),
        downsample_factor: 1.0,
        features: Arc::new(vec![ControlObjectFeature {
            id: "shape-1".to_string(),
            bbox_world: [1.0, 1.0, 9.0, 9.0],
            centroid_world: [5.0, 5.0],
            polygons_world: Arc::new(vec![vec![
                [1.0, 1.0],
                [9.0, 1.0],
                [9.0, 9.0],
                [1.0, 9.0],
                [1.0, 1.0],
            ]]),
            point_position_world: None,
            area_px: 64.0,
            perimeter_px: 32.0,
            properties: serde_json::Map::from_iter([("score".to_string(), json!(2.5))]),
        }]),
        property_names: Arc::new(vec!["id".to_string(), "score".to_string()]),
        numeric_summaries: Arc::new(BTreeMap::from([(
            "score".to_string(),
            crate::model::ControlObjectNumericSummary {
                minimum: 2.5,
                maximum: 2.5,
                positive_minimum: Some(2.5),
                positive_count: 1,
                numeric_count: 1,
                missing_count: 0,
            },
        )])),
        renderer_payload: None,
    });
    model
        .install_document_object_layers(&[
            DocumentObjectLayerResource {
                layer_id: "segmentation_objects".to_string(),
                name: "Primary cells".to_string(),
                kind: "objects".to_string(),
                primary: true,
                resource: Arc::clone(&resource),
            },
            DocumentObjectLayerResource {
                layer_id: "spatial_shape:7".to_string(),
                name: "Cells".to_string(),
                kind: "spatial_shape".to_string(),
                primary: false,
                resource,
            },
        ])
        .expect("secondary object layer installs");

    let target = json!({"target":"spatial_shape","layer_id":7});
    let properties = model
        .dispatch("viewer.objects.properties.list", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(properties["target"], "spatial_shape");
    assert_eq!(properties["layer_id"], 7);
    assert_eq!(properties["total"], 2);

    let selected = model
        .dispatch(
            "viewer.objects.select_rect",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "world_rect":[0.0,0.0,10.0,10.0],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(selected["objects"]["target"], "spatial_shape");
    assert_eq!(
        selected["objects"]["result"]["selection"]["selection_count"],
        1
    );

    model
        .dispatch(
            "viewer.objects.style.set",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "fill_cells":true,
                "color_property":"score",
            }),
        )
        .unwrap()
        .unwrap();
    let style = model
        .dispatch("viewer.objects.style.get", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(style["fill_cells"], true);
    assert_eq!(style["color_property"], "score");

    model
        .dispatch(
            "viewer.objects.legend.set",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "entries":[{"value":"high","visible":true,"color_rgb":[1,2,3]}],
            }),
        )
        .unwrap()
        .unwrap();

    model
        .dispatch(
            "viewer.objects.style.set",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "color_mapping":{
                    "mode":"continuous",
                    "property":"score",
                    "palette":"viridis",
                    "domain":[0.0,5.0],
                    "scale":"linear",
                    "reverse":false,
                    "out_of_range":"clamp",
                    "missing_color_rgb":null,
                },
            }),
        )
        .unwrap()
        .unwrap();
    let continuous = model
        .dispatch("viewer.objects.style.get", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(continuous["color_mapping"]["mode"], "continuous");
    assert_eq!(continuous["color_mapping"]["property"], "score");
    assert_eq!(
        continuous["color_mapping"]["resolved_domain"],
        json!([0.0, 5.0])
    );
    assert_eq!(continuous["color_mapping"]["numeric_count"], 1);
    assert_eq!(continuous["legend"], json!([]));

    let invalid = model
        .dispatch(
            "viewer.objects.style.set",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "color_mapping":{
                    "mode":"continuous",
                    "property":"score",
                    "palette":"viridis",
                    "domain":[5.0,5.0]
                },
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);
    let unchanged = model
        .dispatch("viewer.objects.style.get", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(unchanged["color_mapping"]["domain"], json!([0.0, 5.0]));

    let filter = model
        .begin_object_filter_evaluation(&json!({
            "target":"spatial_shape",
            "layer_id":7,
            "mode":"query",
            "query":"score > 2",
        }))
        .expect("spatial filter starts");
    assert_eq!(filter.4, ObjectTarget::SpatialShape(7));
    let filtered = model.install_object_filter_for_generation(
        filter.0,
        filter.1,
        filter.2,
        &filter.3,
        filter.4,
        filter.5,
        ControlObjectFilterResult {
            model: json!({"mode":"query","query":"score > 2"}),
            matching_indices: Arc::new(vec![0]),
            active: true,
        },
    );
    assert_eq!(filtered.unwrap()["result"]["visible_count"], 1);
    let projected_layer = model
        .dispatch(
            "viewer.native_layers.get",
            &json!({"layer_id":"spatial_shape:7"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        projected_layer["layer"]["presentation"]["objects"]["fill_cells"],
        true
    );
    assert_eq!(
        projected_layer["layer"]["presentation"]["objects"]["filter"]["query"],
        "score > 2"
    );

    let analysis = model
        .prepare_analysis_resource_operation(&target, "histogram")
        .expect("spatial analysis starts");
    assert_eq!(analysis.target, ObjectTarget::SpatialShape(7));
    assert_eq!(analysis.resource.features.len(), 1);

    let primary_analysis = json!({
        "target":"segmentation_objects",
        "state":{"threshold_set_name":"Primary analysis"},
    });
    model
        .dispatch("viewer.analysis.set", &primary_analysis)
        .unwrap()
        .unwrap();
    let spatial_analysis = json!({
        "target":"spatial_shape",
        "layer_id":7,
        "state":{"threshold_set_name":"Spatial analysis"},
    });
    model
        .dispatch("viewer.analysis.set", &spatial_analysis)
        .unwrap()
        .unwrap();
    let primary = model
        .dispatch(
            "viewer.analysis.get",
            &json!({"target":"segmentation_objects"}),
        )
        .unwrap()
        .unwrap()
        .response;
    let spatial = model
        .dispatch("viewer.analysis.get", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(primary["state"]["threshold_set_name"], "Primary analysis");
    assert_eq!(spatial["state"]["threshold_set_name"], "Spatial analysis");
    let secondary = model.secondary_object_projections();
    assert_eq!(
        secondary[0].analysis_state["threshold_set_name"],
        "Spatial analysis"
    );
    assert_eq!(secondary[0].analysis_generation, 2);

    let measurement = model
        .prepare_measurement(&target)
        .expect("spatial measurement starts");
    assert_eq!(measurement.target, ObjectTarget::SpatialShape(7));
    assert_eq!(measurement.target_indices.as_ref(), &[0]);

    let export = model
        .prepare_object_export(
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "scope":"selected",
            }),
            PathBuf::from("spatial-shape.csv"),
            Some(ObjectExportFormat::Csv),
        )
        .expect("spatial export starts");
    assert_eq!(export.target, ObjectTarget::SpatialShape(7));
    assert_eq!(export.row_indices.as_ref(), &[0]);

    model
        .dispatch(
            "viewer.native_layers.set_active",
            &json!({"layer_id":"spatial_shape:7"}),
        )
        .unwrap()
        .unwrap();
    let active = model
        .dispatch("viewer.objects.get_selection", &json!({"target":"active"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(active["objects"]["target"], "spatial_shape");
    assert_eq!(active["objects"]["layer_id"], 7);
}

#[test]
fn mask_io_readiness_is_scoped_and_cancelled_by_document_replacement() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let (document_generation, mask_generation, import_generation, import_scope) =
        model.begin_mask_import_operation().unwrap();
    let (export_generation, export_scope) = model.begin_mask_export_operation().unwrap();
    assert!(model.finish_mask_io_for_generation(
        &export_scope,
        export_generation,
        "Mask export ready",
    ));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("mask_io:{export_scope}")]["phase"],
        "ready"
    );

    let layer = model
        .dispatch("viewer.masks.layers.create", &json!({"name":"Drawn"}))
        .unwrap()
        .unwrap()
        .response;
    model
        .dispatch(
            "viewer.masks.polygons.add",
            &json!({"id":layer["id"],"vertices":[[0,0],[4,0],[4,4]]}),
        )
        .unwrap()
        .unwrap();
    let (_, _, append_generation, append_scope, _) = model.begin_mask_append_operation().unwrap();
    let duplicate_append = model.begin_mask_append_operation().unwrap_err();
    assert_eq!(duplicate_append.kind, ControlErrorKind::NotReady);
    assert!(model.finish_mask_io_for_generation(
        &append_scope,
        append_generation,
        "Mask append ready",
    ));

    model.begin_dataset_open("replacement");
    assert_eq!(
        model.loading_state()["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
        "cancelled"
    );
    assert!(
        model
            .install_imported_masks_for_generation(
                document_generation,
                mask_generation,
                import_generation,
                &import_scope,
                "stale".to_string(),
                true,
                None,
                vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
                PathBuf::from("stale.geojson"),
            )
            .is_none()
    );
}

#[test]
fn actor_model_enforces_scoped_viewport_revision_guards() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let workspace = model.workspace_snapshot().unwrap();
    let id = workspace["active_viewport_id"].as_str().unwrap();
    let navigation = workspace["viewports"][0]["navigation_revision"]
        .as_u64()
        .unwrap();
    let presentation = workspace["viewports"][0]["presentation_revision"]
        .as_u64()
        .unwrap();

    model
        .dispatch(
            "viewer.viewports.camera.set",
            &json!({"viewport_id":id,"zoom":2.0,"if_navigation_revision":navigation}),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        model
            .dispatch(
                "viewer.viewports.camera.set",
                &json!({"viewport_id":id,"zoom":3.0,"if_navigation_revision":navigation}),
            )
            .unwrap()
            .unwrap_err()
            .kind,
        ControlErrorKind::Conflict
    );

    model
        .dispatch(
            "viewer.viewports.channels.set_color",
            &json!({"viewport_id":id,"channel":0,"color_rgb":[1,2,3],"if_presentation_revision":presentation}),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        model
            .dispatch(
                "viewer.viewports.channels.set_color",
                &json!({"viewport_id":id,"channel":0,"color_rgb":[3,2,1],"if_presentation_revision":presentation}),
            )
            .unwrap()
            .unwrap_err()
            .kind,
        ControlErrorKind::Conflict
    );
}

#[test]
fn stale_dataset_worker_results_cannot_replace_a_newer_document_request() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    let stale = model.begin_dataset_open("first");
    let loading_error = model
        .dispatch("viewer.workspace.get", &json!({}))
        .expect("actor-owned methods never fall back while loading")
        .unwrap_err();
    assert_eq!(loading_error.kind, ControlErrorKind::NotReady);
    assert_eq!(
        loading_error.data.as_ref().unwrap()["loading"]["resources_ready"],
        false
    );
    let current = model.begin_dataset_open("second");
    assert!(!model.install_dataset_for_generation(stale, &dataset, Vec::new(), None));
    assert_eq!(model.mode(), ModelMode::Transition);
    assert!(model.install_dataset_for_generation(current, &dataset, Vec::new(), None));
    assert_eq!(model.mode(), ModelMode::Single);
}

#[test]
fn unequal_splits_derive_each_viewport_from_the_retained_workspace_geometry() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let left = model.render_workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    model.report_viewport_geometry(&left, 0.0, 0.0, 1200.0, 800.0);
    let right = model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"horizontal","ratio":0.6}),
        )
        .unwrap()
        .unwrap()
        .response["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let workspace = model.render_workspace_snapshot().unwrap();
    let viewport = |id: &str| {
        workspace["viewports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|viewport| viewport["viewport_id"] == id)
            .unwrap()
    };
    assert_eq!(
        viewport(&left)["camera"]["viewport"],
        json!([0.0, 0.0, 720.0, 800.0])
    );
    let right_width = viewport(&right)["camera"]["viewport"][2].as_f64().unwrap();
    assert!((right_width - 480.0).abs() < 1.0e-3);
    assert_eq!(viewport(&right)["camera"]["viewport"][3], json!(800.0));

    model
        .dispatch("viewer.viewports.remove", &json!({"viewport_id":left}))
        .unwrap()
        .unwrap();
    let remaining = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        remaining["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1200.0, 800.0])
    );
}

#[test]
fn dataset_replacement_preserves_observed_logical_geometry() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    model.report_viewport_geometry("viewport-1", 0.0, 0.0, 1234.0, 777.0);
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "observed"
    );

    model.install_dataset(&dataset);
    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        workspace["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1234.0, 777.0])
    );
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "observed"
    );
}

#[test]
fn panel_changes_derive_background_geometry_without_a_frame() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    model.report_viewport_geometry("viewport-1", 0.0, 0.0, 1000.0, 700.0);

    let hidden = model
        .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(hidden["result"]["changed"], true);
    assert_eq!(
        hidden["result"]["panels"],
        json!({"left":false,"right":false})
    );
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1740.0, 700.0])
    );
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "derived"
    );

    model
        .dispatch("viewer.camera.fit", &json!({}))
        .unwrap()
        .unwrap();
    let fitted = model
        .dispatch("viewer.camera.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert!(
        fitted["camera"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn dataset_bootstrap_restores_actor_project_workspace_and_supersedes_workers() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut source = AppModel::project();
    source.install_dataset(&dataset);
    let left = source.render_workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    source
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"vertical","ratio":0.7,"title":"Native second"}),
        )
        .unwrap()
        .unwrap();
    source
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":left,"channels":[3],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    let saved_workspace = project_workspace_view_json(&source.dataset().unwrap().workspace);
    let source_key = dataset.source.source_key();
    let mut roi = ProjectRoi {
        id: "fixture-roi".to_string(),
        display_name: Some("Fixture ROI".to_string()),
        mask_layers: vec![ProjectMaskLayer {
            id: 17,
            name: "Persisted mask".to_string(),
            visible: true,
            opacity: 0.4,
            width_screen_px: 2.0,
            display_mode: Some("translucent_fill".to_string()),
            color_rgb: [12, 34, 56],
            offset_world: [0.0, 0.0],
            editable: true,
            polygons_world: vec![vec![[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [1.0, 1.0]]],
            source_geojson: None,
        }],
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(dataset.source.clone());

    let mut target = AppModel::project();
    assert!(
        target.bootstrap_project_from_renderer(ProjectModelSnapshot {
            state: json!({"roi_views": {source_key: {"workspace": saved_workspace}}}),
            rois: vec![roi],
            ..ProjectModelSnapshot::default()
        })
    );
    let stale_generation = target.begin_dataset_open("superseded");
    target
        .bootstrap_dataset(&dataset)
        .expect("actor project state bootstraps atomically");
    assert!(!target.install_dataset_for_generation(stale_generation, &dataset, Vec::new(), None));
    let restored = target.render_workspace_snapshot().unwrap();
    let expected = source.render_workspace_snapshot().unwrap();
    for field in ["layout", "ratio", "active_viewport_id", "links"] {
        assert_eq!(restored[field], expected[field], "workspace field {field}");
    }
    assert_eq!(restored["viewports"].as_array().unwrap().len(), 2);
    assert_eq!(
        restored["viewports"][0]["channels"],
        expected["viewports"][0]["channels"]
    );
    assert_eq!(
        target.workspace_snapshot().unwrap()["masks"]["layers"][0]["id"],
        17
    );
}

#[test]
fn plane_commands_retain_per_axis_slices_and_clamp_to_dataset_extents() {
    let (mut dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    dataset.dims.z = Some(1);
    dataset.dims.y = 2;
    dataset.dims.x = 3;
    dataset.dims.ndim = 4;
    for level in &mut dataset.levels {
        level.shape.insert(1, 7);
        level.chunks.insert(1, 1);
        level.scale.insert(1, 1.0);
        level.translation.insert(1, 0.0);
    }
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let set = |model: &mut AppModel, mode: &str, slice: u64| {
        model
            .dispatch(
                "viewer.viewports.planes.set",
                &json!({"viewport_id":"viewport-1","mode":mode,"slice":slice}),
            )
            .unwrap()
            .unwrap()
            .response
    };
    let xy = set(&mut model, "xy", 99);
    assert_eq!(xy["result"]["plane"]["slice"], 6);
    assert_eq!(xy["result"]["plane"]["extent"], 7);
    assert_eq!(
        xy["result"]["plane"]["supported_modes"],
        json!(["xy", "xz", "yz"])
    );

    let xz = set(&mut model, "xz", 1234);
    assert_eq!(xz["result"]["plane"]["slice"], 511);
    assert_eq!(xz["result"]["plane"]["slice_axis"], "y");
    let yz = set(&mut model, "yz", 42);
    assert_eq!(yz["result"]["plane"]["slice"], 42);
    assert_eq!(yz["result"]["plane"]["slice_axis"], "x");

    let back_to_xy = model
        .dispatch(
            "viewer.viewports.planes.set",
            &json!({"viewport_id":"viewport-1","mode":"xy"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(back_to_xy["result"]["plane"]["slice"], 6);
}

#[test]
fn invalid_presentation_commands_are_atomic() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let before = model.render_workspace_snapshot().unwrap();

    let invalid_mode = model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":"viewport-1","channels":[0],"mode":"toggle"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid_mode.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.render_workspace_snapshot().unwrap(), before);

    let invalid_rendering = model
        .dispatch(
            "viewer.viewports.rendering.set",
            &json!({"viewport_id":"viewport-1","show_hud":"yes"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid_rendering.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.render_workspace_snapshot().unwrap(), before);
}

#[test]
fn complete_channel_presentation_executes_and_roundtrips_without_a_renderer() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let note = model
        .dispatch(
            "viewer.channels.set_note",
            &json!({"channel": 1, "note": "T-cell marker"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(note["channel"]["note"], "T-cell marker");

    let transform = model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({
                "viewport_id": "viewport-1",
                "if_presentation_revision": 1,
                "channel": 1,
                "offset_world": [12.5, -3.0],
                "scale": [1.25, 0.75],
                "rotation_rad": 0.5,
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(transform["changed"], true);
    assert_eq!(transform["transform"]["offset_world"], json!([12.5, -3.0]));
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["presentation_revision"],
        2
    );
    let stale_transform = model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({
                "viewport_id":"viewport-1",
                "if_presentation_revision":1,
                "channel":1,
                "offset_world":[0.0,0.0],
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(stale_transform.kind, ControlErrorKind::Conflict);

    let order = model
        .dispatch(
            "viewer.viewports.channels.set_order",
            &json!({
                "viewport_id": "viewport-1",
                "channels": [4, 3, 2, 1, 0],
                "mode": "exact",
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(order["result"]["order"][0]["index"], 4);

    let group = model
        .dispatch(
            "viewer.viewports.channels.set_group",
            &json!({
                "viewport_id": "viewport-1",
                "channels": [1, 2],
                "name": "Immune",
                "color_rgb": [10, 20, 30],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(group["result"]["groups"][0]["name"], "Immune");
    assert_eq!(
        group["result"]["groups"][0]["members"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    model
        .dispatch(
            "viewer.channels.presentation.set",
            &json!({"search": "CD", "sort": "visible_first"}),
        )
        .unwrap()
        .unwrap();
    let projection = model.render_workspace_snapshot().unwrap();
    assert_eq!(projection["channel_presentation"]["search"], "CD");
    assert_eq!(projection["channel_transforms"][1]["rotation_rad"], 0.5);
    assert_eq!(
        projection["viewports"][0]["channel_order"],
        json!([4, 3, 2, 1, 0])
    );

    let mut restored = AppModel::project();
    restored.install_dataset(&dataset);
    restored
        .restore_workspace_snapshot(&projection)
        .expect("complete presentation projection roundtrips");
    assert_eq!(restored.render_workspace_snapshot().unwrap(), projection);

    let before = restored.render_workspace_snapshot().unwrap();
    let invalid = restored
        .dispatch(
            "viewer.channels.set_transform",
            &json!({"channel": 1, "scale": [0.0, 1.0]}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);
    assert_eq!(restored.render_workspace_snapshot().unwrap(), before);
}

#[test]
fn stale_renderer_observation_cannot_revert_actor_owned_state() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let stale = model.render_workspace_snapshot().unwrap();

    model
        .dispatch(
            "viewer.channels.set_note",
            &json!({"channel":1,"note":"new actor value"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({"channel":1,"offset_world":[8.0,-4.0]}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_order",
            &json!({
                "viewport_id":"viewport-1",
                "channels":[4,3,2,1,0],
                "mode":"exact",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.presentation.set",
            &json!({"search":"actor search","sort":"visible_first"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
        .unwrap()
        .unwrap();
    let current_revision = model.mark_projection_dirty();

    assert!(model.observe_renderer_state(&stale, current_revision - 1));
    let current = model.render_workspace_snapshot().unwrap();
    assert_eq!(current["channel_metadata"][1]["note"], "new actor value");
    assert_eq!(
        current["channel_transforms"][1]["offset_world"],
        json!([8.0, -4.0])
    );
    assert_eq!(
        current["viewports"][0]["channel_order"],
        json!([4, 3, 2, 1, 0])
    );
    assert_eq!(current["channel_presentation"]["search"], "actor search");
    assert_eq!(current["panels"], json!({"left":false,"right":false}));

    assert!(!model.observe_renderer_state(&stale, current_revision + 1));
    let mut wrong_document = stale;
    wrong_document["shared_resources"]["dataset_source"] = json!("another dataset");
    assert!(!model.observe_renderer_state(&wrong_document, current_revision));
    assert_eq!(model.render_workspace_snapshot().unwrap(), current);
}

#[test]
fn project_metadata_and_roi_transactions_execute_without_a_renderer() {
    let mut model = AppModel::project();
    let created = model
        .dispatch("project.create", &json!({"default_dataset":"cohort-a"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        created["project"]["metadata"]["default_dataset"],
        "cohort-a"
    );

    for (id, path) in [("roi-a", "/tmp/a.ome.zarr"), ("roi-b", "/tmp/b.ome.zarr")] {
        let added = model
            .dispatch(
                "project.rois.add",
                &json!({"id":id,"path":path,"metadata":{"group":"test"}}),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(added["roi"]["id"], id);
    }
    let selected = model
        .dispatch(
            "project.rois.select",
            &json!({"ids":["roi-b"],"mode":"replace"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(selected["selected"], json!(["roi-b"]));
    assert_eq!(selected["focused"], "roi-b");

    let updated = model
        .dispatch(
            "project.rois.update",
            &json!({
                "target_id":"roi-b",
                "changes":{"display_name":"B","segmentation_path":"/tmp/b.parquet"},
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(updated["roi"]["display_name"], "B");
    model
        .dispatch("project.rois.reorder", &json!({"ids":["roi-b","roi-a"]}))
        .unwrap()
        .unwrap();
    let rois = model
        .dispatch("project.rois.list", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(rois["rois"][0]["id"], "roi-b");
    assert_eq!(rois["rois"][0]["selected"], true);

    let before = rois.clone();
    let duplicate = model
        .dispatch(
            "project.rois.add",
            &json!({"id":"roi-a","path":"/tmp/c.ome.zarr"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(duplicate.kind, ControlErrorKind::Conflict);
    assert_eq!(
        model
            .dispatch("project.rois.list", &json!({}))
            .unwrap()
            .unwrap()
            .response,
        before
    );

    let stepped = model
        .dispatch("project.rois.next", &json!({"wrap":true}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(stepped["focused"], "roi-a");
}

#[test]
fn project_create_installs_a_complete_typed_config_atomically() {
    let mut roi = crate::data::project_config::ProjectRoi {
        id: "roi-from-config".to_string(),
        display_name: Some("Configured ROI".to_string()),
        ..Default::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        PathBuf::from("/tmp/configured.ome.zarr"),
    ));
    let mut config = crate::data::project_config::ProjectConfig {
        default_dataset: Some("configured-dataset".to_string()),
        secondary_dataset: Some("secondary".to_string()),
        default_threshold_marker: Some("DAPI".to_string()),
        rois: vec![roi.clone()],
        ..Default::default()
    };
    config
        .mosaic_segmentation_search_roots
        .push(PathBuf::from("/tmp/segmentations"));
    config
        .datasets
        .insert("configured-dataset".to_string(), Default::default());
    config.control_resources = vec![json!({"id":"resource-from-config"})];

    let mut model = AppModel::project();
    let response = model
        .dispatch("project.create", &json!({"config":config}))
        .unwrap()
        .unwrap()
        .response;
    let snapshot = model.project_snapshot();

    assert_eq!(response["created"], true);
    assert_eq!(snapshot.load_generation, 1);
    assert_eq!(
        serde_json::to_value(&snapshot.rois).unwrap(),
        serde_json::to_value([roi]).unwrap()
    );
    assert_eq!(
        snapshot.default_dataset.as_deref(),
        Some("configured-dataset")
    );
    assert_eq!(snapshot.secondary_dataset.as_deref(), Some("secondary"));
    assert_eq!(snapshot.default_threshold_marker.as_deref(), Some("DAPI"));
    assert_eq!(
        snapshot.mosaic_segmentation_search_roots,
        vec![PathBuf::from("/tmp/segmentations")]
    );
    assert_eq!(snapshot.dataset_keys, vec!["configured-dataset"]);
    assert_eq!(snapshot.config.control_resources.len(), 1);
    assert!(!snapshot.dirty);
}

#[test]
fn renderer_project_bootstrap_cannot_revert_actor_owned_commits() {
    let mut roi = crate::data::project_config::ProjectRoi {
        id: "roi".to_string(),
        display_name: Some("Initial".to_string()),
        ..Default::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        PathBuf::from("/tmp/bootstrap.ome.zarr"),
    ));
    let bootstrap = ProjectModelSnapshot {
        rois: vec![roi],
        ..ProjectModelSnapshot::default()
    };
    let mut model = AppModel::project();
    assert!(model.bootstrap_project_from_renderer(bootstrap.clone()));
    model
        .dispatch(
            "project.rois.update",
            &json!({"target_id":"roi","changes":{"display_name":"Actor"}}),
        )
        .unwrap()
        .unwrap();

    assert!(!model.bootstrap_project_from_renderer(bootstrap));
    assert_eq!(
        model.project_snapshot().rois[0].display_name.as_deref(),
        Some("Actor")
    );
}

#[test]
fn renderer_project_bootstrap_normalizes_persisted_state_and_saved_views() {
    let mut model = AppModel::project();
    assert!(model.bootstrap_project_from_renderer(ProjectModelSnapshot {
        state: json!({
            "browser": "invalid legacy value",
            "view_presets": [{
                "name": "Actor view",
                "description": "",
                "spec": {"channel_ref": {"label": "DAPI"}},
            }],
        }),
        ..ProjectModelSnapshot::default()
    }));

    let snapshot = model.project_snapshot();
    assert!(snapshot.state["browser"].is_object());
    assert_eq!(snapshot.view_count, 1);
    assert_eq!(snapshot.view_presets[0]["name"], "Actor view");

    let replacement = ProjectModelSnapshot {
        state: Value::String("invalid legacy state".to_string()),
        view_count: 99,
        ..ProjectModelSnapshot::default()
    };
    let mut replacement_model = AppModel::project();
    assert!(replacement_model.bootstrap_project_from_renderer(replacement));
    let replacement = replacement_model.project_snapshot();
    assert!(replacement.state.is_object());
    assert_eq!(replacement.view_count, 0);
}

#[test]
fn native_object_layer_replacement_applies_full_display_and_legend_state() {
    let mut objects = default_object_snapshot();
    let changed = apply_native_object_layer_presentation(
        &mut objects,
        &json!({
            "visible":true,
            "opacity":0.6,
            "width_screen_px":2.0,
            "color_rgb":[4,5,6],
            "show_selection_overlay":false,
            "display":{
                "color_property_key":"phenotype",
                "color_level_overrides":{
                    "tumour":{"visible":false,"color_rgb":[220,40,60]}
                },
                "fill_cells":true,
                "fill_opacity":0.45,
                "selected_fill_opacity":0.8,
                "fast_rendering":false
            },
            "filter":{"mode":"query","query":"ignored by presentation replacement"}
        }),
    )
    .expect("native object presentation is valid");

    assert!(changed);
    assert_eq!(objects["visible"], true);
    assert_eq!(objects["opacity"], json!(0.6_f32));
    assert_eq!(objects["fill_cells"], true);
    assert_eq!(objects["fill_opacity"], json!(0.45_f32));
    assert_eq!(objects["selected_fill_opacity"], json!(0.8_f32));
    assert_eq!(objects["fast_rendering"], false);
    assert_eq!(objects["color_property"], "phenotype");
    assert_eq!(
        objects["color_level_overrides"]["tumour"],
        json!({"visible":false,"color_rgb":[220,40,60]})
    );
    assert_eq!(
        objects["filter"],
        default_object_filter_model(),
        "filter evaluation remains on its worker-backed command path"
    );
}
