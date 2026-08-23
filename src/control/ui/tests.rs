use super::*;
use crate::control::Ownership;
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
}
