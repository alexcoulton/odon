//! Typed control-command validation tests.

use super::*;

#[test]
fn typed_commands_validate_representative_parameters() {
    assert!(ControlCommand::decode("get_camera", json!({})).is_ok());
    assert!(ControlCommand::decode("get_camera", json!({"extra": true})).is_err());
    assert!(ControlCommand::decode("set_side_panels", json!({})).is_err());
    assert!(ControlCommand::decode("set_side_panels", json!({"left": false})).is_ok());
    assert!(
        ControlCommand::decode(
            "set_visible_channels",
            json!({"channels": ["DAPI", 2], "mode": "only"})
        )
        .is_ok()
    );
    assert!(ControlCommand::decode("set_camera", json!({"zoom": 0.0})).is_err());
    let command = ControlCommand::decode("set_camera", json!({"zoom": 2.0, "if_revision": 4}))
        .expect("revision precondition");
    assert_eq!(command.if_revision(), Some(4));
    assert_eq!(command.params(), &json!({"zoom": 2.0}));
    assert_eq!(command.method(), "viewer.camera.set");
    assert_eq!(command.event_name(), Some("viewer.camera.changed"));
    assert!(command.available_in().contains(&"single"));
    assert!(ControlCommand::decode("get_camera", json!({"if_revision": 4})).is_err());
}

#[test]
fn actor_service_commands_share_the_typed_command_envelope() {
    let resource = ControlCommand::decode(
        "data.resources.register",
        json!({
            "resource_id": "resource:test",
            "uri": "file:///tmp/test.zarr",
            "format": "ome-zarr",
            "coordinate_space": {"axes": ["y", "x"]},
        }),
    )
    .expect("resource commands are accepted by the actor mailbox");
    assert_eq!(resource.method(), "data.resources.register");
    assert!(resource.mutates());
    assert!(resource.available_in().contains(&"project"));

    let layer = ControlCommand::decode(
        "viewer.layers.update",
        json!({"layer_id":"layer:test","visible":false,"if_revision":12}),
    )
    .expect("layer-local revision is retained by the typed envelope");
    assert_eq!(layer.if_revision(), Some(12));
    assert_eq!(
        layer.params(),
        &json!({"layer_id":"layer:test","visible":false})
    );
    assert!(ControlCommand::decode("data.resources.list", json!({"if_revision": 1})).is_err());
    assert!(ControlCommand::decode("system.hello", json!({})).is_err());
}

#[test]
fn phase_g_commands_have_typed_validation() {
    assert!(ControlCommand::decode("app.settings.set", json!({
        "auto_contrast": {"method": "p1_to_p99", "lower_percentile": 1, "upper_percentile": 99},
        "fast_object_rendering": false
    })).is_ok());
    assert!(
        ControlCommand::decode(
            "app.settings.set",
            json!({
                "auto_contrast": {"lower_percentile": 99, "upper_percentile": 20}
            })
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode("app.lifecycle.request_close", json!({"save": "prompt"})).is_ok()
    );
    assert!(
        ControlCommand::decode("app.lifecycle.request_close", json!({"save": "maybe"})).is_err()
    );
    assert!(ControlCommand::decode("viewer.scale_bar.set", json!({"visible": true})).is_ok());
    assert!(
        ControlCommand::decode(
            "viewer.screenshot.settings.set",
            json!({"legend_scale": 4.0})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "memory.tiles.set",
            json!({"workers": 6, "prefetch_mode": "target_halo"})
        )
        .is_ok()
    );
    assert!(ControlCommand::decode("memory.pin", json!({"level": 1, "scope": "item"})).is_err());
    assert!(
        ControlCommand::decode(
            "memory.pin",
            json!({"level": 1, "scope": "all", "force": true})
        )
        .is_ok()
    );
    assert!(ControlCommand::decode("memory.unpin", json!({"level": 1, "unknown": true})).is_err());
}

#[test]
fn plane_commands_have_typed_validation() {
    assert!(ControlCommand::decode("viewer.planes.get", json!({})).is_ok());
    assert!(ControlCommand::decode("viewer.planes.set", json!({})).is_err());
    assert!(
        ControlCommand::decode("viewer.planes.set", json!({"mode": "XZ", "slice": 12})).is_ok()
    );
    assert!(ControlCommand::decode("viewer.planes.set", json!({"mode": "time"})).is_err());
    assert!(ControlCommand::decode("viewer.planes.next", json!({"step": 0})).is_err());
    let command = ControlCommand::decode(
        "viewer.planes.previous",
        json!({"step": 3, "wrap": true, "if_revision": 9}),
    )
    .expect("valid plane step");
    assert_eq!(command.if_revision(), Some(9));
    assert_eq!(command.event_name(), Some("viewer.planes.changed"));
    assert_eq!(command.available_in(), &["single"]);
}

#[test]
fn channel_property_commands_have_typed_validation() {
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_color",
            json!({"name": "DAPI", "color_rgb": [1, 2, 255]})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_color",
            json!({"name": "DAPI", "color_rgb": [1, 2, 256]})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_note",
            json!({"note": "missing selector"})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_transform",
            json!({"index": 0, "scale": [0.01, 100.0], "rotation_rad": 0.5})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_transform",
            json!({
                "viewport_id":"viewport-1",
                "if_presentation_revision":1,
                "index":0,
                "offset_world":[2.0,3.0],
            })
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_transform",
            json!({"if_presentation_revision":1,"index":0,"offset_world":[2.0,3.0]})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "viewer.channels.set_transform",
            json!({"index": 0, "scale": [0.0, 1.0]})
        )
        .is_err()
    );
}

#[test]
fn native_layer_commands_have_typed_validation() {
    assert!(
        ControlCommand::decode(
            "viewer.native_layers.set_visibility",
            json!({"layer_id": "channel:0", "visible": false})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "viewer.native_layers.set_visibility",
            json!({"layer_id": "channel:0"})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "viewer.native_layers.set_order",
            json!({"stack": "channels", "layers": ["channel:1", "channel:1"]})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "viewer.native_layers.set_offset",
            json!({"layer_id": "mask:2", "offset_world": [1.0, 2.0]})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "viewer.native_layers.get",
            json!({"layer_id": "channel:0", "id": "channel:0"})
        )
        .is_err()
    );
}

#[test]
fn project_view_commands_have_typed_validation() {
    assert!(ControlCommand::decode("project.views.get", json!({"name": "Review"})).is_ok());
    assert!(
        ControlCommand::decode(
            "project.views.capture",
            json!({"name": "Review", "viewport_id": "viewport-2"})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode("project.views.get", json!({"index": 0, "name": "Review"})).is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.views.create",
            json!({"name": "Review", "spec": {"visible_channels": ["DAPI"]}})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode("project.views.create", json!({"name": "Review", "spec": 4}))
            .is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.views.rename",
            json!({"index": 0, "new_name": "Overview"})
        )
        .is_ok()
    );
}

#[test]
fn project_roi_mosaic_and_deep_link_commands_are_typed() {
    assert!(
        ControlCommand::decode(
            "project.create",
            json!({"config":{"default_dataset":"default","rois":[]}})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "project.create",
            json!({
                "default_dataset":"default",
                "config":{"default_dataset":"default","rois":[]}
            })
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.add",
            json!({"id": "ROI-1", "path": "/tmp/roi.zarr"})
        )
        .is_ok()
    );
    assert!(ControlCommand::decode("project.rois.add", json!({"id": "ROI-1"})).is_err());
    let replacement = json!({
        "id": "ROI-native",
        "source": {"Http": {"base_url": "https://example.test/roi.zarr"}},
        "display_name": "Native ROI",
        "mask_layers": [],
        "meta": {"cohort": "A"}
    });
    assert!(
        ControlCommand::decode(
            "project.rois.add",
            json!({"replacement": replacement.clone()})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.add",
            json!({"replacement": replacement.clone(), "id": "also-an-id"})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.update",
            json!({"target_id": "ROI-1", "changes": {}})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.update",
            json!({"target_id": "ROI-1", "replacement": replacement.clone()})
        )
        .is_ok()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.update",
            json!({
                "target_id": "ROI-1",
                "changes": {"display_name": "patch"},
                "replacement": replacement
            })
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode(
            "project.rois.select",
            json!({"ids": ["ROI-1"], "mode": "unexpected"})
        )
        .is_err()
    );
    assert!(
        ControlCommand::decode("mosaic.focus.set", json!({"index": 0, "roi_id": "ROI-1"})).is_err()
    );
    assert!(
        ControlCommand::decode("deep_links.parse", json!({"url": "odon://open?roi=ROI-1"})).is_ok()
    );
    assert!(
        ControlCommand::decode("deep_links.parse", json!({"url": "https://example.com"})).is_err()
    );
    assert!(
        ControlCommand::decode(
            "deep_links.apply",
            json!({"url": "odon://open", "request": {}})
        )
        .is_err()
    );
    assert!(ControlCommand::decode("deep_links.apply", json!({"request": {}})).is_ok());
}
