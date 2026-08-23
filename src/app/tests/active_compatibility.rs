use super::*;

#[test]
fn actor_active_view_compatibility_methods_project_into_the_renderer() {
    let mut app = fixture_actor_app();

    let channels = app.actor_query("viewer.channels.list", serde_json::json!({}));
    assert_eq!(channels["channels"], app.control_channel_snapshot());
    let visible = app.actor_query("viewer.channels.list_visible", serde_json::json!({}));
    assert_eq!(visible["channels"], app.control_visible_channel_snapshot());
    let active = app.actor_query("viewer.channels.get_active", serde_json::json!({}));
    assert_eq!(
        active["active_channel"],
        app.control_active_channel_snapshot()
    );

    let visible = app.actor_command(
        "viewer.channels.set_visible",
        serde_json::json!({"channels":["CD3","PanCK"],"mode":"only"}),
    );
    assert_eq!(visible["result"]["changed"], true);
    assert_eq!(
        app.control_visible_channel_snapshot(),
        app.actor_query("viewer.channels.list_visible", serde_json::json!({}))["channels"]
    );

    let active = app.actor_command(
        "viewer.channels.set_active",
        serde_json::json!({"channel":"PanCK"}),
    );
    assert_eq!(active["result"]["active_channel"]["name"], "PanCK");
    assert_eq!(app.control_active_channel_snapshot()["name"], "PanCK");

    let contrast = app.actor_command(
        "viewer.channels.set_contrast",
        serde_json::json!({"channel":"PanCK","min":100.0,"max":1000.0}),
    );
    assert_eq!(contrast["contrast"]["max"], 1000.0);
    assert_eq!(app.channels[2].window, Some((100.0, 1000.0)));

    let note = app.actor_command(
        "viewer.channels.set_note",
        serde_json::json!({"channel":"PanCK","note":"epithelial marker"}),
    );
    assert_eq!(note["channel"]["note"], "epithelial marker");
    assert_eq!(app.channels[2].note, "epithelial marker");

    let transform = app.actor_command(
        "viewer.channels.set_transform",
        serde_json::json!({
            "channel":"PanCK",
            "offset_world":[4.0,-2.0],
            "scale":[1.2,0.8],
            "rotation_rad":0.25,
        }),
    );
    assert_eq!(transform["changed"], true);
    let selector = serde_json::json!({"channel":"PanCK"});
    assert_eq!(
        app.actor_query("viewer.channels.get_transform", selector.clone()),
        app.channel_transform_snapshot(2)
    );

    let order = app.actor_command(
        "viewer.channels.set_order",
        serde_json::json!({"channels":[4,3,2,1,0],"mode":"exact"}),
    );
    assert_eq!(order["changed"], true);
    assert_eq!(app.channel_layer_order, vec![4, 3, 2, 1, 0]);

    let presentation = app.actor_command(
        "viewer.channels.presentation.set",
        serde_json::json!({"search":"CD","sort":"visible_first"}),
    );
    assert_eq!(presentation["search"], "CD");
    assert_eq!(
        app.actor_query("viewer.channels.presentation.get", serde_json::json!({})),
        app.control_channel_presentation_json()
    );

    let group = app.actor_command(
        "viewer.channels.set_group",
        serde_json::json!({
            "channels":["CD3","PanCK"],
            "name":"Comparison markers",
            "color_rgb":[20,40,60],
        }),
    );
    assert_eq!(group["result"]["changed"], true);
    assert_eq!(
        app.actor_query("viewer.channels.list_groups", serde_json::json!({}))["groups"],
        app.control_channel_groups_snapshot()
    );

    let reset = app.actor_command("viewer.channels.reset_transform", selector);
    assert_eq!(
        reset["transform"]["offset_world"],
        serde_json::json!([0.0, 0.0])
    );

    let camera = app.actor_command(
        "viewer.camera.set",
        serde_json::json!({"center_x":123.0,"center_y":234.0,"zoom":2.5}),
    );
    assert_eq!(
        camera["camera"]["center_world_lvl0"],
        serde_json::json!([123.0, 234.0])
    );
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(123.0, 234.0));

    let plane = app.actor_command(
        "viewer.planes.set",
        serde_json::json!({"mode":"xy","slice":99}),
    );
    assert_eq!(plane["result"]["plane"]["slice"], 0);
    let stepped = app.actor_command("viewer.planes.next", serde_json::json!({"step":1}));
    assert_eq!(stepped["result"]["changed"], false);
    assert_eq!(
        app.actor_query("viewer.planes.get", serde_json::json!({}))["plane"],
        app.control_plane_snapshot()
    );

    let smooth = app.actor_command(
        "viewer.rendering.set_smooth_pixels",
        serde_json::json!({"smooth":false}),
    );
    assert_eq!(smooth["result"]["smooth_pixels"]["smooth"], false);
    assert!(!app.smooth_pixels);

    let panels = app.actor_command(
        "viewer.panels.set",
        serde_json::json!({"left":false,"right":true}),
    );
    assert_eq!(
        panels["result"]["panels"],
        serde_json::json!({"left":false,"right":true})
    );
    assert_eq!(
        app.actor_query("viewer.panels.get", serde_json::json!({}))["panels"],
        app.control_side_panels_snapshot()
    );
}
