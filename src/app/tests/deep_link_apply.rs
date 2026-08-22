use super::*;
#[test]
fn deep_link_application_updates_channels_groups_contrast_and_camera() {
    let mut app = fixture_app();
    let request = DeepLinkRequest::parse_arg(
        "odon://open?channel=CD3&visible_channels=PanCK%7CCD3&channel_order=listed&group_visible_channels=1&visible_channel_group=T%20cell%20markers&visible_channel_group_color=%23abcdef&hidden_channels=DAPI&channel_color=CD3:%23112233&channel_contrast=CD3:100:1000&center=12.5,25&zoom=0.5&fast_rendering=0",
    )
    .expect("parse deep link")
    .expect("Odon deep link");

    app.apply_deep_link_request(&request);

    assert_eq!(app.control_active_channel_snapshot()["name"], "CD3");
    assert_eq!(visible_channel_names(&app), vec!["CD3", "PanCK"]);
    assert_eq!(&app.channel_layer_order[..2], &[2, 1]);
    assert_eq!(app.channels[1].color_rgb, [0x11, 0x22, 0x33]);
    assert_eq!(app.channels[1].window, Some((100.0, 1000.0)));
    assert_eq!(app.camera.center_world_lvl0, egui::pos2(12.5, 25.0));
    assert_eq!(app.camera.zoom_screen_per_lvl0_px, 0.5);
    assert!(!app.fast_object_rendering);

    let groups = app.current_layer_groups();
    let group = groups
        .channel_groups
        .iter()
        .find(|group| group.name == "T cell markers")
        .expect("deep-link channel group");
    assert_eq!(group.color_rgb, [0xab, 0xcd, 0xef]);
    assert_eq!(groups.channel_members["PanCK"].group_id, group.id);
    assert_eq!(groups.channel_members["CD3"].group_id, group.id);
    assert!(!groups.channel_members["CD3"].inherit_color);
}
