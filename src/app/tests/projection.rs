use super::*;
#[test]
fn coalesced_actor_projection_replaces_the_renderer_workspace_atomically() {
    let mut app = fixture_app();
    let mut model = odon::model::AppModel::project();
    model.install_dataset(&app.dataset);
    let left = model.render_workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = model
        .dispatch(
            "viewer.viewports.clone",
            &serde_json::json!({
                "source_viewport_id": left,
                "layout": "horizontal",
                "ratio": 0.6,
                "title": "Marker B",
            }),
        )
        .unwrap()
        .unwrap()
        .response["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    for (viewport_id, channel, color) in [(&left, 1, [255, 0, 0]), (&right, 2, [0, 255, 0])] {
        model
            .dispatch(
                "viewer.viewports.channels.set_visible",
                &serde_json::json!({
                    "viewport_id": viewport_id,
                    "channels": [channel],
                    "mode": "only",
                }),
            )
            .unwrap()
            .unwrap();
        model
            .dispatch(
                "viewer.viewports.channels.set_color",
                &serde_json::json!({
                    "viewport_id": viewport_id,
                    "channel": channel,
                    "color_rgb": color,
                }),
            )
            .unwrap()
            .unwrap();
    }
    model
        .dispatch(
            "viewer.channels.set_note",
            &serde_json::json!({"channel":1,"note":"projection note"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.set_transform",
            &serde_json::json!({
                "channel":1,
                "offset_world":[5.0,-4.0],
                "scale":[1.1,0.9],
                "rotation_rad":0.2,
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_order",
            &serde_json::json!({
                "viewport_id":left,
                "channels":[4,3,2,1,0],
                "mode":"exact",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_group",
            &serde_json::json!({
                "viewport_id":right,
                "channels":[1,2],
                "name":"Projected group",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.presentation.set",
            &serde_json::json!({"search":"CD","sort":"visible_first"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewport_links.set",
            &serde_json::json!({"camera": true, "plane": true, "selection": true}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.ui.set_right_tab",
            &serde_json::json!({"tab":"measurements"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.objects.style.set",
            &serde_json::json!({
                "viewport_id":left,
                "visible":true,
                "fill_cells":true,
                "fill_opacity":0.25,
                "color_property":"marker_a",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.objects.style.set",
            &serde_json::json!({
                "viewport_id":right,
                "visible":true,
                "fill_cells":true,
                "fill_opacity":0.75,
                "color_property":"marker_b",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.objects.legend.set",
            &serde_json::json!({
                "viewport_id":left,
                "entries":[{"value":"positive","color_rgb":[255,0,0]}],
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.camera.fit",
            &serde_json::json!({"viewport_id": left}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.objects.set_visibility",
            &serde_json::json!({"target":"labels","visible":false}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.objects.set_visibility",
            &serde_json::json!({"target":"geojson","visible":true}),
        )
        .unwrap()
        .unwrap();
    let mask_id = model
        .dispatch(
            "viewer.masks.layers.create",
            &serde_json::json!({"name":"Projected mask"}),
        )
        .unwrap()
        .unwrap()
        .response["id"]
        .as_u64()
        .unwrap();
    model
        .dispatch(
            "viewer.masks.polygons.add",
            &serde_json::json!({
                "id":mask_id,
                "vertices":[[10.0,20.0],[30.0,20.0],[30.0,40.0]],
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.masks.selection.set",
            &serde_json::json!({"id":mask_id,"index":0,"vertex_index":1}),
        )
        .unwrap()
        .unwrap();

    let projection = model.render_workspace_snapshot().unwrap();
    app.apply_control_actor_workspace_projection(&projection)
        .expect("latest actor projection applies without command replay");
    let rendered = app.control_viewport_workspace_snapshot();
    assert_eq!(rendered["layout"], "horizontal");
    assert_eq!(rendered["ratio"], projection["ratio"]);
    assert_eq!(rendered["active_viewport_id"], right);
    assert_eq!(rendered["links"], projection["links"]);
    assert_eq!(app.right_tab, RightTab::Measurements);
    assert_eq!(rendered["masks"]["layers"], projection["masks"]["layers"]);
    assert_eq!(
        rendered["masks"]["selection"],
        projection["masks"]["selection"]
    );
    assert_eq!(rendered["channel_metadata"], projection["channel_metadata"]);
    assert_eq!(
        rendered["channel_transforms"],
        projection["channel_transforms"]
    );
    assert_eq!(
        rendered["channel_presentation"],
        projection["channel_presentation"]
    );
    for viewport_id in [&left, &right] {
        let expected = projection["viewports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|viewport| viewport["viewport_id"] == *viewport_id)
            .unwrap();
        let actual = rendered["viewports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|viewport| viewport["viewport_id"] == *viewport_id)
            .unwrap();
        assert_eq!(actual["title"], expected["title"]);
        assert_eq!(
            actual["camera"]["center_world_lvl0"],
            expected["camera"]["center_world_lvl0"]
        );
        assert_eq!(
            actual["camera"]["zoom_screen_per_lvl0_px"],
            expected["camera"]["zoom_screen_per_lvl0_px"]
        );
        assert_eq!(actual["plane"], expected["plane"]);
        assert_eq!(actual["channels"], expected["channels"]);
        assert_eq!(actual["channel_order"], expected["channel_order"]);
        assert_eq!(actual["channel_sort"], expected["channel_sort"]);
        assert_eq!(actual["channel_groups"], expected["channel_groups"]);
        assert_eq!(actual["objects"], expected["objects"]);
        assert_eq!(
            actual["object_overlay_visibility"],
            expected["object_overlay_visibility"]
        );
        assert_eq!(actual["rendering"], expected["rendering"]);
        assert_eq!(
            actual["navigation_revision"],
            expected["navigation_revision"]
        );
        assert_eq!(
            actual["presentation_revision"],
            expected["presentation_revision"]
        );
    }
}
