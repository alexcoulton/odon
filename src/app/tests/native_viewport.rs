use super::*;
#[test]
fn native_viewport_commits_replay_through_the_actor_model() {
    let mut app = fixture_app();
    let before = app.viewport_workspace.as_ref().unwrap().clone();
    let left = before.active_id().clone();
    let mut after = before.clone();
    after
        .clone_viewport(&left, Some("Right".to_string()), ViewportLayout::Horizontal)
        .unwrap();
    after.set_links(ViewportLinks {
        camera: false,
        plane: true,
        selection: true,
    });
    after.rename(&left, "Left".to_string()).unwrap();
    let left_state = &mut after.get_mut(&left).unwrap().state;
    left_state.camera.center_world_lvl0 = egui::pos2(123.0, 456.0);
    left_state.camera.zoom_screen_per_lvl0_px = 2.5;
    for channel in &mut left_state.channels {
        channel.visible = channel.index == 1;
    }
    left_state.channels[1].color_rgb = [12, 34, 56];
    left_state.channel_layer_order.reverse();
    left_state.channel_sort_mode = ChannelSortMode::NameDesc;
    left_state
        .layer_groups
        .channel_groups
        .push(ProjectChannelGroup {
            id: 1,
            name: "Native group".to_string(),
            expanded: true,
            color_rgb: [20, 30, 40],
        });
    left_state.layer_groups.channel_members.insert(
        left_state.channels[1].name.clone(),
        ProjectChannelGroupMember {
            group_id: 1,
            inherit_color: true,
        },
    );
    left_state.object_visible = true;
    left_state.object_display.fill_cells = true;
    left_state.object_display.fill_opacity = 0.45;
    left_state.object_display.color_property_key = Some("phenotype".to_string());
    left_state.object_display.color_level_overrides.insert(
        "tumour".to_string(),
        crate::objects::ObjectColorLevelOverride {
            visible: true,
            color_rgb: Some([220, 40, 60]),
        },
    );

    app.record_native_viewport_intents(&before, &after);
    let intents = app.take_native_control_intents();
    assert!(
        intents
            .iter()
            .any(|intent| intent.method == "viewer.viewports.clone")
    );
    assert!(
        intents
            .iter()
            .any(|intent| intent.method == "viewer.viewports.camera.set")
    );
    assert!(
        intents
            .iter()
            .any(|intent| intent.method == "viewer.viewports.channels.set_color")
    );
    assert!(
        intents
            .iter()
            .any(|intent| intent.method == "viewer.viewports.channels.set_order")
    );
    assert!(intents.iter().any(|intent| {
        intent.method == "viewer.viewports.channels.set_group"
            && intent.params["replace_all"] == true
    }));
    assert!(intents.iter().any(|intent| {
        intent.method == "viewer.viewports.objects.style.set"
            && intent.params["color_property"] == "phenotype"
    }));
    assert!(
        intents
            .iter()
            .any(|intent| intent.method == "viewer.viewports.objects.legend.set")
    );

    let mut model = odon::model::AppModel::project();
    model.install_dataset(&app.dataset);
    for intent in intents {
        model
            .dispatch(intent.method, &intent.params)
            .expect("native intent is actor-owned")
            .expect("native intent applies");
    }
    let workspace = model
        .dispatch("viewer.workspace.get", &serde_json::json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(workspace["layout"], "horizontal");
    assert_eq!(workspace["links"]["camera"], false);
    let model_left = workspace["viewports"]
        .as_array()
        .unwrap()
        .iter()
        .find(|viewport| viewport["viewport_id"] == left.as_str())
        .unwrap();
    assert_eq!(model_left["title"], "Left");
    assert_eq!(
        model_left["camera"]["center_world_lvl0"],
        serde_json::json!([123.0, 456.0])
    );
    assert_eq!(
        model_left["channels"][1]["color_rgb"],
        serde_json::json!([12, 34, 56])
    );
    assert_eq!(
        model_left["channel_order"],
        serde_json::json!([4, 3, 2, 1, 0])
    );
    assert_eq!(model_left["channel_sort"], "name_desc");
    assert_eq!(model_left["channel_groups"][0]["name"], "Native group");
    assert_eq!(model_left["objects"]["visible"], true);
    assert_eq!(model_left["objects"]["fill_cells"], true);
    assert_eq!(model_left["objects"]["color_property"], "phenotype");
    assert_eq!(
        model_left["objects"]["color_level_overrides"]["tumour"]["color_rgb"],
        serde_json::json!([220, 40, 60])
    );
}
