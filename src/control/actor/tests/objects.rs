use super::*;
#[test]
fn object_resources_load_and_clear_without_draining_the_ui_queue() {
    let channels = spawn_test_actor_with_objects();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (load, load_rx) = request(
        "viewer.objects.source.load",
        json!({"path":"objects.geojson","downsample_factor":2.0}),
    );
    channels.request_tx.send(load).unwrap();
    let loaded = load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(loaded["object_count"], 2);
    assert_eq!(loaded["resources_ready"], true);

    let (state, state_rx) = request("viewer.objects.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(state["state"]["object_count"], 2);
    assert_eq!(state["state"]["available_properties"][1], "phenotype");

    let (style, style_rx) = request(
        "viewer.objects.style.set",
        json!({"visible":true,"fill_cells":true,"fill_opacity":0.4}),
    );
    channels.request_tx.send(style).unwrap();
    let styled = style_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(styled["style"]["visible"], true);
    assert_eq!(styled["style"]["fill_cells"], true);

    let (visibility, visibility_rx) = request("viewer.objects.get_visibility", json!({}));
    channels.request_tx.send(visibility).unwrap();
    let visibility = visibility_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(visibility["mode"], "single");
    assert_eq!(visibility["overlay"]["segmentation_labels"], true);
    assert_eq!(visibility["overlay"]["segmentation_geojson"], false);
    assert_eq!(visibility["overlay"]["segmentation_objects"], true);
    assert_eq!(visibility["overlay"]["object_count"], 2);

    for target in ["all", "labels", "geojson", "objects"] {
        let visible = target != "all";
        let (set_visibility, set_visibility_rx) = request(
            "viewer.objects.set_visibility",
            json!({"target":target,"visible":visible}),
        );
        channels.request_tx.send(set_visibility).unwrap();
        let response = set_visibility_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(response["overlay"]["target"], target);
    }

    let (fast, fast_rx) = request(
        "viewer.objects.rendering.set_fast",
        json!({"enabled":false}),
    );
    channels.request_tx.send(fast).unwrap();
    assert_eq!(
        fast_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["enabled"],
        false
    );

    let (global_filter, global_filter_rx) = request(
        "viewer.objects.set_filter",
        json!({"query":"phenotype == 'immune'"}),
    );
    channels.request_tx.send(global_filter).unwrap();
    let global_filter = global_filter_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(global_filter["filter"]["visible_count"], 1);
    assert_eq!(global_filter["target"], "segmentation_objects");

    let (properties, properties_rx) = request(
        "viewer.objects.properties.list",
        json!({"offset":0,"limit":10}),
    );
    channels.request_tx.send(properties).unwrap();
    let properties = properties_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(properties["total"], 3);
    assert_eq!(properties["columns"][2]["numeric"], true);
    let (values, values_rx) = request(
        "viewer.objects.properties.values",
        json!({"property":"phenotype","offset":0,"limit":10}),
    );
    channels.request_tx.send(values).unwrap();
    let values = values_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(values["values"][0]["value"], "tumour");
    assert_eq!(values["values"][1]["value"], "immune");

    let (filter, filter_rx) = request(
        "viewer.viewports.objects.filter.set",
        json!({
            "viewport_id":"viewport-1",
            "mode":"query",
            "query":"phenotype == 'tumour'",
        }),
    );
    channels.request_tx.send(filter).unwrap();
    let filtered = filter_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(filtered["result"]["active"], true);
    assert_eq!(filtered["result"]["total_count"], 2);
    assert_eq!(filtered["result"]["visible_count"], 1);

    let (get_filter, get_filter_rx) = request(
        "viewer.viewports.objects.filter.get",
        json!({"viewport_id":"viewport-1"}),
    );
    channels.request_tx.send(get_filter).unwrap();
    let filter_state = get_filter_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        filter_state["result"]["query"]["text"],
        "phenotype == 'tumour'"
    );

    let (select_filtered, select_filtered_rx) = request(
        "viewer.objects.selection.select_filtered",
        json!({
            "target":"segmentation_objects",
            "filter_query":"phenotype == 'immune'",
            "mode":"replace",
        }),
    );
    channels.request_tx.send(select_filtered).unwrap();
    let standalone_selection = select_filtered_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(standalone_selection["matched_count"], 1);
    assert_eq!(standalone_selection["selection"]["primary"]["id"], "cell-b");

    let (select_ids, select_ids_rx) = request(
        "viewer.objects.selection.select_ids",
        json!({"target":"segmentation_objects","ids":["cell-a"]}),
    );
    channels.request_tx.send(select_ids).unwrap();
    let selected = select_ids_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(selected["selection"]["selection_count"], 1);
    assert_eq!(selected["selection"]["primary"]["id"], "cell-a");

    let (query, query_rx) = request(
        "viewer.objects.query_rect",
        json!({
            "target":"segmentation_objects",
            "viewport_id":"viewport-1",
            "rect":[-1.0,-1.0,11.0,11.0],
        }),
    );
    channels.request_tx.send(query).unwrap();
    let queried = query_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(queried["objects"]["query"]["match_count"], 1);
    assert_eq!(queried["objects"]["query"]["matches"][0]["id"], "cell-a");

    let (camera, camera_rx) = request(
        "viewer.camera.set",
        json!({"center_world_lvl0":[50.0,50.0],"zoom_screen_per_lvl0_px":2.0}),
    );
    channels.request_tx.send(camera).unwrap();
    camera_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    channels
        .model_tx
        .send(ActorModelUpdate::ViewportGeometry {
            viewport_id: "viewport-1".to_string(),
            x: 100.0,
            y: 200.0,
            width: 200.0,
            height: 100.0,
        })
        .unwrap();
    let mut observed_screen_rect = Value::Null;
    for _ in 0..100 {
        let (camera, response) = request("viewer.camera.get", json!({}));
        channels.request_tx.send(camera).unwrap();
        let value = response
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        if value["camera"]["viewport"]["screen_rect"] == json!([100.0, 200.0, 300.0, 300.0]) {
            observed_screen_rect = value;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        observed_screen_rect["camera"]["viewport"]["screen_rect"],
        json!([100.0, 200.0, 300.0, 300.0])
    );
    let (screen_query, screen_query_rx) = request(
        "viewer.objects.query_rect",
        json!({
            "target":"segmentation_objects",
            "viewport_id":"viewport-1",
            "screen_rect":[98.0,148.0,122.0,172.0],
        }),
    );
    channels.request_tx.send(screen_query).unwrap();
    let screen_query = screen_query_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(screen_query["objects"]["query"]["match_count"], 1);
    assert_eq!(
        screen_query["objects"]["query"]["matches"][0]["id"],
        "cell-a"
    );

    let (focus, focus_rx) = request(
        "viewer.objects.focus.set",
        json!({"target":"segmentation_objects","id":"cell-b","fit":false}),
    );
    channels.request_tx.send(focus).unwrap();
    let focused = focus_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(focused["focused"]["id"], "cell-b");
    assert_eq!(focused["selection_count"], 2);

    let (get_selection, get_selection_rx) =
        request("viewer.objects.get_selection", json!({"limit":10}));
    channels.request_tx.send(get_selection).unwrap();
    let selection = get_selection_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(selection["objects"]["selection"]["selection_count"], 2);
    assert_eq!(selection["objects"]["selection"]["primary"]["id"], "cell-b");

    let (clear_filter, clear_filter_rx) = request(
        "viewer.viewports.objects.filter.clear",
        json!({"viewport_id":"viewport-1"}),
    );
    channels.request_tx.send(clear_filter).unwrap();
    let cleared_filter = clear_filter_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(cleared_filter["result"]["active"], false);
    assert_eq!(cleared_filter["result"]["visible_count"], 2);

    let projection = channels.presentation_rx.try_recv().unwrap();
    let resource = &projection.workspace.as_ref().unwrap()["object_resource"];
    assert_eq!(resource["object_count"], 2);
    assert_eq!(resource["source"], "objects.geojson");
    assert_eq!(
        projection.workspace.as_ref().unwrap()["object_selection"]["selected_indices"],
        json!([0, 1])
    );
    assert_eq!(
        projection.workspace.as_ref().unwrap()["object_selection"]["primary_index"],
        1
    );
    let viewport = &projection.workspace.as_ref().unwrap()["viewports"][0];
    assert_eq!(viewport["objects"]["visible"], true);
    assert_eq!(
        viewport["object_overlay_visibility"]["segmentation_labels"],
        true
    );
    assert_eq!(
        viewport["object_overlay_visibility"]["segmentation_geojson"],
        true
    );

    let (clear, clear_rx) = request("viewer.objects.source.clear", json!({}));
    channels.request_tx.send(clear).unwrap();
    let cleared = clear_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(cleared["cleared"], true);
    assert_eq!(cleared["previous_count"], 2);
}
