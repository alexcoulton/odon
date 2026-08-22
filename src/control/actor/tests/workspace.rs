use super::*;
#[test]
fn actor_opens_and_configures_viewports_without_draining_the_ui_queue() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    let opened = open_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("actor reply")
        .unwrap();
    assert_eq!(opened["resources_ready"], true);

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let left = workspace_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let (clone, clone_rx) = request(
        "viewer.viewports.clone",
        json!({"source_viewport_id":left,"layout":"horizontal"}),
    );
    channels.request_tx.send(clone).unwrap();
    let cloned = clone_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let right = cloned["viewport_id"].as_str().unwrap().to_string();
    let (layers, layers_rx) = request("viewer.viewports.layers.list", json!({"viewport_id":right}));
    channels.request_tx.send(layers).unwrap();
    let layers = layers_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(layers["result"].as_array().unwrap().len(), 5);
    let (hide, hide_rx) = request(
        "viewer.viewports.layers.set_visibility",
        json!({"viewport_id":right,"layer_id":"channel:0","visible":false}),
    );
    channels.request_tx.send(hide).unwrap();
    assert_eq!(
        hide_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["result"]["layer"]["visible"],
        false
    );
    let (recolor, recolor_rx) = request(
        "viewer.viewports.layers.set",
        json!({
            "viewport_id":left,
            "layer_id":"channel:1",
            "presentation":{"color_rgb":[12,34,56]},
        }),
    );
    channels.request_tx.send(recolor).unwrap();
    assert_eq!(
        recolor_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["result"]["layer"]["presentation"]["color_rgb"],
        json!([12, 34, 56])
    );
    let (offset, offset_rx) = request(
        "viewer.native_layers.set_offset",
        json!({"layer_id":"channel:0","offset_world":[4.0,5.0]}),
    );
    channels.request_tx.send(offset).unwrap();
    assert_eq!(
        offset_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["result"]["layer"]["offset_world"],
        json!([4.0, 5.0])
    );
    for (viewport_id, fill_opacity, property) in [
        (left.as_str(), 0.2, "marker_a"),
        (right.as_str(), 0.8, "marker_b"),
    ] {
        let (style, style_rx) = request(
            "viewer.viewports.objects.style.set",
            json!({
                "viewport_id":viewport_id,
                "visible":true,
                "fill_cells":true,
                "fill_opacity":fill_opacity,
                "color_property":property,
            }),
        );
        channels.request_tx.send(style).unwrap();
        let styled = style_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(styled["result"]["style"]["color_property"], property);
    }
    let (legend, legend_rx) = request(
        "viewer.viewports.objects.legend.set",
        json!({
            "viewport_id":left,
            "entries":[{"value":"positive","color_rgb":[255,0,0]}],
        }),
    );
    channels.request_tx.send(legend).unwrap();
    legend_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (fit, fit_rx) = request("viewer.viewports.camera.fit", json!({"viewport_id":right}));
    channels.request_tx.send(fit).unwrap();
    fit_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (scale_bar, scale_bar_rx) = request("viewer.scale_bar.set", json!({"visible":false}));
    channels.request_tx.send(scale_bar).unwrap();
    assert_eq!(
        scale_bar_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["visible"],
        false
    );
    let (levels, levels_rx) = request("viewer.thresholds.levels.list", json!({}));
    channels.request_tx.send(levels).unwrap();
    let levels = levels_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(levels["levels"].as_array().unwrap().len(), 4);
    assert_eq!(levels["levels"][0]["width"], 512);
    assert_eq!(levels["default_full_level"], 0);
    let (tab, tab_rx) = request("viewer.ui.set_right_tab", json!({"tab":"measurements"}));
    channels.request_tx.send(tab).unwrap();
    assert_eq!(
        tab_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["tab"]["right_tab"],
        "measurements"
    );
    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(state["mode"], "single");
    assert_eq!(state["view"]["channel_count"], 5);
    assert_eq!(state["view"]["dataset_descriptor"]["pyramid_levels"], 4);
    let (rendering, rendering_rx) = request("viewer.rendering.get_state", json!({}));
    channels.request_tx.send(rendering).unwrap();
    let rendering = rendering_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(rendering["mode"], "single");
    assert_eq!(rendering["compositing"], "additive");
    assert_eq!(
        rendering["deterministic_capture"]["readiness"]["mode"],
        "single"
    );
    assert_eq!(channels.legacy_rx.len(), 0);
    assert_eq!(channels.presentation_rx.len(), 1);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.mode, ModelMode::Single);
    assert!(projection.document.is_some());
    assert_eq!(projection.document_generation, 1);
    let projected_viewports = projection.workspace.as_ref().unwrap()["viewports"]
        .as_array()
        .unwrap();
    let left_projection = projected_viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == left)
        .unwrap();
    let right_projection = projected_viewports
        .iter()
        .find(|viewport| viewport["viewport_id"] == right)
        .unwrap();
    assert!((left_projection["objects"]["fill_opacity"].as_f64().unwrap() - 0.2).abs() < 1.0e-6);
    assert!(
        (right_projection["objects"]["fill_opacity"]
            .as_f64()
            .unwrap()
            - 0.8)
            .abs()
            < 1.0e-6
    );
    assert_eq!(
        left_projection["objects"]["color_level_overrides"]["positive"]["color_rgb"],
        json!([255, 0, 0])
    );
    assert_eq!(
        left_projection["native_layers"][0]["offset_world"],
        json!([4.0, 5.0])
    );
    assert_eq!(right_projection["native_layers"][0]["visible"], false);
    assert_eq!(right_projection["rendering"]["show_scale_bar"], false);
    assert_eq!(
        projection.workspace.as_ref().unwrap()["ui"]["right_tab"],
        "measurements"
    );
    let (canonical, canonical_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(canonical).unwrap();
    let mut canonical = canonical_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    canonical.as_object_mut().unwrap().remove("_control");
    assert_eq!(projection.workspace.unwrap(), canonical);

    let (loading, loading_rx) = request("app.get_loading_state", json!({}));
    channels.request_tx.send(loading).unwrap();
    let loading = loading_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(loading["loading"]["resources_ready"], true);
    assert_eq!(loading["loading"]["presentation_ready"], false);
    let projection_revision = loading["loading"]["projection_revision"].as_u64().unwrap();

    channels
        .model_tx
        .send(ActorModelUpdate::PresentationApplied(projection_revision))
        .unwrap();
    let (presented, presented_rx) = request("app.get_loading_state", json!({}));
    channels.request_tx.send(presented).unwrap();
    assert_eq!(
        presented_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["loading"]["presentation_ready"],
        true
    );

    let (show_project, show_project_rx) = request("app.navigation.show_project", json!({}));
    channels.request_tx.send(show_project).unwrap();
    let shown = show_project_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(shown["mode"], "project");
    assert_eq!(shown["changed"], true);
    assert_eq!(channels.legacy_rx.len(), 0);
    assert_eq!(
        channels
            .presentation_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .mode,
        ModelMode::Project
    );

    let diagnostics = execution_diagnostics(&channels.diagnostics);
    assert_eq!(diagnostics["metrics"]["alive"], true);
    assert!(
        diagnostics["metrics"]["requests"]["actor"]
            .as_u64()
            .unwrap()
            >= 6
    );
    assert_eq!(diagnostics["metrics"]["requests"]["legacy_ui"], 0);
    assert!(
        diagnostics["metrics"]["timing_ms"]["queue_wait"]["samples"]
            .as_u64()
            .unwrap()
            >= 6
    );
    assert_eq!(
        diagnostics["method_routes"]["viewer.viewports.camera.fit"],
        "actor"
    );
    assert_eq!(diagnostics["method_routes"]["viewer.panels.get"], "actor");
}
