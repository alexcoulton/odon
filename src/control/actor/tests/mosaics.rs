use super::*;
use crate::data::dataset_source::DatasetSource;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn call(channels: &ControlActorChannels, method: &str, params: Value) -> Value {
    let (request, response) = request(method, params);
    channels.request_tx.send(request).unwrap();
    response
        .recv_timeout(Duration::from_secs(15))
        .unwrap_or_else(|error| panic!("{method} did not complete without a frame: {error}"))
        .unwrap_or_else(|error| panic!("{method} failed: {error:?}"))
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/synthetic_5ch.ome.zarr")
        .canonicalize()
        .unwrap()
}

fn temp_dir(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "odon-actor-mosaic-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn complete_mosaic_surface_executes_without_a_render_frame() {
    let directory = temp_dir("surface");
    let fixture = fixture_path();
    let sheet = directory.join("mosaic.csv");
    fs::write(
        &sheet,
        format!(
            "id,path,cohort,segpath\nROI-A,{},A,objects-a.geojson\nROI-B,{},B,objects-b.geojson\nROI-C,{},A,objects-c.geojson\n",
            fixture.display(),
            fixture.display(),
            fixture.display(),
        ),
    )
    .unwrap();

    let channels = spawn_test_actor_with_objects();
    let opened = call(
        &channels,
        "datasets.open_mosaic_samplesheet",
        json!({"path":sheet,"columns":2}),
    );
    assert_eq!(opened["mode"], "mosaic");
    assert_eq!(opened["roi_count"], 3);
    assert_eq!(opened["presentation_ready"], false);

    assert_eq!(
        call(&channels, "app.get_state", json!({}))["mode"],
        "mosaic"
    );
    let tile_policy = call(
        &channels,
        "memory.tiles.set",
        json!({
            "cache_mode":"custom",
            "cache_budget_bytes":268435456,
            "channel_history":"current_only",
        }),
    );
    assert_eq!(tile_policy["cache_mode"], "custom");
    assert_eq!(tile_policy["cache_budget_bytes"], 268435456);
    assert_eq!(tile_policy["channel_history"], "current_only");
    assert_eq!(
        call(&channels, "memory.tiles.get", json!({}))["cache_mode"],
        "custom"
    );
    let channel_state = call(&channels, "viewer.channels.list", json!({}));
    assert_eq!(channel_state["mode"], "mosaic");
    assert_eq!(channel_state["channels"].as_array().unwrap().len(), 5);
    call(
        &channels,
        "viewer.channels.set_visible",
        json!({"channels":[0,1],"mode":"only"}),
    );
    assert_eq!(
        call(&channels, "viewer.channels.list_visible", json!({}))["channels"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    call(&channels, "viewer.channels.set_active", json!({"index":1}));
    assert_eq!(
        call(&channels, "viewer.channels.get_active", json!({}))["active_channel"]["index"],
        1
    );
    call(
        &channels,
        "viewer.channels.set_contrast",
        json!({"index":1,"min":10.0,"max":200.0}),
    );
    assert_eq!(
        call(
            &channels,
            "viewer.channels.get_contrast",
            json!({"index":1})
        )["contrast"]["max"],
        200.0
    );
    let batch_contrast = call(
        &channels,
        "viewer.channels.set_contrast",
        json!({"channels":[0,2,4],"min":20.0,"max":300.0}),
    );
    assert_eq!(batch_contrast["contrast"]["count"], 3);
    for index in [0, 2, 4] {
        assert_eq!(
            call(
                &channels,
                "viewer.channels.get_contrast",
                json!({"index":index})
            )["contrast"]["max"],
            300.0
        );
    }
    let distinct_contrast = call(
        &channels,
        "viewer.channels.set_contrast",
        json!({"windows":[
            {"index":0,"min":5.0,"max":250.0},
            {"index":2,"min":15.0,"max":250.0},
        ]}),
    );
    assert_eq!(distinct_contrast["contrast"]["count"], 2);
    assert_eq!(distinct_contrast["contrast"]["windows"][0]["min"], 5.0);
    assert_eq!(distinct_contrast["contrast"]["windows"][1]["min"], 15.0);
    call(
        &channels,
        "viewer.channels.set_color",
        json!({"index":1,"color_rgb":[1,2,3]}),
    );
    call(
        &channels,
        "viewer.channels.set_note",
        json!({"index":1,"note":"actor mosaic"}),
    );
    call(
        &channels,
        "viewer.channels.set_order",
        json!({"order":[4,3,2,1,0]}),
    );
    call(
        &channels,
        "viewer.channels.presentation.set",
        json!({"search":"CD","sort":"name_asc"}),
    );
    let presentation = call(&channels, "viewer.channels.presentation.get", json!({}));
    assert_eq!(presentation["presentation"]["search"], "CD");
    assert_eq!(presentation["presentation"]["sort"], "name_asc");
    call(
        &channels,
        "viewer.channels.set_group",
        json!({"name":"Markers","channels":[0,1],"color_rgb":[9,8,7]}),
    );
    let groups = call(&channels, "viewer.channels.list_groups", json!({}));
    assert_eq!(groups["groups"][0]["name"], "Markers");
    assert_eq!(groups["groups"][0]["members"].as_array().unwrap().len(), 2);

    let layers = call(&channels, "viewer.native_layers.list", json!({}));
    assert_eq!(layers["layers"].as_array().unwrap().len(), 7);
    assert_eq!(
        call(
            &channels,
            "viewer.native_layers.get",
            json!({"layer_id":"text_labels"})
        )["layer"]["kind"],
        "text_labels"
    );
    call(
        &channels,
        "viewer.native_layers.set_visibility",
        json!({"layer_id":"text_labels","visible":false}),
    );
    call(
        &channels,
        "viewer.native_layers.set_active",
        json!({"layer_id":"channel:1"}),
    );
    call(
        &channels,
        "viewer.native_layers.set_order",
        json!({"stack":"overlays","layers":["text_labels","segmentation_geojson"]}),
    );

    let screenshot = call(
        &channels,
        "viewer.screenshot.settings.set",
        json!({"output_dir":directory,"include_legend":false,"legend_scale":1.5}),
    );
    assert_eq!(screenshot["include_legend"], false);
    assert_eq!(
        call(&channels, "viewer.screenshot.settings.get", json!({}))["legend_scale"],
        1.5
    );

    assert_eq!(call(&channels, "memory.get", json!({}))["mode"], "mosaic");
    let pinned = call(
        &channels,
        "memory.pin",
        json!({"level":0,"channels":[0],"scope":"focused","force":true}),
    );
    assert_eq!(pinned["completed"], true);
    assert_eq!(
        pinned["memory"]["items"][0]["levels"][0]["status"],
        "loaded"
    );
    let unpinned = call(
        &channels,
        "memory.unpin",
        json!({"level":0,"scope":"focused"}),
    );
    assert_eq!(unpinned["unloaded_items"], 1);
    assert_eq!(
        call(&channels, "memory.unpin_all", json!({}))["unloaded_item_levels"],
        0
    );
    call(
        &channels,
        "viewer.camera.set",
        json!({"center_world_lvl0":[12.0,34.0],"zoom":0.25}),
    );
    assert_eq!(
        call(&channels, "viewer.camera.get", json!({}))["camera"]["center_world_lvl0"],
        json!([12.0, 34.0])
    );
    call(&channels, "viewer.camera.zoom_in", json!({"factor":2.0}));
    call(&channels, "viewer.camera.zoom_out", json!({"factor":2.0}));
    assert!(call(&channels, "viewer.camera.fit", json!({}))["camera"].is_object());
    call(
        &channels,
        "viewer.panels.set",
        json!({"left":false,"right":true}),
    );
    assert_eq!(
        call(&channels, "viewer.panels.get", json!({}))["panels"]["left"],
        false
    );
    call(
        &channels,
        "viewer.rendering.set_smooth_pixels",
        json!({"smooth":false}),
    );
    assert_eq!(
        call(&channels, "viewer.rendering.get_smooth_pixels", json!({}))["smooth_pixels"]["smooth"],
        false
    );
    assert_eq!(
        call(&channels, "viewer.rendering.get_state", json!({}))["mode"],
        "mosaic"
    );
    call(
        &channels,
        "viewer.objects.set_visibility",
        json!({"target":"objects","visible":true}),
    );
    assert_eq!(
        call(
            &channels,
            "viewer.objects.get_visibility",
            json!({"target":"objects"})
        )["overlay"]["segmentation_objects"],
        true
    );
    call(
        &channels,
        "viewer.objects.rendering.set_fast",
        json!({"enabled":false}),
    );
    assert_eq!(
        call(&channels, "viewer.objects.rendering.get_fast", json!({}))["enabled"],
        false
    );

    let state = call(&channels, "mosaic.get_state", json!({}));
    assert_eq!(state["mosaic"]["roi_count"], 3);
    assert_eq!(state["mosaic"]["layout"]["columns"], 2);
    assert_eq!(state["mosaic"]["focused"]["roi_id"], "ROI-A");

    let tab = call(
        &channels,
        "mosaic.ui.set_right_tab",
        json!({"tab":"layout"}),
    );
    assert_eq!(tab["tab"]["right_tab"], "layout");
    let tab = call(
        &channels,
        "mosaic.ui.set_left_tab",
        json!({"tab":"project"}),
    );
    assert_eq!(tab["tab"]["left_tab"], "project");
    let rendering = call(
        &channels,
        "mosaic.rendering.set",
        json!({"smooth_pixels":true,"show_tile_debug":true}),
    );
    assert_eq!(rendering["result"]["rendering"]["show_tile_debug"], true);
    assert_eq!(
        call(&channels, "viewer.rendering.get_state", json!({}))["show_tile_debug"],
        true
    );
    let layout = call(
        &channels,
        "mosaic.layout.configure",
        json!({
            "group_by":"cohort",
            "sort_by":"id",
            "layout":"native_pixels",
            "columns":1,
            "fit":true,
        }),
    );
    assert_eq!(layout["layout"]["layout"], "native_pixels");
    assert_eq!(layout["layout"]["columns"], 1);

    let items = call(
        &channels,
        "mosaic.items.list",
        json!({"offset":0,"limit":2}),
    );
    assert_eq!(items["result"]["total"], 3);
    assert_eq!(items["result"]["items"].as_array().unwrap().len(), 2);
    assert_eq!(items["result"]["has_more"], true);

    let selection = call(&channels, "mosaic.selection.set", json!({"mode":"all"}));
    assert_eq!(selection["selection"]["count"], 3);
    assert_eq!(
        call(&channels, "mosaic.selection.get", json!({}))["selection"]["count"],
        3
    );
    assert_eq!(
        call(&channels, "mosaic.selection.clear", json!({}))["selection"]["count"],
        0
    );
    call(
        &channels,
        "mosaic.selection.set",
        json!({"mode":"replace","ids":["ROI-A","ROI-B"]}),
    );

    let focused = call(
        &channels,
        "mosaic.focus.set",
        json!({"roi_id":"ROI-B","fit":true}),
    );
    assert_eq!(focused["result"]["focused"]["roi_id"], "ROI-B");
    assert_eq!(
        call(&channels, "mosaic.focus.get", json!({}))["focused"]["roi_id"],
        "ROI-B"
    );
    assert!(
        call(
            &channels,
            "mosaic.focus.next",
            json!({"step":1,"wrap":true})
        )["result"]["focused"]
            .is_object()
    );
    assert!(
        call(
            &channels,
            "mosaic.focus.previous",
            json!({"step":1,"wrap":true})
        )["result"]["focused"]
            .is_object()
    );
    assert!(call(&channels, "mosaic.focus.fit", json!({}))["result"]["camera"].is_object());
    assert!(call(&channels, "mosaic.fit_all", json!({}))["result"]["camera"].is_object());
    assert_eq!(
        call(&channels, "mosaic.focus.clear", json!({}))["result"]["focused"],
        Value::Null
    );

    let loaded = call(&channels, "mosaic.objects.load_selected", json!({}));
    assert_eq!(loaded["settled"], true);
    assert_eq!(loaded["loaded"], 2);
    let objects = call(&channels, "mosaic.objects.get_state", json!({}));
    assert_eq!(objects["objects"]["loaded_count"], 2);
    assert_eq!(objects["objects"]["settled"], true);
    assert_eq!(objects["objects"]["property_loading"], "lazy_on_demand");
    assert_eq!(objects["objects"]["property_cache"]["policy"], "unbounded");
    let property_cache = call(
        &channels,
        "mosaic.objects.property_cache.set",
        json!({"policy":"lru","capacity":2}),
    );
    assert_eq!(property_cache["result"]["cache"]["capacity"], 2);
    assert_eq!(
        call(&channels, "mosaic.objects.property_cache.get", json!({}))["objects"]["policy"],
        "lru"
    );
    let explicit = call(
        &channels,
        "mosaic.objects.load",
        json!({"roi_ids":["ROI-A"],"downsample_factor":2.0}),
    );
    assert_eq!(explicit["settled"], true);
    assert_eq!(explicit["loaded"], 1);
    let style = call(
        &channels,
        "mosaic.objects.style.set",
        json!({"style":{
            "fill_cells":true,
            "opacity":0.42,
            "continuous_domains_by_roi":{
                "ROI-A":[1.0,10.0],
                "ROI-B":[100.0,1000.0],
            },
        }}),
    );
    assert_eq!(style["result"]["style"]["fill_cells"], true);
    assert_eq!(
        style["result"]["style"]["continuous_domains_by_roi"]["ROI-A"],
        json!([1.0, 10.0])
    );
    let (invalid_style, invalid_style_reply) = request(
        "mosaic.objects.style.set",
        json!({"style":{"continuous_domains_by_roi":{"missing-roi":[0.0,1.0]}}}),
    );
    channels.request_tx.send(invalid_style).unwrap();
    let invalid_style_error = invalid_style_reply
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert!(invalid_style_error.message.contains("was not found"));
    assert_eq!(
        call(&channels, "mosaic.objects.style.get", json!({}))["objects"]["style"]["opacity"],
        0.42
    );
    let selected = call(
        &channels,
        "mosaic.objects.selection.replace",
        json!({"roi_id":"ROI-A","state":{"selected_indices":[0],"primary_index":0}}),
    );
    assert_eq!(
        selected["result"]["result"]["selection"]["selection_count"],
        1
    );
    assert_eq!(
        call(
            &channels,
            "mosaic.objects.selection.get",
            json!({"roi_id":"ROI-A"})
        )["objects"]["selection"]["selection_count"],
        1
    );
    call(
        &channels,
        "mosaic.objects.selection.replace",
        json!({"roi_id":"ROI-B","state":{"selected_indices":[0],"primary_index":0}}),
    );
    call(
        &channels,
        "mosaic.objects.selection.replace",
        json!({
            "roi_id":"ROI-A",
            "state":{"selected_indices":[0],"primary_index":0},
            "clear_others":true,
        }),
    );
    assert_eq!(
        call(
            &channels,
            "mosaic.objects.selection.get",
            json!({"roi_id":"ROI-B"})
        )["objects"]["selection"]["selection_count"],
        0
    );
    assert_eq!(
        call(
            &channels,
            "mosaic.objects.selection.clear",
            json!({"roi_id":"ROI-A"})
        )["result"]["result"]["selection"]["selection_count"],
        0
    );
    let cancelled = call(&channels, "mosaic.objects.cancel_load", json!({}));
    assert_eq!(cancelled["result"]["cancelled_requests"], 0);

    assert_eq!(
        channels.diagnostics.legacy_requests.load(Ordering::Relaxed),
        0
    );
    assert!(channels.presentation_rx.len() <= 1);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn selected_project_rois_open_through_the_same_actor_resource_path() {
    let fixture = fixture_path();
    let mut first = ProjectRoi {
        id: "first".to_string(),
        display_name: Some("First ROI".to_string()),
        ..ProjectRoi::default()
    };
    first.set_dataset_source(DatasetSource::Local(fixture.clone()));
    let mut second = ProjectRoi {
        id: "second".to_string(),
        display_name: Some("Second ROI".to_string()),
        ..ProjectRoi::default()
    };
    second.set_dataset_source(DatasetSource::Local(fixture));
    let mut project = ProjectModelSnapshot::default();
    project.rois = vec![first.clone(), second.clone()];
    project.selected_source_keys = project
        .rois
        .iter()
        .filter_map(ProjectRoi::source_key)
        .collect();
    project.state = json!({
        "mosaic":{
            "channel_order":[4,3,2,1,0],
            "channels":[
                {"name":"DAPI","visible":false,"color_rgb":[1,2,3],"window":[10.0,100.0],"note":"restored"}
            ],
            "active_channel":1,
            "group_by":"",
            "sort_by":"id",
            "layout_mode":"native_pixels",
            "columns":1,
            "show_text_labels":false,
            "camera":{"center_world_lvl0":[12.0,34.0],"zoom_screen_per_lvl0_px":0.25},
            "ui":{
                "show_left_panel":false,
                "show_right_panel":true,
                "left_tab":"project",
                "right_tab":"layout",
                "channel_sort":"name_desc",
                "smooth_pixels":false,
                "show_tile_debug":true
            }
        }
    });

    let channels = spawn_test_actor();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(project))
        .unwrap();
    let opened = call(&channels, "project.rois.open_selected_mosaic", json!({}));
    assert_eq!(opened["mode"], "mosaic");
    assert_eq!(opened["roi_count"], 2);
    let state = call(&channels, "mosaic.get_state", json!({}));
    assert_eq!(state["mosaic"]["rois"][0]["roi_id"], "First ROI");
    assert_eq!(state["mosaic"]["rois"][1]["roi_id"], "Second ROI");
    assert_eq!(state["mosaic"]["left_tab"], "project");
    assert_eq!(state["mosaic"]["right_tab"], "layout");
    assert_eq!(state["mosaic"]["layout"]["layout"], "native_pixels");
    assert_eq!(state["mosaic"]["layout"]["columns"], 1);
    assert_eq!(
        state["mosaic"]["camera"]["center_world_lvl0"],
        json!([12.0, 34.0])
    );
    let channels_state = call(&channels, "viewer.channels.list", json!({}));
    assert_eq!(channels_state["channels"][0]["visible"], false);
    assert_eq!(channels_state["channels"][0]["color_rgb"], json!([1, 2, 3]));
    assert_eq!(channels_state["channels"][0]["note"], "restored");
    assert_eq!(
        call(&channels, "viewer.channels.get_active", json!({}))["active_channel"]["index"],
        1
    );
    let presentation = call(&channels, "viewer.channels.presentation.get", json!({}));
    assert_eq!(presentation["presentation"]["sort"], "name_desc");
    assert_eq!(presentation["presentation"]["order"][0]["index"], 4);
    let panels = call(&channels, "viewer.panels.get", json!({}));
    assert_eq!(panels["panels"]["left"], false);
    let rendering = call(&channels, "viewer.rendering.get_state", json!({}));
    assert_eq!(rendering["smooth_pixels"], false);
    assert_eq!(rendering["show_tile_debug"], true);
    assert_eq!(
        channels.diagnostics.legacy_requests.load(Ordering::Relaxed),
        0
    );
}

mod auto_contrast {
    use super::*;
    use crate::model::MosaicAutoContrastChannelResult;

    fn opened_fixture() -> ControlOpenedDocument {
        crate::data::document::open_local_ome_zarr(&fixture_path())
            .unwrap()
            .into_control()
    }

    fn resource(generation: u64, roi_ids: &[&str]) -> ControlMosaicResource {
        let items = roi_ids
            .iter()
            .enumerate()
            .map(|(id, roi_id)| ControlMosaicItemResource {
                id,
                roi_id: (*roi_id).to_string(),
                metadata: std::collections::HashMap::new(),
                document: opened_fixture(),
                segmentation_path: None,
            })
            .collect();
        ControlMosaicResource {
            generation,
            source: "auto-contrast-test".to_string(),
            base_dir: None,
            initial_columns: Some(1),
            metadata_columns: Arc::new(Vec::new()),
            items: Arc::new(items),
        }
    }

    fn model_with(roi_ids: &[&str]) -> AppModel {
        let mut model = AppModel::project();
        model.bootstrap_mosaic(resource(1, roi_ids)).unwrap();
        model
    }

    fn windows(model: &mut AppModel) -> Vec<Value> {
        model
            .dispatch("viewer.channels.list", &json!({}))
            .unwrap()
            .unwrap()
            .response["channels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|channel| channel["window"].clone())
            .collect()
    }

    fn set_contrast(model: &mut AppModel, channel: usize, min: f32, max: f32) {
        model
            .dispatch(
                "viewer.channels.set_contrast",
                &json!({"channel":channel,"min":min,"max":max}),
            )
            .unwrap()
            .unwrap();
    }

    fn computed(index: usize, min: u16, max: u16) -> MosaicAutoContrastChannelResult {
        MosaicAutoContrastChannelResult {
            index,
            min,
            max,
            sample_count: 100,
        }
    }

    fn started_spec(model: &mut AppModel, params: Value) -> MosaicAutoContrastSpec {
        let spec = model.prepare_mosaic_auto_contrast(&params).unwrap();
        assert!(model.mark_mosaic_auto_contrast_started(&spec));
        spec
    }

    fn result(channels: Vec<MosaicAutoContrastChannelResult>) -> MosaicAutoContrastResult {
        MosaicAutoContrastResult {
            channels,
            failed_items: Vec::new(),
        }
    }

    #[test]
    fn on_open_keeps_hand_set_limits_but_the_explicit_request_replaces_them() {
        let mut model = model_with(&["A"]);
        set_contrast(&mut model, 0, 10.0, 20.0);

        let on_open = model.mosaic_auto_contrast_on_open_spec().unwrap().unwrap();
        assert!(on_open.channels.iter().all(|channel| channel.index != 0));
        assert_eq!(on_open.channels.len(), 4);

        let explicit = started_spec(&mut model, json!({"overwrite_manual":true}));
        assert!(explicit.channels.iter().any(|channel| channel.index == 0));
        let response = model
            .install_mosaic_auto_contrast(&explicit, &result(vec![computed(0, 5, 500)]))
            .unwrap();
        assert_eq!(response["applied"][0]["max"], 500.0);
        assert_eq!(windows(&mut model)[0], json!([5.0, 500.0]));
    }

    #[test]
    fn a_contrast_edit_during_the_job_is_not_overwritten() {
        let mut model = model_with(&["A"]);
        let spec = started_spec(&mut model, json!({"overwrite_manual":false}));
        set_contrast(&mut model, 0, 10.0, 20.0);

        let response = model
            .install_mosaic_auto_contrast(
                &spec,
                &result(vec![computed(0, 1, 99), computed(1, 1, 99)]),
            )
            .unwrap();

        assert_eq!(windows(&mut model)[0], json!([10.0, 20.0]));
        assert_eq!(windows(&mut model)[1], json!([1.0, 99.0]));
        assert!(
            response["skipped"].as_array().unwrap().iter().any(
                |item| item["index"] == 0 && item["reason"] == "contrast_changed_after_request"
            )
        );
    }

    #[test]
    fn blank_channels_keep_their_window() {
        let mut model = model_with(&["A"]);
        let before = windows(&mut model);
        let spec = started_spec(&mut model, json!({}));

        let response = model
            .install_mosaic_auto_contrast(&spec, &result(vec![computed(0, 0, 0)]))
            .unwrap();

        assert_eq!(windows(&mut model), before);
        assert!(
            response["skipped"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["index"] == 0 && item["reason"] == "no_signal")
        );
    }

    #[test]
    fn results_for_a_replaced_mosaic_or_superseded_job_are_ignored() {
        let mut model = model_with(&["A"]);
        let old_job = started_spec(&mut model, json!({}));
        let newer_job = started_spec(&mut model, json!({}));
        assert!(
            model
                .install_mosaic_auto_contrast(&old_job, &result(vec![computed(0, 1, 9)]))
                .is_none()
        );

        model.bootstrap_mosaic(resource(2, &["B"])).unwrap();
        let before = windows(&mut model);
        assert!(
            model
                .install_mosaic_auto_contrast(&newer_job, &result(vec![computed(0, 1, 9)]))
                .is_none()
        );
        assert!(!model.fail_mosaic_auto_contrast(&newer_job, "late failure"));
        assert_eq!(windows(&mut model), before);
    }

    #[test]
    fn method_overrides_and_invalid_requests_are_validated() {
        let mut model = model_with(&["A"]);
        let spec = model
            .prepare_mosaic_auto_contrast(&json!({"method":"zero_to_max"}))
            .unwrap();
        assert_eq!(
            spec.settings.method,
            crate::settings::AutoContrastMethod::ZeroToMax
        );

        assert!(
            model
                .prepare_mosaic_auto_contrast(&json!({"viewport_id":"main"}))
                .is_err()
        );
        assert!(
            model
                .prepare_mosaic_auto_contrast(&json!({"method":"unknown"}))
                .is_err()
        );
    }

    #[test]
    fn large_mosaics_read_coarser_levels_to_stay_within_budget() {
        let mut model = model_with(&["A", "B"]);
        let spec = model.prepare_mosaic_auto_contrast(&json!({})).unwrap();
        assert!(spec.items.iter().all(|item| item.level == 0));
        let budget_exceeded = (0..64).map(|i| format!("R{i}")).collect::<Vec<_>>();
        let names = budget_exceeded
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut model = model_with(&names);
        let spec = model.prepare_mosaic_auto_contrast(&json!({})).unwrap();
        let level = spec.items[0].level;
        assert!(level > 0, "64 ROIs must not each read 512 x 512 x 5 pixels");
        let shape = &spec.items[0].document.descriptor.levels[level].shape;
        assert!(shape[1] * shape[2] <= 16_000_000 / 64);
    }

    #[test]
    fn worker_pools_rois_and_matches_the_shared_percentile_calculation() {
        let mut model = model_with(&["A", "B"]);
        let spec = model
            .prepare_mosaic_auto_contrast(&json!({"channels":[0]}))
            .unwrap();
        let single = {
            let mut model = model_with(&["A"]);
            let spec = model
                .prepare_mosaic_auto_contrast(&json!({"channels":[0]}))
                .unwrap();
            read_mosaic_auto_contrast(&spec, &|| false).unwrap()
        };

        let pooled = read_mosaic_auto_contrast(&spec, &|| false).unwrap();

        assert!(pooled.failed_items.is_empty());
        let (pooled, single) = (&pooled.channels[0], &single.channels[0]);
        assert_eq!(pooled.sample_count, 512 * 512 * 2);
        assert_eq!(single.sample_count, 512 * 512);
        assert!(
            pooled.max > 0,
            "the fixture has signal in its first channel"
        );
        assert_eq!(
            (pooled.min, pooled.max),
            (single.min, single.max),
            "identical ROIs have the same percentile as one"
        );
    }

    #[test]
    fn rois_without_a_channel_are_skipped_and_cancelling_stops_the_job() {
        let mut model = model_with(&["A", "B"]);
        let mut spec = model
            .prepare_mosaic_auto_contrast(&json!({"channels":[0]}))
            .unwrap();
        spec.items[1].channel_map = vec![None; 5];

        let result = read_mosaic_auto_contrast(&spec, &|| false).unwrap();
        assert_eq!(result.channels[0].sample_count, 512 * 512);
        assert!(result.failed_items.is_empty());

        assert!(read_mosaic_auto_contrast(&spec, &|| true).is_err());
    }

    #[test]
    fn one_unreadable_roi_is_reported_while_the_others_still_count() {
        let mut model = model_with(&["A", "B"]);
        let mut spec = model
            .prepare_mosaic_auto_contrast(&json!({"channels":[0]}))
            .unwrap();
        spec.items[1].level = 99;

        let result = read_mosaic_auto_contrast(&spec, &|| false).unwrap();
        assert_eq!(result.failed_items.len(), 1);
        assert_eq!(result.failed_items[0].0, spec.items[1].item_id);
        assert_eq!(result.channels[0].sample_count, 512 * 512);

        spec.items[0].level = 99;
        assert!(read_mosaic_auto_contrast(&spec, &|| false).is_err());
    }

    #[test]
    fn actor_applies_on_open_and_explicit_contrast_without_a_render_frame() {
        let directory = temp_dir("auto-contrast");
        let fixture = fixture_path();
        let sheet = directory.join("mosaic.csv");
        fs::write(
            &sheet,
            format!(
                "id,path\nROI-A,{}\nROI-B,{}\n",
                fixture.display(),
                fixture.display()
            ),
        )
        .unwrap();
        let channels = spawn_test_actor();
        super::super::channels::bootstrap_auto_contrast(&channels, true);
        call(
            &channels,
            "datasets.open_mosaic_samplesheet",
            json!({"path":sheet}),
        );
        let list = |channels: &ControlActorChannels| {
            call(channels, "viewer.channels.list", json!({}))["channels"][0]["window"].clone()
        };
        let metadata = json!([0.0, 60000.0]);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while list(&channels) == metadata {
            assert!(
                std::time::Instant::now() < deadline,
                "opening should apply automatic contrast"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let automatic = list(&channels);

        call(
            &channels,
            "viewer.channels.set_contrast",
            json!({"channel":0,"min":1.0,"max":2.0}),
        );
        assert_eq!(list(&channels), json!([1.0, 2.0]));

        let response = call(
            &channels,
            "viewer.channels.auto_contrast",
            json!({"channels":[0],"overwrite_manual":true}),
        );
        assert_eq!(response["completed"], true);
        assert_eq!(response["mode"], "mosaic");
        assert_eq!(list(&channels), automatic);
        fs::remove_dir_all(directory).ok();
    }
}

mod project_save {
    use super::*;

    struct SavedMosaicProject {
        directory: PathBuf,
        path: PathBuf,
    }

    impl Drop for SavedMosaicProject {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).ok();
        }
    }

    fn copy_directory(from: &std::path::Path, to: &std::path::Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_directory(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    fn write_project(label: &str) -> SavedMosaicProject {
        let directory = temp_dir(label);
        let path = directory.join("mosaic.odon.project.json");
        let fixture = fixture_path();
        // ROIs sharing one image are merged into a single project record.
        let second = directory.join("second.ome.zarr");
        copy_directory(&fixture, &second);
        let project = json!({
            "version": 6,
            "config": {"rois": [
                {"id":"first","path":fixture,"display_name":"First"},
                {"id":"second","path":second,"display_name":"Second"},
            ]},
            "state": {"browser":{"selected":[],"focused":null},"view_presets":[]},
        });
        fs::write(&path, serde_json::to_string_pretty(&project).unwrap()).unwrap();
        SavedMosaicProject { directory, path }
    }

    fn open_mosaic(channels: &ControlActorChannels, path: &PathBuf) {
        call(channels, "project.open", json!({"path":path}));
        call(
            channels,
            "project.rois.select",
            json!({"ids":["first","second"],"mode":"replace"}),
        );
        let opened = call(channels, "project.rois.open_selected_mosaic", json!({}));
        assert_eq!(opened["mode"], "mosaic");
    }

    fn channel<'a>(channels: &'a Value, name: &str) -> &'a Value {
        channels["channels"]
            .as_array()
            .unwrap()
            .iter()
            .find(|channel| channel["name"] == name)
            .unwrap_or_else(|| panic!("missing channel {name}"))
    }

    fn saved_project(path: &PathBuf) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn save_and_reopen_restores_channels_layout_and_camera() {
        let project = write_project("roundtrip");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        let defaults = call(&actor, "mosaic.get_state", json!({}));
        assert_eq!(defaults["mosaic"]["layout"]["label_font_size"], 13.0);
        assert_eq!(defaults["mosaic"]["layout"]["group_label_font_size"], 15.0);
        call(
            &actor,
            "viewer.channels.set_contrast",
            json!({"channel":"CD3","min":12.0,"max":345.0}),
        );
        call(
            &actor,
            "viewer.channels.set_color",
            json!({"channel":"CD3","color_rgb":[1,2,3]}),
        );
        call(
            &actor,
            "viewer.channels.set_note",
            json!({"channel":"CD3","note":"checked"}),
        );
        call(
            &actor,
            "viewer.channels.set_visible",
            json!({"mode":"only","channels":["CD3"]}),
        );
        call(
            &actor,
            "viewer.channels.set_active",
            json!({"channel":"CD3"}),
        );
        let reversed = call(&actor, "viewer.channels.list", json!({}))["channels"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .map(|channel| channel["name"].clone())
            .collect::<Vec<_>>();
        call(
            &actor,
            "viewer.channels.set_order",
            json!({"order":reversed}),
        );
        call(
            &actor,
            "mosaic.layout.configure",
            json!({"columns":1,"group_gap":7.0,"label_font_size":18.5,
                "group_label_font_size":26.0,"fit":false}),
        );
        call(
            &actor,
            "viewer.camera.set",
            json!({"center_world_lvl0":[12.0,34.0],"zoom":0.25}),
        );
        let before = call(&actor, "viewer.channels.list", json!({}));
        let camera_before = call(&actor, "viewer.camera.get", json!({}));
        // Updating another layout property must preserve both font sizes.
        call(
            &actor,
            "mosaic.layout.configure",
            json!({"columns":1,"fit":false}),
        );

        call(&actor, "project.save", json!({}));
        let saved = saved_project(&project.path);
        assert_eq!(saved["state"]["mosaic"]["label_font_size"], 18.5);
        assert_eq!(saved["state"]["mosaic"]["group_label_font_size"], 26.0);

        assert!(
            saved_project(&project.path)["state"]["mosaic"].is_object(),
            "the save must write the mosaic view"
        );
        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let after = call(&reopened, "viewer.channels.list", json!({}));
        let cd3 = channel(&after, "CD3");
        assert_eq!(cd3["window"], json!([12.0, 345.0]));
        assert_eq!(cd3["color_rgb"], json!([1, 2, 3]));
        assert_eq!(cd3["note"], "checked");
        assert_eq!(cd3["visible"], true);
        assert_eq!(cd3["active"], true);
        assert_eq!(channel(&after, "DAPI")["visible"], false);
        for key in ["window", "color_rgb", "visible", "note"] {
            for channel_before in before["channels"].as_array().unwrap() {
                let name = channel_before["name"].as_str().unwrap();
                assert_eq!(
                    channel(&after, name)[key],
                    channel_before[key],
                    "{name} {key}"
                );
            }
        }
        let state = call(&reopened, "mosaic.get_state", json!({}));
        assert_eq!(state["mosaic"]["layout"]["columns"], 1);
        assert_eq!(state["mosaic"]["layout"]["group_gap"], 7.0);
        assert_eq!(state["mosaic"]["layout"]["label_font_size"], 18.5);
        assert_eq!(state["mosaic"]["layout"]["group_label_font_size"], 26.0);
        let camera_after = call(&reopened, "viewer.camera.get", json!({}));
        assert_eq!(
            camera_after["camera"]["center_world_lvl0"],
            camera_before["camera"]["center_world_lvl0"]
        );
        let order = call(&reopened, "viewer.channels.presentation.get", json!({}));
        let order_text = order.to_string();
        assert!(
            order_text.find("PanCK").unwrap() < order_text.find("CD3").unwrap(),
            "saved channel order is restored: {order_text}"
        );
    }

    #[test]
    fn automatic_contrast_on_open_keeps_saved_limits_and_sets_the_rest() {
        let project = write_project("auto-contrast");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        call(
            &actor,
            "viewer.channels.set_contrast",
            json!({"channel":"DAPI","min":0.0,"max":19200.0}),
        );
        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        super::super::channels::bootstrap_auto_contrast(&reopened, true);
        open_mosaic(&reopened, &project.path);
        let metadata_cd3 = json!([0.0, 42000.0]);
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            let list = call(&reopened, "viewer.channels.list", json!({}));
            if channel(&list, "CD3")["window"] != metadata_cd3 {
                assert_eq!(channel(&list, "DAPI")["window"], json!([0.0, 19200.0]));
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "auto contrast never ran"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn group_colour_and_inheritance_survive_save_and_reopen() {
        let project = write_project("groups");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        call(
            &actor,
            "viewer.channels.set_group",
            json!({"state":{
                "channel_groups":[
                    {"id":2,"name":"T cells","expanded":true,"color_rgb":[255,255,0]}
                ],
                "channel_members":{
                    "CD3":{"group_id":2,"inherit_color":true},
                    "DAPI":{"group_id":2,"inherit_color":false},
                },
            }}),
        );

        call(&actor, "project.save", json!({}));

        let saved = saved_project(&project.path);
        assert_eq!(
            saved["config"]["layer_groups"]["channel_groups"][0]["color_rgb"],
            json!([255, 255, 0])
        );
        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let groups = call(&reopened, "viewer.channels.list_groups", json!({}));
        let group = &groups["groups"][0];
        assert_eq!(group["name"], "T cells");
        assert_eq!(group["color_rgb"], json!([255, 255, 0]));
        let inherits = |name: &str| {
            group["members"]
                .as_array()
                .unwrap()
                .iter()
                .find(|member| member["name"] == name)
                .unwrap_or_else(|| panic!("missing member {name}"))["inherit_color"]
                .clone()
        };
        assert_eq!(inherits("CD3"), true);
        assert_eq!(inherits("DAPI"), false);
    }

    #[test]
    fn saving_a_mosaic_keeps_groups_it_did_not_edit() {
        let project = write_project("group-keep");
        let mut file = saved_project(&project.path);
        file["config"]["layer_groups"] = json!({
            "channel_groups":[{"id":5,"name":"Kept","expanded":true,"color_rgb":[1,2,3]}],
            "channel_members":{"CD3":{"group_id":5,"inherit_color":false}},
        });
        fs::write(&project.path, file.to_string()).unwrap();
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);

        call(&actor, "project.save", json!({}));

        let saved = saved_project(&project.path);
        assert_eq!(
            saved["config"]["layer_groups"]["channel_groups"][0]["name"],
            "Kept"
        );
        assert_eq!(
            saved["config"]["layer_groups"]["channel_members"]["CD3"]["group_id"],
            5
        );
    }

    #[test]
    fn selected_property_and_colouring_survive_save_and_reopen() {
        let project = write_project("object-style");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        let mapping = json!({
            "mode":"continuous",
            "property":"marker_CD3_p90_intensity",
            "palette":"magma",
            "domain":[1.0,9.0],
            "scale":"linear",
            "reverse":true,
            "out_of_range":"clamp",
        });
        call(
            &actor,
            "mosaic.objects.style.set",
            json!({"style":{"color_mapping":mapping,"fill_cells":true,"opacity":0.4}}),
        );
        let before = call(&actor, "mosaic.objects.style.get", json!({}));

        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let after = call(&reopened, "mosaic.objects.style.get", json!({}));
        assert_eq!(after["objects"]["style"], before["objects"]["style"]);
        assert_eq!(
            after["objects"]["style"]["color_mapping"]["property"],
            "marker_CD3_p90_intensity"
        );
    }

    #[test]
    fn categorical_colouring_survives_save_and_reopen() {
        let project = write_project("object-style-categorical");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        call(
            &actor,
            "mosaic.objects.style.set",
            json!({"style":{"color_mapping":{"mode":"categorical","property":"pos_CD3"}}}),
        );

        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let style = call(&reopened, "mosaic.objects.style.get", json!({}));
        assert_eq!(
            style["objects"]["style"]["color_mapping"]["property"],
            "pos_CD3"
        );
        assert_eq!(style["objects"]["style"]["color_property_key"], "pos_CD3");
    }

    #[test]
    fn unticked_category_mode_survives_save_and_reopen_in_a_mosaic() {
        let project = write_project("unticked-mosaic");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        let style = call(&actor, "mosaic.objects.style.get", json!({}));
        assert_eq!(style["objects"]["style"]["unticked_categories"], "hidden");
        call(
            &actor,
            "mosaic.objects.style.set",
            json!({"style":{"unticked_categories":"hover_only"}}),
        );

        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let style = call(&reopened, "mosaic.objects.style.get", json!({}));
        assert_eq!(
            style["objects"]["style"]["unticked_categories"],
            "hover_only"
        );
    }

    #[test]
    fn unticked_category_mode_survives_save_and_reopen_in_a_single_image() {
        let project = write_project("unticked-single");
        let open_roi = |actor: &ControlActorChannels| {
            call(actor, "project.open", json!({"path":project.path}));
            call(actor, "project.rois.open", json!({"id":"first"}));
        };
        let actor = spawn_test_actor();
        open_roi(&actor);
        let styled = call(
            &actor,
            "viewer.objects.style.set",
            json!({"unticked_categories":"hover_only"}),
        );
        assert_eq!(styled["style"]["unticked_categories"], "hover_only");

        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        open_roi(&reopened);
        let style = call(&reopened, "viewer.objects.style.get", json!({}));
        assert_eq!(style["unticked_categories"], "hover_only");
    }

    #[test]
    fn a_saved_single_image_without_labels_can_still_be_reopened() {
        let project = write_project("empty-label");
        let actor = spawn_test_actor();
        call(&actor, "project.open", json!({"path":project.path}));
        call(&actor, "project.rois.open", json!({"id":"first"}));
        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        call(&reopened, "project.open", json!({"path":project.path}));
        let opened = call(&reopened, "project.rois.open", json!({"id":"first"}));
        assert_eq!(opened["mode"], "single");
    }

    #[test]
    fn outline_only_mode_survives_save_and_reopen_in_a_mosaic() {
        let project = write_project("outline-only");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        call(
            &actor,
            "mosaic.objects.style.set",
            json!({"style":{"unticked_categories":"outline_only"}}),
        );

        call(&actor, "project.save", json!({}));

        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        let style = call(&reopened, "mosaic.objects.style.get", json!({}));
        assert_eq!(
            style["objects"]["style"]["unticked_categories"],
            "outline_only"
        );
    }

    #[test]
    fn an_unknown_unticked_category_mode_is_rejected() {
        let project = write_project("unticked-invalid");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        let (bad, reply) = request(
            "mosaic.objects.style.set",
            json!({"style":{"unticked_categories":"sometimes"}}),
        );
        actor.request_tx.send(bad).unwrap();
        assert!(reply.recv_timeout(Duration::from_secs(5)).unwrap().is_err());
    }

    #[test]
    fn an_invalid_saved_style_does_not_stop_the_mosaic_opening() {
        let (actor, _project) = restored_with(json!({
            "object_style":{"color_mapping":"not a mapping","unknown_field":1},
        }));

        let style = call(&actor, "mosaic.objects.style.get", json!({}));
        assert_eq!(style["objects"]["style"]["color_mapping"]["mode"], "single");
    }

    #[test]
    fn pinned_and_recent_properties_survive_save_and_reopen() {
        let project = write_project("property-preferences");
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        let preferences = json!({"pinned":["pos_CD3","CD8a"],"recent":["pos_B7_H3"]});
        call(
            &actor,
            "project.update_metadata",
            json!({"object_property_preferences":preferences}),
        );

        call(&actor, "project.save", json!({}));

        assert_eq!(
            saved_project(&project.path)["config"]["object_property_preferences"],
            preferences
        );
        let reopened = spawn_test_actor();
        open_mosaic(&reopened, &project.path);
        call(&reopened, "project.save", json!({}));
        assert_eq!(
            saved_project(&project.path)["config"]["object_property_preferences"],
            preferences,
            "reopening and saving keeps them"
        );
        let got = call(&reopened, "project.get", json!({}));
        assert_eq!(got["metadata"]["object_property_preferences"], preferences);
    }

    #[test]
    fn invalid_property_preferences_are_rejected() {
        assert!(
            odon_control_decode(json!({"object_property_preferences":{"pinned":["  "]}})).is_err()
        );
        assert!(
            odon_control_decode(json!({"object_property_preferences":{"pinned":["a"]}})).is_ok()
        );
    }

    fn odon_control_decode(params: Value) -> Result<ControlCommand, ControlError> {
        ControlCommand::decode("project.update_metadata", params)
    }

    #[test]
    fn saving_keeps_annotation_layers_already_in_the_project() {
        let project = write_project("annotations");
        let mut file = saved_project(&project.path);
        file["state"]["mosaic"] = json!({"annotation_layers":[]});
        fs::write(&project.path, file.to_string()).unwrap();
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);

        call(&actor, "project.save", json!({}));

        let saved = saved_project(&project.path);
        assert!(saved["state"]["mosaic"]["annotation_layers"].is_array());
        assert!(saved["state"]["mosaic"]["channels"].is_array());
    }

    fn restored_with(state: Value) -> (ControlActorChannels, SavedMosaicProject) {
        let project = write_project("restore");
        let mut file = saved_project(&project.path);
        file["state"]["mosaic"] = state;
        fs::write(&project.path, file.to_string()).unwrap();
        let actor = spawn_test_actor();
        open_mosaic(&actor, &project.path);
        (actor, project)
    }

    #[test]
    fn saved_channels_missing_from_the_mosaic_change_nothing() {
        let fresh = spawn_test_actor();
        let project = write_project("baseline");
        open_mosaic(&fresh, &project.path);
        let baseline = call(&fresh, "viewer.channels.list", json!({}));

        let (actor, _project) = restored_with(json!({
            "channels":[
                {"name":"Not in this mosaic","visible":false,"color_rgb":[9,9,9],"window":[1.0,2.0]},
            ],
            "active_channel_name":"Not in this mosaic",
        }));

        let after = call(&actor, "viewer.channels.list", json!({}));
        assert_eq!(after["channels"], baseline["channels"]);
    }

    #[test]
    fn saved_order_is_by_name_and_keeps_channels_it_does_not_mention() {
        let (actor, _project) = restored_with(json!({
            "channel_order_names":["PanCK","Not in this mosaic","DAPI"],
        }));

        let text = call(&actor, "viewer.channels.presentation.get", json!({})).to_string();
        assert!(text.find("PanCK").unwrap() < text.find("DAPI").unwrap());
        assert!(text.contains("CD3"), "unmentioned channels stay available");
    }

    #[test]
    fn a_camera_saved_outside_this_mosaic_is_not_restored() {
        let fresh = spawn_test_actor();
        let project = write_project("camera-baseline");
        open_mosaic(&fresh, &project.path);
        let fitted = call(&fresh, "viewer.camera.get", json!({}));

        let (actor, _project) = restored_with(json!({
            "camera":{"center_world_lvl0":[9000000.0,9000000.0],"zoom_screen_per_lvl0_px":3.0},
        }));

        let camera = call(&actor, "viewer.camera.get", json!({}));
        assert_eq!(camera["camera"], fitted["camera"]);
    }

    #[test]
    fn older_projects_with_positional_order_still_restore() {
        let (actor, _project) = restored_with(json!({
            "channel_order":[4,3,2,1,0],
            "active_channel":1,
        }));

        let list = call(&actor, "viewer.channels.list", json!({}));
        let active = list["channels"]
            .as_array()
            .unwrap()
            .iter()
            .position(|channel| channel["active"] == true);
        assert_eq!(active, Some(1));
    }
}
