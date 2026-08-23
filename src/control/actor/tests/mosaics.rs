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
    let style = call(
        &channels,
        "mosaic.objects.style.set",
        json!({"style":{"fill_cells":true,"opacity":0.42}}),
    );
    assert_eq!(style["result"]["style"]["fill_cells"], true);
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
