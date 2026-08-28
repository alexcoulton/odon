use super::*;

fn bootstrap_project(channels: &ControlActorChannels, snapshot: ProjectModelSnapshot) {
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(snapshot))
        .unwrap();
    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
}

fn current_project(dataset_path: PathBuf, object_path: PathBuf) -> ProjectModelSnapshot {
    let mut roi = ProjectRoi {
        id: "roi-a".to_string(),
        display_name: Some("ROI A".to_string()),
        segpath: Some(object_path),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(dataset_path));
    let config = ProjectConfig {
        rois: vec![roi.clone()],
        ..ProjectConfig::default()
    };
    ProjectModelSnapshot {
        config,
        state: json!({"browser":{},"roi_views":{}}),
        load_generation: 1,
        rois: vec![roi],
        saved_path: Some(std::env::temp_dir().join("actor-deep-link.odon.json")),
        config_generation: 1,
        ..ProjectModelSnapshot::default()
    }
}

#[test]
fn deep_link_apply_commits_project_resources_channels_objects_and_camera_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let object_path = std::env::temp_dir().join(format!(
        "odon-deep-link-objects-{}-{}.parquet",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&object_path, b"fixture").unwrap();
    bootstrap_project(
        &channels,
        current_project(dataset_path.clone(), object_path.clone()),
    );

    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{
            "roi":"roi-a",
            "channel":"CD3",
            "visible_channels":["DAPI","CD3"],
            "group_visible_channels":true,
            "visible_channel_group":"Comparison",
            "visible_channel_group_color":[8,40,220],
            "channel_order":"listed",
            "hidden_channels":["PanCK"],
            "channel_colors":[{"channel":"DAPI","color_rgb":[10,20,30]}],
            "channel_contrasts":[{"channel":"CD3","min":12.0,"max":1200.0}],
            "segmentation_source":"objects",
            "cell_color_by":"phenotype",
            "fill_cells":true,
            "object_level_colors":[{"value":"immune","color_rgb":[0,255,0]}],
            "visible_cell_types":["immune"],
            "object_query":"immune",
            "center_world":[42.0,84.0],
            "zoom":3.25
        }}),
    );
    channels.request_tx.send(apply).unwrap();
    let applied = apply_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(applied["applied"], true);
    assert_eq!(applied["settled"], true);
    assert_eq!(applied["resolution"]["project_source"], "current");
    assert_eq!(applied["opened"]["roi"], "roi-a");

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let workspace = workspace_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let viewport = &workspace["viewports"][0];
    assert_eq!(viewport["camera"]["center_world_lvl0"], json!([42.0, 84.0]));
    assert_eq!(viewport["camera"]["zoom_screen_per_lvl0_px"], 3.25);
    assert_eq!(viewport["channels"][1]["selected"], true);
    assert_eq!(viewport["channels"][0]["color_rgb"], json!([10, 20, 30]));
    assert_eq!(viewport["channel_order"][0], 0);
    assert_eq!(viewport["channel_order"][1], 1);
    assert_eq!(viewport["channel_groups"][0]["name"], "Comparison");
    assert_eq!(viewport["objects"]["color_property"], "phenotype");
    assert_eq!(viewport["objects"]["fill_cells"], true);
    assert_eq!(viewport["objects"]["filter"]["mode"], "query");
    assert_eq!(viewport["objects"]["filter"]["query"], "immune");
    assert_eq!(
        viewport["objects"]["color_level_overrides"]["immune"]["visible"],
        true
    );
    assert_eq!(
        viewport["objects"]["color_level_overrides"]["tumour"]["visible"],
        false
    );

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.mode, ModelMode::Single);
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
    assert_eq!(projection.object_resource.unwrap().features.len(), 2);
    let _ = std::fs::remove_file(object_path);
}

#[test]
fn failed_external_deep_link_retains_the_previous_usable_document() {
    let channels = spawn_test_actor();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request(
        "datasets.open_ome_zarr",
        json!({"path":dataset_path.clone()}),
    );
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let missing = std::env::temp_dir().join("odon-missing-deep-link-project.json");
    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{"project_path":missing,"roi":"missing"}}),
    );
    channels.request_tx.send(apply).unwrap();
    let error = apply_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Application);

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    assert_eq!(
        state_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["mode"],
        "single"
    );
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}

#[test]
fn external_project_deep_link_replaces_project_and_document_atomically() {
    let channels = spawn_test_actor();
    let project_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.project.json");
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{
            "project_path":project_path.clone(),
            "roi":"synthetic_5ch.ome.zarr",
            "channel":"PanCK",
            "center_world":[12.0,34.0],
            "zoom":1.75
        }}),
    );
    channels.request_tx.send(apply).unwrap();
    let applied = apply_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(applied["settled"], true);
    assert_eq!(applied["resolution"]["project_source"], "project_file");
    assert_eq!(
        applied["resolution"]["project_path"],
        json!(project_path.to_string_lossy())
    );

    let (project, project_rx) = request("project.get", json!({}));
    channels.request_tx.send(project).unwrap();
    let project = project_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(project["path"], json!(project_path.to_string_lossy()));
    assert_eq!(project["roi_count"], 1);
    let (camera, camera_rx) = request("viewer.camera.get", json!({}));
    channels.request_tx.send(camera).unwrap();
    let camera = camera_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(camera["camera"]["center_world_lvl0"], json!([12.0, 34.0]));
    assert_eq!(camera["camera"]["zoom_screen_per_lvl0_px"], 1.75);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
    assert_eq!(projection.project.saved_path, Some(project_path));
}

#[test]
fn same_roi_deep_link_reuses_the_document_and_preserves_the_workspace() {
    let channels = spawn_test_actor();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let mut roi = ProjectRoi {
        id: "roi-a".to_string(),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(dataset_path));
    bootstrap_project(
        &channels,
        ProjectModelSnapshot {
            config: ProjectConfig {
                rois: vec![roi.clone()],
                ..ProjectConfig::default()
            },
            state: json!({"browser":{},"roi_views":{}}),
            load_generation: 1,
            rois: vec![roi],
            config_generation: 1,
            ..ProjectModelSnapshot::default()
        },
    );

    let (open, open_rx) = request(
        "deep_links.apply",
        json!({"request":{"roi":"roi-a","channel":"DAPI"}}),
    );
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
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
    clone_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{"roi":"roi-a","channel":"CD3"}}),
    );
    channels.request_tx.send(apply).unwrap();
    let applied = apply_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(applied["opened"]["reused_document"], true);

    let (workspace, workspace_rx) = request("viewer.workspace.get", json!({}));
    channels.request_tx.send(workspace).unwrap();
    let workspace = workspace_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(workspace["viewports"].as_array().unwrap().len(), 2);
}

#[test]
fn deep_link_application_loads_requested_bundled_labels_in_the_same_transaction() {
    let channels = spawn_test_actor();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let mut roi = ProjectRoi {
        id: "label-roi".to_string(),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(dataset_path));
    bootstrap_project(
        &channels,
        ProjectModelSnapshot {
            config: ProjectConfig {
                rois: vec![roi.clone()],
                ..ProjectConfig::default()
            },
            state: json!({"browser":{},"roi_views":{}}),
            load_generation: 1,
            rois: vec![roi],
            config_generation: 1,
            ..ProjectModelSnapshot::default()
        },
    );
    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{
            "roi":"label-roi",
            "segmentation":"cells",
            "load_segmentation_labels":true
        }}),
    );
    channels.request_tx.send(apply).unwrap();
    apply_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (labels, labels_rx) = request("viewer.labels.get", json!({}));
    channels.request_tx.send(labels).unwrap();
    let labels = labels_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(labels["loaded"], "cells");
    assert_eq!(labels["selected"], "cells");
    assert_eq!(labels["visible"], true);
}

struct BlockingDeepLinkObjectLoader {
    started: Sender<()>,
    release: Mutex<Receiver<()>>,
}

impl ObjectResourceLoader for BlockingDeepLinkObjectLoader {
    fn load(&self, path: PathBuf, downsample_factor: f32) -> anyhow::Result<ControlObjectResource> {
        self.started.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        Ok(ControlObjectResource {
            source: path,
            downsample_factor,
            features: Arc::new(Vec::new()),
            property_names: Arc::new(Vec::new()),
            property_source: Arc::new(crate::model::EmptyControlObjectPropertySource),
            numeric_summaries: Arc::new(Default::default()),
            memory_diagnostics: Arc::new(Default::default()),
            renderer_payload: None,
        })
    }
}

fn blocking_actor() -> (ControlActorChannels, Receiver<()>, Sender<()>) {
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    let channels = spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        Some(Arc::new(BlockingDeepLinkObjectLoader {
            started: started_tx,
            release: Mutex::new(release_rx),
        })),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    (channels, started_rx, release_tx)
}

fn bootstrap_previous_document_and_blocking_project(
    channels: &ControlActorChannels,
) -> (PathBuf, PathBuf) {
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request(
        "datasets.open_ome_zarr",
        json!({"path":dataset_path.clone()}),
    );
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let object_path = std::env::temp_dir().join("odon-blocking-deep-link.parquet");
    bootstrap_project(
        channels,
        current_project(dataset_path.clone(), object_path.clone()),
    );
    (dataset_path, object_path)
}

#[test]
fn cancelled_deep_link_cannot_replace_the_previous_document() {
    let (channels, started_rx, release_tx) = blocking_actor();
    let (dataset_path, _) = bootstrap_previous_document_and_blocking_project(&channels);
    let task = channels
        .task_service
        .create("blocked deep link", "test", true)
        .unwrap();
    let event_hub = EventHub::shared();
    let (reply, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "deep_links.apply",
                json!({"request":{"roi":"roi-a","channel":"CD3"}}),
            )
            .unwrap(),
            reply,
            session_id: "test".to_string(),
            request_id: None,
            event_hub,
            task_registry: channels.task_service.registry(),
            task_id: Some(task.task_id.clone()),
        })
        .unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("deep-link object load started");
    channels.task_service.cancel(&task.task_id).unwrap();
    release_tx.send(()).unwrap();
    let error = reply_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Cancelled);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}

#[test]
fn project_edit_supersedes_a_late_deep_link_without_losing_the_edit() {
    let (channels, started_rx, release_tx) = blocking_actor();
    let (dataset_path, _) = bootstrap_previous_document_and_blocking_project(&channels);
    let (apply, apply_rx) = request(
        "deep_links.apply",
        json!({"request":{"roi":"roi-a","channel":"CD3"}}),
    );
    channels.request_tx.send(apply).unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("deep-link object load started");

    let (rename, rename_rx) = request(
        "project.rois.update",
        json!({"target_id":"roi-a","changes":{"display_name":"Renamed while loading"}}),
    );
    channels.request_tx.send(rename).unwrap();
    rename_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("actor remains responsive")
        .unwrap();
    release_tx.send(()).unwrap();
    let error = apply_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Conflict);

    let (roi, roi_rx) = request("project.rois.get", json!({"id":"roi-a"}));
    channels.request_tx.send(roi).unwrap();
    assert_eq!(
        roi_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["roi"]["display_name"],
        "Renamed while loading"
    );
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}
