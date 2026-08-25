use super::*;

fn project_roi_snapshot(
    dataset_path: PathBuf,
    segmentation_path: Option<PathBuf>,
) -> ProjectModelSnapshot {
    let mut roi = ProjectRoi {
        id: "roi-a".to_string(),
        display_name: Some("ROI A".to_string()),
        segpath: segmentation_path,
        mask_layers: vec![crate::data::project_config::ProjectMaskLayer {
            id: 7,
            name: "Nucleus boundary".to_string(),
            visible: true,
            opacity: 0.6,
            width_screen_px: 2.0,
            display_mode: Some("translucent_fill".to_string()),
            color_rgb: [20, 180, 240],
            offset_world: [0.0, 0.0],
            editable: true,
            polygons_world: vec![vec![[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 1.0]]],
            source_geojson: None,
        }],
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(dataset_path.clone()));
    let source_key = roi.source_key().unwrap();
    let state = json!({
        "browser":{"focused":source_key.clone(),"selected":[source_key.clone()]},
        "roi_views":{
            (source_key):{
                "active_channel":1,
                "channels":[
                    {"visible":false,"color_rgb":[1,2,3]},
                    {"visible":true,"window":[10.0,500.0]}
                ],
                "camera":{"center_world_lvl0":[123.0,45.0],"zoom_screen_per_lvl0_px":2.5},
                "segmentation":{"object_display":{"fill_cells":true,"fill_opacity":0.4}},
                "ui":{"show_left_panel":false,"right_tab":"properties"}
            }
        }
    });
    let config = ProjectConfig {
        rois: vec![roi.clone()],
        ..ProjectConfig::default()
    };
    ProjectModelSnapshot {
        config,
        state,
        load_generation: 1,
        rois: vec![roi],
        saved_path: Some(std::env::temp_dir().join("actor-roi-open.odon.json")),
        config_generation: 1,
        ..ProjectModelSnapshot::default()
    }
}

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

#[test]
fn project_roi_open_commits_document_resources_and_saved_view_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let object_path = std::env::temp_dir().join(format!(
        "odon-roi-open-objects-{}-{}.parquet",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&object_path, b"fixture").unwrap();
    bootstrap_project(
        &channels,
        project_roi_snapshot(dataset_path.clone(), Some(object_path.clone())),
    );

    let (open, open_rx) = request("project.rois.open", json!({"roi":"roi-a"}));
    channels.request_tx.send(open).unwrap();
    let opened = open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert_eq!(opened["opened"], true);
    assert_eq!(opened["roi"], "roi-a");
    assert_eq!(opened["resources_ready"], true);

    let (camera, camera_rx) = request("viewer.camera.get", json!({}));
    channels.request_tx.send(camera).unwrap();
    let camera = camera_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(camera["camera"]["center_world_lvl0"], json!([123.0, 45.0]));
    assert_eq!(camera["camera"]["zoom_screen_per_lvl0_px"], 2.5);

    let (masks, masks_rx) = request("viewer.masks.layers.list", json!({}));
    channels.request_tx.send(masks).unwrap();
    assert_eq!(
        masks_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["layers"][0]["id"],
        7
    );
    let (objects, objects_rx) = request("viewer.objects.get_state", json!({}));
    channels.request_tx.send(objects).unwrap();
    assert_eq!(
        objects_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["state"]["object_count"],
        2
    );

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.mode, ModelMode::Single);
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
    assert_eq!(projection.object_resource.unwrap().features.len(), 2);
    assert_eq!(
        projection.project.focused_source_key,
        projection.project.rois[0].source_key()
    );

    let _ = std::fs::remove_file(object_path);
}

#[test]
fn failed_project_roi_open_retains_the_previous_usable_document() {
    struct FailingObjectLoader;
    impl ObjectResourceLoader for FailingObjectLoader {
        fn load(
            &self,
            _path: PathBuf,
            _downsample_factor: f32,
        ) -> anyhow::Result<ControlObjectResource> {
            anyhow::bail!("synthetic object load failure")
        }
    }
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    let channels = spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        Some(Arc::new(FailingObjectLoader)),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (initial, initial_rx) = request(
        "datasets.open_ome_zarr",
        json!({"path":dataset_path.clone()}),
    );
    channels.request_tx.send(initial).unwrap();
    initial_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    bootstrap_project(
        &channels,
        project_roi_snapshot(dataset_path.clone(), Some(PathBuf::from("broken.parquet"))),
    );

    let (open, open_rx) = request("project.rois.open", json!({"roi":"roi-a"}));
    channels.request_tx.send(open).unwrap();
    let error = open_rx
        .recv_timeout(Duration::from_secs(10))
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
    assert_eq!(projection.mode, ModelMode::Single);
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}

#[test]
fn cancelled_project_roi_open_cannot_replace_the_previous_document() {
    struct BlockingObjectLoader {
        started: Sender<()>,
        release: Mutex<Receiver<()>>,
    }
    impl ObjectResourceLoader for BlockingObjectLoader {
        fn load(
            &self,
            path: PathBuf,
            downsample_factor: f32,
        ) -> anyhow::Result<ControlObjectResource> {
            let _ = self.started.send(());
            let _ = self.release.lock().unwrap().recv();
            Ok(ControlObjectResource {
                source: path,
                downsample_factor,
                features: Arc::new(Vec::new()),
                property_names: Arc::new(Vec::new()),
                numeric_summaries: Arc::new(Default::default()),
                renderer_payload: None,
            })
        }
    }
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(Arc::clone(&events));
    let channels = spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        Some(Arc::new(BlockingObjectLoader {
            started: started_tx,
            release: Mutex::new(release_rx),
        })),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (initial, initial_rx) = request(
        "datasets.open_ome_zarr",
        json!({"path":dataset_path.clone()}),
    );
    channels.request_tx.send(initial).unwrap();
    initial_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    bootstrap_project(
        &channels,
        project_roi_snapshot(dataset_path.clone(), Some(PathBuf::from("blocked.parquet"))),
    );

    let task = channels
        .task_service
        .create("blocked ROI open", "test", true)
        .unwrap();
    let (reply, reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode("project.rois.open", json!({"roi":"roi-a"})).unwrap(),
            reply,
            session_id: "test".to_string(),
            request_id: None,
            event_hub: events,
            task_registry: channels.task_service.registry(),
            task_id: Some(task.task_id.clone()),
        })
        .unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("project ROI object load started");
    channels.task_service.cancel(&task.task_id).unwrap();
    release_tx.send(()).unwrap();
    let error = reply_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Cancelled);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.mode, ModelMode::Single);
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}

#[test]
fn superseding_dataset_open_rejects_a_late_project_roi_completion() {
    struct BlockingObjectLoader {
        started: Sender<()>,
        release: Mutex<Receiver<()>>,
    }
    impl ObjectResourceLoader for BlockingObjectLoader {
        fn load(
            &self,
            path: PathBuf,
            downsample_factor: f32,
        ) -> anyhow::Result<ControlObjectResource> {
            let _ = self.started.send(());
            let _ = self.release.lock().unwrap().recv();
            Ok(ControlObjectResource {
                source: path,
                downsample_factor,
                features: Arc::new(Vec::new()),
                property_names: Arc::new(Vec::new()),
                numeric_summaries: Arc::new(Default::default()),
                renderer_payload: None,
            })
        }
    }
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let events = EventHub::shared();
    let resources = ResourceRegistry::shared(events);
    let channels = spawn_control_actor_with_services(
        Arc::new(|| {}),
        resources,
        Some(Arc::new(BlockingObjectLoader {
            started: started_tx,
            release: Mutex::new(release_rx),
        })),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    bootstrap_project(
        &channels,
        project_roi_snapshot(dataset_path.clone(), Some(PathBuf::from("blocked.parquet"))),
    );
    let (roi_open, roi_open_rx) = request("project.rois.open", json!({"roi":"roi-a"}));
    channels.request_tx.send(roi_open).unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("project ROI object load started");

    let (dataset_open, dataset_open_rx) = request(
        "datasets.open_ome_zarr",
        json!({"path":dataset_path.clone()}),
    );
    channels.request_tx.send(dataset_open).unwrap();
    dataset_open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    release_tx.send(()).unwrap();
    let error = roi_open_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Conflict);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.document.unwrap().path(),
        Some(dataset_path.as_path())
    );
}
