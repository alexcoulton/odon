use super::*;

fn temporary_preload_paths(label: &str) -> (PathBuf, PathBuf) {
    let unique = format!(
        "odon-actor-preload-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir();
    (
        root.join(format!("{unique}.odon.json")),
        root.join(format!("{unique}.parquet")),
    )
}

fn prepare_saved_project(
    channels: &ControlActorChannels,
    project_path: &PathBuf,
    source_path: &PathBuf,
) {
    std::fs::write(source_path, b"fixture").unwrap();
    let (create, create_rx) = request("project.create", json!({}));
    channels.request_tx.send(create).unwrap();
    create_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (add, add_rx) = request(
        "project.rois.add",
        json!({
            "id":"roi-a",
            "path":"/tmp/roi-a.zarr",
            "segmentation_path":source_path,
        }),
    );
    channels.request_tx.send(add).unwrap();
    add_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (save, save_rx) = request("project.save_as", json!({"path":project_path}));
    channels.request_tx.send(save).unwrap();
    save_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
}

fn wait_for_preload_projection(
    channels: &ControlActorChannels,
    matches: impl Fn(&RenderProjection) -> bool,
) -> RenderProjection {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let projection = channels
            .presentation_rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("object source discovery publishes its result without another command");
        if matches(&projection) {
            return projection;
        }
    }
}

#[test]
fn project_open_discovers_preload_sources_without_an_api_scan() {
    let (project_path, source_path) = temporary_preload_paths("open");
    let source_path = source_path.with_extension("geoparquet");
    fs::write(&source_path, b"fixture").unwrap();
    let relative_source = source_path.file_name().unwrap().to_string_lossy();
    let project = json!({
        "version": 6,
        "config": {"rois": [
            {"id":"a", "path":"/tmp/preload-a.zarr", "segpath":relative_source},
            {"id":"b", "path":"/tmp/preload-b.zarr", "segpath":relative_source},
            {"id":"missing", "path":"/tmp/preload-c.zarr", "segpath":"missing.parquet"},
            {"id":"geojson", "path":"/tmp/preload-d.zarr", "segpath":"objects.geojson"}
        ]},
        "state": {}
    });
    fs::write(&project_path, serde_json::to_vec(&project).unwrap()).unwrap();
    let channels = spawn_test_actor();
    // Reopening the same project must also schedule discovery after resetting the catalog.
    for generation in 1..=2 {
        let (open, open_rx) = request("project.open", json!({"path":project_path}));
        channels.request_tx.send(open).unwrap();
        open_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let projection = wait_for_preload_projection(&channels, |projection| {
            projection.project.load_generation == generation
                && projection.project_object_preload.state["available_count"] == 1
        });
        assert_eq!(projection.project_object_preload.state["on_disk_bytes"], 7);
        assert_eq!(projection.project_object_preload.state["cached"], 0);
        assert_eq!(projection.project_object_preload.state["loading"], false);
        assert!(projection.project.saved_path.is_some());
    }
    let _ = fs::remove_file(project_path);
    let _ = fs::remove_file(source_path);
}

#[test]
fn bootstrapped_project_discovers_preload_sources_without_an_api_scan() {
    let (project_path, source_path) = temporary_preload_paths("bootstrap");
    fs::write(&source_path, b"fixture").unwrap();
    let mut roi = ProjectRoi {
        id: "roi-a".to_string(),
        segpath: Some(source_path.clone()),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(PathBuf::from("/tmp/preload-a.zarr")));
    let channels = spawn_test_actor();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(ProjectModelSnapshot {
            load_generation: 1,
            rois: vec![roi],
            saved_path: Some(project_path),
            ..ProjectModelSnapshot::default()
        }))
        .unwrap();
    let projection = wait_for_preload_projection(&channels, |projection| {
        projection.project_object_preload.state["available_count"] == 1
    });
    assert_eq!(projection.project_object_preload.state["on_disk_bytes"], 7);
    let _ = fs::remove_file(source_path);
}

#[test]
fn editing_project_segmentation_refreshes_preload_sources() {
    let channels = spawn_test_actor();
    let (project_path, source_path) = temporary_preload_paths("edit");
    prepare_saved_project(&channels, &project_path, &source_path);
    wait_for_preload_projection(&channels, |projection| {
        projection.project.saved_path.is_some()
            && projection.project_object_preload.state["available_count"] == 1
    });
    let replacement = source_path.with_extension("geoparquet");
    fs::write(&replacement, b"replacement").unwrap();
    let (update, update_rx) = request(
        "project.rois.update",
        json!({"target_id":"roi-a", "changes":{"segmentation_path":replacement}}),
    );
    channels.request_tx.send(update).unwrap();
    update_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    wait_for_preload_projection(&channels, |projection| {
        projection.project_object_preload.state["available_count"] == 1
            && projection.project_object_preload.state["on_disk_bytes"] == 11
    });
    let _ = fs::remove_file(project_path);
    let _ = fs::remove_file(source_path);
    let _ = fs::remove_file(replacement);
}

#[test]
fn explicit_preload_inspection_publishes_updated_file_sizes() {
    let channels = spawn_test_actor();
    let (project_path, source_path) = temporary_preload_paths("inspect");
    prepare_saved_project(&channels, &project_path, &source_path);
    wait_for_preload_projection(&channels, |projection| {
        projection.project.saved_path.is_some()
            && projection.project_object_preload.state["available_count"] == 1
    });
    fs::write(&source_path, b"updated fixture").unwrap();
    let (inspect, inspect_rx) = request("project.objects.preload.get", json!({}));
    channels.request_tx.send(inspect).unwrap();
    let inspected = inspect_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(inspected["on_disk_bytes"], 15);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.project_object_preload.state["on_disk_bytes"], 15);
    let _ = fs::remove_file(project_path);
    let _ = fs::remove_file(source_path);
}

#[test]
fn project_object_preload_completes_and_projects_shared_resources_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    let (project_path, source_path) = temporary_preload_paths("complete");
    prepare_saved_project(&channels, &project_path, &source_path);

    let (list, list_rx) = request(
        "project.objects.preload.list_sources",
        json!({"offset":0,"limit":10}),
    );
    channels.request_tx.send(list).unwrap();
    let listed = list_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(listed["total"], 1);
    assert_eq!(listed["sources"][0]["cached"], false);

    let (start, start_rx) = request(
        "project.objects.preload.start",
        json!({"mode":"centroid_points","lazy_properties":true}),
    );
    channels.request_tx.send(start).unwrap();
    let completed = start_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(completed["completed"], true);
    assert_eq!(completed["preload"]["cached"], 1);
    assert_eq!(completed["preload"]["loading"], false);

    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(projection.project_object_preload.state["cached"], 1);
    assert_eq!(projection.project_object_preload.resources.len(), 1);
    assert_eq!(
        projection.project_object_preload.settings.mode,
        crate::model::ProjectObjectPreloadMode::CentroidPoints
    );

    let (clear, clear_rx) = request("project.objects.preload.clear", json!({}));
    channels.request_tx.send(clear).unwrap();
    let cleared = clear_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(cleared["removed"], 1);
    assert_eq!(cleared["cancelled"], false);
    assert_eq!(cleared["preload"]["cached"], 0);
    assert_eq!(cleared["preload"]["available_count"], 1);
    let projection = channels.presentation_rx.try_recv().unwrap();
    assert_eq!(
        projection.project_object_preload.state["available_count"],
        1
    );
    assert_eq!(projection.project_object_preload.state["loading"], false);

    let _ = std::fs::remove_file(project_path);
    let _ = std::fs::remove_file(source_path);
}

#[test]
fn clearing_project_object_preload_rejects_its_late_worker_completion() {
    struct BlockingLoader {
        started: Sender<()>,
        release: Mutex<Receiver<()>>,
    }
    impl ObjectResourceLoader for BlockingLoader {
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
                property_source: Arc::new(crate::model::EmptyControlObjectPropertySource),
                numeric_summaries: Arc::new(Default::default()),
                memory_diagnostics: Arc::new(Default::default()),
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
        Some(Arc::new(BlockingLoader {
            started: started_tx,
            release: Mutex::new(release_rx),
        })),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let (project_path, source_path) = temporary_preload_paths("stale");
    prepare_saved_project(&channels, &project_path, &source_path);

    let (start, start_rx) = request("project.objects.preload.start", json!({}));
    channels.request_tx.send(start).unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("preload worker started");

    let (clear, clear_rx) = request("project.objects.preload.clear", json!({}));
    channels.request_tx.send(clear).unwrap();
    let cleared = clear_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(cleared["cancelled"], true);

    release_tx.send(()).unwrap();
    let error = start_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::Conflict);

    let _ = std::fs::remove_file(project_path);
    let _ = std::fs::remove_file(source_path);
}
