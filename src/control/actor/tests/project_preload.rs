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
