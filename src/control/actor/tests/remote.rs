use super::*;
use crate::data::dataset_source::DatasetSource;
use crate::data::document::{OmeZarrDocumentResource, OpenedDocument, open_local_ome_zarr};
use crate::data::remote_store::{
    RemoteDatasetBackend, S3BrowseEntry, S3BrowseListing, S3SessionCredentials,
};

struct FixtureRemoteBackend {
    fixture: PathBuf,
}

impl FixtureRemoteBackend {
    fn opened(
        &self,
        source: DatasetSource,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        let mut opened = open_local_ome_zarr(&self.fixture)?;
        opened.descriptor.source = source.clone();
        opened.resource.dataset.source = source;
        Ok(opened)
    }
}

impl RemoteDatasetBackend for FixtureRemoteBackend {
    fn list_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<S3BrowseListing> {
        if prefix == "secret-error" {
            anyhow::bail!(
                "backend echoed {} and {}",
                credentials.access_key,
                credentials.secret_key
            );
        }
        Ok(S3BrowseListing {
            prefix: prefix.to_string(),
            parent_prefix: Some("study".to_string()),
            entries: vec![S3BrowseEntry {
                prefix: "study/roi.ome.zarr".to_string(),
                name: "roi.ome.zarr".to_string(),
                is_dataset: true,
            }],
            current_is_dataset: false,
        })
    }

    fn open_http(&self, url: &str) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        self.opened(DatasetSource::Http {
            base_url: url.to_string(),
        })
    }

    fn open_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        self.opened(DatasetSource::S3 {
            endpoint: credentials.endpoint.clone(),
            region: credentials.region.clone(),
            bucket: credentials.bucket.clone(),
            prefix: prefix.to_string(),
        })
    }
}

fn fixture_backend() -> Arc<dyn RemoteDatasetBackend> {
    Arc::new(FixtureRemoteBackend {
        fixture: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr"),
    })
}

#[test]
fn remote_sessions_listing_and_opens_complete_without_a_ui_frame() {
    let channels = spawn_test_actor_with_remote(fixture_backend());
    let access_key = "access-key-must-stay-secret";
    let secret_key = "secret-key-must-stay-secret";

    let (configure, configure_rx) = request(
        "datasets.s3.configure_session",
        json!({
            "endpoint":"objects.example.test/",
            "region":"auto",
            "bucket":"images",
            "access_key":access_key,
            "secret_key":secret_key,
        }),
    );
    channels.request_tx.send(configure).unwrap();
    let configured = configure_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(configured["configured"], true);
    assert_eq!(configured["endpoint"], "https://objects.example.test");
    assert_eq!(configured["credentials"], "session_only_redacted");
    let configured_text = configured.to_string();
    assert!(!configured_text.contains(access_key));
    assert!(!configured_text.contains(secret_key));

    let event_hub = EventHub::shared();
    let (event_tx, event_rx) = crossbeam_channel::bounded(2);
    event_hub.register("observer".to_string(), event_tx);
    event_hub
        .subscribe("observer", vec!["datasets.credentials.changed".to_string()])
        .unwrap();
    let (configure_reply, configure_reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode(
                "datasets.s3.configure_session",
                json!({
                    "endpoint":"objects.example.test/",
                    "region":"auto",
                    "bucket":"images",
                    "access_key":access_key,
                    "secret_key":secret_key,
                }),
            )
            .unwrap(),
            reply: configure_reply,
            session_id: "initiator".to_string(),
            request_id: Some(json!(7)),
            event_hub: Arc::clone(&event_hub),
            task_registry: TaskRegistry::shared(event_hub),
            task_id: None,
        })
        .unwrap();
    configure_reply_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let event_text = event_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("credential event")
        .to_string();
    assert!(!event_text.contains(access_key));
    assert!(!event_text.contains(secret_key));

    let (list, list_rx) = request("datasets.s3.list", json!({"prefix":"/study/"}));
    channels.request_tx.send(list).unwrap();
    let listing = list_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(listing["prefix"], "study");
    assert_eq!(listing["entries"][0]["is_dataset"], true);
    assert!(!listing.to_string().contains(access_key));
    assert!(!listing.to_string().contains(secret_key));

    let (failure, failure_rx) = request("datasets.s3.list", json!({"prefix":"secret-error"}));
    channels.request_tx.send(failure).unwrap();
    let failure = failure_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert!(failure.message.contains("[redacted]"));
    assert!(!failure.message.contains(access_key));
    assert!(!failure.message.contains(secret_key));

    let (open_s3, open_s3_rx) = request("datasets.open_s3", json!({"prefix":"study/roi.ome.zarr"}));
    channels.request_tx.send(open_s3).unwrap();
    let opened_s3 = open_s3_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(opened_s3["kind"], "s3_ome_zarr");
    assert_eq!(opened_s3["resources_ready"], true);
    assert!(!opened_s3.to_string().contains(access_key));
    assert!(!opened_s3.to_string().contains(secret_key));

    let (open_http, open_http_rx) = request(
        "datasets.open_http",
        json!({"url":"https://images.example.test/remote.ome.zarr/"}),
    );
    channels.request_tx.send(open_http).unwrap();
    let opened_http = open_http_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(opened_http["kind"], "http_ome_zarr");
    assert_eq!(
        opened_http["url"],
        "https://images.example.test/remote.ome.zarr"
    );

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(state["mode"], "single");
    assert!(!state.to_string().contains(access_key));
    assert!(!state.to_string().contains(secret_key));
    assert_eq!(channels.legacy_rx.len(), 0);
}

struct BlockingRemoteBackend {
    fixture: PathBuf,
    started: Sender<()>,
    release: Mutex<Receiver<()>>,
}

impl RemoteDatasetBackend for BlockingRemoteBackend {
    fn list_s3(
        &self,
        _credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<S3BrowseListing> {
        self.started.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        Ok(S3BrowseListing {
            prefix: prefix.to_string(),
            parent_prefix: None,
            entries: Vec::new(),
            current_is_dataset: false,
        })
    }

    fn open_http(&self, url: &str) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        let mut opened = open_local_ome_zarr(&self.fixture)?;
        let source = DatasetSource::Http {
            base_url: url.to_string(),
        };
        opened.descriptor.source = source.clone();
        opened.resource.dataset.source = source;
        Ok(opened)
    }

    fn open_s3(
        &self,
        _credentials: &S3SessionCredentials,
        _prefix: &str,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        unreachable!("this test blocks only S3 listing")
    }
}

#[test]
fn clearing_a_session_invalidates_in_flight_remote_work() {
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let backend: Arc<dyn RemoteDatasetBackend> = Arc::new(BlockingRemoteBackend {
        fixture: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr"),
        started: started_tx,
        release: Mutex::new(release_rx),
    });
    let channels = spawn_test_actor_with_remote(backend);
    let (configure, configure_rx) = request(
        "datasets.s3.configure_session",
        json!({
            "endpoint":"https://objects.example.test",
            "region":"auto",
            "bucket":"images",
            "access_key":"access",
            "secret_key":"secret",
        }),
    );
    channels.request_tx.send(configure).unwrap();
    configure_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (list, list_rx) = request("datasets.s3.list", json!({"prefix":"study"}));
    channels.request_tx.send(list).unwrap();
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let (clear, clear_rx) = request("datasets.s3.clear_session", json!({}));
    channels.request_tx.send(clear).unwrap();
    let cleared = clear_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("actor remains responsive while worker is blocked")
        .unwrap();
    assert_eq!(cleared["configured"], false);

    release_tx.send(()).unwrap();
    let stale = list_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(stale.kind, ControlErrorKind::Conflict);

    let (loading, loading_rx) = request("app.get_loading_state", json!({}));
    channels.request_tx.send(loading).unwrap();
    let loading = loading_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        loading["loading"]["operations"]["remote_listing:1:study"]["phase"],
        "cancelled"
    );

    let (configure, configure_rx) = request(
        "datasets.s3.configure_session",
        json!({
            "endpoint":"https://objects.example.test",
            "region":"auto",
            "bucket":"images",
            "access_key":"access",
            "secret_key":"secret",
        }),
    );
    channels.request_tx.send(configure).unwrap();
    configure_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let task = channels
        .task_service
        .create("remote listing", "test", true)
        .unwrap();
    let task_events = EventHub::shared();
    let (task_reply, task_reply_rx) = crossbeam_channel::bounded(1);
    channels
        .request_tx
        .send(OdonControlRequest {
            command: ControlCommand::decode("datasets.s3.list", json!({"prefix":"cancel-me"}))
                .unwrap(),
            reply: task_reply,
            session_id: "test".to_string(),
            request_id: None,
            event_hub: task_events,
            task_registry: channels.task_service.registry(),
            task_id: Some(task.task_id.clone()),
        })
        .unwrap();
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        channels.task_service.cancel(&task.task_id).unwrap().state,
        TaskState::Cancelled
    );
    release_tx.send(()).unwrap();
    let cancelled = task_reply_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(cancelled.kind, ControlErrorKind::Cancelled);
    assert_eq!(channels.legacy_rx.len(), 0);
}

struct BlockingS3OpenBackend {
    fixture: PathBuf,
    started: Sender<()>,
    release: Mutex<Receiver<()>>,
}

impl RemoteDatasetBackend for BlockingS3OpenBackend {
    fn list_s3(
        &self,
        _credentials: &S3SessionCredentials,
        _prefix: &str,
    ) -> anyhow::Result<S3BrowseListing> {
        unreachable!("this test blocks only S3 opening")
    }

    fn open_http(&self, _url: &str) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        unreachable!("this test blocks only S3 opening")
    }

    fn open_s3(
        &self,
        credentials: &S3SessionCredentials,
        prefix: &str,
    ) -> anyhow::Result<OpenedDocument<OmeZarrDocumentResource>> {
        self.started.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        let mut opened = open_local_ome_zarr(&self.fixture)?;
        let source = DatasetSource::S3 {
            endpoint: credentials.endpoint.clone(),
            region: credentials.region.clone(),
            bucket: credentials.bucket.clone(),
            prefix: prefix.to_string(),
        };
        opened.descriptor.source = source.clone();
        opened.resource.dataset.source = source;
        Ok(opened)
    }
}

#[test]
fn clearing_a_session_prevents_a_stale_s3_document_install() {
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let backend: Arc<dyn RemoteDatasetBackend> = Arc::new(BlockingS3OpenBackend {
        fixture: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr"),
        started: started_tx,
        release: Mutex::new(release_rx),
    });
    let channels = spawn_test_actor_with_remote(backend);
    let (configure, configure_rx) = request(
        "datasets.s3.configure_session",
        json!({
            "endpoint":"https://objects.example.test",
            "region":"auto",
            "bucket":"images",
            "access_key":"access",
            "secret_key":"secret",
        }),
    );
    channels.request_tx.send(configure).unwrap();
    configure_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (open, open_rx) = request("datasets.open_s3", json!({"prefix":"study/roi.ome.zarr"}));
    channels.request_tx.send(open).unwrap();
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let (clear, clear_rx) = request("datasets.s3.clear_session", json!({}));
    channels.request_tx.send(clear).unwrap();
    clear_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("session clear does not wait for remote open")
        .unwrap();
    release_tx.send(()).unwrap();
    let stale = open_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(stale.kind, ControlErrorKind::Conflict);

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    assert_eq!(
        state_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["mode"],
        "project"
    );
    assert_eq!(channels.legacy_rx.len(), 0);
}
