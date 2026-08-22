use super::*;
use crate::data::document::{
    AlternateDocumentResource, DocumentDescriptor, DocumentKind, OpenedDocument,
    SpatialDataOpenIdentity, SpatialDataOpenOptions, XeniumOpenIdentity, XeniumOpenOptions,
};

struct FixtureTiffBackend;

fn fixture_document(
    path: &std::path::Path,
    kind: DocumentKind,
) -> anyhow::Result<OpenedDocument<AlternateDocumentResource>> {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (mut dataset, store) = OmeZarrDataset::open_local(&fixture)?;
    dataset.source = DatasetSource::Local(path.to_path_buf());
    let descriptor = DocumentDescriptor::from_alternate(&dataset, kind);
    Ok(OpenedDocument {
        descriptor,
        resource: AlternateDocumentResource::new(dataset, store, Arc::new(())),
    })
}

impl AlternateDatasetBackend for FixtureTiffBackend {
    fn open_tiff(
        &self,
        path: &std::path::Path,
        _z: usize,
        _t: usize,
    ) -> anyhow::Result<OpenedDocument<AlternateDocumentResource>> {
        fixture_document(path, DocumentKind::Tiff)
    }

    fn open_spatialdata(
        &self,
        path: &std::path::Path,
        options: &SpatialDataOpenOptions,
    ) -> anyhow::Result<(
        OpenedDocument<AlternateDocumentResource>,
        SpatialDataOpenIdentity,
    )> {
        Ok((
            fixture_document(path, DocumentKind::SpatialData)?,
            SpatialDataOpenIdentity {
                root: path.to_path_buf(),
                image: options.image.clone(),
                extra_images: options.extra_images.clone(),
                labels: options.labels.clone(),
                shapes: options.shapes.clone(),
                points: options.points.clone(),
                points_max: options.points_max,
            },
        ))
    }

    fn open_xenium(
        &self,
        path: &std::path::Path,
        options: &XeniumOpenOptions,
    ) -> anyhow::Result<(
        OpenedDocument<AlternateDocumentResource>,
        XeniumOpenIdentity,
    )> {
        Ok((
            fixture_document(path, DocumentKind::Xenium)?,
            XeniumOpenIdentity {
                root: path.to_path_buf(),
                imagery: options.imagery.clone(),
                imagery_path: path.join("morphology.ome.zarr"),
                cells_loaded: options.load_cells,
                transcripts_loaded: options.load_transcripts,
                pixel_size_um: 0.2125,
            },
        ))
    }
}

struct BlockingTiffBackend {
    started: Sender<()>,
    release: Mutex<Receiver<()>>,
}

impl AlternateDatasetBackend for BlockingTiffBackend {
    fn open_tiff(
        &self,
        path: &std::path::Path,
        z: usize,
        t: usize,
    ) -> anyhow::Result<OpenedDocument<AlternateDocumentResource>> {
        let _ = self.started.send(());
        let _ = self.release.lock().unwrap().recv();
        FixtureTiffBackend.open_tiff(path, z, t)
    }
}

fn temporary_tiff_path(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "odon-actor-{label}-{}-{}.tiff",
        std::process::id(),
        crate::control::discovery::random_uuid_like().expect("random test suffix")
    ));
    std::fs::File::create(&path).expect("create TIFF-shaped test path");
    path
}

#[test]
fn tiff_open_reaches_resource_readiness_without_a_ui_frame() {
    let path = temporary_tiff_path("tiff");
    let channels = spawn_test_actor_with_alternate(Arc::new(FixtureTiffBackend));
    let (open, open_rx) = request("datasets.open_tiff", json!({"path":path, "z":2, "t":3}));
    channels.request_tx.send(open).unwrap();

    let response = open_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("TIFF open must not need a UI frame")
        .expect("TIFF fixture open succeeds");
    assert_eq!(response["kind"], "tiff");
    assert_eq!(response["plane"], json!({"z":2,"t":3}));
    assert_eq!(response["model_ready"], true);
    assert_eq!(response["resources_ready"], true);
    assert_eq!(response["presentation_ready"], false);
    assert!(channels.legacy_rx.try_recv().is_err());

    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("actor publishes prepared TIFF projection");
    let document = projection.document.expect("prepared TIFF document");
    assert_eq!(document.opened.descriptor.kind, DocumentKind::Tiff);
    assert!(matches!(
        document.opened.resource,
        crate::data::document::DocumentResource::Alternate(_)
    ));

    let (state, state_rx) = request("app.get_loading_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("state reply")
        .expect("state succeeds");
    assert_eq!(state["mode"], "single");
    assert_eq!(state["loading"]["resources_ready"], true);

    let _ = std::fs::remove_file(path);
}

#[test]
fn superseding_open_rejects_a_stale_tiff_completion() {
    let path = temporary_tiff_path("stale-tiff");
    let (started_tx, started_rx) = crossbeam_channel::bounded(1);
    let (release_tx, release_rx) = crossbeam_channel::bounded(1);
    let channels = spawn_test_actor_with_alternate(Arc::new(BlockingTiffBackend {
        started: started_tx,
        release: Mutex::new(release_rx),
    }));

    let (tiff, tiff_rx) = request("datasets.open_tiff", json!({"path":path}));
    channels.request_tx.send(tiff).unwrap();
    started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("TIFF worker started");

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (replacement, replacement_rx) =
        request("datasets.open_ome_zarr", json!({"path":fixture.clone()}));
    channels.request_tx.send(replacement).unwrap();
    replacement_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("replacement open finishes without a frame")
        .expect("replacement open succeeds");

    release_tx.send(()).unwrap();
    let error = tiff_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("stale TIFF completion replies")
        .expect_err("stale TIFF result is rejected");
    assert_eq!(error.kind, ControlErrorKind::Conflict);

    let (state, state_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("state reply")
        .expect("state succeeds");
    assert_eq!(state["mode"], "single");
    assert_eq!(
        state["view"]["dataset"],
        format!("local:{}", fixture.to_string_lossy())
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn spatialdata_and_xenium_open_without_a_ui_frame() {
    let channels = spawn_test_actor_with_alternate(Arc::new(FixtureTiffBackend));
    let spatial_root = std::env::temp_dir().join("odon-spatialdata-fixture");
    let (spatial, spatial_rx) = request(
        "datasets.open_spatialdata",
        json!({
            "path":spatial_root,
            "image":"morphology",
            "extra_images":["nuclei"],
            "labels":"cells",
            "shapes":["cell_boundaries"],
            "points":"transcripts",
            "points_max":1234,
        }),
    );
    channels.request_tx.send(spatial).unwrap();
    let response = spatial_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("SpatialData does not wait for a frame")
        .expect("SpatialData fixture succeeds");
    assert_eq!(response["kind"], "spatialdata");
    assert_eq!(response["image"], "morphology");
    assert_eq!(response["extra_images"], json!(["nuclei"]));
    assert_eq!(response["points_max"], 1234);
    assert_eq!(response["presentation_ready"], false);
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("SpatialData projection");
    assert_eq!(
        projection.document.unwrap().opened.descriptor.kind,
        DocumentKind::SpatialData
    );

    let xenium_root = std::env::temp_dir().join("odon-xenium-fixture");
    let (xenium, xenium_rx) = request(
        "datasets.open_xenium",
        json!({
            "path":xenium_root,
            "imagery":"tiff",
            "load_cells":false,
            "load_transcripts":true,
        }),
    );
    channels.request_tx.send(xenium).unwrap();
    let response = xenium_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("Xenium does not wait for a frame")
        .expect("Xenium fixture succeeds");
    assert_eq!(response["kind"], "xenium");
    assert_eq!(response["imagery"], "tiff");
    assert_eq!(response["cells_loaded"], false);
    assert_eq!(response["transcripts_loaded"], true);
    assert_eq!(response["presentation_ready"], false);
    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("Xenium projection");
    assert_eq!(
        projection.document.unwrap().opened.descriptor.kind,
        DocumentKind::Xenium
    );
    assert!(channels.legacy_rx.try_recv().is_err());
}
