use super::*;
#[test]
fn background_worker_preserves_channel_intensity_statistics() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (dataset, store) = OmeZarrDataset::open_local(&fixture).expect("open OME-Zarr fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let app = OmeZarrViewerApp::new_runtime(
        &egui::Context::default(),
        false,
        dataset.clone(),
        Arc::clone(&store),
        AutoContrastSettings {
            enabled_on_open: false,
            ..AutoContrastSettings::default()
        },
    );
    let params = serde_json::json!({"channel":"PanCK","level":0});
    let renderer = app.control_get_channel_intensity_stats(&params);
    let spec = model
        .channel_intensity_spec(&dataset, &params)
        .expect("actor plans the same image subset");
    let document = odon::control::actor::RenderDocument {
        generation: model.document_generation(),
        opened: odon::data::document::OpenedDocument {
            descriptor: odon::data::document::DocumentDescriptor::from_ome_zarr(&dataset),
            resource: odon::data::document::OmeZarrDocumentResource { dataset, store },
        },
    };
    let actor = odon::control::actor::read_channel_intensity_stats(&document, &spec)
        .expect("background worker reads fixture");
    assert_eq!(actor, renderer);
}
