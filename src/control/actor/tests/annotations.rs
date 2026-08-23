use super::*;
use arrow_array::{Float32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::fs::File;

fn annotation_fixture() -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("odon-annotations-{nonce}.parquet"));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("x_centroid", DataType::Float32, false),
        Field::new("y_centroid", DataType::Float32, false),
        Field::new("cluster_label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["roi-a", "roi-a", "roi-b"])),
            Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float32Array::from(vec![4.0, 5.0, 6.0])),
            Arc::new(StringArray::from(vec!["tumour", "immune", "tumour"])),
        ],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

fn annotation_state(id: u64, path: &str) -> Value {
    json!({
        "id":id,
        "name":"Restored cells",
        "visible":true,
        "radius_screen_px":4.0,
        "opacity":0.9,
        "stroke_width":1.0,
        "stroke_color_rgb":[0,0,0],
        "stroke_color_alpha":140,
        "offset_world":[0.0,0.0],
        "parquet_path":path,
        "roi_id_column":"id",
        "x_column":"x_centroid",
        "y_column":"y_centroid",
        "value_column":"cluster_label",
        "selected_value_column":"cluster_label",
        "category_styles":[],
        "continuous_shape":"circle"
    })
}

#[test]
fn annotation_crud_and_parquet_loading_complete_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();

    let (create, create_rx) = request(
        "viewer.annotations.layers.create",
        json!({"name":"Cell classes","opacity":0.6}),
    );
    channels.request_tx.send(create).unwrap();
    let created = create_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let id = created["id"].as_u64().unwrap();
    assert_eq!(created["name"], "Cell classes");
    assert!((created["opacity"].as_f64().unwrap() - 0.6).abs() < 1e-5);

    let path = annotation_fixture();
    let (inspect, inspect_rx) = request(
        "viewer.annotations.source.inspect",
        json!({"layer_id":id,"path":path}),
    );
    channels.request_tx.send(inspect).unwrap();
    let inspected = inspect_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(
        inspected["schema"],
        json!(["id", "x_centroid", "y_centroid", "cluster_label"])
    );
    assert!(inspected["resource"].is_null());

    let (load, load_rx) = request(
        "viewer.annotations.source.load",
        json!({"layer_id":id,"path":path}),
    );
    channels.request_tx.send(load).unwrap();
    let loaded = load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
    assert_eq!(loaded["resource"]["mode"], "categorical");
    assert_eq!(loaded["resource"]["total_points"], 3);
    assert_eq!(loaded["resource"]["total_rois"], 2);
    assert_eq!(
        loaded["resource"]["categories"],
        json!(["tumour", "immune"])
    );

    let projection = channels.presentation_rx.try_recv().unwrap();
    let annotation = projection
        .annotation_layers
        .iter()
        .find(|layer| layer.state.id == id)
        .unwrap();
    assert_eq!(
        annotation.resource.as_ref().unwrap().dataset.total_points,
        3
    );
    assert_eq!(annotation.state.category_styles.len(), 2);
    let native = projection.workspace.as_ref().unwrap()["viewports"][0]["native_layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|layer| layer["layer_id"] == format!("annotation:{id}"))
        .unwrap();
    assert_eq!(native["kind"], "annotation");

    let (update, update_rx) = request(
        "viewer.annotations.layers.update",
        json!({"layer_id":id,"state":{"visible":false,"radius_screen_px":7.0}}),
    );
    channels.request_tx.send(update).unwrap();
    let updated = update_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(updated["visible"], false);
    assert_eq!(updated["radius_screen_px"], 7.0);
    assert_eq!(updated["resource"]["total_points"], 3);

    let (clear, clear_rx) = request("viewer.annotations.source.clear", json!({"layer_id":id}));
    channels.request_tx.send(clear).unwrap();
    let cleared = clear_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert!(cleared["parquet_path"].is_null());
    assert!(cleared["resource"].is_null());

    let (delete, delete_rx) = request("viewer.annotations.layers.delete", json!({"layer_id":id}));
    channels.request_tx.send(delete).unwrap();
    assert_eq!(
        delete_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["deleted"],
        true
    );
    let (list, list_rx) = request("viewer.annotations.layers.list", json!({}));
    channels.request_tx.send(list).unwrap();
    assert_eq!(
        list_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap()["layers"],
        json!([])
    );

    std::fs::remove_file(path).unwrap();
}

#[test]
fn persisted_annotation_source_restores_on_the_actor_without_a_ui_frame() {
    let channels = spawn_test_actor();
    let dataset_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let source_key = DatasetSource::Local(dataset_path.clone()).source_key();
    let annotation_path = annotation_fixture();
    let filename = annotation_path.file_name().unwrap().to_string_lossy();
    let project_path = annotation_path.with_file_name("restored-annotations.odon.json");
    let snapshot = ProjectModelSnapshot {
        state: json!({
            "browser":{},
            "roi_views":{
                source_key.clone():{
                    "annotation_layers":[annotation_state(7, &filename)]
                }
            }
        }),
        saved_path: Some(project_path),
        load_generation: 1,
        ..ProjectModelSnapshot::default()
    };
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapProject(snapshot))
        .unwrap();
    let (barrier, barrier_rx) = request("app.get_state", json!({}));
    channels.request_tx.send(barrier).unwrap();
    barrier_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let opened = open_local_ome_zarr(&dataset_path).unwrap();
    channels
        .model_tx
        .send(ActorModelUpdate::BootstrapDataset {
            dataset: opened.resource.dataset,
            store: opened.resource.store,
        })
        .unwrap();

    let projection = channels
        .presentation_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("restored annotation worker publishes a projection without a frame");
    let restored = projection
        .annotation_layers
        .iter()
        .find(|layer| layer.state.id == 7)
        .expect("persisted annotation remains in the projection");
    assert_eq!(restored.resource.as_ref().unwrap().dataset.total_points, 3);
    assert_eq!(
        restored.state.parquet_path.as_deref(),
        Some(filename.as_ref())
    );

    std::fs::remove_file(annotation_path).unwrap();
}
