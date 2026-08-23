use super::*;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn open_export_fixture(channels: &ControlActorChannels) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr");
    let (open, open_rx) = request("datasets.open_ome_zarr", json!({"path":fixture}));
    channels.request_tx.send(open).unwrap();
    open_rx
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let (load, load_rx) = request(
        "viewer.objects.source.load",
        json!({"path":"export-polygons.geojson"}),
    );
    channels.request_tx.send(load).unwrap();
    load_rx
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
}

fn output_path(extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "odon-object-export-{}-{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        extension
    ))
}

#[test]
fn object_export_columns_csv_and_geoparquet_complete_without_a_frame() {
    let channels = spawn_test_actor_with_objects();
    open_export_fixture(&channels);

    let (analysis, analysis_rx) = request(
        "viewer.analysis.set",
        json!({"state":{
            "threshold_elements":[{
                "name":"Score positive",
                "scope":{"kind":"composite"},
                "rules":[{"column_key":"score","op":"greater_equal","value":0.5,"value_transform":"none"}]
            }],
            "threshold_selected_element":0,
            "selection_elements":[{"name":"Review cells","object_ids":["cell-a"]}]
        }}),
    );
    channels.request_tx.send(analysis).unwrap();
    analysis_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let (select, select_rx) = request(
        "viewer.objects.selection.select_ids",
        json!({"ids":["cell-a"]}),
    );
    channels.request_tx.send(select).unwrap();
    select_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();

    let (columns, columns_rx) = request("exports.objects.columns", json!({}));
    channels.request_tx.send(columns).unwrap();
    let columns = columns_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let names = columns["columns"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    for expected in [
        "id",
        "score",
        "_odon_geometry_type",
        "_odon_selected",
        "_odon_live_call",
        "_odon_call_score_positive",
        "_odon_selection_review_cells",
    ] {
        assert!(
            names.contains(&expected),
            "missing export column {expected}"
        );
    }

    let csv_path = output_path("csv");
    let csv_columns = [
        "id",
        "score",
        "_odon_selected",
        "_odon_call_score_positive",
        "_odon_selection_review_cells",
    ];
    let (csv_export, csv_rx) = request(
        "exports.objects.export_csv",
        json!({
            "path":csv_path,
            "scope":"selected",
            "columns":csv_columns,
        }),
    );
    channels.request_tx.send(csv_export).unwrap();
    let csv = csv_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(csv["completed"], true);
    assert_eq!(csv["format"], "csv");
    assert_eq!(csv["object_count"], 1);
    let text = std::fs::read_to_string(&csv_path).unwrap();
    assert!(text.starts_with(&csv_columns.join(",")));
    assert!(text.lines().nth(1).unwrap().starts_with("cell-a,"));

    let (no_clobber, no_clobber_rx) = request(
        "exports.objects.export_csv",
        json!({"path":csv_path,"scope":"all"}),
    );
    channels.request_tx.send(no_clobber).unwrap();
    let error = no_clobber_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap_err();
    assert!(error.message.contains("destination exists"));
    assert_eq!(std::fs::read_to_string(&csv_path).unwrap(), text);

    let parquet_path = output_path("geoparquet");
    let (parquet_export, parquet_rx) = request(
        "exports.objects.start",
        json!({
            "path":parquet_path,
            "format":"geoparquet",
            "scope":"all",
            "columns":["id","score"]
        }),
    );
    channels.request_tx.send(parquet_export).unwrap();
    let parquet = parquet_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    assert_eq!(parquet["format"], "geoparquet");
    assert_eq!(parquet["object_count"], 2);
    let reader = SerializedFileReader::new(std::fs::File::open(&parquet_path).unwrap()).unwrap();
    assert_eq!(reader.metadata().file_metadata().num_rows(), 2);
    assert!(
        reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .unwrap()
            .iter()
            .any(|metadata| metadata.key == "geo")
    );

    let (state, state_rx) = request("exports.objects.get_state", json!({}));
    channels.request_tx.send(state).unwrap();
    let state = state_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert_eq!(state["running"], false);
    assert_eq!(state["last_output"]["format"], "geoparquet");

    let _ = std::fs::remove_file(csv_path);
    let _ = std::fs::remove_file(parquet_path);
}

#[test]
fn spatial_shape_object_export_target_is_resolved_by_the_actor() {
    let channels = spawn_test_actor_with_objects();
    open_export_fixture(&channels);
    let (request, reply) = request(
        "exports.objects.columns",
        json!({"target":"spatial_shape","layer_id":7}),
    );
    channels.request_tx.send(request).unwrap();
    let error = reply
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::ResourceNotFound);
}
