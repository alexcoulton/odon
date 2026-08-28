use super::geometry::centroid_summary_from_wkb;
use super::load_shapes_f32_property_column;
use crate::spatialdata::SpatialDataTransform2;
use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn le_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn le_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

#[test]
fn centroid_summary_reads_polygon_without_persisting_rings() {
    let mut wkb = Vec::new();
    wkb.push(1);
    le_u32(&mut wkb, 3);
    le_u32(&mut wkb, 1);
    le_u32(&mut wkb, 5);
    for (x, y) in [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 20.0),
        (0.0, 20.0),
        (0.0, 0.0),
    ] {
        le_f64(&mut wkb, x);
        le_f64(&mut wkb, y);
    }

    let summary = centroid_summary_from_wkb(&wkb, &SpatialDataTransform2::default(), None)
        .expect("valid WKB")
        .expect("polygon summary");

    assert!((summary.centroid_world.x - 5.0).abs() < 1e-4);
    assert!((summary.centroid_world.y - 10.0).abs() < 1e-4);
    assert!((summary.area_px - 200.0).abs() < 1e-4);
    assert!((summary.perimeter_px - 60.0).abs() < 1e-4);
}

#[test]
fn floating_parquet_property_streams_to_compact_values_and_validity() {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "odon-compact-property-{}-{}.parquet",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let schema = Arc::new(Schema::new(vec![Field::new(
        "score",
        DataType::Float64,
        true,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Float64Array::from(vec![
            Some(1.25),
            None,
            Some(-2.5),
        ]))],
    )
    .expect("record batch");
    let file = std::fs::File::create(&path).expect("create parquet fixture");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write parquet fixture");
    writer.close().expect("close parquet fixture");

    let values = load_shapes_f32_property_column(&path, "score", &AtomicBool::new(false))
        .expect("load floating property")
        .expect("floating fast path");
    let _ = std::fs::remove_file(path);

    assert_eq!(values.len(), 3);
    assert_eq!(values.validity_word_len(), 1);
    assert_eq!(values.get(0), Some(1.25));
    assert_eq!(values.get(1), None);
    assert_eq!(values.get(2), Some(-2.5));
}
