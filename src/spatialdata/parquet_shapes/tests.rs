use super::geometry::centroid_summary_from_wkb;
use crate::spatialdata::SpatialDataTransform2;

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
