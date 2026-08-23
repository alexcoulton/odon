use arrow_array::Array;
use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, UInt32Array, UInt64Array,
};
use serde_json::Value;

use super::{ShapesLoadOptions, ShapesRenderKind};
use crate::spatialdata::SpatialDataTransform2;

pub(super) fn parse_wkb_object_polygons(
    bytes: &[u8],
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    let mut out = Vec::new();
    flatten_geom_object_polygons(&geom, xform, radius_world, &mut out);
    Ok(out)
}

pub(super) fn render_kind_from_wkb(
    bytes: &[u8],
    has_radius: bool,
) -> anyhow::Result<Option<ShapesRenderKind>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    Ok(match classify_geom_kind(&geom) {
        GeomKind::Pointish if has_radius => Some(ShapesRenderKind::Circles),
        GeomKind::Pointish => Some(ShapesRenderKind::Points),
        GeomKind::Linear | GeomKind::Polygonal => Some(ShapesRenderKind::Lines),
        GeomKind::Unsupported => None,
    })
}

pub(super) fn circle_polylines_from_wkb(
    bytes: &[u8],
    radius_world: f32,
    xform: &SpatialDataTransform2,
    segments: usize,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    let mut centers = Vec::new();
    flatten_geom_points(&geom, xform, &mut centers);
    Ok(centers
        .into_iter()
        .map(|center| circle_polyline(center, radius_world, segments))
        .collect())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CentroidSummary {
    pub(super) centroid_world: eframe::egui::Pos2,
    pub(super) bbox_world: eframe::egui::Rect,
    pub(super) area_px: f32,
    pub(super) perimeter_px: f32,
}

pub(super) fn centroid_summary_from_wkb(
    bytes: &[u8],
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
) -> anyhow::Result<Option<CentroidSummary>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    Ok(summarize_geom_centroid(&geom, xform, radius_world))
}

#[derive(Debug, Clone)]
struct SummaryBuilder {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    area_sum: f32,
    perimeter_sum: f32,
    centroid_num: eframe::egui::Vec2,
    point_sum: eframe::egui::Vec2,
    point_count: usize,
}

impl SummaryBuilder {
    fn new() -> Self {
        Self {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
            area_sum: 0.0,
            perimeter_sum: 0.0,
            centroid_num: eframe::egui::Vec2::ZERO,
            point_sum: eframe::egui::Vec2::ZERO,
            point_count: 0,
        }
    }

    fn add_point(&mut self, point: eframe::egui::Pos2) {
        if !(point.x.is_finite() && point.y.is_finite()) {
            return;
        }
        self.min_x = self.min_x.min(point.x);
        self.min_y = self.min_y.min(point.y);
        self.max_x = self.max_x.max(point.x);
        self.max_y = self.max_y.max(point.y);
        self.point_sum += point.to_vec2();
        self.point_count += 1;
    }

    fn add_polyline(&mut self, points: &[eframe::egui::Pos2]) {
        if points.is_empty() {
            return;
        }
        for &point in points {
            self.add_point(point);
        }
        for window in points.windows(2) {
            self.perimeter_sum += (window[1] - window[0]).length();
        }
        if let Some((area, centroid)) = polygon_area_and_centroid(points) {
            self.area_sum += area;
            self.centroid_num += centroid.to_vec2() * area;
        }
    }

    fn finish(self) -> Option<CentroidSummary> {
        if self.point_count == 0 {
            return None;
        }
        let bbox = eframe::egui::Rect::from_min_max(
            eframe::egui::pos2(self.min_x, self.min_y),
            eframe::egui::pos2(self.max_x, self.max_y),
        );
        let centroid = if self.area_sum > 1e-6 {
            (self.centroid_num / self.area_sum).to_pos2()
        } else {
            (self.point_sum / self.point_count as f32).to_pos2()
        };
        let min_side = 2.0f32;
        let bbox_world = if bbox.is_positive() {
            bbox.expand(0.5)
        } else {
            eframe::egui::Rect::from_center_size(centroid, eframe::egui::Vec2::splat(min_side))
        };
        Some(CentroidSummary {
            centroid_world: centroid,
            bbox_world,
            area_px: self.area_sum.max(0.0),
            perimeter_px: self.perimeter_sum.max(0.0),
        })
    }
}

fn summarize_geom_centroid(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
) -> Option<CentroidSummary> {
    let mut builder = SummaryBuilder::new();
    add_geom_summary(geom, xform, radius_world, &mut builder);
    builder.finish()
}

fn add_geom_summary(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
    out: &mut SummaryBuilder,
) {
    match geom {
        Geom::Point { pt } => {
            let q = xform.apply([pt[0] as f32, pt[1] as f32]);
            let point = eframe::egui::pos2(q[0], q[1]);
            out.add_point(point);
            if let Some(radius) = radius_world.filter(|radius| radius.is_finite() && *radius > 0.0)
            {
                out.area_sum += std::f32::consts::PI * radius * radius;
                out.perimeter_sum += std::f32::consts::TAU * radius;
                out.min_x = out.min_x.min(point.x - radius);
                out.min_y = out.min_y.min(point.y - radius);
                out.max_x = out.max_x.max(point.x + radius);
                out.max_y = out.max_y.max(point.y + radius);
            }
        }
        Geom::MultiPoint { pts } => {
            for pt in pts {
                add_geom_summary(&Geom::Point { pt: *pt }, xform, radius_world, out);
            }
        }
        Geom::Polygon { rings } => {
            if let Some(ring) = rings.first() {
                let points = ring
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                out.add_polyline(&points);
            }
        }
        Geom::MultiPolygon { polys } => {
            for poly in polys {
                add_geom_summary(poly, xform, radius_world, out);
            }
        }
        Geom::LineString { pts } => {
            let points = pts
                .iter()
                .map(|p| {
                    let q = xform.apply([p[0] as f32, p[1] as f32]);
                    eframe::egui::pos2(q[0], q[1])
                })
                .collect::<Vec<_>>();
            out.add_polyline(&points);
        }
        Geom::MultiLineString { lines } => {
            for line in lines {
                let points = line
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                out.add_polyline(&points);
            }
        }
        Geom::Unsupported => {}
    }
}

fn polygon_area_and_centroid(points: &[eframe::egui::Pos2]) -> Option<(f32, eframe::egui::Pos2)> {
    if points.len() < 4 {
        return None;
    }
    let mut cross_sum = 0.0f32;
    let mut cx_sum = 0.0f32;
    let mut cy_sum = 0.0f32;

    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        let cross = a.x * b.y - b.x * a.y;
        cross_sum += cross;
        cx_sum += (a.x + b.x) * cross;
        cy_sum += (a.y + b.y) * cross;
    }
    let area_signed = cross_sum * 0.5;
    let area = area_signed.abs();
    if area <= 1e-6 {
        return None;
    }
    let denom = 3.0 * cross_sum;
    if denom.abs() <= 1e-6 {
        return None;
    }
    Some((area, eframe::egui::pos2(cx_sum / denom, cy_sum / denom)))
}

pub(super) fn geometry_bytes_at<'a>(geom: &'a dyn Array, row: usize) -> Option<&'a [u8]> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        return (!col.is_null(row)).then(|| col.value(row));
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        return (!col.is_null(row)).then(|| col.value(row));
    }
    None
}

pub(super) fn geometry_array_len(geom: &dyn Array) -> anyhow::Result<usize> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        return Ok(col.len());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        return Ok(col.len());
    }
    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)");
}

pub(super) fn array_value_to_json(arr: &dyn Array, row: usize) -> Option<Value> {
    if row >= arr.len() || arr.is_null(row) {
        return None;
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Some(Value::from(col.value(row) as i64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Some(Value::from(col.value(row) as i64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt64Array>() {
        return Some(Value::from(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt8Array>() {
        return Some(Value::from(col.value(row) as u64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt16Array>() {
        return Some(Value::from(col.value(row) as u64));
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float32Array>() {
        return serde_json::Number::from_f64(col.value(row) as f64).map(Value::Number);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float64Array>() {
        return serde_json::Number::from_f64(col.value(row)).map(Value::Number);
    }
    if let Some(col) = arr.as_any().downcast_ref::<BooleanArray>() {
        return Some(Value::Bool(col.value(row)));
    }
    if let Some(col) = arr.as_any().downcast_ref::<StringArray>() {
        return Some(Value::String(col.value(row).to_string()));
    }
    if let Some(col) = arr.as_any().downcast_ref::<LargeStringArray>() {
        return Some(Value::String(col.value(row).to_string()));
    }
    None
}

pub(super) fn array_value_to_f64(arr: &dyn Array, row: usize) -> Option<f64> {
    if row >= arr.len() || arr.is_null(row) {
        return None;
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Int64Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int8Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::Int16Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<UInt64Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt8Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<arrow_array::UInt16Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float32Array>() {
        return Some(col.value(row) as f64);
    }
    if let Some(col) = arr.as_any().downcast_ref::<Float64Array>() {
        return Some(col.value(row));
    }
    None
}

pub(super) fn append_geoms(
    geom: &dyn Array,
    options: &ShapesLoadOptions,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) -> anyhow::Result<()> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            out.extend(parse_wkb_polylines_exterior(bytes, &options.transform)?);
        }
        return Ok(());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            out.extend(parse_wkb_polylines_exterior(bytes, &options.transform)?);
        }
        return Ok(());
    }

    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)")
}

pub(super) fn append_geom_points(
    geom: &dyn Array,
    options: &ShapesLoadOptions,
    out: &mut Vec<eframe::egui::Pos2>,
) -> anyhow::Result<()> {
    if let Some(col) = geom.as_any().downcast_ref::<arrow_array::BinaryArray>() {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            flatten_geom_points(&geom, &options.transform, out);
        }
        return Ok(());
    }
    if let Some(col) = geom
        .as_any()
        .downcast_ref::<arrow_array::LargeBinaryArray>()
    {
        for i in 0..col.len() {
            if col.is_null(i) {
                continue;
            }
            let bytes = col.value(i);
            let mut cur = Cursor::new(bytes);
            let geom = read_geom(&mut cur)?;
            flatten_geom_points(&geom, &options.transform, out);
        }
        return Ok(());
    }
    anyhow::bail!("unsupported geometry column type (expected binary/largebinary)")
}

fn parse_wkb_polylines_exterior(
    bytes: &[u8],
    xform: &SpatialDataTransform2,
) -> anyhow::Result<Vec<Vec<eframe::egui::Pos2>>> {
    let mut cur = Cursor::new(bytes);
    let geom = read_geom(&mut cur)?;
    let mut out = Vec::new();
    flatten_geom_exterior(&geom, xform, &mut out);
    Ok(out)
}

#[derive(Debug)]
enum Geom {
    Point { pt: [f64; 2] },
    MultiPoint { pts: Vec<[f64; 2]> },
    Polygon { rings: Vec<Vec<[f64; 2]>> },
    MultiPolygon { polys: Vec<Geom> },
    LineString { pts: Vec<[f64; 2]> },
    MultiLineString { lines: Vec<Vec<[f64; 2]>> },
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeomKind {
    Pointish,
    Linear,
    Polygonal,
    Unsupported,
}

fn flatten_geom_exterior(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) {
    match geom {
        Geom::Point { .. } | Geom::MultiPoint { .. } => {}
        Geom::Polygon { rings } => {
            if let Some(r0) = rings.first() {
                let pts = r0
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                if pts.len() >= 2 {
                    out.push(pts);
                }
            }
        }
        Geom::MultiPolygon { polys } => {
            for g in polys {
                flatten_geom_exterior(g, xform, out);
            }
        }
        Geom::LineString { pts } => {
            let pts = pts
                .iter()
                .map(|p| {
                    let q = xform.apply([p[0] as f32, p[1] as f32]);
                    eframe::egui::pos2(q[0], q[1])
                })
                .collect::<Vec<_>>();
            if pts.len() >= 2 {
                out.push(pts);
            }
        }
        Geom::MultiLineString { lines } => {
            for l in lines {
                let pts = l
                    .iter()
                    .map(|p| {
                        let q = xform.apply([p[0] as f32, p[1] as f32]);
                        eframe::egui::pos2(q[0], q[1])
                    })
                    .collect::<Vec<_>>();
                if pts.len() >= 2 {
                    out.push(pts);
                }
            }
        }
        Geom::Unsupported => {}
    }
}

fn flatten_geom_object_polygons(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    radius_world: Option<f32>,
    out: &mut Vec<Vec<eframe::egui::Pos2>>,
) {
    match geom {
        Geom::Point { pt } => {
            let r = radius_world.unwrap_or(4.0).max(1e-3);
            out.push(circle_polyline_transformed(*pt, r, xform, 24));
        }
        Geom::MultiPoint { pts } => {
            let r = radius_world.unwrap_or(4.0).max(1e-3);
            for &pt in pts {
                out.push(circle_polyline_transformed(pt, r, xform, 24));
            }
        }
        Geom::Polygon { .. } | Geom::MultiPolygon { .. } => {
            flatten_geom_exterior(geom, xform, out);
        }
        Geom::LineString { .. } | Geom::MultiLineString { .. } | Geom::Unsupported => {}
    }
}

fn flatten_geom_points(
    geom: &Geom,
    xform: &SpatialDataTransform2,
    out: &mut Vec<eframe::egui::Pos2>,
) {
    match geom {
        Geom::Point { pt } => {
            let q = xform.apply([pt[0] as f32, pt[1] as f32]);
            out.push(eframe::egui::pos2(q[0], q[1]));
        }
        Geom::MultiPoint { pts } => {
            for pt in pts {
                let q = xform.apply([pt[0] as f32, pt[1] as f32]);
                out.push(eframe::egui::pos2(q[0], q[1]));
            }
        }
        Geom::MultiPolygon { polys } => {
            for g in polys {
                flatten_geom_points(g, xform, out);
            }
        }
        _ => {}
    }
}

fn classify_geom_kind(geom: &Geom) -> GeomKind {
    match geom {
        Geom::Point { .. } | Geom::MultiPoint { .. } => GeomKind::Pointish,
        Geom::Polygon { .. } | Geom::MultiPolygon { .. } => GeomKind::Polygonal,
        Geom::LineString { .. } | Geom::MultiLineString { .. } => GeomKind::Linear,
        Geom::Unsupported => GeomKind::Unsupported,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, i: 0 }
    }

    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        if self.i + n > self.bytes.len() {
            anyhow::bail!("unexpected end of WKB");
        }
        let out = &self.bytes[self.i..self.i + n];
        self.i += n;
        Ok(out)
    }

    fn u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, le: bool) -> anyhow::Result<u32> {
        let b = self.take(4)?;
        Ok(if le {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        } else {
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    fn f64(&mut self, le: bool) -> anyhow::Result<f64> {
        let b = self.take(8)?;
        Ok(if le {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        } else {
            f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        })
    }
}

fn read_geom(cur: &mut Cursor<'_>) -> anyhow::Result<Geom> {
    let endian = cur.u8()?;
    let le = match endian {
        0 => false,
        1 => true,
        _ => anyhow::bail!("invalid WKB endian byte: {endian}"),
    };

    let raw_ty = cur.u32(le)?;
    // EWKB flags: strip Z/M/SRID bits if present.
    let ty = raw_ty & 0x1FFF_FFFF;
    // OGC WKB uses +1000/+2000/+3000 for Z/M/ZM.
    let base = ty % 1000;
    let has_z = (ty >= 1000) || (raw_ty & 0x8000_0000) != 0;
    let coords = if has_z { 3 } else { 2 };

    match base {
        1 => {
            let x = cur.f64(le)?;
            let y = cur.f64(le)?;
            if coords == 3 {
                let _ = cur.f64(le)?;
            }
            Ok(Geom::Point { pt: [x, y] })
        }
        2 => {
            // LineString
            let n = cur.u32(le)? as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                let x = cur.f64(le)?;
                let y = cur.f64(le)?;
                if coords == 3 {
                    let _z = cur.f64(le)?;
                    let _ = _z;
                }
                pts.push([x, y]);
            }
            Ok(Geom::LineString { pts })
        }
        3 => {
            // Polygon
            let rings_n = cur.u32(le)? as usize;
            let mut rings: Vec<Vec<[f64; 2]>> = Vec::with_capacity(rings_n);
            for _ in 0..rings_n {
                let n = cur.u32(le)? as usize;
                let mut pts = Vec::with_capacity(n);
                for _ in 0..n {
                    let x = cur.f64(le)?;
                    let y = cur.f64(le)?;
                    if coords == 3 {
                        let _z = cur.f64(le)?;
                        let _ = _z;
                    }
                    pts.push([x, y]);
                }
                rings.push(pts);
            }
            Ok(Geom::Polygon { rings })
        }
        5 => {
            // MultiLineString
            let n = cur.u32(le)? as usize;
            let mut lines = Vec::with_capacity(n);
            for _ in 0..n {
                let g = read_geom(cur)?;
                if let Geom::LineString { pts } = g {
                    lines.push(pts);
                }
            }
            Ok(Geom::MultiLineString { lines })
        }
        4 => {
            // MultiPoint
            let n = cur.u32(le)? as usize;
            let mut pts = Vec::with_capacity(n);
            for _ in 0..n {
                match read_geom(cur)? {
                    Geom::Point { pt } => pts.push(pt),
                    _ => {}
                }
            }
            Ok(Geom::MultiPoint { pts })
        }
        6 => {
            // MultiPolygon
            let n = cur.u32(le)? as usize;
            let mut polys = Vec::with_capacity(n);
            for _ in 0..n {
                polys.push(read_geom(cur)?);
            }
            Ok(Geom::MultiPolygon { polys })
        }
        _ => Ok(Geom::Unsupported),
    }
}

pub(super) fn circle_polyline(
    center: eframe::egui::Pos2,
    radius_world: f32,
    segments: usize,
) -> Vec<eframe::egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        pts.push(eframe::egui::pos2(
            center.x + radius_world * t.cos(),
            center.y + radius_world * t.sin(),
        ));
    }
    pts
}

fn circle_polyline_transformed(
    center: [f64; 2],
    radius: f32,
    xform: &SpatialDataTransform2,
    segments: usize,
) -> Vec<eframe::egui::Pos2> {
    let n = segments.max(8);
    let mut pts = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = (i as f32) * std::f32::consts::TAU / (n as f32);
        let src = [
            center[0] as f32 + radius * t.cos(),
            center[1] as f32 + radius * t.sin(),
        ];
        let q = xform.apply(src);
        pts.push(eframe::egui::pos2(q[0], q[1]));
    }
    pts
}
