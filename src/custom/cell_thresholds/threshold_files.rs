use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Context;

use super::data::normalize_roi_label;
use super::{AutoThreshold, AutoThresholdRecord, ThresholdCsvRow};

fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;

    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        field.push('"');
                        let _ = chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                row.push(std::mem::take(&mut field));
            }
            '\n' if !in_quotes => {
                row.push(std::mem::take(&mut field));
                // Skip fully empty trailing line.
                if !(row.len() == 1 && row[0].is_empty() && out.is_empty()) {
                    out.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            '\r' if !in_quotes => {
                if chars.peek() == Some(&'\n') {
                    let _ = chars.next();
                }
                row.push(std::mem::take(&mut field));
                out.push(std::mem::take(&mut row));
            }
            other => field.push(other),
        }
    }

    // Best-effort: accept unterminated quotes as literal.
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        out.push(row);
    }
    out
}

pub(super) fn load_thresholds_csv(
    path: &Path,
) -> anyhow::Result<HashMap<(String, String, String), ThresholdCsvRow>> {
    if !path.exists() {
        return Ok(HashMap::new());
    }

    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read thresholds csv: {}", path.to_string_lossy()))?;
    let mut recs = parse_csv(&text);
    if recs.is_empty() {
        return Ok(HashMap::new());
    }
    let header_row = recs.remove(0);
    let headers = header_row
        .iter()
        .enumerate()
        .map(|(i, h)| (h.to_ascii_lowercase(), i))
        .collect::<HashMap<_, _>>();

    let idx = |name: &str| headers.get(&name.to_ascii_lowercase()).copied();
    let i_roi = idx("roi").context("thresholds.csv missing 'roi' column")?;
    let i_marker = idx("marker").context("thresholds.csv missing 'marker' column")?;

    let i_raw = idx("raw_threshold");
    let i_arc = idx("arcsinh_threshold");
    let i_method = idx("method");
    let i_k = idx("kmeans_k");
    let i_ge = idx("positive_ge");
    let i_source = idx("source");

    let mut rows = HashMap::new();

    for rec in recs {
        let roi = normalize_roi_label(rec.get(i_roi).map(|s| s.as_str()).unwrap_or("").trim());
        let marker = rec
            .get(i_marker)
            .map(|s| s.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if roi.is_empty() || marker.is_empty() {
            continue;
        }

        let raw_threshold = i_raw
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or(0.0);
        let arcsinh_threshold = i_arc
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<f32>().ok())
            .unwrap_or_else(|| (raw_threshold as f64).asinh() as f32);
        let method = i_method
            .and_then(|i| rec.get(i))
            .map_or("manual", |v| v.as_str())
            .trim()
            .to_string();
        let source = i_source
            .and_then(|i| rec.get(i))
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let kmeans_k = i_k
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<u8>().ok());
        let positive_ge = i_ge
            .and_then(|i| rec.get(i))
            .and_then(|s| s.trim().parse::<u8>().ok());

        let row = ThresholdCsvRow {
            arcsinh_threshold,
            method: method.clone(),
            kmeans_k,
            positive_ge,
        };
        rows.insert((roi.clone(), marker.clone(), source.clone()), row);
    }

    Ok(rows)
}

pub(super) fn load_auto_thresholds_json(
    path: &Path,
) -> anyhow::Result<(
    u8,
    Option<String>,
    HashMap<(String, String), AutoThresholdRecord>,
)> {
    if !path.exists() {
        return Ok((6, None, HashMap::new()));
    }
    let text = fs::read_to_string(path).with_context(|| {
        format!(
            "failed to read auto thresholds json: {}",
            path.to_string_lossy()
        )
    })?;

    let root: serde_json::Value =
        serde_json::from_str(&text).context("failed to parse auto thresholds JSON")?;
    let marker_stat = root
        .get("marker_stat")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut out: HashMap<(String, String), AutoThresholdRecord> = HashMap::new();
    let mut global_kmeans_k = root.get("kmeans_k").and_then(|v| v.as_u64()).unwrap_or(6) as u8;

    let thresholds = root
        .get("thresholds")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let parse_auto = |node: &serde_json::Value, fallback_k: u8| -> Option<AutoThreshold> {
        let km = node.get("kmeans").and_then(|v| v.as_object());
        let otsu = node.get("otsu").and_then(|v| v.as_object());
        let km_k = km
            .and_then(|o| o.get("k"))
            .and_then(|v| v.as_u64())
            .unwrap_or(fallback_k as u64) as u8;
        let cutoffs = km
            .and_then(|o| o.get("cutoffs_arcsinh"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|v| v as f32)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let thr = otsu
            .and_then(|o| o.get("threshold_arcsinh"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        Some(AutoThreshold {
            kmeans_cutoffs_arcsinh: cutoffs,
            otsu_arcsinh: thr,
            kmeans_k: km_k.max(2),
        })
    };

    for rec in thresholds {
        let roi = rec
            .get("roi")
            .and_then(|v| v.as_str())
            .map(normalize_roi_label)
            .unwrap_or_default();
        let marker_key = rec
            .get("marker_key")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if roi.is_empty() || marker_key.is_empty() {
            continue;
        }

        let preferred_source = rec
            .get("preferred_source")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty());

        let mut sources: HashMap<String, AutoThreshold> = HashMap::new();
        if let Some(obj) = rec.get("sources").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(thr) = parse_auto(v, global_kmeans_k) {
                    global_kmeans_k = global_kmeans_k.max(thr.kmeans_k);
                    sources.insert(k.trim().to_ascii_lowercase(), thr);
                }
            }
        }
        // Legacy / fallback: treat record itself as a "standard" threshold set.
        if sources.is_empty() {
            if let Some(thr) = parse_auto(&rec, global_kmeans_k) {
                global_kmeans_k = global_kmeans_k.max(thr.kmeans_k);
                sources.insert("standard".to_string(), thr);
            }
        }

        out.insert(
            (roi, marker_key),
            AutoThresholdRecord {
                preferred_source,
                sources,
            },
        );
    }

    Ok((global_kmeans_k.max(2), marker_stat, out))
}
