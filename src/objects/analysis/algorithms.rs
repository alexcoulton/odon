//! Pure analysis, histogram, threshold, and scatter algorithms.

use super::*;

#[derive(Debug, Clone)]
pub(in crate::objects) struct SimpleHistogram {
    pub(in crate::objects) min: f32,
    pub(in crate::objects) max: f32,
    pub(in crate::objects) median: f32,
    pub(in crate::objects) bins: Vec<u32>,
    pub(in crate::objects) max_count: u32,
}

pub(in crate::objects) fn compute_threshold_selection_indices(
    rules: &[AnalysisSelectionJobRule],
    objects: &[GeoJsonObjectFeature],
    property_store: &ObjectPropertyStore,
    filtered_mask: Option<&[bool]>,
) -> (Arc<Vec<usize>>, Arc<Vec<egui::Pos2>>, Arc<Vec<f32>>) {
    if rules.is_empty() {
        return (
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
            Arc::new(Vec::new()),
        );
    }

    let indices = if rules.len() == 1 {
        let job = &rules[0];
        let pairs = analysis_selection_column_pairs(
            objects,
            property_store,
            filtered_mask,
            &job.rule.column_key,
        );
        let out = pairs
            .iter()
            .filter_map(|(object_index, value)| {
                threshold_rule_matches(&job.rule, *value).then_some(*object_index)
            })
            .collect::<Vec<_>>();
        out
    } else {
        let mut selected: Option<HashSet<usize>> = None;
        for job in rules {
            let pairs = analysis_selection_column_pairs(
                objects,
                property_store,
                filtered_mask,
                &job.rule.column_key,
            );
            let rule_matches = pairs
                .iter()
                .filter_map(|(object_index, value)| {
                    threshold_rule_matches(&job.rule, *value).then_some(*object_index)
                })
                .collect::<HashSet<_>>();
            selected = Some(match selected {
                Some(mut current) => {
                    current.retain(|idx| rule_matches.contains(idx));
                    current
                }
                None => rule_matches,
            });
        }
        selected.unwrap_or_default().into_iter().collect()
    };

    let proxy_positions = indices
        .iter()
        .filter_map(|idx| objects.get(*idx).map(object_proxy_position_world))
        .collect::<Vec<_>>();
    let proxy_values = vec![1.0f32; proxy_positions.len()];
    (
        Arc::new(indices),
        Arc::new(proxy_positions),
        Arc::new(proxy_values),
    )
}

pub(in crate::objects) fn analysis_selection_column_pairs(
    objects: &[GeoJsonObjectFeature],
    property_store: &ObjectPropertyStore,
    filtered_mask: Option<&[bool]>,
    key: &str,
) -> Vec<(usize, f32)> {
    if let Some(mut pairs) = property_store.numeric_pairs(key) {
        if let Some(mask) = filtered_mask {
            pairs.retain(|(idx, _)| mask.get(*idx).copied().unwrap_or(false));
        }
        return pairs;
    }

    let mut out = Vec::new();
    for (idx, obj) in objects.iter().enumerate() {
        if filtered_mask.is_some_and(|mask| !mask.get(idx).copied().unwrap_or(false)) {
            continue;
        }
        let Some(value) = obj.inline_properties.get(key).and_then(numeric_json_value) else {
            continue;
        };
        if value.is_finite() {
            out.push((idx, value));
        }
    }
    out
}

pub(in crate::objects) fn threshold_rule_matches(
    rule: &ObjectPropertyThresholdRule,
    value: f32,
) -> bool {
    match rule.op {
        AnalysisThresholdOp::GreaterEqual => value >= rule.value,
        AnalysisThresholdOp::LessEqual => value <= rule.value,
    }
}

pub(in crate::objects) fn build_polygon_mask(
    polygons_world: &[Vec<egui::Pos2>],
    x0: u64,
    y0: u64,
    width: usize,
    height: usize,
) -> Vec<bool> {
    let mut mask = vec![false; width.saturating_mul(height)];
    for yy in 0..height {
        for xx in 0..width {
            let world = egui::pos2(x0 as f32 + xx as f32 + 0.5, y0 as f32 + yy as f32 + 0.5);
            mask[yy * width + xx] = point_in_any_polygon(world, polygons_world);
        }
    }
    mask
}

pub(in crate::objects) fn point_in_any_polygon(
    p: egui::Pos2,
    polygons: &[Vec<egui::Pos2>],
) -> bool {
    polygons.iter().any(|poly| point_in_polygon(p, poly))
}

pub(in crate::objects) fn point_in_polygon(p: egui::Pos2, poly: &[egui::Pos2]) -> bool {
    if poly.len() < 4 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let dy = pj.y - pi.y;
        let intersects = ((pi.y > p.y) != (pj.y > p.y))
            && dy.abs() > 1e-12
            && (p.x < (pj.x - pi.x) * (p.y - pi.y) / dy + pi.x);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub(in crate::objects) fn numeric_json_value(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(v) => v.as_f64().map(|v| v as f32),
        serde_json::Value::String(v) => v.parse::<f32>().ok(),
        _ => None,
    }
}

pub(in crate::objects) fn normalize_analysis_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

pub(in crate::objects) fn analysis_name_tokens(value: &str) -> HashSet<String> {
    let raw_tokens = value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect::<Vec<_>>();

    let mut tokens = HashSet::new();
    for token in &raw_tokens {
        tokens.insert(token.clone());
    }
    for window in raw_tokens.windows(2) {
        let joined = format!("{}{}", window[0], window[1]);
        if joined.len() >= 3 {
            tokens.insert(joined);
        }
    }
    let collapsed = normalize_analysis_name(value);
    if collapsed.len() >= 3 {
        tokens.insert(collapsed);
    }
    tokens
}

pub(in crate::objects) fn analysis_token_frequencies(
    channels: &[ChannelInfo],
    numeric_columns: &[String],
) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for token_set in channels
        .iter()
        .map(|channel| analysis_name_tokens(&channel.name))
        .chain(
            numeric_columns
                .iter()
                .map(|column| analysis_name_tokens(column)),
        )
    {
        for token in token_set {
            *frequencies.entry(token).or_insert(0) += 1;
        }
    }
    frequencies
}

pub(in crate::objects) fn fuzzy_name_score(query: &str, candidate: &str) -> Option<i32> {
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return Some(0);
    }
    let candidate_lower = candidate.to_ascii_lowercase();
    if candidate_lower == query {
        return Some(100_000);
    }
    if let Some(rest) = candidate_lower.strip_prefix(&query) {
        return Some(90_000 - rest.len() as i32);
    }
    if let Some(pos) = candidate_lower.find(&query) {
        return Some(80_000 - pos as i32);
    }

    let mut score = 50_000i32;
    let mut search_from = 0usize;
    for ch in query.chars() {
        let hay = &candidate_lower[search_from..];
        let rel = hay.find(ch)?;
        score -= rel as i32;
        search_from += rel + ch.len_utf8();
    }
    Some(score)
}

pub(in crate::objects) fn finite_min_max_f32(values: &[f32]) -> Option<(f32, f32)> {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any = false;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        any = true;
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    any.then(|| {
        if (max_v - min_v).abs() <= 1e-12 {
            (min_v, min_v + 1.0)
        } else {
            (min_v, max_v)
        }
    })
}

pub(in crate::objects) fn compute_histogram_f32(
    values: &[f32],
    bin_count: usize,
) -> SimpleHistogram {
    // Histogram statistics are computed over finite values only. The median is derived from the
    // sorted finite sample so UI annotations stay consistent with the plotted bins.
    let (min_v, max_v) = finite_min_max_f32(values).unwrap_or((0.0, 1.0));
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let bin_count = bin_count.max(8);
    let mut bins = vec![0u32; bin_count];
    let inv = (bin_count as f32 - 1.0) / (max_v - min_v).max(1e-6);
    for &v in &sorted {
        let idx = (((v - min_v) * inv).floor() as usize).min(bin_count - 1);
        bins[idx] = bins[idx].saturating_add(1);
    }
    let max_count = bins.iter().copied().max().unwrap_or(1).max(1);
    SimpleHistogram {
        min: min_v,
        max: max_v,
        median,
        bins,
        max_count,
    }
}

pub(in crate::objects) fn quantile_threshold_levels(
    values: &[f32],
    level_count: usize,
) -> Vec<f32> {
    if values.len() < 2 || level_count < 2 {
        return Vec::new();
    }
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if sorted.len() < 2 {
        return Vec::new();
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let bins = level_count.max(2);
    let mut out = Vec::with_capacity(bins.saturating_sub(1));
    for q in 1..bins {
        let pos = ((sorted.len() - 1) as f32 * (q as f32 / bins as f32)).round() as usize;
        let value = sorted[pos.min(sorted.len() - 1)];
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

pub(in crate::objects) fn kmeans_threshold_levels(
    values: &[f32],
    k: usize,
    iterations: usize,
) -> Vec<f32> {
    // Use 1D k-means centroids to propose threshold boundaries halfway between neighboring
    // clusters. This is heuristic UI assistance, not a persisted analysis model.
    let mut samples = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if samples.len() < 2 {
        return Vec::new();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.clamp(2, samples.len().min(12));

    let mut centroids = (0..k)
        .map(|i| {
            let pos = ((samples.len() - 1) as f32 * (i as f32 / (k - 1) as f32)).round() as usize;
            samples[pos.min(samples.len() - 1)]
        })
        .collect::<Vec<_>>();
    let mut assignments = vec![0usize; samples.len()];

    for _ in 0..iterations.max(1) {
        let mut changed = false;
        for (i, value) in samples.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f32::INFINITY;
            for (idx, centroid) in centroids.iter().enumerate() {
                let dist = (*value - *centroid).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            if assignments[i] != best_idx {
                assignments[i] = best_idx;
                changed = true;
            }
        }

        let mut sums = vec![0.0f32; k];
        let mut counts = vec![0usize; k];
        for (value, &cluster) in samples.iter().zip(assignments.iter()) {
            sums[cluster] += *value;
            counts[cluster] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centroids[i] = sums[i] / counts[i] as f32;
            }
        }
        if !changed {
            break;
        }
    }

    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(k.saturating_sub(1));
    for pair in centroids.windows(2) {
        let value = 0.5 * (pair[0] + pair[1]);
        if out.last().copied() != Some(value) {
            out.push(value);
        }
    }
    out
}

pub(in crate::objects) fn histogram_level_label(
    method: HistogramLevelMethod,
    level_count: usize,
    level_index: usize,
    total_levels: usize,
) -> String {
    match method {
        HistogramLevelMethod::Quantiles => {
            let denom = level_count.max(2) as f32;
            let pct = (((level_index + 1) as f32 / denom) * 100.0).round() as i32;
            format!("Quantile {pct}%")
        }
        HistogramLevelMethod::KMeans => {
            let left_cluster = (level_index + 1).min(total_levels.max(1));
            format!("K={} boundary {}", level_count.max(2), left_cluster)
        }
    }
}

pub(in crate::objects) fn apply_histogram_value_transform(
    value: f32,
    transform: HistogramValueTransform,
) -> f32 {
    // The transform is applied symmetrically anywhere histogram values are compared or displayed
    // so brushing and threshold handles remain in the same value space as the plotted axis.
    match transform {
        HistogramValueTransform::None => value,
        HistogramValueTransform::Arcsinh => value.asinh(),
    }
}

pub(in crate::objects) fn invert_histogram_value_transform(
    value: f32,
    transform: HistogramValueTransform,
) -> f32 {
    match transform {
        HistogramValueTransform::None => value,
        HistogramValueTransform::Arcsinh => value.sinh(),
    }
}

pub(in crate::objects) fn analysis_picker_popup_width(
    ui: &egui::Ui,
    button_width: f32,
    names: &[String],
    search: &str,
    selected_label: Option<&str>,
) -> f32 {
    let viewport_width = ui.ctx().content_rect().width().max(button_width);
    let max_width = (viewport_width - 32.0).max(button_width);
    let mut candidates = names
        .iter()
        .filter_map(|name| fuzzy_name_score(search, name).map(|score| (score, name.as_str())))
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));

    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let mut target_width = button_width.max(240.0);
    for label in selected_label
        .into_iter()
        .chain(candidates.into_iter().take(12).map(|(_, name)| name))
    {
        let galley = ui.ctx().fonts_mut(|fonts| {
            fonts.layout_no_wrap(label.to_owned(), font_id.clone(), egui::Color32::WHITE)
        });
        target_width = target_width.max(galley.size().x + 72.0);
    }

    let min_width = button_width.max(320.0).min(max_width);
    target_width.clamp(min_width, max_width)
}

pub(in crate::objects) fn order_pair(a: f32, b: f32) -> (f32, f32) {
    if a <= b { (a, b) } else { (b, a) }
}

pub(in crate::objects) fn scatter_axis_drag_speed(min_v: f32, max_v: f32) -> f64 {
    let span = (max_v - min_v).abs().max(1.0);
    (span / 200.0).max(0.01) as f64
}

pub(in crate::objects) fn normalize_scatter_axis_pair(
    min_v: f32,
    max_v: f32,
    fallback: (f32, f32),
) -> (f32, f32) {
    let (fallback_min, fallback_max) = order_pair(fallback.0, fallback.1);
    if !(min_v.is_finite() && max_v.is_finite()) {
        return (fallback_min, fallback_max.max(fallback_min + 1.0));
    }

    let (mut lo, mut hi) = order_pair(min_v, max_v);
    if (hi - lo).abs() <= 1e-6 {
        let span = (fallback_max - fallback_min).abs().max(1.0);
        let pad = (span * 0.005).max(0.5);
        lo -= pad;
        hi += pad;
    }
    (lo, hi)
}

pub(in crate::objects) fn normalize_scatter_view_rect(
    view_rect: egui::Rect,
    fallback_rect: egui::Rect,
) -> egui::Rect {
    let (x_min, x_max) = normalize_scatter_axis_pair(
        view_rect.min.x,
        view_rect.max.x,
        (fallback_rect.min.x, fallback_rect.max.x),
    );
    let (y_min, y_max) = normalize_scatter_axis_pair(
        view_rect.min.y,
        view_rect.max.y,
        (fallback_rect.min.y, fallback_rect.max.y),
    );
    egui::Rect::from_min_max(egui::pos2(x_min, y_min), egui::pos2(x_max, y_max))
}

pub(in crate::objects) fn value_to_screen_x(
    value: f32,
    rect: egui::Rect,
    min_v: f32,
    max_v: f32,
) -> f32 {
    let t = ((value - min_v) / (max_v - min_v).max(1e-6)).clamp(0.0, 1.0);
    rect.left() + t * rect.width()
}

pub(in crate::objects) fn screen_x_to_value(
    x: f32,
    rect: egui::Rect,
    min_v: f32,
    max_v: f32,
) -> f32 {
    let t = ((x - rect.left()) / rect.width().max(1e-6)).clamp(0.0, 1.0);
    min_v + t * (max_v - min_v)
}

pub(in crate::objects) fn value_to_screen_y(
    value: f32,
    rect: egui::Rect,
    min_v: f32,
    max_v: f32,
) -> f32 {
    let t = ((value - min_v) / (max_v - min_v).max(1e-6)).clamp(0.0, 1.0);
    rect.bottom() - t * rect.height()
}

pub(in crate::objects) fn screen_y_to_value(
    y: f32,
    rect: egui::Rect,
    min_v: f32,
    max_v: f32,
) -> f32 {
    let t = ((rect.bottom() - y) / rect.height().max(1e-6)).clamp(0.0, 1.0);
    min_v + t * (max_v - min_v)
}

pub(in crate::objects) fn screen_rect_to_value_rect(
    screen_rect: egui::Rect,
    plot_rect: egui::Rect,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
) -> Option<egui::Rect> {
    if !screen_rect.is_positive() {
        return None;
    }
    let min_x = screen_x_to_value(screen_rect.min.x, plot_rect, x_min, x_max);
    let max_x = screen_x_to_value(screen_rect.max.x, plot_rect, x_min, x_max);
    let min_y = screen_y_to_value(screen_rect.max.y, plot_rect, y_min, y_max);
    let max_y = screen_y_to_value(screen_rect.min.y, plot_rect, y_min, y_max);
    Some(egui::Rect::from_min_max(
        egui::pos2(min_x.min(max_x), min_y.min(max_y)),
        egui::pos2(min_x.max(max_x), min_y.max(max_y)),
    ))
}

pub(in crate::objects) fn value_rect_to_screen_rect(
    value_rect: egui::Rect,
    plot_rect: egui::Rect,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
) -> egui::Rect {
    egui::Rect::from_min_max(
        egui::pos2(
            value_to_screen_x(value_rect.min.x, plot_rect, x_min, x_max),
            value_to_screen_y(value_rect.max.y, plot_rect, y_min, y_max),
        ),
        egui::pos2(
            value_to_screen_x(value_rect.max.x, plot_rect, x_min, x_max),
            value_to_screen_y(value_rect.min.y, plot_rect, y_min, y_max),
        ),
    )
}
