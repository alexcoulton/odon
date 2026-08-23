use super::*;

#[allow(dead_code)] // The native binary also compiles this shared module without the actor model.
pub(crate) fn object_segmentation_requested(request: &DeepLinkRequest) -> bool {
    let source = request
        .segmentation_source
        .as_deref()
        .or(request.segmentation.as_deref())
        .map(normalize_public_name);
    source.as_deref().is_some_and(|source| {
        matches!(
            source,
            "objects"
                | "object"
                | "geoparquet"
                | "parquet"
                | "project"
                | "projectobjects"
                | "cellsgeoparquet"
        )
    }) || !request.object_filters.is_empty()
        || request.object_query.is_some()
        || !request.object_level_colors.is_empty()
}

#[allow(dead_code)] // See `object_segmentation_requested`.
pub(crate) fn requested_bundled_label(request: &DeepLinkRequest) -> Option<String> {
    let object_requested = object_segmentation_requested(request);
    let source = request
        .segmentation_source
        .as_deref()
        .or(request.segmentation.as_deref())
        .map(normalize_public_name);
    let bundled_requested = source
        .as_deref()
        .is_none_or(|source| !object_requested && source != "none");
    let load = request
        .load_segmentation_labels
        .unwrap_or(bundled_requested);
    (load && bundled_requested)
        .then(|| request.segmentation.as_deref())
        .flatten()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

#[allow(dead_code)] // See `object_segmentation_requested`.
pub(crate) fn object_filter_model(request: &DeepLinkRequest) -> Option<serde_json::Value> {
    if let Some(query) = request.object_query.as_deref() {
        return Some(serde_json::json!({"mode":"query","query":query}));
    }
    if request.object_filters.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "mode":"simple",
        "logic":match request.object_filter_logic.unwrap_or(DeepLinkObjectFilterLogic::All) {
            DeepLinkObjectFilterLogic::All => "all",
            DeepLinkObjectFilterLogic::Any => "any",
        },
        "clauses":request.object_filters.iter().map(|clause| serde_json::json!({
            "enabled":true,
            "property":clause.property_key,
            "query":clause.query,
        })).collect::<Vec<_>>(),
    }))
}

#[allow(dead_code)] // See `object_segmentation_requested`.
fn normalize_public_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}
