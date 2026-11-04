//! Validation for `project.update_metadata`.

use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectMetadataRequest {
    default_dataset: Option<Value>,
    secondary_dataset: Option<Value>,
    default_threshold_marker: Option<Value>,
    mosaic_segmentation_search_roots: Option<Vec<String>>,
    category_colors: Option<std::collections::BTreeMap<String, [u8; 3]>>,
}

pub(super) fn validate(method: &str, params: &Value) -> Result<(), ControlError> {
    let invalid = |error: serde_json::Error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    };
    let request: ProjectMetadataRequest =
        serde_json::from_value(params.clone()).map_err(invalid)?;
    validate_optional_nullable_string(method, "default_dataset", request.default_dataset.as_ref())?;
    validate_optional_nullable_string(
        method,
        "secondary_dataset",
        request.secondary_dataset.as_ref(),
    )?;
    validate_optional_nullable_string(
        method,
        "default_threshold_marker",
        request.default_threshold_marker.as_ref(),
    )?;
    if request
        .mosaic_segmentation_search_roots
        .as_ref()
        .is_some_and(|roots| roots.len() > 4096 || roots.iter().any(|root| root.trim().is_empty()))
    {
        return Err(ControlError::invalid_params(
            method,
            "search roots must contain at most 4096 non-empty paths",
        ));
    }
    if request.category_colors.as_ref().is_some_and(|colors| {
        colors.len() > 4096 || colors.keys().any(|label| label.trim().is_empty())
    }) {
        return Err(ControlError::invalid_params(
            method,
            "category colors must contain at most 4096 non-empty labels",
        ));
    }
    Ok(())
}
