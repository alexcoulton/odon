//! Native actor object-resource loading and filter evaluation.

use super::*;

/// Native implementation of the actor's Send-only object boundary. Parsing and filtering happen
/// on bounded resource workers; renderer caches remain owned by `ObjectsLayer` on the UI thread.
pub struct NativeObjectControlService;

#[derive(Debug, Clone)]
pub enum ObjectUiAction {
    Load {
        path: PathBuf,
        options: Option<serde_json::Value>,
    },
    Reload,
    Clear,
    ClearSelection,
    SelectFiltered,
    SelectIds {
        ids: Vec<String>,
    },
}

impl ObjectResourceLoader for NativeObjectControlService {
    fn load(&self, path: PathBuf, downsample_factor: f32) -> anyhow::Result<ControlObjectResource> {
        load_control_object_resource(path, downsample_factor)
    }

    fn evaluate_filter(
        &self,
        resource: Arc<ControlObjectResource>,
        model: serde_json::Value,
    ) -> anyhow::Result<ControlObjectFilterResult> {
        evaluate_control_object_filter(&resource, &model)
    }

    fn load_with_options(
        &self,
        path: PathBuf,
        downsample_factor: f32,
        options: Option<serde_json::Value>,
    ) -> anyhow::Result<ControlObjectResource> {
        load_control_object_resource_with_options(path, downsample_factor, options.as_ref())
    }
}

pub(super) fn evaluate_control_object_filter(
    resource: &ControlObjectResource,
    requested: &serde_json::Value,
) -> anyhow::Result<ControlObjectFilterResult> {
    let requested = requested.get("model").unwrap_or(requested);
    let mode = requested
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| {
            if requested.get("query").is_some() || requested.get("expression").is_some() {
                "query"
            } else {
                "simple"
            }
        });
    match mode {
        "query" => {
            let query = requested
                .get("query")
                .or_else(|| requested.get("expression"))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("query mode requires query"))?
                .trim()
                .to_string();
            let expression = if query.is_empty() {
                None
            } else {
                let expression = ObjectFilterQueryExpr::parse(&query)
                    .map_err(|error| anyhow!(error.to_string()))?;
                let missing = expression
                    .referenced_properties()
                    .into_iter()
                    .filter(|property| !control_object_property_available(resource, property))
                    .collect::<Vec<_>>();
                if !missing.is_empty() {
                    anyhow::bail!(
                        "Unknown object propert{}: {}",
                        if missing.len() == 1 { "y" } else { "ies" },
                        missing.join(", ")
                    );
                }
                Some(expression)
            };
            let matching_indices = resource
                .features
                .iter()
                .enumerate()
                .filter_map(|(index, _feature)| {
                    expression
                        .as_ref()
                        .is_none_or(|expression| {
                            expression.matches_control_feature(resource, index)
                        })
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            Ok(ControlObjectFilterResult {
                model: serde_json::json!({"mode": "query", "query": query}),
                matching_indices: Arc::new(matching_indices),
                active: expression.is_some(),
            })
        }
        "simple" => {
            let logic = requested
                .get("logic")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("all");
            if !matches!(logic, "all" | "any") {
                anyhow::bail!("logic must be 'all' or 'any'");
            }
            let clauses = requested
                .get("clauses")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow!("simple mode requires clauses"))?;
            let mut canonical = Vec::with_capacity(clauses.len().max(1));
            for clause in clauses {
                let property = clause
                    .get("property")
                    .or_else(|| clause.get("property_key"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|property| !property.is_empty())
                    .ok_or_else(|| anyhow!("each filter clause requires property"))?;
                if !control_object_property_available(resource, property) {
                    anyhow::bail!("unknown object property '{property}'");
                }
                let query = clause
                    .get("query")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| anyhow!("each filter clause requires query"))?
                    .trim();
                canonical.push(serde_json::json!({
                    "enabled": clause
                        .get("enabled")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(true),
                    "property": property,
                    "query": query,
                }));
            }
            if canonical.is_empty() {
                canonical.push(serde_json::json!({
                    "enabled": true,
                    "property": "id",
                    "query": "",
                }));
            }
            let active_clauses = canonical
                .iter()
                .filter(|clause| {
                    clause["enabled"].as_bool() == Some(true)
                        && clause["query"]
                            .as_str()
                            .is_some_and(|query| !query.trim().is_empty())
                })
                .collect::<Vec<_>>();
            let matching_indices = resource
                .features
                .iter()
                .enumerate()
                .filter_map(|(index, feature)| {
                    let matches = |clause: &&serde_json::Value| {
                        let property = clause["property"].as_str().unwrap_or("id");
                        let needle = clause["query"]
                            .as_str()
                            .unwrap_or_default()
                            .to_ascii_lowercase();
                        let value = resource
                            .property_value(index, property)
                            .as_ref()
                            .map(column_value_to_display_text)
                            .unwrap_or_else(|| {
                                (property == "id")
                                    .then(|| feature.id.clone())
                                    .unwrap_or_default()
                            });
                        value.to_ascii_lowercase().contains(&needle)
                    };
                    let visible = if active_clauses.is_empty() {
                        true
                    } else if logic == "all" {
                        active_clauses.iter().all(matches)
                    } else {
                        active_clauses.iter().any(matches)
                    };
                    visible.then_some(index)
                })
                .collect::<Vec<_>>();
            Ok(ControlObjectFilterResult {
                model: serde_json::json!({
                    "mode": "simple",
                    "logic": logic,
                    "clauses": canonical,
                }),
                matching_indices: Arc::new(matching_indices),
                active: !active_clauses.is_empty(),
            })
        }
        _ => anyhow::bail!("mode must be 'simple' or 'query'"),
    }
}

pub(super) fn control_object_property_available(
    resource: &ControlObjectResource,
    property: &str,
) -> bool {
    property == "id"
        || resource
            .property_names
            .iter()
            .any(|candidate| candidate == property)
}
