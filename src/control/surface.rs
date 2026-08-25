//! Machine-readable parity manifest for Odon's complete semantic surface.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::registry;
use super::{ControlError, ControlErrorKind};

const MANIFEST_JSON: &str = include_str!("../../api/application-surface.json");

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationSurfaceManifest {
    pub schema_version: u32,
    pub generated_for: String,
    pub entries: Vec<ApplicationSurfaceEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceStatus {
    Covered,
    Partial,
    Planned,
    PresentationOnly,
    AdapterOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplicationSurfaceEntry {
    pub id: String,
    pub feature_id: String,
    pub domain: String,
    pub title: String,
    pub phase: String,
    pub status: SurfaceStatus,
    #[serde(default = "default_true")]
    pub semantic: bool,
    #[serde(default)]
    pub native_entry_points: Vec<String>,
    #[serde(default)]
    pub methods: Vec<String>,
    #[serde(default)]
    pub events: Vec<String>,
    #[serde(default)]
    pub permissions: Vec<String>,
    #[serde(default)]
    pub python_sync: Vec<String>,
    #[serde(default)]
    pub python_async: Vec<String>,
    #[serde(default)]
    pub tests: Vec<String>,
    #[serde(default)]
    pub notes: Option<String>,
}

fn default_true() -> bool {
    true
}

pub fn application_surface() -> Result<ApplicationSurfaceManifest, ControlError> {
    let manifest: ApplicationSurfaceManifest =
        serde_json::from_str(MANIFEST_JSON).map_err(|error| {
            ControlError::new(
                ControlErrorKind::Internal,
                format!("invalid embedded application-surface manifest: {error}"),
            )
        })?;
    validate_application_surface(&manifest).map_err(|message| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("invalid embedded application-surface manifest: {message}"),
        )
    })?;
    Ok(manifest)
}

pub fn application_surface_json() -> Result<Value, ControlError> {
    serde_json::to_value(application_surface()?).map_err(|error| {
        ControlError::new(
            ControlErrorKind::Internal,
            format!("failed to serialize application-surface manifest: {error}"),
        )
    })
}

pub fn validate_application_surface(manifest: &ApplicationSurfaceManifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported schema_version {}; expected 1",
            manifest.schema_version
        ));
    }
    if manifest.entries.is_empty() {
        return Err("entries must not be empty".to_string());
    }

    let mut ids = BTreeSet::new();
    let mut registered_methods = BTreeMap::<&str, &str>::new();
    for entry in &manifest.entries {
        if entry.id.trim().is_empty() || !ids.insert(entry.id.as_str()) {
            return Err(format!(
                "surface entry ID is empty or duplicated: {:?}",
                entry.id
            ));
        }
        if entry.feature_id.trim().is_empty()
            || entry.domain.trim().is_empty()
            || entry.title.trim().is_empty()
            || entry.phase.trim().is_empty()
        {
            return Err(format!(
                "surface entry {} has incomplete metadata",
                entry.id
            ));
        }
        if entry.semantic
            && matches!(
                entry.status,
                SurfaceStatus::Covered | SurfaceStatus::Partial
            )
            && entry.methods.is_empty()
        {
            return Err(format!(
                "covered semantic entry {} must declare at least one method",
                entry.id
            ));
        }
        if !entry.semantic
            && !matches!(
                entry.status,
                SurfaceStatus::PresentationOnly | SurfaceStatus::AdapterOnly
            )
        {
            return Err(format!(
                "non-semantic entry {} must be presentation_only or adapter_only",
                entry.id
            ));
        }
        if matches!(
            entry.status,
            SurfaceStatus::Covered | SurfaceStatus::Partial
        ) && entry.semantic
            && (entry.python_sync.is_empty() || entry.python_async.is_empty())
        {
            return Err(format!(
                "implemented semantic entry {} needs sync and async Python coverage",
                entry.id
            ));
        }

        for method_name in &entry.methods {
            if let Some(previous) = registered_methods.insert(method_name, entry.id.as_str()) {
                return Err(format!(
                    "method {} is assigned to both {} and {}",
                    method_name, previous, entry.id
                ));
            }
            if let Some(descriptor) = registry::method(method_name) {
                if descriptor.name != method_name {
                    return Err(format!(
                        "surface entry {} uses deprecated alias {}; use canonical {}",
                        entry.id, method_name, descriptor.name
                    ));
                }
                if !entry
                    .permissions
                    .iter()
                    .any(|permission| permission == descriptor.capability)
                {
                    return Err(format!(
                        "surface entry {} omits method permission {}",
                        entry.id, descriptor.capability
                    ));
                }
                if let Some(event) = descriptor.event
                    && !entry.events.iter().any(|candidate| candidate == event)
                {
                    return Err(format!(
                        "surface entry {} omits method event {}",
                        entry.id, event
                    ));
                }
                continue;
            }
            let Some(protocol) = registry::PROTOCOL_METHODS
                .iter()
                .find(|descriptor| descriptor.0 == method_name)
            else {
                return Err(format!(
                    "surface entry {} references unknown method {}",
                    entry.id, method_name
                ));
            };
            if !entry
                .permissions
                .iter()
                .any(|permission| permission == protocol.2)
            {
                return Err(format!(
                    "surface entry {} omits method permission {}",
                    entry.id, protocol.2
                ));
            }
        }
    }

    let missing = registry::METHODS
        .iter()
        .filter(|descriptor| !registered_methods.contains_key(descriptor.name))
        .map(|descriptor| descriptor.name)
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "registered methods missing from application surface: {}",
            missing.join(", ")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_valid_and_covers_the_registry() {
        let manifest = application_surface().expect("valid application surface");
        let declared_actor_methods = manifest
            .entries
            .iter()
            .flat_map(|entry| &entry.methods)
            .filter(|method| {
                registry::METHODS
                    .iter()
                    .any(|item| item.name == method.as_str())
            })
            .count();
        assert_eq!(declared_actor_methods, registry::METHODS.len());
        for method in [
            "ui.extensions.layouts.register",
            "ui.extensions.layouts.list",
            "ui.extensions.layouts.remove",
        ] {
            assert!(
                manifest
                    .entries
                    .iter()
                    .any(|entry| entry.methods.iter().any(|item| item == method)),
                "application surface omits protocol service {method}"
            );
        }
    }

    #[test]
    fn every_feature_matrix_id_is_classified() {
        let manifest = application_surface().expect("valid application surface");
        let represented = manifest
            .entries
            .iter()
            .map(|entry| entry.feature_id.as_str())
            .collect::<BTreeSet<_>>();
        let matrix = include_str!("../../docs/design/test-coverage-matrix.md");
        let matrix_ids = matrix
            .lines()
            .filter_map(|line| {
                let mut columns = line.split('|');
                let _ = columns.next()?;
                let id = columns.next()?.trim();
                (id.contains('-')
                    && id.chars().any(|character| character.is_ascii_uppercase())
                    && id.chars().any(|character| character.is_ascii_digit())
                    && id.chars().all(|character| {
                        character.is_ascii_uppercase()
                            || character.is_ascii_digit()
                            || character == '-'
                    }))
                .then_some(id)
            })
            .collect::<BTreeSet<_>>();
        let missing = matrix_ids
            .difference(&represented)
            .copied()
            .collect::<Vec<_>>();
        assert!(
            missing.is_empty(),
            "unclassified feature matrix IDs: {missing:?}"
        );
    }
}
