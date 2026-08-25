//! Versioned portable shell-layout documents and protected recovery layouts.

use serde_json::{Map, Value, json};

use super::layout::{ShellLayout, recovery_layout};
use super::*;

const DOCUMENT_FORMAT: &str = "odon.shell-layout";
const DOCUMENT_SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShellImportOutcome {
    pub mode: String,
    pub source_schema_version: u64,
    pub migrated: bool,
}

impl ShellModel {
    pub(crate) fn export_layout_document(&self, mode: &str) -> Result<Value, ControlError> {
        let shell = self
            .modes
            .get(mode)
            .ok_or_else(|| internal(format!("shell mode '{mode}' is missing")))?;
        Ok(json!({
            "format":DOCUMENT_FORMAT,
            "schema_version":DOCUMENT_SCHEMA_VERSION,
            "mode":mode,
            "layout":shell.layout.to_json(),
        }))
    }

    pub(crate) fn import_layout_document(
        &mut self,
        params: &Value,
        current_mode: ModelMode,
    ) -> Result<ShellImportOutcome, ControlError> {
        validate_revision_guard(params, self.revision, "ui.shell.import_layout")?;
        let document = params
            .get("document")
            .ok_or_else(|| invalid("ui.shell.import_layout", "document is required"))?;
        let parsed = parse_document(document, "ui.shell.import_layout")?;
        let requested = requested_mode(params, current_mode)?;
        if requested != parsed.mode {
            return Err(invalid(
                "ui.shell.import_layout",
                format!(
                    "document mode '{}' does not match requested mode '{requested}'",
                    parsed.mode
                ),
            ));
        }
        let candidate =
            ShellLayout::from_value(&parsed.layout, &parsed.mode, "ui.shell.import_layout")?;
        let shell = self
            .modes
            .get_mut(&parsed.mode)
            .ok_or_else(|| internal(format!("shell mode '{}' is missing", parsed.mode)))?;
        if shell.layout != candidate {
            shell.layout = candidate;
            shell.active_region_id = shell.layout.preferred_active_region_id().to_string();
            shell.focused_node_id = None;
            self.touch();
        }
        Ok(ShellImportOutcome {
            mode: parsed.mode,
            source_schema_version: parsed.source_schema_version,
            migrated: parsed.source_schema_version != DOCUMENT_SCHEMA_VERSION,
        })
    }

    pub(crate) fn recover_layout(
        &mut self,
        params: &Value,
        current_mode: ModelMode,
    ) -> Result<String, ControlError> {
        validate_revision_guard(params, self.revision, "ui.shell.recover")?;
        let mode = requested_mode(params, current_mode)?;
        let replacement = recovery_layout(mode);
        let shell = self
            .modes
            .get_mut(mode)
            .ok_or_else(|| internal(format!("shell mode '{mode}' is missing")))?;
        let active_region_id = replacement.preferred_active_region_id().to_string();
        let changed = shell.layout != replacement
            || shell.active_region_id != active_region_id
            || shell.focused_node_id.is_some();
        if changed {
            shell.layout = replacement;
            shell.active_region_id = active_region_id;
            shell.focused_node_id = None;
            self.touch();
        }
        Ok(mode.to_string())
    }
}

struct ParsedDocument {
    mode: String,
    layout: Value,
    source_schema_version: u64,
}

fn parse_document(document: &Value, method: &str) -> Result<ParsedDocument, ControlError> {
    let object = document
        .as_object()
        .ok_or_else(|| invalid(method, "document must be an object"))?;
    let version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            invalid(
                method,
                "document.schema_version must be an unsigned integer",
            )
        })?;
    match version {
        1 => parse_v1_document(object, method),
        0 => parse_v0_document(object, method),
        _ => Err(ControlError::new(
            ControlErrorKind::Unsupported,
            format!("shell layout document schema version {version} is not supported"),
        )
        .with_data(json!({
            "method":method,
            "schema_version":version,
            "supported_schema_versions":[0,1],
            "recovery_method":"ui.shell.recover",
        }))),
    }
}

fn parse_v1_document(
    object: &Map<String, Value>,
    method: &str,
) -> Result<ParsedDocument, ControlError> {
    let allowed = ["format", "schema_version", "mode", "layout"];
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            method,
            "version-1 document contains an unknown field",
        ));
    }
    if object.get("format").and_then(Value::as_str) != Some(DOCUMENT_FORMAT) {
        return Err(invalid(
            method,
            format!("document.format must be '{DOCUMENT_FORMAT}'"),
        ));
    }
    parsed_fields(object, "layout", 1, method)
}

fn parse_v0_document(
    object: &Map<String, Value>,
    method: &str,
) -> Result<ParsedDocument, ControlError> {
    let allowed = ["schema_version", "mode", "desired_tree"];
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            method,
            "version-0 document contains an unknown field",
        ));
    }
    parsed_fields(object, "desired_tree", 0, method)
}

fn parsed_fields(
    object: &Map<String, Value>,
    layout_field: &str,
    source_schema_version: u64,
    method: &str,
) -> Result<ParsedDocument, ControlError> {
    let mode = object
        .get("mode")
        .and_then(Value::as_str)
        .filter(|mode| matches!(*mode, "project" | "single" | "mosaic"))
        .ok_or_else(|| invalid(method, "document.mode must be project, single, or mosaic"))?;
    let layout = object
        .get(layout_field)
        .filter(|layout| layout.is_object())
        .cloned()
        .ok_or_else(|| invalid(method, format!("document.{layout_field} must be an object")))?;
    Ok(ParsedDocument {
        mode: mode.to_string(),
        layout,
        source_schema_version,
    })
}

pub(super) fn shell_document_schema() -> Value {
    json!({
        "$schema":"https://json-schema.org/draft/2020-12/schema",
        "$id":"https://odon.app/schemas/ui-shell-layout-document-v1.json",
        "type":"object",
        "properties":{
            "format":{"const":DOCUMENT_FORMAT},
            "schema_version":{"const":DOCUMENT_SCHEMA_VERSION},
            "mode":{"type":"string","enum":["project","single","mosaic"]},
            "layout":super::layout_schema(),
        },
        "required":["format","schema_version","mode","layout"],
        "additionalProperties":false,
    })
}

pub(super) fn validate_layout_document(document: &Value) -> Result<(), ControlError> {
    validate_layout_document_for(document, "ui.extensions.layouts.register")
}

pub(super) fn validate_layout_document_for(
    document: &Value,
    method: &str,
) -> Result<(), ControlError> {
    let parsed = parse_document(document, method)?;
    ShellLayout::from_value(&parsed.layout, &parsed.mode, method)?;
    Ok(())
}

pub(super) fn normalize_layout_document(document: &Value) -> Result<Value, ControlError> {
    let parsed = parse_document(document, "ui.extensions.layouts.register")?;
    let layout = ShellLayout::from_value(
        &parsed.layout,
        &parsed.mode,
        "ui.extensions.layouts.register",
    )?;
    Ok(json!({
        "format":DOCUMENT_FORMAT,
        "schema_version":DOCUMENT_SCHEMA_VERSION,
        "mode":parsed.mode,
        "layout":layout.to_json(),
    }))
}
