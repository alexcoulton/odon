//! Declarative UI extension request and snapshot types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DisconnectPolicy {
    Remove,
    Disable,
    Retain,
}

impl Default for DisconnectPolicy {
    fn default() -> Self {
        Self::Remove
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub value: Value,
    #[serde(default)]
    pub minimum: Option<f64>,
    #[serde(default)]
    pub maximum: Option<f64>,
    #[serde(default)]
    pub options: Vec<Value>,
    #[serde(default)]
    pub columns: Option<usize>,
    #[serde(default)]
    pub action: Option<Value>,
    #[serde(default)]
    pub event_policy: Option<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub state_bindings: BTreeMap<String, Value>,
    #[serde(default)]
    pub children: Vec<Component>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterExtension {
    pub id: String,
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub disconnect_policy: DisconnectPolicy,
    #[serde(default = "default_true")]
    pub ready: bool,
    #[serde(default)]
    pub readiness_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtensionSnapshot {
    pub id: String,
    pub name: String,
    pub version: String,
    pub requested_capabilities: Vec<String>,
    pub granted_capabilities: Vec<String>,
    pub disconnect_policy: DisconnectPolicy,
    pub owner_session_id: String,
    pub connected: bool,
    pub ready: bool,
    pub readiness_reason: Option<String>,
    pub revision: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterContribution {
    pub extension_id: String,
    #[serde(default)]
    pub contribution_id: Option<String>,
    #[serde(default = "default_shell_location")]
    pub location: String,
    pub root: Component,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContributionSnapshot {
    pub contribution_id: String,
    pub extension_id: String,
    pub extension_version: String,
    pub shell_mount: String,
    pub location: String,
    pub root: Component,
    pub ownership: Value,
    pub readiness: String,
    pub revision: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterExtensionLayout {
    pub extension_id: String,
    pub name: String,
    pub document: Value,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetExtensionReadiness {
    pub extension_id: String,
    pub ready: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtensionLayoutSnapshot {
    pub extension_id: String,
    pub extension_version: String,
    pub name: String,
    pub document: Value,
    pub ownership: Value,
    pub readiness: String,
    pub revision: u64,
}

#[derive(Debug, Clone)]
pub struct UiAction {
    pub extension_id: String,
    pub owner_session_id: String,
    pub component_id: String,
    pub action: Value,
    pub value: Value,
}

#[derive(Debug, Clone, Default)]
pub struct UiSessionCleanup {
    pub unavailable_mounts: Vec<String>,
    pub extensions: Vec<UiExtensionCleanup>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UiExtensionCleanup {
    pub extension_id: String,
    pub disconnect_policy: DisconnectPolicy,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExtensionCommandContext {
    pub extension_id: String,
    pub extension_version: String,
    pub owner_session_id: String,
    pub disconnect_policy: DisconnectPolicy,
    pub ready: bool,
}

fn default_true() -> bool {
    true
}

fn default_shell_location() -> String {
    "shell".to_string()
}
