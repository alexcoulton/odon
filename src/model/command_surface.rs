//! Actor-owned command descriptors and their platform-menu presentation.
//!
//! Commands describe meaning and execution independently from the places that present them. The
//! first presentation is the platform application menu; toolbars, palettes, and shortcuts can use
//! the same stable command IDs without cloning action semantics.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::control::{
    ControlError, ControlErrorKind, DisconnectPolicy, ExtensionCommandContext, UiExtensionCleanup,
};

const MAX_MENU_NODES: usize = 256;
const MAX_MENU_DEPTH: usize = 12;
const MAX_TOOLBAR_GROUPS: usize = 32;
const MAX_TOOLBAR_ITEMS: usize = 128;
const MAX_PALETTE_RESULTS: u64 = 100;
const MAX_PREDICATE_NODES: usize = 32;
const MAX_PREDICATE_DEPTH: usize = 8;
const MAX_TEXT_BYTES: usize = 256;

const COMMAND_STATE_PATHS: &[&str] = &[
    "mode",
    "resources.project",
    "resources.dataset",
    "resources.mosaic",
    "resources.objects",
    "resources.labels",
    "resources.masks",
    "resources.gpu",
    "selection.objects.count",
    "selection.mosaic_items.count",
    "presentation.scale_bar.checked",
    "presentation.left_panel.visible",
    "presentation.right_panel.visible",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShortcutPlatform {
    MacOs,
    Windows,
    Linux,
    Other,
}

impl ShortcutPlatform {
    const fn current() -> Self {
        if cfg!(target_os = "macos") {
            Self::MacOs
        } else if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else {
            Self::Other
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::MacOs => "macos",
            Self::Windows => "windows",
            Self::Linux => "linux",
            Self::Other => "other",
        }
    }

    const fn primary(self) -> &'static str {
        if matches!(self, Self::MacOs) {
            "super"
        } else {
            "control"
        }
    }

    const fn supports_super(self) -> bool {
        matches!(self, Self::MacOs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CommandEvaluationContext {
    pub mode: String,
    pub native: bool,
    pub capabilities: BTreeSet<String>,
    pub state: Value,
}

impl CommandEvaluationContext {
    pub(crate) fn native(mode: impl Into<String>, state: Value) -> Self {
        Self {
            mode: mode.into(),
            native: true,
            capabilities: BTreeSet::new(),
            state,
        }
    }

    pub(crate) fn session(
        mode: impl Into<String>,
        capabilities: impl IntoIterator<Item = String>,
        state: Value,
    ) -> Self {
        Self {
            mode: mode.into(),
            native: false,
            capabilities: capabilities.into_iter().collect(),
            state,
        }
    }

    fn has_capability(&self, capability: &str) -> bool {
        self.native
            || self.capabilities.contains("ui.shell.application_control")
            || self.capabilities.contains(capability)
    }
}

#[derive(Debug, Default)]
struct PredicateEvaluation {
    value: bool,
    reasons: Vec<String>,
    missing_capabilities: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct CommandSurfaceModel {
    revision: u64,
    commands: BTreeMap<String, Value>,
    menu: Value,
    toolbar: Value,
    palette: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct ExtensionCommandInvocation {
    pub command_id: String,
    pub extension_id: String,
    pub owner_session_id: String,
    pub event: String,
}

#[derive(Debug, Clone)]
pub(crate) enum CommandInvocation {
    Native {
        command_id: String,
        action: String,
        checked: Option<bool>,
    },
    Control {
        command_id: String,
        method: String,
        params: Value,
    },
    ExtensionEvent(ExtensionCommandInvocation),
}

impl Default for CommandSurfaceModel {
    fn default() -> Self {
        let commands = default_commands()
            .into_iter()
            .map(|command| {
                let id = command["id"]
                    .as_str()
                    .expect("built-in command ID")
                    .to_string();
                (id, command)
            })
            .collect::<BTreeMap<_, _>>();
        for command in commands.values() {
            validate_command_predicates(command.get("predicates"), "ui.commands.register")
                .expect("valid built-in command predicates");
            if let Some(shortcut) = command.get("shortcut").filter(|value| !value.is_null()) {
                validate_shortcut_for_platform(
                    shortcut,
                    ShortcutPlatform::current(),
                    "ui.commands.register",
                )
                .expect("valid built-in command shortcut");
            }
        }
        let menu = default_menu();
        validate_menu(&menu, &commands).expect("valid built-in command menu");
        let toolbar = default_toolbar();
        let palette = default_palette();
        validate_palette(&palette, &commands).expect("valid built-in command palette");
        Self {
            revision: 1,
            commands,
            menu,
            toolbar,
            palette,
        }
    }
}

impl CommandSurfaceModel {
    pub(crate) fn projection(&self) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "commands":self.commands.values().collect::<Vec<_>>(),
            "menu":self.menu,
            "toolbar":self.toolbar,
            "palette":self.palette,
        })
    }

    pub(crate) fn evaluated_projection(&self, context: &CommandEvaluationContext) -> Value {
        let mut projection = self.projection();
        projection["commands"] = Value::Array(self.evaluated_commands(context));
        projection["active_mode"] = json!(context.mode);
        projection["evaluation_context"] = context.state.clone();
        projection
    }

    #[cfg(test)]
    pub(crate) fn commands_snapshot(&self) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "commands":self.commands.values().collect::<Vec<_>>(),
        })
    }

    pub(crate) fn evaluated_commands_snapshot(&self, context: &CommandEvaluationContext) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "commands":self.evaluated_commands(context),
            "evaluation_context":context.state,
        })
    }

    fn evaluated_commands(&self, context: &CommandEvaluationContext) -> Vec<Value> {
        self.commands
            .values()
            .map(|command| evaluated_command(command, context))
            .collect()
    }

    pub(crate) fn menu_snapshot(&self) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "menu":self.menu,
        })
    }

    pub(crate) fn toolbar_snapshot(&self) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "toolbar":self.toolbar,
        })
    }

    pub(crate) fn palette_snapshot(&self) -> Value {
        json!({
            "schema_version":1,
            "revision":self.revision,
            "palette":self.palette,
        })
    }

    pub(crate) fn replace_toolbar(&mut self, params: &Value) -> Result<Value, ControlError> {
        let method = "ui.toolbars.replace";
        validate_revision(params, self.revision, method, "ui.toolbars.get")?;
        let transaction_id = transaction_id(params, method)?;
        let toolbar = params
            .get("toolbar")
            .ok_or_else(|| ControlError::invalid_params(method, "toolbar is required"))?;
        validate_toolbar(toolbar, &self.commands)?;
        let changed = &self.toolbar != toolbar;
        if changed {
            self.toolbar = toolbar.clone();
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        let mut snapshot = self.toolbar_snapshot();
        snapshot.as_object_mut().unwrap().insert(
            "change".to_string(),
            json!({
                "operation":"replace",
                "changed":changed,
                "transaction_id":transaction_id,
            }),
        );
        Ok(snapshot)
    }

    pub(crate) fn replace_palette(&mut self, params: &Value) -> Result<Value, ControlError> {
        let method = "ui.palette.replace";
        validate_revision(params, self.revision, method, "ui.palette.get")?;
        let transaction_id = transaction_id(params, method)?;
        let palette = params
            .get("palette")
            .ok_or_else(|| ControlError::invalid_params(method, "palette is required"))?;
        validate_palette(palette, &self.commands)?;
        let changed = &self.palette != palette;
        if changed {
            self.palette = palette.clone();
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        let mut snapshot = self.palette_snapshot();
        snapshot["change"] = json!({
            "operation":"replace",
            "changed":changed,
            "revision":self.revision,
            "transaction_id":transaction_id,
        });
        Ok(snapshot)
    }

    pub(crate) fn replace_menu(&mut self, params: &Value) -> Result<Value, ControlError> {
        validate_revision(params, self.revision, "ui.menus.replace", "ui.menus.get")?;
        let transaction_id = transaction_id(params, "ui.menus.replace")?;
        let menu = params
            .get("menu")
            .ok_or_else(|| ControlError::invalid_params("ui.menus.replace", "menu is required"))?;
        validate_menu(menu, &self.commands)?;
        let changed = &self.menu != menu;
        if changed {
            self.menu = menu.clone();
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        let mut snapshot = self.menu_snapshot();
        snapshot.as_object_mut().unwrap().insert(
            "change".to_string(),
            json!({
                "operation":"replace",
                "changed":changed,
                "transaction_id":transaction_id,
            }),
        );
        Ok(snapshot)
    }

    pub(crate) fn schema(&self) -> Value {
        command_surface_schema()
    }

    pub(crate) fn register_extension_command(
        &mut self,
        params: &Value,
        context: &ExtensionCommandContext,
    ) -> Result<Value, ControlError> {
        let method = "ui.commands.register";
        validate_revision(params, self.revision, method, "ui.commands.list")?;
        let transaction_id = transaction_id(params, method)?;
        let raw = params
            .get("command")
            .and_then(Value::as_object)
            .ok_or_else(|| ControlError::invalid_params(method, "command must be an object"))?;
        validate_keys_for(
            raw,
            &[
                "id",
                "title",
                "description",
                "event",
                "modes",
                "shortcut",
                "icon",
                "predicates",
            ],
            method,
        )?;
        let local_id = validate_command_local_id(raw.get("id"), method)?;
        let id = format!("extension:{}/{local_id}", context.extension_id);
        if id.len() > MAX_TEXT_BYTES {
            return Err(ControlError::invalid_params(
                method,
                "canonical extension command ID exceeds 256 bytes",
            ));
        }
        let title = validate_text_for(raw.get("title"), "command title", method)?;
        let description = validate_text_for(raw.get("description"), "command description", method)?;
        let event = validate_text_for(raw.get("event"), "command event", method)?;
        let modes = validate_command_modes(raw.get("modes"), method)?;
        let shortcut = validate_shortcut(raw.get("shortcut"), method)?;
        let predicates = validate_command_predicates(raw.get("predicates"), method)?;
        if let Some(shortcut) = shortcut.as_ref() {
            ensure_shortcut_available(&id, shortcut, &modes, &self.commands, method)?;
            ensure_palette_shortcut_available(&id, shortcut, &modes, &self.palette, method)?;
        }
        let icon = match raw.get("icon") {
            Some(Value::String(icon)) => {
                validate_bounded_text(icon, "command icon", method)?;
                Some(icon.clone())
            }
            Some(Value::Null) | None => None,
            _ => {
                return Err(ControlError::invalid_params(
                    method,
                    "command icon must be a string or null",
                ));
            }
        };
        if let Some(existing) = self.commands.get(&id)
            && existing
                .pointer("/ownership/owner_id")
                .and_then(Value::as_str)
                != Some(context.extension_id.as_str())
        {
            return Err(ControlError::new(
                ControlErrorKind::Conflict,
                format!("command '{id}' is owned by another authority"),
            ));
        }
        let readiness = if context.ready { "ready" } else { "not_ready" };
        let command = json!({
            "id":id,
            "title":title,
            "description":description,
            "handler":{"type":"event","event":event},
            "availability":{"modes":modes},
            "protected":false,
            "shortcut":shortcut,
            "icon":icon,
            "predicates":predicates,
            "ownership":{
                "scope":"extension",
                "owner_id":context.extension_id,
                "owner_session_id":context.owner_session_id,
                "protected":false,
            },
            "readiness":{
                "state":readiness,
                "expected_extension_version":context.extension_version,
                "current_extension_version":context.extension_version,
            },
            "disconnect_policy":disconnect_policy_name(&context.disconnect_policy),
        });
        let changed = self.commands.get(&id) != Some(&command);
        if changed {
            self.commands.insert(id.clone(), command.clone());
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        Ok(json!({
            "schema_version":1,
            "revision":self.revision,
            "command":command,
            "change":{
                "operation":"register",
                "changed":changed,
                "transaction_id":transaction_id,
            }
        }))
    }

    pub(crate) fn remove_extension_command(
        &mut self,
        params: &Value,
        context: &ExtensionCommandContext,
    ) -> Result<Value, ControlError> {
        let method = "ui.commands.remove";
        validate_revision(params, self.revision, method, "ui.commands.list")?;
        let transaction_id = transaction_id(params, method)?;
        let id = validate_text_for(params.get("command_id"), "command_id", method)?.to_string();
        let command = self.commands.get(&id).ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("command '{id}' was not found"),
            )
        })?;
        if command
            .pointer("/ownership/owner_id")
            .and_then(Value::as_str)
            != Some(context.extension_id.as_str())
        {
            return Err(ControlError::new(
                ControlErrorKind::PermissionDenied,
                format!(
                    "command '{id}' is not owned by extension '{}'",
                    context.extension_id
                ),
            ));
        }
        self.commands.remove(&id);
        remove_command_presentations(&mut self.menu, &BTreeSet::from([id.clone()]));
        remove_toolbar_presentations(&mut self.toolbar, &BTreeSet::from([id.clone()]));
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(json!({
            "schema_version":1,
            "revision":self.revision,
            "command_id":id,
            "removed":true,
            "menu":self.menu,
            "toolbar":self.toolbar,
            "change":{
                "operation":"remove",
                "changed":true,
                "transaction_id":transaction_id,
            }
        }))
    }

    pub(crate) fn cleanup_extensions(&mut self, extensions: &[UiExtensionCleanup]) -> Value {
        let remove_owners = extensions
            .iter()
            .filter(|extension| matches!(extension.disconnect_policy, DisconnectPolicy::Remove))
            .map(|extension| extension.extension_id.as_str())
            .collect::<BTreeSet<_>>();
        let disconnect_owners = extensions
            .iter()
            .filter(|extension| !matches!(extension.disconnect_policy, DisconnectPolicy::Remove))
            .map(|extension| extension.extension_id.as_str())
            .collect::<BTreeSet<_>>();
        let removed = self
            .commands
            .iter()
            .filter_map(|(id, command)| {
                command
                    .pointer("/ownership/owner_id")
                    .and_then(Value::as_str)
                    .is_some_and(|owner| remove_owners.contains(owner))
                    .then_some(id.clone())
            })
            .collect::<BTreeSet<_>>();
        let mut changed = !removed.is_empty();
        self.commands.retain(|id, _| !removed.contains(id));
        if !removed.is_empty() {
            remove_command_presentations(&mut self.menu, &removed);
            remove_toolbar_presentations(&mut self.toolbar, &removed);
        }
        for command in self.commands.values_mut() {
            let disconnected = command
                .pointer("/ownership/owner_id")
                .and_then(Value::as_str)
                .is_some_and(|owner| disconnect_owners.contains(owner));
            if disconnected {
                if command.pointer("/readiness/state").and_then(Value::as_str)
                    != Some("disconnected")
                {
                    command["readiness"]["state"] = json!("disconnected");
                    command["ownership"]["owner_session_id"] = Value::Null;
                    changed = true;
                }
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        json!({
            "schema_version":1,
            "revision":self.revision,
            "changed":changed,
            "removed_command_ids":removed,
            "menu":self.menu,
            "toolbar":self.toolbar,
        })
    }

    pub(crate) fn sync_extension(&mut self, context: &ExtensionCommandContext) -> Value {
        let mut changed = false;
        let mut command_ids = Vec::new();
        for (id, command) in &mut self.commands {
            if command
                .pointer("/ownership/owner_id")
                .and_then(Value::as_str)
                != Some(context.extension_id.as_str())
            {
                continue;
            }
            let expected_version = command
                .pointer("/readiness/expected_extension_version")
                .and_then(Value::as_str);
            let state = if expected_version != Some(context.extension_version.as_str()) {
                "incompatible_version"
            } else if context.ready {
                "ready"
            } else {
                "not_ready"
            };
            let previous = command.clone();
            command["readiness"]["state"] = json!(state);
            command["readiness"]["current_extension_version"] = json!(context.extension_version);
            command["ownership"]["owner_session_id"] = json!(context.owner_session_id);
            command["disconnect_policy"] =
                json!(disconnect_policy_name(&context.disconnect_policy));
            if *command != previous {
                changed = true;
                command_ids.push(id.clone());
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        json!({
            "schema_version":1,
            "revision":self.revision,
            "changed":changed,
            "extension_id":context.extension_id,
            "command_ids":command_ids,
        })
    }

    pub(crate) fn invocation(
        &self,
        params: &Value,
        context: &CommandEvaluationContext,
    ) -> Result<CommandInvocation, ControlError> {
        let method = "ui.commands.execute";
        let id = validate_text_for(params.get("command_id"), "command_id", method)?;
        let command = self.commands.get(id).ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::ResourceNotFound,
                format!("command '{id}' was not found"),
            )
        })?;
        let evaluated = evaluated_command(command, context);
        let state = evaluated.get("state").cloned().unwrap_or_else(|| json!({}));
        if state.get("visible").and_then(Value::as_bool) != Some(true)
            || state.get("enabled").and_then(Value::as_bool) != Some(true)
        {
            let missing_capabilities = state
                .get("missing_capabilities")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let kind = if missing_capabilities.is_empty() {
                ControlErrorKind::NotReady
            } else {
                ControlErrorKind::PermissionDenied
            };
            return Err(
                ControlError::new(kind, format!("command '{id}' is not available")).with_data(
                    json!({
                        "command_id":id,
                        "active_mode":context.mode,
                        "state":state,
                        "availability":command.get("availability"),
                        "readiness":command.get("readiness"),
                        "resolution":"satisfy the reported command predicates and retry",
                    }),
                ),
            );
        }
        let handler = command
            .get("handler")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ControlError::new(
                    ControlErrorKind::Application,
                    format!("command '{id}' has no valid handler"),
                )
            })?;
        match handler.get("type").and_then(Value::as_str) {
            Some("native") => Ok(CommandInvocation::Native {
                command_id: id.to_string(),
                action: handler
                    .get("action")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::Application,
                            format!("native command '{id}' has no action"),
                        )
                    })?
                    .to_string(),
                checked: params.get("checked").and_then(Value::as_bool),
            }),
            Some("control") => Ok(CommandInvocation::Control {
                command_id: id.to_string(),
                method: handler
                    .get("method")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::Application,
                            format!("control command '{id}' has no method"),
                        )
                    })?
                    .to_string(),
                params: handler.get("params").cloned().unwrap_or_else(|| json!({})),
            }),
            Some("event") => Ok(CommandInvocation::ExtensionEvent(
                ExtensionCommandInvocation {
                    command_id: id.to_string(),
                    extension_id: command
                        .pointer("/ownership/owner_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    owner_session_id: command
                        .pointer("/ownership/owner_session_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    event: handler
                        .get("event")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                },
            )),
            kind => Err(ControlError::new(
                ControlErrorKind::Unsupported,
                format!("command '{id}' uses unsupported handler type {kind:?}"),
            )),
        }
    }
}

fn validate_revision(
    params: &Value,
    current: u64,
    method: &str,
    snapshot_method: &str,
) -> Result<(), ControlError> {
    let Some(expected) = params.get("if_command_revision") else {
        return Ok(());
    };
    let expected = expected.as_u64().ok_or_else(|| {
        ControlError::invalid_params(method, "if_command_revision must be an unsigned integer")
    })?;
    if expected == current {
        return Ok(());
    }
    Err(ControlError::new(
        ControlErrorKind::Conflict,
        format!(
            "command-surface revision conflict: expected {expected}, current revision is {current}"
        ),
    )
    .with_data(json!({
        "expected_revision":expected,
        "current_revision":current,
        "conflicting_domain":"application_command_surface",
        "snapshot_method":snapshot_method,
        "retry_strategy":"refetch_merge_retry",
    })))
}

fn transaction_id<'a>(params: &'a Value, method: &str) -> Result<Option<&'a str>, ControlError> {
    let Some(value) = params.get("transaction_id") else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .ok_or_else(|| ControlError::invalid_params(method, "transaction_id must be a string"))?;
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(ControlError::invalid_params(
            method,
            "transaction_id must contain 1 to 256 non-control bytes",
        ));
    }
    Ok(Some(value))
}

fn validate_menu(menu: &Value, commands: &BTreeMap<String, Value>) -> Result<(), ControlError> {
    let object = menu
        .as_object()
        .ok_or_else(|| invalid("menu must be an object"))?;
    validate_keys(object, &["id", "type", "children"])?;
    if object.get("type").and_then(Value::as_str) != Some("menu_bar") {
        return Err(invalid("menu.type must be 'menu_bar'"));
    }
    validate_text(object.get("id"), "menu.id")?;
    let children = object
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("menu.children must be an array"))?;
    let mut ids = BTreeSet::new();
    ids.insert(object["id"].as_str().unwrap().to_string());
    let mut presented = BTreeSet::new();
    let mut count = 1;
    for child in children {
        if child.get("type").and_then(Value::as_str) != Some("menu") {
            return Err(invalid("menu_bar children must be menu nodes"));
        }
        validate_menu_node(child, commands, &mut ids, &mut presented, &mut count, 1)?;
    }
    for command in commands.values().filter(|command| {
        command
            .get("protected")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }) {
        let id = command["id"].as_str().unwrap();
        if !presented.contains(id) {
            return Err(invalid(format!(
                "protected command '{id}' must remain presented in the platform menu"
            )));
        }
    }
    Ok(())
}

fn validate_menu_node(
    node: &Value,
    commands: &BTreeMap<String, Value>,
    ids: &mut BTreeSet<String>,
    presented: &mut BTreeSet<String>,
    count: &mut usize,
    depth: usize,
) -> Result<(), ControlError> {
    *count += 1;
    if *count > MAX_MENU_NODES {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            format!("platform menu exceeds the {MAX_MENU_NODES}-node limit"),
        ));
    }
    if depth > MAX_MENU_DEPTH {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            format!("platform menu exceeds the maximum depth of {MAX_MENU_DEPTH}"),
        ));
    }
    let object = node
        .as_object()
        .ok_or_else(|| invalid("menu nodes must be objects"))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("menu node type is required"))?;
    let id = validate_text(object.get("id"), "menu node id")?;
    if !ids.insert(id.to_string()) {
        return Err(invalid(format!("duplicate menu node id '{id}'")));
    }
    match kind {
        "menu" => {
            validate_keys(object, &["id", "type", "title", "children"])?;
            validate_text(object.get("title"), "menu title")?;
            let children = object
                .get("children")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid(format!("menu '{id}' children must be an array")))?;
            for child in children {
                validate_menu_node(child, commands, ids, presented, count, depth + 1)?;
            }
        }
        "command" => {
            validate_keys(
                object,
                &["id", "type", "command_id", "label", "icon", "show_shortcut"],
            )?;
            let command_id = validate_text(object.get("command_id"), "command_id")?;
            if !commands.contains_key(command_id) {
                return Err(invalid(format!(
                    "menu node '{id}' references unknown command '{command_id}'"
                )));
            }
            if let Some(label) = object.get("label") {
                validate_text(Some(label), "menu command label")?;
            }
            if let Some(icon) = object.get("icon") {
                validate_text(Some(icon), "menu command icon")?;
            }
            if object
                .get("show_shortcut")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err(invalid("show_shortcut must be a boolean"));
            }
            presented.insert(command_id.to_string());
        }
        "separator" => validate_keys(object, &["id", "type"])?,
        _ => {
            return Err(invalid(format!(
                "menu node '{id}' has unsupported type '{kind}'"
            )));
        }
    }
    Ok(())
}

fn validate_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), ControlError> {
    validate_keys_for(object, allowed, "ui.menus.replace")
}

fn validate_keys_for(
    object: &Map<String, Value>,
    allowed: &[&str],
    method: &str,
) -> Result<(), ControlError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ControlError::invalid_params(
            method,
            format!("unknown property '{key}'"),
        ));
    }
    Ok(())
}

fn validate_text<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a str, ControlError> {
    validate_text_for(value, field, "ui.menus.replace")
}

fn validate_text_for<'a>(
    value: Option<&'a Value>,
    field: &str,
    method: &str,
) -> Result<&'a str, ControlError> {
    let value = value
        .and_then(Value::as_str)
        .ok_or_else(|| ControlError::invalid_params(method, format!("{field} must be a string")))?;
    validate_bounded_text(value, field, method)?;
    Ok(value)
}

fn validate_bounded_text(value: &str, field: &str, method: &str) -> Result<(), ControlError> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        return Err(ControlError::invalid_params(
            method,
            format!("{field} must contain 1 to {MAX_TEXT_BYTES} non-control bytes"),
        ));
    }
    Ok(())
}

fn validate_command_local_id<'a>(
    value: Option<&'a Value>,
    method: &str,
) -> Result<&'a str, ControlError> {
    let id = validate_text_for(value, "command id", method)?;
    if id.starts_with('.')
        || id.ends_with('.')
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ControlError::invalid_params(
            method,
            "command id must use ASCII letters, digits, '.', '_', or '-' and cannot begin or end with '.'",
        ));
    }
    Ok(id)
}

fn validate_command_modes(
    value: Option<&Value>,
    method: &str,
) -> Result<Vec<String>, ControlError> {
    const ALL_MODES: [&str; 3] = ["project", "single", "mosaic"];
    let Some(value) = value else {
        return Ok(ALL_MODES.into_iter().map(str::to_string).collect());
    };
    let modes = value
        .as_array()
        .ok_or_else(|| ControlError::invalid_params(method, "command modes must be an array"))?;
    if modes.is_empty() {
        return Err(ControlError::invalid_params(
            method,
            "command modes must contain at least one mode",
        ));
    }
    let mut normalized = Vec::with_capacity(modes.len());
    for mode in modes {
        let mode = mode.as_str().ok_or_else(|| {
            ControlError::invalid_params(method, "each command mode must be a string")
        })?;
        if !ALL_MODES.contains(&mode) {
            return Err(ControlError::invalid_params(
                method,
                format!("unsupported command mode '{mode}'"),
            ));
        }
        if normalized.iter().any(|existing| existing == mode) {
            return Err(ControlError::invalid_params(
                method,
                format!("duplicate command mode '{mode}'"),
            ));
        }
        normalized.push(mode.to_string());
    }
    Ok(normalized)
}

fn validate_shortcut(value: Option<&Value>, method: &str) -> Result<Option<Value>, ControlError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value.as_object().ok_or_else(|| {
        ControlError::invalid_params(method, "shortcut must be an object or null")
    })?;
    validate_keys_for(object, &["key", "modifiers"], method)?;
    let key = validate_text_for(object.get("key"), "shortcut key", method)?;
    if !(key == "comma" || (key.len() == 1 && key.as_bytes()[0].is_ascii_alphabetic())) {
        return Err(ControlError::invalid_params(
            method,
            "shortcut key must be one ASCII letter or 'comma'",
        ));
    }
    let modifiers = object
        .get("modifiers")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ControlError::invalid_params(method, "shortcut modifiers must be an array")
        })?;
    let mut normalized = BTreeSet::new();
    for modifier in modifiers {
        let modifier = modifier.as_str().ok_or_else(|| {
            ControlError::invalid_params(method, "each shortcut modifier must be a string")
        })?;
        if !["primary", "shift", "alt", "control", "super"].contains(&modifier) {
            return Err(ControlError::invalid_params(
                method,
                format!("unsupported shortcut modifier '{modifier}'"),
            ));
        }
        if !normalized.insert(modifier.to_string()) {
            return Err(ControlError::invalid_params(
                method,
                format!("duplicate shortcut modifier '{modifier}'"),
            ));
        }
    }
    let shortcut = json!({
        "key":key.to_ascii_lowercase(),
        "modifiers":normalized.into_iter().collect::<Vec<_>>(),
    });
    validate_shortcut_for_platform(&shortcut, ShortcutPlatform::current(), method)?;
    Ok(Some(shortcut))
}

fn validate_shortcut_for_platform(
    shortcut: &Value,
    platform: ShortcutPlatform,
    method: &str,
) -> Result<(), ControlError> {
    let mut effective = BTreeSet::new();
    for modifier in shortcut
        .get("modifiers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        let resolved = match modifier {
            "primary" => platform.primary(),
            "super" if !platform.supports_super() => {
                return Err(ControlError::new(
                    ControlErrorKind::Unsupported,
                    format!(
                        "shortcut modifier 'super' is not available on {}; use 'primary' for a portable platform command modifier",
                        platform.name()
                    ),
                )
                .with_data(json!({
                    "method":method,
                    "platform":platform.name(),
                    "unsupported_modifier":"super",
                    "supported_modifiers":["primary","shift","alt","control"],
                    "resolution":"replace_super_with_primary_or_choose_a_supported_modifier",
                })));
            }
            other => other,
        };
        if !effective.insert(resolved) {
            return Err(ControlError::invalid_params(
                method,
                format!(
                    "shortcut modifiers resolve to duplicate '{resolved}' on {}",
                    platform.name()
                ),
            )
            .with_data(json!({
                "method":method,
                "platform":platform.name(),
                "effective_modifier":resolved,
                "resolution":"remove_the_redundant_platform_alias",
            })));
        }
    }
    Ok(())
}

fn effective_shortcut_signature(
    shortcut: &Value,
    platform: ShortcutPlatform,
) -> Option<(String, BTreeSet<String>)> {
    let key = shortcut.get("key")?.as_str()?.to_ascii_lowercase();
    let mut modifiers = BTreeSet::new();
    for modifier in shortcut.get("modifiers")?.as_array()? {
        let modifier = match modifier.as_str()? {
            "primary" => platform.primary(),
            "super" if !platform.supports_super() => return None,
            other => other,
        };
        modifiers.insert(modifier.to_string());
    }
    Some((key, modifiers))
}

fn shortcuts_equivalent(left: &Value, right: &Value, platform: ShortcutPlatform) -> bool {
    effective_shortcut_signature(left, platform)
        .zip(effective_shortcut_signature(right, platform))
        .is_some_and(|(left, right)| left == right)
}

fn ensure_shortcut_available(
    id: &str,
    shortcut: &Value,
    modes: &[String],
    commands: &BTreeMap<String, Value>,
    method: &str,
) -> Result<(), ControlError> {
    let requested_modes = modes.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if let Some((conflicting_id, conflicting_modes)) =
        commands.iter().find_map(|(other_id, command)| {
            if other_id == id
                || !command.get("shortcut").is_some_and(|other| {
                    shortcuts_equivalent(other, shortcut, ShortcutPlatform::current())
                })
            {
                return None;
            }
            let overlap = command
                .pointer("/availability/modes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .filter(|mode| requested_modes.contains(mode))
                .map(str::to_string)
                .collect::<Vec<_>>();
            (!overlap.is_empty()).then_some((other_id, overlap))
        })
    {
        return Err(ControlError::new(
            ControlErrorKind::Conflict,
            format!("shortcut conflicts with command '{conflicting_id}'"),
        )
        .with_data(json!({
            "method":method,
            "command_id":id,
            "conflicting_command_id":conflicting_id,
            "overlapping_modes":conflicting_modes,
            "resolution":"choose_an_unclaimed_shortcut_or_non_overlapping_modes",
        })));
    }
    Ok(())
}

fn validate_command_predicates(value: Option<&Value>, method: &str) -> Result<Value, ControlError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(json!({}));
    };
    let predicates = value.as_object().ok_or_else(|| {
        ControlError::invalid_params(method, "command predicates must be an object")
    })?;
    validate_keys_for(predicates, &["visible", "enabled", "checked"], method)?;
    let mut nodes = 0;
    for (name, predicate) in predicates {
        if predicate.is_null() {
            continue;
        }
        validate_predicate(predicate, method, name, 1, &mut nodes)?;
    }
    Ok(value.clone())
}

fn validate_predicate(
    predicate: &Value,
    method: &str,
    label: &str,
    depth: usize,
    nodes: &mut usize,
) -> Result<(), ControlError> {
    *nodes += 1;
    if *nodes > MAX_PREDICATE_NODES {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            format!("command predicates exceed the {MAX_PREDICATE_NODES}-node limit"),
        ));
    }
    if depth > MAX_PREDICATE_DEPTH {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            format!("command predicates exceed the depth-{MAX_PREDICATE_DEPTH} limit"),
        ));
    }
    let object = predicate.as_object().ok_or_else(|| {
        ControlError::invalid_params(method, format!("{label} predicate must be an object"))
    })?;
    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        ControlError::invalid_params(method, format!("{label} predicate requires type"))
    })?;
    if let Some(reason) = object.get("reason") {
        validate_text_for(Some(reason), "predicate reason", method)?;
    }
    match kind {
        "always" => {
            validate_keys_for(object, &["type", "value", "reason"], method)?;
            if !object.get("value").is_some_and(Value::is_boolean) {
                return Err(ControlError::invalid_params(
                    method,
                    format!("{label} always predicate requires a boolean value"),
                ));
            }
        }
        "capability" => {
            validate_keys_for(object, &["type", "capability", "reason"], method)?;
            validate_text_for(object.get("capability"), "predicate capability", method)?;
        }
        "state" => {
            validate_keys_for(
                object,
                &["type", "path", "operator", "value", "reason"],
                method,
            )?;
            let path = validate_text_for(object.get("path"), "predicate state path", method)?;
            if !COMMAND_STATE_PATHS.contains(&path) {
                return Err(ControlError::invalid_params(
                    method,
                    format!("unsupported command state path '{path}'"),
                ));
            }
            let operator = object
                .get("operator")
                .and_then(Value::as_str)
                .unwrap_or("truthy");
            if ![
                "truthy",
                "falsy",
                "equals",
                "not_equals",
                "greater_than",
                "at_least",
            ]
            .contains(&operator)
            {
                return Err(ControlError::invalid_params(
                    method,
                    format!("unsupported command state operator '{operator}'"),
                ));
            }
            let comparison = matches!(
                operator,
                "equals" | "not_equals" | "greater_than" | "at_least"
            );
            if comparison != object.contains_key("value") {
                return Err(ControlError::invalid_params(
                    method,
                    format!(
                        "{label} state predicate operator '{operator}' {} a value",
                        if comparison {
                            "requires"
                        } else {
                            "does not accept"
                        }
                    ),
                ));
            }
            if matches!(operator, "greater_than" | "at_least")
                && !object.get("value").is_some_and(Value::is_number)
            {
                return Err(ControlError::invalid_params(
                    method,
                    format!("{label} numeric state predicate requires a numeric value"),
                ));
            }
            if object
                .get("value")
                .is_some_and(|value| value.is_array() || value.is_object() || value.is_null())
            {
                return Err(ControlError::invalid_params(
                    method,
                    format!("{label} predicate value must be a scalar"),
                ));
            }
        }
        "all" | "any" => {
            validate_keys_for(object, &["type", "predicates", "reason"], method)?;
            let children = object
                .get("predicates")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    ControlError::invalid_params(
                        method,
                        format!("{label} {kind} predicate requires a predicates array"),
                    )
                })?;
            if children.is_empty() {
                return Err(ControlError::invalid_params(
                    method,
                    format!("{label} {kind} predicate must contain at least one child"),
                ));
            }
            for child in children {
                validate_predicate(child, method, label, depth + 1, nodes)?;
            }
        }
        "not" => {
            validate_keys_for(object, &["type", "predicate", "reason"], method)?;
            let child = object.get("predicate").ok_or_else(|| {
                ControlError::invalid_params(
                    method,
                    format!("{label} not predicate requires predicate"),
                )
            })?;
            validate_predicate(child, method, label, depth + 1, nodes)?;
        }
        _ => {
            return Err(ControlError::invalid_params(
                method,
                format!("unsupported command predicate type '{kind}'"),
            ));
        }
    }
    Ok(())
}

fn evaluated_command(command: &Value, context: &CommandEvaluationContext) -> Value {
    let mut evaluated = command.clone();
    let in_mode = command
        .pointer("/availability/modes")
        .and_then(Value::as_array)
        .is_some_and(|modes| {
            modes
                .iter()
                .any(|candidate| candidate.as_str() == Some(context.mode.as_str()))
        });
    let readiness = command.pointer("/readiness/state").and_then(Value::as_str);
    let ready = readiness.is_none_or(|state| state == "ready");
    let predicates = command.get("predicates").and_then(Value::as_object);
    let visible = predicates
        .and_then(|predicates| predicates.get("visible"))
        .filter(|predicate| !predicate.is_null())
        .map(|predicate| evaluate_predicate(predicate, context))
        .unwrap_or(PredicateEvaluation {
            value: true,
            ..Default::default()
        });
    let enabled_predicate = predicates
        .and_then(|predicates| predicates.get("enabled"))
        .filter(|predicate| !predicate.is_null())
        .map(|predicate| evaluate_predicate(predicate, context))
        .unwrap_or(PredicateEvaluation {
            value: true,
            ..Default::default()
        });
    let checked = predicates
        .and_then(|predicates| predicates.get("checked"))
        .filter(|predicate| !predicate.is_null())
        .map(|predicate| evaluate_predicate(predicate, context));

    let mut reasons = Vec::new();
    let mut missing_capabilities = BTreeSet::new();
    if !in_mode {
        reasons.push(format!("not available in {} mode", context.mode));
    }
    if !ready {
        reasons.push(
            command
                .pointer("/readiness/reason")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!("command readiness is {}", readiness.unwrap_or("unknown"))
                }),
        );
    }
    if !visible.value {
        reasons.extend(visible.reasons.clone());
        missing_capabilities.extend(visible.missing_capabilities.iter().cloned());
    }
    if !enabled_predicate.value {
        reasons.extend(enabled_predicate.reasons.clone());
        missing_capabilities.extend(enabled_predicate.missing_capabilities.iter().cloned());
    }
    evaluated["state"] = json!({
        "visible":visible.value,
        "enabled":visible.value && in_mode && ready && enabled_predicate.value,
        "checkable":checked.is_some(),
        "checked":checked.as_ref().map(|result| result.value),
        "reasons":reasons,
        "missing_capabilities":missing_capabilities,
    });
    evaluated
}

fn evaluate_predicate(
    predicate: &Value,
    context: &CommandEvaluationContext,
) -> PredicateEvaluation {
    let Some(object) = predicate.as_object() else {
        return PredicateEvaluation {
            value: false,
            reasons: vec!["invalid command predicate".to_string()],
            ..Default::default()
        };
    };
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let explicit_reason = object.get("reason").and_then(Value::as_str);
    let failed = |fallback: String| PredicateEvaluation {
        value: false,
        reasons: vec![explicit_reason.unwrap_or(&fallback).to_string()],
        ..Default::default()
    };
    match kind {
        "always" => {
            let value = object
                .get("value")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if value {
                PredicateEvaluation {
                    value: true,
                    ..Default::default()
                }
            } else {
                failed("command condition is false".to_string())
            }
        }
        "capability" => {
            let capability = object
                .get("capability")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if context.has_capability(capability) {
                PredicateEvaluation {
                    value: true,
                    ..Default::default()
                }
            } else {
                let mut result = failed(format!("requires capability '{capability}'"));
                result.missing_capabilities.insert(capability.to_string());
                result
            }
        }
        "state" => {
            let path = object
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let operator = object
                .get("operator")
                .and_then(Value::as_str)
                .unwrap_or("truthy");
            let actual = command_state_value(&context.state, path);
            let expected = object.get("value");
            let value = match operator {
                "truthy" => actual.is_some_and(value_truthy),
                "falsy" => actual.is_none_or(|value| !value_truthy(value)),
                "equals" => actual
                    .zip(expected)
                    .is_some_and(|(actual, expected)| actual == expected),
                "not_equals" => actual
                    .zip(expected)
                    .is_some_and(|(actual, expected)| actual != expected),
                "greater_than" => {
                    numeric_comparison(actual, expected, |actual, expected| actual > expected)
                }
                "at_least" => {
                    numeric_comparison(actual, expected, |actual, expected| actual >= expected)
                }
                _ => false,
            };
            if value {
                PredicateEvaluation {
                    value: true,
                    ..Default::default()
                }
            } else {
                failed(format!("state '{path}' must satisfy '{operator}'"))
            }
        }
        "all" | "any" => {
            let children = object
                .get("predicates")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|predicate| evaluate_predicate(predicate, context))
                .collect::<Vec<_>>();
            let value = if kind == "all" {
                children.iter().all(|child| child.value)
            } else {
                children.iter().any(|child| child.value)
            };
            if value {
                return PredicateEvaluation {
                    value: true,
                    ..Default::default()
                };
            }
            let mut result = PredicateEvaluation {
                value: false,
                reasons: explicit_reason
                    .map(|reason| vec![reason.to_string()])
                    .unwrap_or_else(|| {
                        children
                            .iter()
                            .filter(|child| !child.value)
                            .flat_map(|child| child.reasons.iter().cloned())
                            .collect()
                    }),
                ..Default::default()
            };
            for child in children.iter().filter(|child| !child.value) {
                result
                    .missing_capabilities
                    .extend(child.missing_capabilities.iter().cloned());
            }
            result
        }
        "not" => {
            let child = object
                .get("predicate")
                .map(|predicate| evaluate_predicate(predicate, context))
                .unwrap_or_default();
            if !child.value {
                PredicateEvaluation {
                    value: true,
                    ..Default::default()
                }
            } else {
                failed("negated command condition is true".to_string())
            }
        }
        _ => failed("unsupported command predicate".to_string()),
    }
}

fn command_state_value<'a>(state: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.')
        .try_fold(state, |value, segment| value.get(segment))
}

fn value_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Null => false,
    }
}

fn numeric_comparison(
    actual: Option<&Value>,
    expected: Option<&Value>,
    compare: impl FnOnce(f64, f64) -> bool,
) -> bool {
    actual
        .and_then(Value::as_f64)
        .zip(expected.and_then(Value::as_f64))
        .is_some_and(|(actual, expected)| compare(actual, expected))
}

fn ensure_palette_shortcut_available(
    command_id: &str,
    shortcut: &Value,
    modes: &[String],
    palette: &Value,
    method: &str,
) -> Result<(), ControlError> {
    if !palette.get("shortcut").is_some_and(|palette_shortcut| {
        shortcuts_equivalent(palette_shortcut, shortcut, ShortcutPlatform::current())
    }) {
        return Ok(());
    }
    Err(ControlError::new(
        ControlErrorKind::Conflict,
        "shortcut conflicts with the command palette",
    )
    .with_data(json!({
        "method":method,
        "command_id":command_id,
        "conflicting_presentation_id":palette.get("id"),
        "overlapping_modes":modes,
        "resolution":"choose_an_unclaimed_shortcut",
    })))
}

fn validate_palette(
    palette: &Value,
    commands: &BTreeMap<String, Value>,
) -> Result<(), ControlError> {
    let method = "ui.palette.replace";
    let object = palette
        .as_object()
        .ok_or_else(|| ControlError::invalid_params(method, "palette must be an object"))?;
    validate_keys_for(
        object,
        &[
            "id",
            "type",
            "title",
            "placeholder",
            "shortcut",
            "show_descriptions",
            "max_results",
        ],
        method,
    )?;
    let id = validate_text_for(object.get("id"), "palette id", method)?;
    if object.get("type").and_then(Value::as_str) != Some("command_palette") {
        return Err(ControlError::invalid_params(
            method,
            "palette.type must be 'command_palette'",
        ));
    }
    validate_text_for(object.get("title"), "palette title", method)?;
    validate_text_for(object.get("placeholder"), "palette placeholder", method)?;
    if object
        .get("show_descriptions")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err(ControlError::invalid_params(
            method,
            "palette.show_descriptions must be a boolean",
        ));
    }
    let max_results = object
        .get("max_results")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ControlError::invalid_params(method, "palette.max_results must be an unsigned integer")
        })?;
    if !(1..=MAX_PALETTE_RESULTS).contains(&max_results) {
        return Err(ControlError::invalid_params(
            method,
            format!("palette.max_results must be between 1 and {MAX_PALETTE_RESULTS}"),
        ));
    }
    let shortcut = validate_shortcut(object.get("shortcut"), method)?
        .ok_or_else(|| ControlError::invalid_params(method, "palette.shortcut must not be null"))?;
    ensure_shortcut_available(
        id,
        &shortcut,
        &[
            "project".to_string(),
            "single".to_string(),
            "mosaic".to_string(),
        ],
        commands,
        method,
    )
}

fn disconnect_policy_name(policy: &DisconnectPolicy) -> &'static str {
    match policy {
        DisconnectPolicy::Remove => "remove",
        DisconnectPolicy::Disable => "disable",
        DisconnectPolicy::Retain => "retain",
    }
}

fn remove_command_presentations(node: &mut Value, removed: &BTreeSet<String>) {
    let Some(object) = node.as_object_mut() else {
        return;
    };
    let Some(children) = object.get_mut("children").and_then(Value::as_array_mut) else {
        return;
    };
    children.retain(|child| {
        child
            .get("command_id")
            .and_then(Value::as_str)
            .is_none_or(|command_id| !removed.contains(command_id))
    });
    for child in children {
        remove_command_presentations(child, removed);
    }
}

fn validate_toolbar(
    toolbar: &Value,
    commands: &BTreeMap<String, Value>,
) -> Result<(), ControlError> {
    let method = "ui.toolbars.replace";
    let object = toolbar
        .as_object()
        .ok_or_else(|| ControlError::invalid_params(method, "toolbar must be an object"))?;
    validate_keys_for(object, &["id", "type", "groups"], method)?;
    validate_text_for(object.get("id"), "toolbar id", method)?;
    if object.get("type").and_then(Value::as_str) != Some("toolbar") {
        return Err(ControlError::invalid_params(
            method,
            "toolbar.type must be 'toolbar'",
        ));
    }
    let groups = object
        .get("groups")
        .and_then(Value::as_array)
        .ok_or_else(|| ControlError::invalid_params(method, "toolbar.groups must be an array"))?;
    if groups.len() > MAX_TOOLBAR_GROUPS {
        return Err(ControlError::new(
            ControlErrorKind::ResourceLimit,
            format!("toolbar exceeds the {MAX_TOOLBAR_GROUPS}-group limit"),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut item_count = 0;
    for group in groups {
        let group = group.as_object().ok_or_else(|| {
            ControlError::invalid_params(method, "toolbar groups must be objects")
        })?;
        validate_keys_for(group, &["id", "title", "items"], method)?;
        let group_id = validate_text_for(group.get("id"), "toolbar group id", method)?;
        if !ids.insert(group_id.to_string()) {
            return Err(ControlError::invalid_params(
                method,
                format!("duplicate toolbar presentation ID '{group_id}'"),
            ));
        }
        if let Some(title) = group.get("title") {
            validate_text_for(Some(title), "toolbar group title", method)?;
        }
        let items = group
            .get("items")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ControlError::invalid_params(method, "toolbar group items must be an array")
            })?;
        item_count += items.len();
        if item_count > MAX_TOOLBAR_ITEMS {
            return Err(ControlError::new(
                ControlErrorKind::ResourceLimit,
                format!("toolbar exceeds the {MAX_TOOLBAR_ITEMS}-item limit"),
            ));
        }
        for item in items {
            let item = item.as_object().ok_or_else(|| {
                ControlError::invalid_params(method, "toolbar items must be objects")
            })?;
            validate_keys_for(
                item,
                &["id", "command_id", "label", "icon", "tooltip", "show_label"],
                method,
            )?;
            let item_id = validate_text_for(item.get("id"), "toolbar item id", method)?;
            if !ids.insert(item_id.to_string()) {
                return Err(ControlError::invalid_params(
                    method,
                    format!("duplicate toolbar presentation ID '{item_id}'"),
                ));
            }
            let command_id =
                validate_text_for(item.get("command_id"), "toolbar command_id", method)?;
            if !commands.contains_key(command_id) {
                return Err(ControlError::invalid_params(
                    method,
                    format!("toolbar item '{item_id}' references unknown command '{command_id}'"),
                ));
            }
            for (field, label) in [
                ("label", "toolbar item label"),
                ("icon", "toolbar item icon"),
                ("tooltip", "toolbar item tooltip"),
            ] {
                if let Some(value) = item.get(field) {
                    validate_text_for(Some(value), label, method)?;
                }
            }
            if item
                .get("show_label")
                .is_some_and(|value| !value.is_boolean())
            {
                return Err(ControlError::invalid_params(
                    method,
                    "toolbar item show_label must be a boolean",
                ));
            }
        }
    }
    Ok(())
}

fn remove_toolbar_presentations(toolbar: &mut Value, removed: &BTreeSet<String>) {
    for group in toolbar
        .get_mut("groups")
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
    {
        if let Some(items) = group.get_mut("items").and_then(Value::as_array_mut) {
            items.retain(|item| {
                item.get("command_id")
                    .and_then(Value::as_str)
                    .is_none_or(|command_id| !removed.contains(command_id))
            });
        }
    }
}

fn invalid(message: impl Into<String>) -> ControlError {
    ControlError::invalid_params("ui.menus.replace", message)
}

fn command(
    id: &str,
    title: &str,
    description: &str,
    native_action: &str,
    modes: &[&str],
    protected: bool,
    shortcut: Option<Value>,
) -> Value {
    json!({
        "id":id,
        "title":title,
        "description":description,
        "handler":{"type":"native","action":native_action},
        "availability":{"modes":modes},
        "protected":protected,
        "shortcut":shortcut,
        "icon":null,
        "predicates":{},
    })
}

fn default_commands() -> Vec<Value> {
    let all = ["project", "single", "mosaic"];
    let viewer = ["single", "mosaic"];
    let single = ["single"];
    let mut commands = vec![
        command(
            "app.about",
            "About odon",
            "Show application information.",
            "about",
            &all,
            false,
            None,
        ),
        command(
            "app.settings.open",
            "Settings…",
            "Open Odon settings.",
            "settings",
            &all,
            false,
            Some(json!({"key":"comma","modifiers":["primary"]})),
        ),
        command(
            "dataset.open.ome_zarr",
            "Open OME-Zarr…",
            "Choose and open an OME-Zarr dataset.",
            "open_ome_zarr",
            &all,
            false,
            Some(json!({"key":"o","modifiers":["primary"]})),
        ),
        command(
            "dataset.open.tiff",
            "Open TIFF / OME-TIFF…",
            "Choose and open a TIFF dataset.",
            "open_tiff",
            &all,
            false,
            None,
        ),
        command(
            "project.open",
            "Open Project…",
            "Choose and open an Odon project.",
            "open_project",
            &all,
            false,
            Some(json!({"key":"o","modifiers":["primary","shift"]})),
        ),
        command(
            "project.save",
            "Save Project…",
            "Save the current project.",
            "save_project",
            &all,
            false,
            Some(json!({"key":"s","modifiers":["primary"]})),
        ),
        command(
            "project.save_as",
            "Save New Project…",
            "Save the project to a new path.",
            "save_new_project",
            &all,
            false,
            None,
        ),
        command(
            "viewer.screenshot.save",
            "Save Screenshot…",
            "Choose a destination and capture the current view.",
            "save_screenshot",
            &viewer,
            false,
            None,
        ),
        command(
            "viewer.screenshot.quick",
            "Quick Screenshot",
            "Capture to the configured screenshot directory.",
            "quick_screenshot",
            &viewer,
            false,
            Some(json!({"key":"s","modifiers":["primary","shift"]})),
        ),
        command(
            "viewer.screenshot.settings",
            "Screenshot Settings…",
            "Open screenshot settings.",
            "screenshot_settings",
            &viewer,
            false,
            None,
        ),
        command(
            "viewer.roi_info.show",
            "ROI Info",
            "Show information about the current ROI.",
            "roi_info",
            &single,
            false,
            Some(json!({"key":"i","modifiers":["primary"]})),
        ),
        command(
            "viewer.annotations.add",
            "Annotations",
            "Add an annotation layer.",
            "add_annotations",
            &single,
            false,
            None,
        ),
        command(
            "viewer.segmentation.load_geojson",
            "Load Seg GeoJSON…",
            "Load segmentation geometry.",
            "load_seg_geojson",
            &single,
            false,
            None,
        ),
        command(
            "viewer.segmentation.load_objects",
            "Load Seg Objects…",
            "Load segmentation objects.",
            "load_seg_objects",
            &single,
            false,
            None,
        ),
        command(
            "viewer.masks.export_geojson",
            "Export Masks GeoJSON…",
            "Export masks as GeoJSON.",
            "export_masks_geojson",
            &single,
            false,
            None,
        ),
        command(
            "viewer.scale_bar.toggle",
            "Scale Bar",
            "Toggle the scale bar.",
            "toggle_scale_bar",
            &single,
            false,
            None,
        ),
        command(
            "app.window.close",
            "Close Window",
            "Safely close the current Odon window.",
            "close_window",
            &all,
            true,
            Some(json!({"key":"w","modifiers":["primary"]})),
        ),
        command(
            "app.lifecycle.quit",
            "Quit odon",
            "Safely quit Odon.",
            "quit",
            &all,
            true,
            Some(json!({"key":"q","modifiers":["primary"]})),
        ),
        json!({
            "id":"app.shell.reset",
            "title":"Reset to Default Layout",
            "description":"Restore Odon's default application layout for the active mode.",
            "handler":{"type":"control","method":"ui.shell.reset","params":{}},
            "availability":{"modes":all},
            "protected":true,
            "shortcut":null,
            "icon":null,
            "predicates":{},
        }),
        json!({
            "id":"app.shell.recover",
            "title":"Recover Application Layout",
            "description":"Install Odon's protected recovery layout for the active mode.",
            "handler":{"type":"control","method":"ui.shell.recover","params":{}},
            "availability":{"modes":all},
            "protected":true,
            "shortcut":null,
            "icon":null,
            "predicates":{},
        }),
    ];
    for command in &mut commands {
        match command.get("id").and_then(Value::as_str) {
            Some("viewer.masks.export_geojson") => {
                command["predicates"]["enabled"] = json!({
                    "type":"state",
                    "path":"resources.masks",
                    "operator":"truthy",
                    "reason":"No mask layers are available to export.",
                });
            }
            Some("viewer.scale_bar.toggle") => {
                command["predicates"]["checked"] = json!({
                    "type":"state",
                    "path":"presentation.scale_bar.checked",
                    "operator":"truthy",
                });
            }
            _ => {}
        }
    }
    commands
}

#[doc(hidden)]
pub fn command_surface_native_actions() -> BTreeSet<String> {
    default_commands()
        .into_iter()
        .filter(|command| {
            command.pointer("/handler/type").and_then(Value::as_str) == Some("native")
        })
        .filter_map(|command| {
            command
                .pointer("/handler/action")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

fn menu(id: &str, title: &str, children: Vec<Value>) -> Value {
    json!({"id":id,"type":"menu","title":title,"children":children})
}

fn item(id: &str, command_id: &str) -> Value {
    json!({"id":id,"type":"command","command_id":command_id})
}

fn separator(id: &str) -> Value {
    json!({"id":id,"type":"separator"})
}

fn default_menu() -> Value {
    json!({
        "id":"menu:root",
        "type":"menu_bar",
        "children":[
            menu("menu:application", "odon", vec![
                item("menu-item:about", "app.about"),
                separator("menu-separator:application-1"),
                item("menu-item:settings", "app.settings.open"),
                separator("menu-separator:application-2"),
                item("menu-item:quit", "app.lifecycle.quit"),
            ]),
            menu("menu:file", "File", vec![
                item("menu-item:open-omezarr", "dataset.open.ome_zarr"),
                item("menu-item:open-tiff", "dataset.open.tiff"),
                separator("menu-separator:file-1"),
                item("menu-item:open-project", "project.open"),
                item("menu-item:save-project", "project.save"),
                item("menu-item:save-project-as", "project.save_as"),
                separator("menu-separator:file-2"),
                item("menu-item:export-masks", "viewer.masks.export_geojson"),
                separator("menu-separator:file-3"),
                item("menu-item:save-screenshot", "viewer.screenshot.save"),
                item("menu-item:quick-screenshot", "viewer.screenshot.quick"),
                item("menu-item:screenshot-settings", "viewer.screenshot.settings"),
                separator("menu-separator:file-4"),
                item("menu-item:close", "app.window.close"),
            ]),
            menu("menu:add", "Add", vec![
                item("menu-item:add-annotations", "viewer.annotations.add"),
                separator("menu-separator:add-1"),
                item("menu-item:load-seg-geojson", "viewer.segmentation.load_geojson"),
                item("menu-item:load-seg-objects", "viewer.segmentation.load_objects"),
            ]),
            menu("menu:view", "View", vec![
                item("menu-item:roi-info", "viewer.roi_info.show"),
                separator("menu-separator:view-1"),
                item("menu-item:scale-bar", "viewer.scale_bar.toggle"),
                separator("menu-separator:view-2"),
                item("menu-item:reset-layout", "app.shell.reset"),
            ]),
            menu("menu:help", "Help", vec![
                item("menu-item:recover-layout", "app.shell.recover"),
            ]),
        ]
    })
}

fn default_toolbar() -> Value {
    json!({
        "id":"toolbar:main",
        "type":"toolbar",
        "groups":[],
    })
}

fn default_palette() -> Value {
    json!({
        "id":"palette:main",
        "type":"command_palette",
        "title":"Commands",
        "placeholder":"Search commands…",
        "shortcut":{"key":"p","modifiers":["primary","shift"]},
        "show_descriptions":true,
        "max_results":20,
    })
}

fn command_surface_schema() -> Value {
    let shortcut_platform = ShortcutPlatform::current();
    let supported_modifiers = if shortcut_platform.supports_super() {
        json!(["primary", "shift", "alt", "control", "super"])
    } else {
        json!(["primary", "shift", "alt", "control"])
    };
    json!({
        "schema_version":1,
        "mutation_scopes":["extension_command_catalogue","platform_menu_presentation","toolbar_presentation","palette_presentation"],
        "limits":{
            "max_nodes":MAX_MENU_NODES,
            "max_depth":MAX_MENU_DEPTH,
            "max_toolbar_groups":MAX_TOOLBAR_GROUPS,
            "max_toolbar_items":MAX_TOOLBAR_ITEMS,
            "max_palette_results":MAX_PALETTE_RESULTS,
            "max_predicate_nodes":MAX_PREDICATE_NODES,
            "max_predicate_depth":MAX_PREDICATE_DEPTH,
            "max_text_bytes":MAX_TEXT_BYTES
        },
        "command":{
            "required":["id","title","description","handler","availability","protected"],
            "handler_types":["native","control","event"],
            "extension_command_id":"extension:<extension-id>/<local-id>",
            "extension_readiness_states":["ready","not_ready","disconnected","incompatible_version"],
            "shortcut_keys":["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","comma"],
            "shortcut_modifiers":["primary","shift","alt","control","super"],
            "shortcut_display_labels":{"primary":"Primary","shift":"Shift","alt":"Alt","control":"Ctrl","super":"Super","comma":","},
            "shortcut_platform":{
                "name":shortcut_platform.name(),
                "primary_modifier":shortcut_platform.primary(),
                "supported_modifiers":supported_modifiers,
                "unsupported_modifiers":if shortcut_platform.supports_super() { json!([]) } else { json!(["super"]) },
                "unsupported_policy":"registration and palette replacement return UNSUPPORTED with platform diagnostics",
            },
            "predicate_slots":["visible","enabled","checked"],
            "predicate_types":["always","capability","state","all","any","not"],
            "predicate_state_paths":COMMAND_STATE_PATHS,
            "predicate_state_operators":["truthy","falsy","equals","not_equals","greater_than","at_least"],
            "evaluated_state_fields":["visible","enabled","checkable","checked","reasons","missing_capabilities"],
        },
        "execution":{
            "method":"ui.commands.execute",
            "request_fields":["command_id","checked"],
            "handler_types":["native","control","event"],
            "event":"ui.commands.executed",
            "native_authority":"ui.shell.application_control",
            "control_authority":"target method capability",
        },
        "menu_node_types":["menu_bar","menu","command","separator"],
        "toolbar":{"type":"toolbar","group_fields":["id","title","items"],"item_fields":["id","command_id","label","icon","tooltip","show_label"]},
        "palette":{"type":"command_palette","fields":["id","title","placeholder","shortcut","show_descriptions","max_results"]},
        "protected_policy":"protected command presentations must remain reachable from the platform menu",
        "methods":{
            "commands":"ui.commands.list",
            "register_command":"ui.commands.register",
            "remove_command":"ui.commands.remove",
            "execute_command":"ui.commands.execute",
            "get_menu":"ui.menus.get",
            "replace_menu":"ui.menus.replace",
            "get_toolbar":"ui.toolbars.get",
            "replace_toolbar":"ui.toolbars.replace",
            "get_palette":"ui.palette.get",
            "replace_palette":"ui.palette.replace",
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evaluation_context(mode: &str) -> CommandEvaluationContext {
        CommandEvaluationContext::native(
            mode,
            json!({
                "mode":mode,
                "resources":{
                    "project":true,"dataset":true,"mosaic":true,"objects":true,
                    "labels":true,"masks":true,"gpu":true,
                },
                "selection":{"objects":{"count":2},"mosaic_items":{"count":1}},
                "presentation":{
                    "scale_bar":{"checked":true},
                    "left_panel":{"visible":true},
                    "right_panel":{"visible":true},
                },
            }),
        )
    }

    fn session_evaluation_context(
        mode: &str,
        capabilities: &[&str],
        selection_count: usize,
    ) -> CommandEvaluationContext {
        CommandEvaluationContext::session(
            mode,
            capabilities.iter().map(|value| value.to_string()),
            json!({
                "mode":mode,
                "resources":{
                    "project":true,"dataset":true,"mosaic":false,"objects":true,
                    "labels":false,"masks":false,"gpu":true,
                },
                "selection":{
                    "objects":{"count":selection_count},
                    "mosaic_items":{"count":0},
                },
                "presentation":{
                    "scale_bar":{"checked":false},
                    "left_panel":{"visible":true},
                    "right_panel":{"visible":false},
                },
            }),
        )
    }

    #[test]
    fn built_in_commands_and_menu_are_valid_and_separate() {
        let model = CommandSurfaceModel::default();
        let snapshot = model.projection();
        assert_eq!(snapshot["schema_version"], 1);
        assert!(snapshot["commands"].as_array().unwrap().len() >= 19);
        assert_eq!(snapshot["menu"]["type"], "menu_bar");
        assert!(
            snapshot["commands"]
                .as_array()
                .unwrap()
                .iter()
                .any(|command| {
                    command["id"] == "app.shell.recover" && command["protected"] == true
                })
        );
        assert!(
            snapshot["commands"]
                .as_array()
                .unwrap()
                .iter()
                .any(|command| {
                    command["id"] == "app.shell.reset" && command["protected"] == true
                })
        );

        let CommandInvocation::Native {
            command_id,
            action,
            checked,
        } = model
            .invocation(
                &json!({"command_id":"app.settings.open","checked":true}),
                &evaluation_context("project"),
            )
            .unwrap()
        else {
            panic!("settings should resolve to a native invocation")
        };
        assert_eq!(command_id, "app.settings.open");
        assert_eq!(action, "settings");
        assert_eq!(checked, Some(true));

        let CommandInvocation::Control {
            command_id,
            method,
            params,
        } = model
            .invocation(
                &json!({"command_id":"app.shell.recover"}),
                &evaluation_context("project"),
            )
            .unwrap()
        else {
            panic!("recovery should resolve to a control invocation")
        };
        assert_eq!(command_id, "app.shell.recover");
        assert_eq!(method, "ui.shell.recover");
        assert_eq!(params, json!({}));

        let CommandInvocation::Control {
            command_id,
            method,
            params,
        } = model
            .invocation(
                &json!({"command_id":"app.shell.reset"}),
                &evaluation_context("single"),
            )
            .unwrap()
        else {
            panic!("layout reset should resolve to a control invocation")
        };
        assert_eq!(command_id, "app.shell.reset");
        assert_eq!(method, "ui.shell.reset");
        assert_eq!(params, json!({}));

        let unavailable = model
            .invocation(
                &json!({"command_id":"viewer.screenshot.save"}),
                &evaluation_context("project"),
            )
            .unwrap_err();
        assert_eq!(unavailable.kind, ControlErrorKind::NotReady);
        assert_eq!(unavailable.data.unwrap()["active_mode"], "project");
    }

    #[test]
    fn built_in_command_availability_matches_native_realizer_modes() {
        let model = CommandSurfaceModel::default();
        let expected = BTreeMap::from([
            ("app.about", vec!["project", "single", "mosaic"]),
            ("app.settings.open", vec!["project", "single", "mosaic"]),
            ("dataset.open.ome_zarr", vec!["project", "single", "mosaic"]),
            ("dataset.open.tiff", vec!["project", "single", "mosaic"]),
            ("project.open", vec!["project", "single", "mosaic"]),
            ("project.save", vec!["project", "single", "mosaic"]),
            ("project.save_as", vec!["project", "single", "mosaic"]),
            ("viewer.screenshot.save", vec!["single", "mosaic"]),
            ("viewer.screenshot.quick", vec!["single", "mosaic"]),
            ("viewer.screenshot.settings", vec!["single", "mosaic"]),
            ("viewer.roi_info.show", vec!["single"]),
            ("viewer.annotations.add", vec!["single"]),
            ("viewer.segmentation.load_geojson", vec!["single"]),
            ("viewer.segmentation.load_objects", vec!["single"]),
            ("viewer.masks.export_geojson", vec!["single"]),
            ("viewer.scale_bar.toggle", vec!["single"]),
            ("app.window.close", vec!["project", "single", "mosaic"]),
            ("app.lifecycle.quit", vec!["project", "single", "mosaic"]),
        ]);
        let native_commands = model
            .commands
            .iter()
            .filter_map(|(id, command)| {
                (command.pointer("/handler/type").and_then(Value::as_str) == Some("native"))
                    .then_some((id.as_str(), command))
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            native_commands.keys().copied().collect::<BTreeSet<_>>(),
            expected.keys().copied().collect::<BTreeSet<_>>(),
            "every built-in native action must receive an explicit mode audit"
        );

        for (command_id, expected_modes) in expected {
            let descriptor = native_commands[command_id];
            assert_eq!(
                descriptor["availability"]["modes"],
                json!(expected_modes),
                "descriptor modes must match the native realizer for {command_id}"
            );
            for mode in ["project", "single", "mosaic"] {
                let result =
                    model.invocation(&json!({"command_id":command_id}), &evaluation_context(mode));
                if expected_modes.contains(&mode) {
                    assert!(
                        matches!(result, Ok(CommandInvocation::Native { .. })),
                        "{command_id} should be actor-dispatchable in {mode} mode: {result:?}"
                    );
                } else {
                    let error = result.expect_err("unsupported realizer mode must be rejected");
                    assert_eq!(error.kind, ControlErrorKind::NotReady);
                    assert_eq!(error.data.unwrap()["state"]["enabled"], false);
                }
            }
        }
    }

    #[test]
    fn shortcut_aliases_conflict_by_platform_and_unsupported_modifiers_are_diagnostic() {
        let primary = json!({"key":"k","modifiers":["primary"]});
        let control = json!({"key":"k","modifiers":["control"]});
        let super_key = json!({"key":"k","modifiers":["super"]});
        assert!(shortcuts_equivalent(
            &primary,
            &super_key,
            ShortcutPlatform::MacOs
        ));
        assert!(!shortcuts_equivalent(
            &primary,
            &control,
            ShortcutPlatform::MacOs
        ));
        assert!(shortcuts_equivalent(
            &primary,
            &control,
            ShortcutPlatform::Windows
        ));
        assert!(shortcuts_equivalent(
            &primary,
            &control,
            ShortcutPlatform::Linux
        ));

        let unsupported = validate_shortcut_for_platform(
            &super_key,
            ShortcutPlatform::Linux,
            "ui.commands.register",
        )
        .unwrap_err();
        assert_eq!(unsupported.kind, ControlErrorKind::Unsupported);
        let data = unsupported.data.unwrap();
        assert_eq!(data["platform"], "linux");
        assert_eq!(data["unsupported_modifier"], "super");
        assert_eq!(
            data["resolution"],
            "replace_super_with_primary_or_choose_a_supported_modifier"
        );

        let duplicate_alias = json!({"key":"k","modifiers":["primary","control"]});
        let duplicate = validate_shortcut_for_platform(
            &duplicate_alias,
            ShortcutPlatform::Windows,
            "ui.commands.register",
        )
        .unwrap_err();
        assert_eq!(duplicate.kind, ControlErrorKind::InvalidParams);
        assert_eq!(duplicate.data.unwrap()["effective_modifier"], "control");

        let schema = command_surface_schema();
        assert_eq!(
            schema["command"]["shortcut_platform"]["name"],
            ShortcutPlatform::current().name()
        );
        assert_eq!(
            schema["command"]["shortcut_display_labels"]["primary"],
            "Primary"
        );
    }

    #[test]
    fn menu_replacement_is_guarded_bounded_and_preserves_protected_commands() {
        let mut model = CommandSurfaceModel::default();
        let initial = model.menu_snapshot();
        let mut menu = initial["menu"].clone();
        menu["children"].as_array_mut().unwrap().swap(1, 2);
        let changed = model
            .replace_menu(&json!({
                "if_command_revision":initial["revision"],
                "transaction_id":"menu-test",
                "menu":menu,
            }))
            .unwrap();
        assert_eq!(changed["revision"], 2);
        assert_eq!(changed["change"]["transaction_id"], "menu-test");

        let conflict = model
            .replace_menu(&json!({
                "if_command_revision":1,
                "menu":changed["menu"],
            }))
            .unwrap_err();
        assert_eq!(conflict.kind, ControlErrorKind::Conflict);
        assert_eq!(
            conflict.data.unwrap()["retry_strategy"],
            "refetch_merge_retry"
        );

        let mut unsafe_menu = changed["menu"].clone();
        for top_level in unsafe_menu["children"].as_array_mut().unwrap() {
            if top_level["id"] == "menu:application" {
                top_level["children"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|item| item["command_id"] != "app.lifecycle.quit");
            }
        }
        let error = model
            .replace_menu(&json!({"menu":unsafe_menu}))
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::InvalidParams);
        assert!(
            error
                .message
                .contains("protected command 'app.lifecycle.quit'")
        );
    }

    #[test]
    fn contextual_predicates_are_bounded_evaluated_and_enforced() {
        let mut model = CommandSurfaceModel::default();
        let extension = extension_context(DisconnectPolicy::Retain, true);
        let registered = model
            .register_extension_command(
                &json!({"command":{
                    "id":"measure-selection",
                    "title":"Measure Selection",
                    "description":"Measure the current object selection.",
                    "event":"measure_selection",
                    "modes":["single"],
                    "predicates":{
                        "visible":{
                            "type":"capability",
                            "capability":"viewer.read",
                            "reason":"Viewer access is required."
                        },
                        "enabled":{
                            "type":"all",
                            "predicates":[
                                {"type":"state","path":"resources.objects","operator":"truthy"},
                                {
                                    "type":"state",
                                    "path":"selection.objects.count",
                                    "operator":"greater_than",
                                    "value":0,
                                    "reason":"Select at least one object."
                                }
                            ]
                        },
                        "checked":{
                            "type":"state",
                            "path":"presentation.left_panel.visible",
                            "operator":"truthy"
                        }
                    }
                }}),
                &extension,
            )
            .unwrap();
        let command_id = registered["command"]["id"].as_str().unwrap();

        let no_capability = session_evaluation_context("single", &[], 2);
        let commands = model.evaluated_commands_snapshot(&no_capability);
        let command = commands["commands"]
            .as_array()
            .unwrap()
            .iter()
            .find(|command| command["id"] == command_id)
            .unwrap();
        assert_eq!(command["state"]["visible"], false);
        assert_eq!(command["state"]["enabled"], false);
        assert_eq!(command["state"]["checked"], true);
        assert_eq!(
            command["state"]["missing_capabilities"],
            json!(["viewer.read"])
        );
        assert_eq!(
            model
                .invocation(&json!({"command_id":command_id}), &no_capability)
                .unwrap_err()
                .kind,
            ControlErrorKind::PermissionDenied
        );

        let empty_selection = session_evaluation_context("single", &["viewer.read"], 0);
        let unavailable = model
            .invocation(&json!({"command_id":command_id}), &empty_selection)
            .unwrap_err();
        assert_eq!(unavailable.kind, ControlErrorKind::NotReady);
        assert_eq!(
            unavailable.data.unwrap()["state"]["reasons"],
            json!(["Select at least one object."])
        );

        let ready = session_evaluation_context("single", &["viewer.read"], 3);
        assert!(matches!(
            model
                .invocation(&json!({"command_id":command_id}), &ready)
                .unwrap(),
            CommandInvocation::ExtensionEvent(_)
        ));

        let invalid = model
            .register_extension_command(
                &json!({"command":{
                    "id":"invalid-predicate",
                    "title":"Invalid",
                    "description":"Uses private actor state.",
                    "event":"invalid",
                    "modes":["single"],
                    "predicates":{"enabled":{
                        "type":"state",
                        "path":"private.arbitrary.path",
                        "operator":"truthy"
                    }}
                }}),
                &extension,
            )
            .unwrap_err();
        assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);

        let oversized = json!({
            "type":"all",
            "predicates":(0..MAX_PREDICATE_NODES)
                .map(|_| json!({"type":"always","value":true}))
                .collect::<Vec<_>>(),
        });
        let quota = model
            .register_extension_command(
                &json!({"command":{
                    "id":"oversized-predicate",
                    "title":"Oversized",
                    "description":"Exceeds the predicate quota.",
                    "event":"oversized",
                    "modes":["single"],
                    "predicates":{"enabled":oversized}
                }}),
                &extension,
            )
            .unwrap_err();
        assert_eq!(quota.kind, ControlErrorKind::ResourceLimit);

        let mut too_deep = json!({"type":"always","value":true});
        for _ in 0..MAX_PREDICATE_DEPTH {
            too_deep = json!({"type":"not","predicate":too_deep});
        }
        let depth = model
            .register_extension_command(
                &json!({"command":{
                    "id":"deep-predicate",
                    "title":"Deep",
                    "description":"Exceeds the predicate depth limit.",
                    "event":"deep",
                    "modes":["single"],
                    "predicates":{"enabled":too_deep}
                }}),
                &extension,
            )
            .unwrap_err();
        assert_eq!(depth.kind, ControlErrorKind::ResourceLimit);
    }

    fn extension_context(policy: DisconnectPolicy, ready: bool) -> ExtensionCommandContext {
        ExtensionCommandContext {
            extension_id: "org.example.commands".to_string(),
            extension_version: "1.0.0".to_string(),
            owner_session_id: "extension-session".to_string(),
            disconnect_policy: policy,
            ready,
        }
    }

    #[test]
    fn extension_commands_are_namespaced_conflict_checked_and_lifecycle_owned() {
        let mut model = CommandSurfaceModel::default();
        let context = extension_context(DisconnectPolicy::Retain, true);
        let registered = model
            .register_extension_command(
                &json!({
                    "if_command_revision":1,
                    "transaction_id":"register-measure",
                    "command":{
                        "id":"measure.cells",
                        "title":"Measure Cells",
                        "description":"Measure selected cells.",
                        "event":"measure",
                        "modes":["single","mosaic"],
                        "shortcut":{"key":"m","modifiers":["primary","shift"]},
                    }
                }),
                &context,
            )
            .unwrap();
        let command_id = "extension:org.example.commands/measure.cells";
        assert_eq!(registered["command"]["id"], command_id);
        assert_eq!(registered["command"]["readiness"]["state"], "ready");
        assert_eq!(registered["change"]["transaction_id"], "register-measure");

        let collision = model
            .register_extension_command(
                &json!({
                    "command":{
                        "id":"save-collision",
                        "title":"Collision",
                        "description":"Conflicts with Save Project.",
                        "event":"collision",
                        "modes":["single"],
                        "shortcut":{"key":"s","modifiers":["primary"]},
                    }
                }),
                &context,
            )
            .unwrap_err();
        assert_eq!(collision.kind, ControlErrorKind::Conflict);
        assert_eq!(
            collision.data.unwrap()["conflicting_command_id"],
            "project.save"
        );

        let mut menu = model.menu_snapshot()["menu"].clone();
        menu["children"][4]["children"]
            .as_array_mut()
            .unwrap()
            .push(item("menu-item:measure-cells", command_id));
        model.replace_menu(&json!({"menu":menu})).unwrap();
        model
            .replace_toolbar(&json!({
                "toolbar":{
                    "id":"toolbar:analysis",
                    "type":"toolbar",
                    "groups":[{
                        "id":"toolbar-group:measure",
                        "title":"Analysis",
                        "items":[{
                            "id":"toolbar-item:measure-cells",
                            "command_id":command_id,
                            "label":"Measure"
                        }]
                    }]
                }
            }))
            .unwrap();

        let disconnected = model.cleanup_extensions(&[UiExtensionCleanup {
            extension_id: context.extension_id.clone(),
            disconnect_policy: DisconnectPolicy::Retain,
        }]);
        assert_eq!(disconnected["changed"], true);
        assert_eq!(
            model.commands_snapshot()["commands"]
                .as_array()
                .unwrap()
                .iter()
                .find(|command| command["id"] == command_id)
                .unwrap()["readiness"]["state"],
            "disconnected"
        );
        assert_eq!(
            model
                .invocation(
                    &json!({"command_id":command_id}),
                    &evaluation_context("single"),
                )
                .unwrap_err()
                .kind,
            ControlErrorKind::NotReady
        );

        model
            .register_extension_command(
                &json!({
                    "command":{
                        "id":"measure.cells",
                        "title":"Measure Cells",
                        "description":"Measure selected cells.",
                        "event":"measure",
                        "modes":["single","mosaic"],
                        "shortcut":{"key":"m","modifiers":["primary","shift"]},
                    }
                }),
                &context,
            )
            .unwrap();
        let CommandInvocation::ExtensionEvent(invocation) = model
            .invocation(
                &json!({"command_id":command_id}),
                &evaluation_context("single"),
            )
            .unwrap()
        else {
            panic!("extension command should resolve to an event invocation")
        };
        assert_eq!(invocation.event, "measure");

        let removed = model.cleanup_extensions(&[UiExtensionCleanup {
            extension_id: context.extension_id,
            disconnect_policy: DisconnectPolicy::Remove,
        }]);
        assert_eq!(removed["removed_command_ids"], json!([command_id]));
        assert!(
            !model.menu_snapshot()["menu"]
                .to_string()
                .contains(command_id)
        );
        assert!(
            !model.toolbar_snapshot()["toolbar"]
                .to_string()
                .contains(command_id)
        );
    }

    #[test]
    fn toolbar_replacement_is_revisioned_and_reference_checked() {
        let mut model = CommandSurfaceModel::default();
        let toolbar = json!({
            "id":"toolbar:main",
            "type":"toolbar",
            "groups":[{
                "id":"toolbar-group:file",
                "items":[
                    {"id":"toolbar-item:open","command_id":"dataset.open.ome_zarr"},
                    {"id":"toolbar-item:save","command_id":"project.save","label":"Save"}
                ]
            }]
        });
        let changed = model
            .replace_toolbar(&json!({
                "toolbar":toolbar,
                "if_command_revision":1,
                "transaction_id":"toolbar-test"
            }))
            .unwrap();
        assert_eq!(changed["revision"], 2);
        assert_eq!(changed["change"]["transaction_id"], "toolbar-test");

        let conflict = model
            .replace_toolbar(&json!({"toolbar":toolbar,"if_command_revision":1}))
            .unwrap_err();
        assert_eq!(conflict.kind, ControlErrorKind::Conflict);
        assert_eq!(conflict.data.unwrap()["snapshot_method"], "ui.toolbars.get");

        let unknown = model
            .replace_toolbar(&json!({"toolbar":{
                "id":"toolbar:bad","type":"toolbar","groups":[{
                    "id":"toolbar-group:bad","items":[{
                        "id":"toolbar-item:bad","command_id":"missing.command"
                    }]
                }]
            }}))
            .unwrap_err();
        assert_eq!(unknown.kind, ControlErrorKind::InvalidParams);
    }

    #[test]
    fn maximum_toolbar_and_predicates_reconcile_under_changing_actor_state() {
        let mut model = CommandSurfaceModel::default();
        let boundary_items = (0..MAX_TOOLBAR_ITEMS)
            .map(|index| {
                json!({
                    "id":format!("toolbar-item:stress-{index}"),
                    "command_id":"project.save",
                })
            })
            .collect::<Vec<_>>();
        let boundary_toolbar = json!({
            "id":"toolbar:stress",
            "type":"toolbar",
            "groups":[{
                "id":"toolbar-group:stress",
                "items":boundary_items,
            }],
        });
        model
            .replace_toolbar(&json!({"toolbar":boundary_toolbar}))
            .unwrap();
        assert_eq!(
            model.toolbar_snapshot()["toolbar"]["groups"][0]["items"]
                .as_array()
                .unwrap()
                .len(),
            MAX_TOOLBAR_ITEMS
        );

        const TOOLBAR_SAMPLES: u32 = 128;
        let toolbar_started = std::time::Instant::now();
        for _ in 0..TOOLBAR_SAMPLES {
            validate_toolbar(&boundary_toolbar, &model.commands).unwrap();
        }
        let toolbar_elapsed = toolbar_started.elapsed();
        println!(
            "maximum toolbar validation: items={MAX_TOOLBAR_ITEMS} samples={TOOLBAR_SAMPLES} total_us={} average_us={}",
            toolbar_elapsed.as_micros(),
            toolbar_elapsed.as_micros() / u128::from(TOOLBAR_SAMPLES),
        );
        assert!(
            toolbar_elapsed < std::time::Duration::from_secs(5),
            "{TOOLBAR_SAMPLES} maximum-toolbar validations took {toolbar_elapsed:?}"
        );

        let mut oversized_toolbar = boundary_toolbar;
        oversized_toolbar["groups"][0]["items"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "id":"toolbar-item:too-many",
                "command_id":"project.save",
            }));
        let error = model
            .replace_toolbar(&json!({"toolbar":oversized_toolbar}))
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::ResourceLimit);

        let predicates = (0..(MAX_PREDICATE_NODES - 1))
            .map(|_| {
                json!({
                    "type":"state",
                    "path":"selection.objects.count",
                    "operator":"greater_than",
                    "value":0,
                    "reason":"Select at least one object.",
                })
            })
            .collect::<Vec<_>>();
        let registered = model
            .register_extension_command(
                &json!({"command":{
                    "id":"stress-selection",
                    "title":"Stress Selection",
                    "description":"Exercises the accepted predicate boundary.",
                    "event":"stress_selection",
                    "modes":["single"],
                    "predicates":{"enabled":{"type":"all","predicates":predicates}},
                }}),
                &extension_context(DisconnectPolicy::Retain, true),
            )
            .unwrap();
        let command_id = registered["command"]["id"].as_str().unwrap();

        const PROJECTION_SAMPLES: u32 = 512;
        let projection_started = std::time::Instant::now();
        for selection_count in 0..PROJECTION_SAMPLES {
            let context = session_evaluation_context(
                "single",
                &["viewer.read"],
                (selection_count % 4) as usize,
            );
            let snapshot = model.evaluated_commands_snapshot(&context);
            let command = snapshot["commands"]
                .as_array()
                .unwrap()
                .iter()
                .find(|command| command["id"] == command_id)
                .unwrap();
            assert_eq!(command["state"]["enabled"], json!(selection_count % 4 > 0));
        }
        let projection_elapsed = projection_started.elapsed();
        println!(
            "maximum predicate reconciliation: nodes={MAX_PREDICATE_NODES} samples={PROJECTION_SAMPLES} total_us={} average_us={}",
            projection_elapsed.as_micros(),
            projection_elapsed.as_micros() / u128::from(PROJECTION_SAMPLES),
        );
        assert!(
            projection_elapsed < std::time::Duration::from_secs(5),
            "{PROJECTION_SAMPLES} maximum-predicate projections took {projection_elapsed:?}"
        );

        let mut depth_boundary = json!({
            "type":"state",
            "path":"resources.objects",
            "operator":"truthy",
        });
        for _ in 1..MAX_PREDICATE_DEPTH {
            depth_boundary = json!({"type":"not","predicate":depth_boundary});
        }
        model
            .register_extension_command(
                &json!({"command":{
                    "id":"stress-depth",
                    "title":"Stress Depth",
                    "description":"Exercises the accepted predicate depth boundary.",
                    "event":"stress_depth",
                    "modes":["single"],
                    "predicates":{"enabled":depth_boundary},
                }}),
                &extension_context(DisconnectPolicy::Retain, true),
            )
            .unwrap();
    }

    #[test]
    fn palette_replacement_is_revisioned_bounded_and_shortcut_checked() {
        let mut model = CommandSurfaceModel::default();
        let palette = json!({
            "id":"palette:review",
            "type":"command_palette",
            "title":"Review commands",
            "placeholder":"Find a review command…",
            "shortcut":{"key":"k","modifiers":["primary"]},
            "show_descriptions":false,
            "max_results":12,
        });
        let changed = model
            .replace_palette(&json!({
                "palette":palette,
                "if_command_revision":1,
                "transaction_id":"palette-test",
            }))
            .unwrap();
        assert_eq!(changed["revision"], 2);
        assert_eq!(changed["palette"]["title"], "Review commands");
        assert_eq!(changed["change"]["transaction_id"], "palette-test");

        let stale = model
            .replace_palette(&json!({
                "palette":changed["palette"],
                "if_command_revision":1,
            }))
            .unwrap_err();
        assert_eq!(stale.kind, ControlErrorKind::Conflict);
        assert_eq!(stale.data.unwrap()["snapshot_method"], "ui.palette.get");

        let invalid = model
            .replace_palette(&json!({"palette":{
                "id":"palette:invalid",
                "type":"command_palette",
                "title":"Commands",
                "placeholder":"Search",
                "shortcut":{"key":"j","modifiers":["primary"]},
                "show_descriptions":true,
                "max_results":0,
            }}))
            .unwrap_err();
        assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);

        let command_collision = model
            .replace_palette(&json!({"palette":{
                "id":"palette:collision",
                "type":"command_palette",
                "title":"Commands",
                "placeholder":"Search",
                "shortcut":{"key":"s","modifiers":["primary"]},
                "show_descriptions":true,
                "max_results":20,
            }}))
            .unwrap_err();
        assert_eq!(command_collision.kind, ControlErrorKind::Conflict);
        assert_eq!(
            command_collision.data.unwrap()["conflicting_command_id"],
            "project.save"
        );

        let context = extension_context(DisconnectPolicy::Remove, true);
        let palette_collision = model
            .register_extension_command(
                &json!({"command":{
                    "id":"open-palette",
                    "title":"Open Palette",
                    "description":"Conflicts with the palette shortcut.",
                    "event":"open_palette",
                    "modes":["single"],
                    "shortcut":{"key":"k","modifiers":["primary"]},
                }}),
                &context,
            )
            .unwrap_err();
        assert_eq!(palette_collision.kind, ControlErrorKind::Conflict);
        assert_eq!(
            palette_collision.data.unwrap()["conflicting_presentation_id"],
            "palette:review"
        );
    }
}
