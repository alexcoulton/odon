//! Command-surface request validation kept separate from the general request dispatcher.

use super::*;

pub(super) fn menu_replace(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: MenuReplaceRequest = serde_json::from_value(params.clone()).map_err(|error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    })?;
    revision_and_transaction(
        method,
        request.if_command_revision,
        request.transaction_id.as_deref(),
    )?;
    if !request.menu.is_object() {
        return Err(ControlError::invalid_params(
            method,
            "menu must be an object",
        ));
    }
    Ok(())
}

pub(super) fn toolbar_replace(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: ToolbarReplaceRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    revision_and_transaction(
        method,
        request.if_command_revision,
        request.transaction_id.as_deref(),
    )?;
    if !request.toolbar.is_object() {
        return Err(ControlError::invalid_params(
            method,
            "toolbar must be an object",
        ));
    }
    Ok(())
}

pub(super) fn palette_replace(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: PaletteReplaceRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    revision_and_transaction(
        method,
        request.if_command_revision,
        request.transaction_id.as_deref(),
    )?;
    if !request.palette.is_object() {
        return Err(ControlError::invalid_params(
            method,
            "palette must be an object",
        ));
    }
    Ok(())
}

fn revision_and_transaction(
    method: &str,
    revision: Option<u64>,
    transaction_id: Option<&str>,
) -> Result<(), ControlError> {
    if revision == Some(0) {
        return Err(ControlError::invalid_params(
            method,
            "if_command_revision must be at least 1",
        ));
    }
    validate_shell_transaction_id(method, transaction_id)
}

fn non_empty(method: &str, name: &str, value: &str) -> Result<(), ControlError> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(ControlError::invalid_params(
            method,
            format!("{name} must contain 1 to 256 non-control bytes"),
        ));
    }
    Ok(())
}

pub(super) fn register(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: CommandRegisterRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    non_empty(method, "extension_id", &request.extension_id)?;
    validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
    if request.if_command_revision == Some(0) {
        return Err(ControlError::invalid_params(
            method,
            "if_command_revision must be at least 1",
        ));
    }
    if !request.command.is_object() {
        return Err(ControlError::invalid_params(
            method,
            "command must be an object",
        ));
    }
    Ok(())
}

pub(super) fn remove(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: CommandRemoveRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    non_empty(method, "extension_id", &request.extension_id)?;
    non_empty(method, "command_id", &request.command_id)?;
    validate_shell_transaction_id(method, request.transaction_id.as_deref())?;
    if request.if_command_revision == Some(0) {
        return Err(ControlError::invalid_params(
            method,
            "if_command_revision must be at least 1",
        ));
    }
    Ok(())
}

pub(super) fn execute(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: CommandExecuteRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    let _checked = request.checked;
    non_empty(method, "command_id", &request.command_id)
}

pub(super) fn cleanup(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: CommandCleanupRequest =
        serde_json::from_value(params.clone()).map_err(|error| {
            ControlError::invalid_params(method, format!("invalid parameters: {error}"))
        })?;
    if request
        .extensions
        .iter()
        .any(|extension| !extension.is_object())
    {
        return Err(ControlError::invalid_params(
            method,
            "extensions must contain objects",
        ));
    }
    Ok(())
}

pub(super) fn sync(method: &str, params: &Value) -> Result<(), ControlError> {
    let request: CommandSyncRequest = serde_json::from_value(params.clone()).map_err(|error| {
        ControlError::invalid_params(method, format!("invalid parameters: {error}"))
    })?;
    if !request.context.is_object() {
        return Err(ControlError::invalid_params(
            method,
            "context must be an object",
        ));
    }
    Ok(())
}
