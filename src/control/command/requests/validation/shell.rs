//! Validation helpers shared by the shell request shapes.

use crate::control::ControlError;

pub(super) fn validate_shell_mode(method: &str, mode: Option<&str>) -> Result<(), ControlError> {
    if mode.is_some_and(|mode| !matches!(mode, "project" | "single" | "mosaic")) {
        return Err(ControlError::invalid_params(
            method,
            "mode must be project, single, or mosaic",
        ));
    }
    Ok(())
}

pub(super) fn validate_shell_profile_scope(
    method: &str,
    scope: Option<&str>,
) -> Result<(), ControlError> {
    if scope.is_some_and(|scope| !matches!(scope, "session" | "application" | "project")) {
        return Err(ControlError::invalid_params(
            method,
            "scope must be session, application, or project",
        ));
    }
    Ok(())
}

pub(super) fn validate_shell_profile_name(method: &str, name: &str) -> Result<(), ControlError> {
    if name.trim().is_empty()
        || name.len() > 128
        || name.chars().any(|character| character.is_control())
    {
        return Err(ControlError::invalid_params(
            method,
            "name must contain 1 to 128 characters without control characters",
        ));
    }
    Ok(())
}

pub(super) fn validate_shell_id(method: &str, field: &str, id: &str) -> Result<(), ControlError> {
    if id.trim().is_empty() || id.len() > 256 {
        return Err(ControlError::invalid_params(
            method,
            format!("{field} IDs must contain 1 to 256 bytes"),
        ));
    }
    Ok(())
}

pub(super) fn validate_shell_transaction_id(
    method: &str,
    transaction_id: Option<&str>,
) -> Result<(), ControlError> {
    if transaction_id.is_some_and(|value| {
        value.is_empty() || value.len() > 128 || value.chars().any(char::is_control)
    }) {
        return Err(ControlError::invalid_params(
            method,
            "transaction_id must contain 1 to 128 non-control bytes",
        ));
    }
    Ok(())
}
