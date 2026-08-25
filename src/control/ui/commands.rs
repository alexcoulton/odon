//! Session- and extension-owned command authority at the declarative UI boundary.

use super::*;

impl UiRegistry {
    pub fn validate_session_capability(
        &self,
        session_id: &str,
        capability: &str,
        method: &str,
    ) -> Result<(), ControlError> {
        if session_id == "native-ui" {
            return Ok(());
        }
        let granted = self.session_capabilities(session_id);
        if granted
            .iter()
            .any(|candidate| candidate == capability || candidate == "ui.shell.application_control")
        {
            return Ok(());
        }
        Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            format!("{method} requires the '{capability}' session capability"),
        )
        .with_data(json!({
            "method":method,
            "required_capability":capability,
            "granted_capabilities":granted,
            "resolution":"request the capability during system.hello",
        })))
    }

    pub fn extension_command_context(
        &self,
        extension_id: &str,
        session_id: &str,
    ) -> Result<ExtensionCommandContext, ControlError> {
        let state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        if !extension
            .granted_capabilities
            .iter()
            .any(|capability| capability == "ui.actions")
        {
            return Err(ControlError::new(
                ControlErrorKind::PermissionDenied,
                format!("extension '{extension_id}' did not declare the 'ui.actions' capability"),
            )
            .with_data(json!({
                "extension_id":extension_id,
                "required_capability":"ui.actions",
                "resolution":"include ui.actions when registering the extension",
            })));
        }
        Ok(ExtensionCommandContext {
            extension_id: extension.id.clone(),
            extension_version: extension.version.clone(),
            owner_session_id: extension.owner_session_id.clone(),
            disconnect_policy: extension.disconnect_policy.clone(),
            ready: extension.ready,
        })
    }

    pub fn extension_cleanup_context(
        &self,
        extension_id: &str,
        session_id: &str,
    ) -> Result<UiExtensionCleanup, ControlError> {
        let state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        Ok(UiExtensionCleanup {
            extension_id: extension.id.clone(),
            disconnect_policy: DisconnectPolicy::Remove,
        })
    }
}
