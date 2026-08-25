//! Extension-owned default shell-layout templates.

use serde_json::Value;

use super::*;

impl UiRegistry {
    pub fn register_extension_layout(
        &self,
        params: Value,
        session_id: &str,
    ) -> Result<ExtensionLayoutSnapshot, ControlError> {
        let request: RegisterExtensionLayout = serde_json::from_value(params).map_err(|error| {
            ControlError::invalid_params(
                "ui.extensions.layouts.register",
                format!("invalid extension layout: {error}"),
            )
        })?;
        if request.extension_id.trim().is_empty() || request.extension_id.len() > 256 {
            return Err(ControlError::invalid_params(
                "ui.extensions.layouts.register",
                "extension_id must contain 1 to 256 characters",
            ));
        }
        validate_layout_name(&request.name, "ui.extensions.layouts.register")?;
        let document = crate::model::normalize_shell_layout_document(&request.document)?;
        let desired_tree = document
            .get("layout")
            .expect("normalized layout document contains a desired tree");
        self.validate_extension_shell_layout_access(
            desired_tree,
            session_id,
            &request.extension_id,
        )?;

        let mut state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(&request.extension_id)
            .ok_or_else(|| not_found("extension", &request.extension_id))?;
        ensure_owner(extension, session_id)?;
        ensure_layout_capability(extension)?;
        let existing = state.extension_layouts.iter().position(|layout| {
            layout.extension_id == request.extension_id && layout.name == request.name
        });
        if existing.is_none()
            && state
                .extension_layouts
                .iter()
                .filter(|layout| layout.extension_id == request.extension_id)
                .count()
                >= 64
        {
            return Err(ControlError::new(
                ControlErrorKind::ResourceLimit,
                "extension layout template limit of 64 has been reached",
            ));
        }
        let revision = self.events.next_revision();
        let snapshot = ExtensionLayoutSnapshot {
            extension_id: request.extension_id.clone(),
            extension_version: extension.version.clone(),
            name: request.name,
            document,
            ownership: extension_ownership(&request.extension_id, session_id),
            readiness: extension_content_readiness(&extension.version, extension).to_string(),
            revision,
        };
        if let Some(index) = existing {
            state.extension_layouts[index] = snapshot.clone();
        } else {
            state.extension_layouts.push(snapshot.clone());
        }
        drop(state);
        self.publish(
            "ui.extensions.layouts.changed",
            &request.extension_id,
            revision,
            &snapshot,
            session_id,
        );
        Ok(snapshot)
    }

    pub fn list_extension_layouts(
        &self,
        extension_id: &str,
        session_id: &str,
    ) -> Result<Vec<ExtensionLayoutSnapshot>, ControlError> {
        let state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        ensure_layout_capability(extension)?;
        let mut layouts = state
            .extension_layouts
            .iter()
            .filter(|layout| layout.extension_id == extension_id)
            .cloned()
            .collect::<Vec<_>>();
        layouts.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(layouts)
    }

    pub fn remove_extension_layout(
        &self,
        extension_id: &str,
        name: &str,
        session_id: &str,
    ) -> Result<(), ControlError> {
        validate_layout_name(name, "ui.extensions.layouts.remove")?;
        let mut state = self.state.lock().expect("UI registry poisoned");
        let extension = state
            .extensions
            .get(extension_id)
            .ok_or_else(|| not_found("extension", extension_id))?;
        ensure_owner(extension, session_id)?;
        ensure_layout_capability(extension)?;
        let index = state
            .extension_layouts
            .iter()
            .position(|layout| layout.extension_id == extension_id && layout.name == name)
            .ok_or_else(|| not_found("extension layout", name))?;
        state.extension_layouts.remove(index);
        let revision = self.events.next_revision();
        drop(state);
        self.events.publish(
            "ui.extensions.layouts.changed",
            extension_id,
            revision,
            json!({"extension_id":extension_id,"name":name,"removed":true}),
            Some(session_id.to_string()),
            None,
        );
        Ok(())
    }
}

fn ensure_layout_capability(extension: &ExtensionSnapshot) -> Result<(), ControlError> {
    if extension
        .granted_capabilities
        .iter()
        .any(|capability| capability == "ui.panels")
    {
        Ok(())
    } else {
        Err(ControlError::new(
            ControlErrorKind::PermissionDenied,
            "extension layout templates require the ui.panels capability",
        ))
    }
}

fn validate_layout_name(name: &str, method: &str) -> Result<(), ControlError> {
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
