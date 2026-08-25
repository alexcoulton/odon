//! Application-shell snapshots and mutations over existing actor-owned UI state.

use std::collections::BTreeMap;

use super::*;
use crate::model::shell::{
    selection_patch, shell_component_catalog, shell_schema, visibility_patch,
};

impl AppModel {
    pub(crate) fn shell_snapshot(
        &self,
        requested_mode: Option<&str>,
    ) -> Result<Value, ControlError> {
        let mode = shell_mode(requested_mode, self.mode)?;
        if requested_mode.is_some() && mode != self.mode {
            return self.shell.mode_state(mode.as_str());
        }
        let mut snapshot = match mode {
            ModelMode::Project => self.shell.snapshot(mode, &[], &[]),
            ModelMode::Single => {
                let dataset = self.dataset.as_ref().ok_or_else(|| {
                    ControlError::new(
                        ControlErrorKind::NotReady,
                        "single-view shell state is unavailable until a dataset is open",
                    )
                })?;
                self.shell.snapshot(
                    mode,
                    &[
                        ("builtin:single.left-panel", dataset.show_left_panel),
                        ("builtin:single.right-panel", dataset.show_right_panel),
                    ],
                    &[
                        (
                            "builtin:single.left-tabs",
                            single_left_node(&dataset.left_tab)?,
                        ),
                        (
                            "builtin:single.right-tabs",
                            single_right_node(&dataset.right_tab)?,
                        ),
                    ],
                )
            }
            ModelMode::Mosaic => {
                let state = self.mosaic.projection_state();
                if state.as_object().is_none_or(serde_json::Map::is_empty) {
                    return Err(ControlError::new(
                        ControlErrorKind::NotReady,
                        "mosaic shell state is unavailable until a mosaic is open",
                    ));
                }
                let left = state
                    .pointer("/panels/left")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let right = state
                    .pointer("/panels/right")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let left_tab = state
                    .get("left_tab")
                    .and_then(Value::as_str)
                    .unwrap_or("layers");
                let right_tab = state
                    .get("right_tab")
                    .and_then(Value::as_str)
                    .unwrap_or("properties");
                self.shell.snapshot(
                    mode,
                    &[
                        ("builtin:mosaic.left-panel", left),
                        ("builtin:mosaic.right-panel", right),
                    ],
                    &[
                        ("builtin:mosaic.left-tabs", mosaic_left_node(left_tab)?),
                        ("builtin:mosaic.right-tabs", mosaic_right_node(right_tab)?),
                    ],
                )
            }
            ModelMode::Transition => unreachable!("shell_mode rejects transition"),
        }?;
        apply_shell_command_state_bindings(&mut snapshot, &self.command_surface_projection());
        Ok(snapshot)
    }

    pub(super) fn patch_shell(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.patch")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.patch")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        // Validate a complete candidate before applying any mirrored domain values.
        let mut shell = self.shell.clone();
        shell.patch(params, self.mode)?;
        self.apply_shell_domain_patch(mode, params)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(mode.as_str()))?;
        Ok(with_shell_change(after, &before, "patch", transaction_id))
    }

    pub(super) fn reset_shell(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.reset")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.reset")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        let mut shell = self.shell.clone();
        let mode_name = shell.reset(params, self.mode)?;
        let mode = shell_mode(Some(&mode_name), self.mode)?;
        self.apply_shell_defaults(mode)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(&mode_name))?;
        Ok(with_shell_change(after, &before, "reset", transaction_id))
    }

    pub(super) fn export_shell_layout(
        &self,
        requested_mode: Option<&str>,
    ) -> Result<Value, ControlError> {
        let mode = shell_mode(requested_mode, self.mode)?;
        self.shell.export_layout_document(mode.as_str())
    }

    pub(super) fn import_shell_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.import_layout")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.import_layout")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        let mut shell = self.shell.clone();
        let outcome = shell.import_layout_document(params, self.mode)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(mode.as_str()))?;
        let mut response = with_shell_change(after, &before, "import_layout", transaction_id);
        response
            .as_object_mut()
            .expect("shell snapshots are objects")
            .insert(
                "import".to_string(),
                json!({
                    "mode":outcome.mode,
                    "source_schema_version":outcome.source_schema_version,
                    "schema_version":1,
                    "migrated":outcome.migrated,
                }),
            );
        Ok(response)
    }

    pub(super) fn recover_shell(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.recover")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.recover")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        let mut shell = self.shell.clone();
        let mode_name = shell.recover_layout(params, self.mode)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(&mode_name))?;
        let mut response = with_shell_change(after, &before, "recover", transaction_id);
        response
            .as_object_mut()
            .expect("shell snapshots are objects")
            .insert(
                "recovery".to_string(),
                json!({"protected":true,"mode":mode_name}),
            );
        Ok(response)
    }

    pub(super) fn replace_shell_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.replace_layout")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.replace_layout")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        let mut shell = self.shell.clone();
        shell.replace_layout(params, self.mode)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(mode.as_str()))?;
        Ok(with_shell_change(
            after,
            &before,
            "replace_layout",
            transaction_id,
        ))
    }

    pub(super) fn patch_shell_layout(&mut self, params: &Value) -> Result<Value, ControlError> {
        let transaction_id = shell_transaction_id(params, "ui.shell.patch_layout")?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        require_active_shell(mode, self.mode, "ui.shell.patch_layout")?;
        let before = self.shell_snapshot(Some(mode.as_str()))?;
        let mut shell = self.shell.clone();
        shell.patch_layout(params, self.mode)?;
        self.shell = shell;
        let after = self.shell_snapshot(Some(mode.as_str()))?;
        Ok(with_shell_change(
            after,
            &before,
            "patch_layout",
            transaction_id,
        ))
    }

    pub(super) fn describe_shell_schema(&self) -> Value {
        shell_schema()
    }

    pub(super) fn list_shell_components(
        &self,
        requested_mode: Option<&str>,
    ) -> Result<Value, ControlError> {
        let mode = shell_mode(requested_mode, self.mode)?;
        Ok(shell_component_catalog(mode.as_str()))
    }

    pub(super) fn list_shell_profiles(&self, params: &Value) -> Result<Value, ControlError> {
        let scope = shell_profile_scope(params, "ui.shell.profiles.list")?;
        let profiles = match scope {
            "session" => self
                .session_shell_profiles
                .iter()
                .map(|(name, document)| (name.clone(), document.clone()))
                .collect::<Vec<_>>(),
            "application" => self
                .settings
                .shell_layout_profiles
                .iter()
                .map(|(name, document)| (name.clone(), document.clone()))
                .collect::<Vec<_>>(),
            "project" => self.project.shell_layout_profiles()?,
            _ => unreachable!("validated profile scope"),
        };
        Ok(json!({
            "schema_version":1,
            "scope":scope,
            "profiles":profiles.iter().map(|(name, document)| {
                let startup_modes = if scope == "application" {
                    self.settings.shell_layout_startup_profiles.iter()
                        .filter_map(|(mode, profile)| (profile == name).then_some(mode.as_str()))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                shell_profile_summary(name, document, &startup_modes)
            }).collect::<Vec<_>>(),
        }))
    }

    pub fn prepare_shell_profile_save(
        &mut self,
        params: &Value,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let method = "ui.shell.profiles.save";
        let name = shell_profile_name(params, method)?;
        let scope = shell_profile_scope(params, method)?;
        let mode = shell_mode(params.get("mode").and_then(Value::as_str), self.mode)?;
        let document = self.shell.export_layout_document(mode.as_str())?;
        let response = json!({
            "saved":true,
            "name":name,
            "scope":scope,
            "mode":mode.as_str(),
            "document_schema_version":1,
            "persisted":scope == "application",
            "project_dirty":scope == "project",
        });
        if scope == "session" {
            if self.session_shell_profiles.len() >= 64
                && !self.session_shell_profiles.contains_key(name)
            {
                return Err(ControlError::new(
                    ControlErrorKind::ResourceLimit,
                    "session shell profile limit of 64 has been reached",
                ));
            }
            self.session_shell_profiles
                .insert(name.to_string(), document);
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        if scope == "project" {
            let profiles = self.project.shell_layout_profiles()?;
            if profiles.len() >= 64 && !profiles.iter().any(|(candidate, _)| candidate == name) {
                return Err(ControlError::new(
                    ControlErrorKind::ResourceLimit,
                    "project shell profile limit of 64 has been reached",
                ));
            }
            self.project
                .set_shell_layout_profile(name, Some(document))?;
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        let mut candidate = self.settings.clone();
        if candidate.shell_layout_profiles.len() >= 64
            && !candidate.shell_layout_profiles.contains_key(name)
        {
            return Err(ControlError::new(
                ControlErrorKind::ResourceLimit,
                "application shell profile limit of 64 has been reached",
            ));
        }
        if candidate.shell_layout_profiles.get(name) == Some(&document) {
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        candidate
            .shell_layout_profiles
            .insert(name.to_string(), document);
        let path = self.settings_save_path()?;
        Ok(SettingsMutationOutcome::Persist(
            self.begin_settings_save(candidate, path, response)?,
        ))
    }

    pub fn prepare_shell_profile_remove(
        &mut self,
        params: &Value,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let method = "ui.shell.profiles.remove";
        let name = shell_profile_name(params, method)?;
        let scope = shell_profile_scope(params, method)?;
        let response = json!({"removed":true,"name":name,"scope":scope});
        if scope == "session" {
            if self.session_shell_profiles.remove(name).is_none() {
                return Err(shell_profile_not_found(name, scope));
            }
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        if scope == "project" {
            if !self.project.set_shell_layout_profile(name, None)? {
                return Err(shell_profile_not_found(name, scope));
            }
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        let mut candidate = self.settings.clone();
        if candidate.shell_layout_profiles.remove(name).is_none() {
            return Err(shell_profile_not_found(name, scope));
        }
        candidate
            .shell_layout_startup_profiles
            .retain(|_, profile| profile != name);
        let path = self.settings_save_path()?;
        Ok(SettingsMutationOutcome::Persist(
            self.begin_settings_save(candidate, path, response)?,
        ))
    }

    pub(super) fn load_shell_profile(&mut self, params: &Value) -> Result<Value, ControlError> {
        let method = "ui.shell.profiles.load";
        let name = shell_profile_name(params, method)?;
        let scope = shell_profile_scope(params, method)?;
        let document = match scope {
            "session" => self.session_shell_profiles.get(name).cloned(),
            "application" => self.settings.shell_layout_profiles.get(name).cloned(),
            "project" => self
                .project
                .shell_layout_profiles()?
                .into_iter()
                .find_map(|(candidate, document)| (candidate == name).then_some(document)),
            _ => unreachable!("validated profile scope"),
        }
        .ok_or_else(|| shell_profile_not_found(name, scope))?;
        let mut import = json!({"document":document});
        if let Some(mode) = params.get("mode") {
            import["mode"] = mode.clone();
        }
        if let Some(revision) = params.get("if_shell_revision") {
            import["if_shell_revision"] = revision.clone();
        }
        if let Some(transaction_id) = params.get("transaction_id") {
            import["transaction_id"] = transaction_id.clone();
        }
        let mut response = self.import_shell_layout(&import)?;
        response
            .as_object_mut()
            .expect("shell snapshots are objects")
            .insert("profile".to_string(), json!({"name":name,"scope":scope}));
        if let Some(change) = response.get_mut("change").and_then(Value::as_object_mut) {
            change.insert("operation".to_string(), json!("load_profile"));
        }
        Ok(response)
    }

    pub(crate) fn shell_mutation_layout_candidate(
        &self,
        method: &str,
        params: &Value,
    ) -> Result<Option<Value>, ControlError> {
        let document =
            match method {
                "ui.shell.replace_layout" => return Ok(params.get("desired_tree").cloned()),
                "ui.shell.import_layout" => params.get("document").cloned(),
                "ui.shell.profiles.load" => {
                    let name = shell_profile_name(params, method)?;
                    let scope = shell_profile_scope(params, method)?;
                    match scope {
                        "session" => self.session_shell_profiles.get(name).cloned(),
                        "application" => self.settings.shell_layout_profiles.get(name).cloned(),
                        "project" => self.project.shell_layout_profiles()?.into_iter().find_map(
                            |(candidate, document)| (candidate == name).then_some(document),
                        ),
                        _ => unreachable!("validated profile scope"),
                    }
                }
                _ => None,
            };
        Ok(document.and_then(|document| {
            document
                .get("layout")
                .or_else(|| document.get("desired_tree"))
                .cloned()
        }))
    }

    pub(crate) fn shell_projection(&self) -> Value {
        self.shell_snapshot(None).unwrap_or_else(|_| {
            self.shell
                .mode_state("project")
                .unwrap_or_else(|_| json!({"schema_version":1,"revision":self.shell.revision()}))
        })
    }

    pub(crate) fn apply_startup_shell_layout_if_needed(&mut self) -> bool {
        if !self.settings_bootstrapped || self.mode == ModelMode::Transition {
            return false;
        }
        let mode = self.mode.as_str().to_string();
        if !self.shell_startup_attempted.insert(mode.clone()) {
            return false;
        }
        let Some(profile) = self
            .settings
            .shell_layout_startup_profiles
            .get(&mode)
            .cloned()
        else {
            self.shell_startup_results.insert(
                mode.clone(),
                json!({
                    "mode":mode,
                    "attempted":true,
                    "configured":false,
                    "status":"default",
                    "protected_recovery":false,
                }),
            );
            return false;
        };
        let before = self.shell.revision();
        let restore = self
            .settings
            .shell_layout_profiles
            .get(&profile)
            .cloned()
            .ok_or_else(|| shell_profile_not_found(&profile, "application"))
            .and_then(|document| {
                self.shell
                    .import_layout_document(&json!({"mode":mode,"document":document}), self.mode)
            });
        match restore {
            Ok(outcome) => {
                self.shell_startup_results.insert(
                    mode.clone(),
                    json!({
                        "mode":mode,
                        "attempted":true,
                        "configured":true,
                        "profile":profile,
                        "status":"restored",
                        "protected_recovery":false,
                        "source_schema_version":outcome.source_schema_version,
                        "schema_version":1,
                        "migrated":outcome.migrated,
                    }),
                );
            }
            Err(error) => {
                let recovery = self.shell.recover_layout(&json!({"mode":mode}), self.mode);
                let recovery_failed = recovery.err().map(|failure| failure.message);
                self.settings_status = format!(
                    "Startup shell profile '{profile}' for {mode} failed; protected recovery layout installed: {}",
                    error.message
                );
                self.shell_startup_results.insert(
                    mode.clone(),
                    json!({
                        "mode":mode,
                        "attempted":true,
                        "configured":true,
                        "profile":profile,
                        "status":if recovery_failed.is_none() { "recovered" } else { "recovery_failed" },
                        "protected_recovery":recovery_failed.is_none(),
                        "error":{"kind":error.kind,"message":error.message},
                        "recovery_error":recovery_failed,
                    }),
                );
            }
        }
        self.shell.revision() != before
    }

    pub(crate) fn startup_shell_restore_snapshot(&self) -> Value {
        json!({
            "configured":self.settings.shell_layout_startup_profiles,
            "results":self.shell_startup_results,
        })
    }

    pub(crate) fn sync_active_shell_domain(&mut self) -> Result<Value, ControlError> {
        self.sync_active_shell_domain_inner(false)
    }

    pub(crate) fn sync_active_shell_domain_to_layout(&mut self) -> Result<Value, ControlError> {
        self.sync_active_shell_domain_inner(true)
    }

    fn sync_active_shell_domain_inner(
        &mut self,
        sync_desired_layout: bool,
    ) -> Result<Value, ControlError> {
        let before = match self.mode {
            ModelMode::Transition => return Ok(json!({})),
            mode => self.shell.mode_state(mode.as_str())?,
        };
        match self.mode {
            ModelMode::Project => {}
            ModelMode::Single => {
                let dataset = self.dataset()?;
                let left_visible = dataset.show_left_panel;
                let right_visible = dataset.show_right_panel;
                let left_tab = single_left_node(&dataset.left_tab)?;
                let right_tab = single_right_node(&dataset.right_tab)?;
                self.shell.sync_overlay(
                    ModelMode::Single,
                    &[
                        ("builtin:single.left-panel", left_visible),
                        ("builtin:single.right-panel", right_visible),
                    ],
                    &[
                        ("builtin:single.left-tabs", left_tab),
                        ("builtin:single.right-tabs", right_tab),
                    ],
                    sync_desired_layout,
                )?;
            }
            ModelMode::Mosaic => {
                let state = self.mosaic.projection_state();
                let left_visible = state
                    .pointer("/panels/left")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let right_visible = state
                    .pointer("/panels/right")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let left_tab = mosaic_left_node(
                    state
                        .get("left_tab")
                        .and_then(Value::as_str)
                        .unwrap_or("layers"),
                )?;
                let right_tab = mosaic_right_node(
                    state
                        .get("right_tab")
                        .and_then(Value::as_str)
                        .unwrap_or("properties"),
                )?;
                self.shell.sync_overlay(
                    ModelMode::Mosaic,
                    &[
                        ("builtin:mosaic.left-panel", left_visible),
                        ("builtin:mosaic.right-panel", right_visible),
                    ],
                    &[
                        ("builtin:mosaic.left-tabs", left_tab),
                        ("builtin:mosaic.right-tabs", right_tab),
                    ],
                    sync_desired_layout,
                )?;
            }
            ModelMode::Transition => {}
        }
        let after = self.shell.mode_state(self.mode.as_str())?;
        Ok(shell_change_summary(&before, &after, "native_sync", None))
    }

    fn apply_shell_domain_patch(
        &mut self,
        mode: ModelMode,
        params: &Value,
    ) -> Result<(), ControlError> {
        if let Some(visibility) = visibility_patch(params) {
            let prefix = match mode {
                ModelMode::Single => "builtin:single",
                ModelMode::Mosaic => "builtin:mosaic",
                ModelMode::Project => "builtin:project",
                ModelMode::Transition => unreachable!(),
            };
            let left = visibility
                .get(&format!("{prefix}.left-panel"))
                .and_then(Value::as_bool);
            let right = visibility
                .get(&format!("{prefix}.right-panel"))
                .and_then(Value::as_bool);
            if left.is_some() || right.is_some() {
                let mut panel_params = serde_json::Map::new();
                if let Some(left) = left {
                    panel_params.insert("left".to_string(), Value::Bool(left));
                }
                if let Some(right) = right {
                    panel_params.insert("right".to_string(), Value::Bool(right));
                }
                let panel_params = Value::Object(panel_params);
                match mode {
                    ModelMode::Single => {
                        self.set_panels(&panel_params)?;
                    }
                    ModelMode::Mosaic => {
                        self.mosaic
                            .dispatch("viewer.panels.set", &panel_params)
                            .ok_or_else(|| {
                                ControlError::new(
                                    ControlErrorKind::Internal,
                                    "mosaic panel shell adapter is unavailable",
                                )
                            })??;
                    }
                    ModelMode::Project => {}
                    ModelMode::Transition => unreachable!(),
                }
            }
        }
        if let Some(selected) = selection_patch(params) {
            match mode {
                ModelMode::Single => {
                    if let Some(node) = selected
                        .get("builtin:single.left-tabs")
                        .and_then(Value::as_str)
                    {
                        self.set_left_tab(&json!({"tab":single_left_key(node)?}))?;
                    }
                    if let Some(node) = selected
                        .get("builtin:single.right-tabs")
                        .and_then(Value::as_str)
                    {
                        self.set_right_tab(&json!({"tab":single_right_key(node)?}))?;
                    }
                }
                ModelMode::Mosaic => {
                    if let Some(node) = selected
                        .get("builtin:mosaic.left-tabs")
                        .and_then(Value::as_str)
                    {
                        self.mosaic
                            .dispatch(
                                "mosaic.ui.set_left_tab",
                                &json!({"tab":mosaic_left_key(node)?}),
                            )
                            .transpose()?
                            .ok_or_else(|| {
                                ControlError::new(
                                    ControlErrorKind::Internal,
                                    "mosaic left-tab shell adapter is unavailable",
                                )
                            })?;
                    }
                    if let Some(node) = selected
                        .get("builtin:mosaic.right-tabs")
                        .and_then(Value::as_str)
                    {
                        self.mosaic
                            .dispatch(
                                "mosaic.ui.set_right_tab",
                                &json!({"tab":mosaic_right_key(node)?}),
                            )
                            .transpose()?
                            .ok_or_else(|| {
                                ControlError::new(
                                    ControlErrorKind::Internal,
                                    "mosaic right-tab shell adapter is unavailable",
                                )
                            })?;
                    }
                }
                ModelMode::Project => {}
                ModelMode::Transition => unreachable!(),
            }
        }
        Ok(())
    }

    fn apply_shell_defaults(&mut self, mode: ModelMode) -> Result<(), ControlError> {
        match mode {
            ModelMode::Project => Ok(()),
            ModelMode::Single => {
                self.set_panels(&json!({"left":true,"right":true}))?;
                self.set_left_tab(&json!({"tab":"layers"}))?;
                self.set_right_tab(&json!({"tab":"properties"}))?;
                Ok(())
            }
            ModelMode::Mosaic => {
                self.mosaic
                    .dispatch("viewer.panels.set", &json!({"left":true,"right":true}))
                    .transpose()?
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::Internal,
                            "mosaic panel shell adapter is unavailable",
                        )
                    })?;
                self.mosaic
                    .dispatch("mosaic.ui.set_left_tab", &json!({"tab":"layers"}))
                    .transpose()?
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::Internal,
                            "mosaic left-tab shell adapter is unavailable",
                        )
                    })?;
                self.mosaic
                    .dispatch("mosaic.ui.set_right_tab", &json!({"tab":"properties"}))
                    .transpose()?
                    .ok_or_else(|| {
                        ControlError::new(
                            ControlErrorKind::Internal,
                            "mosaic right-tab shell adapter is unavailable",
                        )
                    })?;
                Ok(())
            }
            ModelMode::Transition => unreachable!(),
        }
    }
}

fn apply_shell_command_state_bindings(shell: &mut Value, command_surface: &Value) {
    let commands = command_surface
        .get("commands")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|command| Some((command.get("id")?.as_str()?, command)))
        .collect::<BTreeMap<_, _>>();
    let Some(nodes) = shell
        .pointer_mut("/layout/nodes")
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    for node in nodes {
        let Some(binding) = node.pointer("/state_bindings/visible") else {
            continue;
        };
        let command = binding
            .get("command_id")
            .and_then(Value::as_str)
            .and_then(|command_id| commands.get(command_id).copied());
        let state = binding.get("state").and_then(Value::as_str);
        let expected = binding
            .get("equals")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let visible = command
            .and_then(|command| state.and_then(|state| command.pointer(&format!("/state/{state}"))))
            .and_then(Value::as_bool)
            .is_some_and(|actual| actual == expected);
        node["visible"] = json!(visible);
    }
}

fn shell_profile_scope<'a>(params: &'a Value, method: &str) -> Result<&'a str, ControlError> {
    match params
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("session")
    {
        scope @ ("session" | "application" | "project") => Ok(scope),
        _ => Err(ControlError::invalid_params(
            method,
            "scope must be session, application, or project",
        )),
    }
}

fn shell_profile_name<'a>(params: &'a Value, method: &str) -> Result<&'a str, ControlError> {
    params
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| {
            !name.trim().is_empty()
                && name.len() <= 128
                && !name.chars().any(|character| character.is_control())
        })
        .ok_or_else(|| {
            ControlError::invalid_params(
                method,
                "name must contain 1 to 128 characters without control characters",
            )
        })
}

fn shell_profile_summary(name: &str, document: &Value, startup_modes: &[&str]) -> Value {
    let mode = document.get("mode").and_then(Value::as_str);
    let schema_version = document.get("schema_version").and_then(Value::as_u64);
    let validation =
        crate::model::shell::validate_layout_document_for(document, "ui.shell.profiles.list");
    let valid = validation.is_ok();
    let (error, error_kind) = validation
        .err()
        .map(|error| (json!(error.message), json!(error.kind)))
        .unwrap_or((Value::Null, Value::Null));
    json!({
        "name":name,
        "mode":mode,
        "document_schema_version":schema_version,
        "valid":valid,
        "error":error,
        "error_kind":error_kind,
        "recovery_method":if valid { Value::Null } else { json!("ui.shell.recover") },
        "startup_modes":startup_modes,
    })
}

fn shell_profile_not_found(name: &str, scope: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::ResourceNotFound,
        format!("shell profile '{name}' was not found in {scope} scope"),
    )
}

fn with_shell_change(
    mut snapshot: Value,
    before: &Value,
    operation: &str,
    transaction_id: Option<&str>,
) -> Value {
    let change = shell_change_summary(before, &snapshot, operation, transaction_id);
    snapshot
        .as_object_mut()
        .expect("shell snapshots are objects")
        .insert("change".to_string(), change);
    snapshot
}

fn shell_change_summary(
    before: &Value,
    after: &Value,
    operation: &str,
    transaction_id: Option<&str>,
) -> Value {
    let before_nodes = nodes_by_id(before);
    let after_nodes = nodes_by_id(after);
    let mut changes = Vec::new();
    for (node_id, after_node) in &after_nodes {
        let Some(before_node) = before_nodes.get(node_id) else {
            continue;
        };
        for (property, field) in [
            ("visibility", "visible"),
            ("order", "children"),
            ("selection", "selected_id"),
        ] {
            if before_node.get(field) != after_node.get(field) {
                changes.push(json!({
                    "node_id":node_id,
                    "property":property,
                    "before":before_node.get(field).cloned().unwrap_or(Value::Null),
                    "after":after_node.get(field).cloned().unwrap_or(Value::Null),
                }));
            }
        }
    }
    let before_layout_nodes = layout_nodes_by_id(before);
    let after_layout_nodes = layout_nodes_by_id(after);
    let layout_topology_changed = before.pointer("/layout/root_id")
        != after.pointer("/layout/root_id")
        || before_layout_nodes.keys().collect::<Vec<_>>()
            != after_layout_nodes.keys().collect::<Vec<_>>()
        || after_layout_nodes.iter().any(|(node_id, after_node)| {
            let Some(before_node) = before_layout_nodes.get(node_id) else {
                return true;
            };
            ["type", "parent_id", "title", "mount"]
                .iter()
                .any(|field| before_node.get(field) != after_node.get(field))
                || string_set(before_node.get("children")) != string_set(after_node.get("children"))
        });
    if layout_topology_changed {
        changes.push(json!({
            "node_id":after.pointer("/layout/root_id").cloned().unwrap_or(Value::Null),
            "property":"layout",
            "before":before.get("layout").cloned().unwrap_or(Value::Null),
            "after":after.get("layout").cloned().unwrap_or(Value::Null),
        }));
    } else {
        for (node_id, after_node) in &after_layout_nodes {
            let Some(before_node) = before_layout_nodes.get(node_id) else {
                continue;
            };
            for (property, field) in [
                ("visibility", "visible"),
                ("order", "children"),
                ("selection", "selected_id"),
                ("size", "size"),
                ("split", "split"),
                ("collapse", "collapsed"),
                ("configuration", "configuration"),
            ] {
                if before_node.get(field) != after_node.get(field) {
                    changes.push(json!({
                        "node_id":node_id,
                        "property":property,
                        "before":before_node.get(field).cloned().unwrap_or(Value::Null),
                        "after":after_node.get(field).cloned().unwrap_or(Value::Null),
                    }));
                }
            }
        }
    }
    for (property, field) in [
        ("active_region", "active_region_id"),
        ("focus", "focused_node_id"),
    ] {
        if before.get(field) != after.get(field) {
            changes.push(json!({
                "node_id":after.get(field).filter(|value| !value.is_null()).cloned()
                    .or_else(|| before.get(field).filter(|value| !value.is_null()).cloned())
                    .unwrap_or_else(|| json!("application:shell")),
                "property":property,
                "before":before.get(field).cloned().unwrap_or(Value::Null),
                "after":after.get(field).cloned().unwrap_or(Value::Null),
            }));
        }
    }
    json!({
        "operation":operation,
        "mode":after.get("mode").cloned().unwrap_or(Value::Null),
        "previous_revision":before.get("revision").cloned().unwrap_or(Value::Null),
        "revision":after.get("revision").cloned().unwrap_or(Value::Null),
        "changed":!changes.is_empty(),
        "changes":changes,
        "transaction_id":transaction_id,
    })
}

fn shell_transaction_id<'a>(
    params: &'a Value,
    method: &str,
) -> Result<Option<&'a str>, ControlError> {
    let Some(value) = params.get("transaction_id") else {
        return Ok(None);
    };
    let transaction_id = value
        .as_str()
        .ok_or_else(|| ControlError::invalid_params(method, "transaction_id must be a string"))?;
    if transaction_id.is_empty()
        || transaction_id.len() > 128
        || transaction_id.chars().any(char::is_control)
    {
        return Err(ControlError::invalid_params(
            method,
            "transaction_id must contain 1 to 128 non-control bytes",
        ));
    }
    Ok(Some(transaction_id))
}

fn nodes_by_id(snapshot: &Value) -> BTreeMap<String, &Value> {
    snapshot
        .get("nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| {
            node.get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_string(), node))
        })
        .collect()
}

fn layout_nodes_by_id(snapshot: &Value) -> BTreeMap<String, &Value> {
    snapshot
        .pointer("/layout/nodes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|node| {
            node.get("id")
                .and_then(Value::as_str)
                .map(|id| (id.to_string(), node))
        })
        .collect()
}

fn string_set(value: Option<&Value>) -> BTreeSet<&str> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn shell_mode(requested: Option<&str>, current: ModelMode) -> Result<ModelMode, ControlError> {
    match requested {
        Some("project") => Ok(ModelMode::Project),
        Some("single") => Ok(ModelMode::Single),
        Some("mosaic") => Ok(ModelMode::Mosaic),
        Some(_) => Err(ControlError::invalid_params(
            "ui.shell",
            "mode must be project, single, or mosaic",
        )),
        None if current == ModelMode::Transition => Err(ControlError::new(
            ControlErrorKind::NotReady,
            "shell mode must be explicit while Odon is transitioning",
        )),
        None => Ok(current),
    }
}

fn require_active_shell(
    requested: ModelMode,
    current: ModelMode,
    method: &str,
) -> Result<(), ControlError> {
    if requested == current {
        Ok(())
    } else {
        Err(ControlError::invalid_params(
            method,
            format!(
                "can only mutate the active shell (active mode is '{}')",
                current.as_str()
            ),
        ))
    }
}

fn single_left_node(tab: &str) -> Result<&'static str, ControlError> {
    match tab {
        "layers" => Ok("builtin:single.layers"),
        "project" => Ok("builtin:single.project"),
        _ => Err(internal_tab("single left", tab)),
    }
}

fn single_right_node(tab: &str) -> Result<&'static str, ControlError> {
    match tab {
        "properties" => Ok("builtin:single.properties"),
        "views" => Ok("builtin:single.views"),
        "analysis" => Ok("builtin:single.analysis"),
        "measurements" => Ok("builtin:single.measurements"),
        "memory" => Ok("builtin:single.memory"),
        "roi_selector" => Ok("builtin:single.roi-selector"),
        _ => Err(internal_tab("single right", tab)),
    }
}

fn mosaic_left_node(tab: &str) -> Result<&'static str, ControlError> {
    match tab {
        "layers" => Ok("builtin:mosaic.layers"),
        "project" => Ok("builtin:mosaic.project"),
        _ => Err(internal_tab("mosaic left", tab)),
    }
}

fn mosaic_right_node(tab: &str) -> Result<&'static str, ControlError> {
    match tab {
        "properties" => Ok("builtin:mosaic.properties"),
        "views" => Ok("builtin:mosaic.views"),
        "layout" => Ok("builtin:mosaic.layout"),
        "memory" => Ok("builtin:mosaic.memory"),
        _ => Err(internal_tab("mosaic right", tab)),
    }
}

fn single_left_key(node: &str) -> Result<&'static str, ControlError> {
    match node {
        "builtin:single.layers" => Ok("layers"),
        "builtin:single.project" => Ok("project"),
        _ => Err(ControlError::invalid_params(
            "ui.shell.patch",
            format!("unknown single-view left-tab node '{node}'"),
        )),
    }
}

fn single_right_key(node: &str) -> Result<&'static str, ControlError> {
    match node {
        "builtin:single.properties" => Ok("properties"),
        "builtin:single.views" => Ok("views"),
        "builtin:single.analysis" => Ok("analysis"),
        "builtin:single.measurements" => Ok("measurements"),
        "builtin:single.memory" => Ok("memory"),
        "builtin:single.roi-selector" => Ok("roi_selector"),
        _ => Err(ControlError::invalid_params(
            "ui.shell.patch",
            format!("unknown single-view right-tab node '{node}'"),
        )),
    }
}

fn mosaic_left_key(node: &str) -> Result<&'static str, ControlError> {
    match node {
        "builtin:mosaic.layers" => Ok("layers"),
        "builtin:mosaic.project" => Ok("project"),
        _ => Err(ControlError::invalid_params(
            "ui.shell.patch",
            format!("unknown mosaic left-tab node '{node}'"),
        )),
    }
}

fn mosaic_right_key(node: &str) -> Result<&'static str, ControlError> {
    match node {
        "builtin:mosaic.properties" => Ok("properties"),
        "builtin:mosaic.views" => Ok("views"),
        "builtin:mosaic.layout" => Ok("layout"),
        "builtin:mosaic.memory" => Ok("memory"),
        _ => Err(ControlError::invalid_params(
            "ui.shell.patch",
            format!("unknown mosaic right-tab node '{node}'"),
        )),
    }
}

fn internal_tab(side: &str, tab: &str) -> ControlError {
    ControlError::new(
        ControlErrorKind::Internal,
        format!("actor {side} tab '{tab}' is not represented by the shell schema"),
    )
}
