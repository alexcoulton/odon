//! Settings lifecycle, recent projects, and deep-link model transactions.

use super::*;

impl AppModel {
    pub fn settings_snapshot(&self) -> Value {
        json!({
            "auto_contrast":self.settings.auto_contrast,
            "fast_object_rendering":self.settings.fast_object_rendering,
            "show_extension_manager":self.settings.show_extension_manager,
            "shell_layout_startup_profiles":self.settings.shell_layout_startup_profiles,
            "shell_layout_startup_restore":self.startup_shell_restore_snapshot(),
            "settings_path":self.settings_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "status":self.settings_status,
            "generation":self.settings_operation_generation,
            "persisting":self.settings_operation_pending,
        })
    }

    pub fn recent_projects_snapshot(&self) -> Value {
        json!({
            "projects":self.settings.recent_projects.iter().map(|project| json!({
                "path":project.path.to_string_lossy(),
                "display_name":project.display_name(),
                "last_opened_unix_ms":project.last_opened_unix_ms,
                "exists":self.recent_project_exists.get(&project.path).copied().unwrap_or(false),
            })).collect::<Vec<_>>(),
        })
    }

    pub fn lifecycle_state(&self) -> Value {
        let project = self.project_snapshot();
        let mask_dirty = self
            .dataset
            .as_ref()
            .is_some_and(|dataset| dataset.masks.dirty());
        json!({
            "dirty":project.dirty || mask_dirty,
            "project_path":project.saved_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "can_save":project.saved_path.is_some(),
            "mode":self.mode.as_str(),
        })
    }

    pub fn deep_link_request_from_params(params: &Value) -> Result<DeepLinkRequest, ControlError> {
        if let Some(url) = params.get("url").and_then(Value::as_str) {
            return match DeepLinkRequest::parse_arg(url) {
                Ok(Some(request)) => Ok(request),
                Ok(None) => Err(invalid("url must use the odon: scheme")),
                Err(error) => Err(invalid(format!("invalid deep link: {error}"))),
            };
        }
        if let Some(value) = params.get("request") {
            return serde_json::from_value::<DeepLinkRequest>(value.clone())
                .map_err(|error| invalid(format!("invalid deep-link request: {error}")));
        }
        Err(invalid("url or request is required"))
    }

    pub(super) fn parse_deep_link(params: &Value) -> Result<Value, ControlError> {
        let url = params
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("url is required"))?;
        let request = match DeepLinkRequest::parse_arg(url) {
            Ok(Some(request)) => request,
            Ok(None) => return Err(invalid("url must use the odon: scheme")),
            Err(error) => return Err(invalid(format!("invalid deep link: {error}"))),
        };
        Ok(json!({
            "valid":true,
            "url":request.to_url(),
            "request":request,
        }))
    }

    pub(super) fn deep_link_filters(params: &Value) -> Result<Value, ControlError> {
        let request = Self::deep_link_request_from_params(params)?;
        Ok(json!({
            "object_filters":request.object_filters,
            "object_filter_logic":request.object_filter_logic,
            "object_query":request.object_query,
            "visible_cell_types":request.visible_cell_types,
            "hidden_cell_types":request.hidden_cell_types,
        }))
    }

    pub(super) fn generate_deep_link(&self, params: &Value) -> Result<Value, ControlError> {
        let explicit = params.get("request").is_some();
        let mut request = if let Some(value) = params.get("request") {
            serde_json::from_value::<DeepLinkRequest>(value.clone())
                .map_err(|error| invalid(format!("invalid deep-link request: {error}")))?
        } else {
            self.current_deep_link_request()?
        };
        if !explicit
            && params
                .get("include_project")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        {
            request.project_path = self.project_snapshot().saved_path;
        }
        if params.get("roi").is_some() {
            request.roi = params
                .get("roi")
                .and_then(Value::as_str)
                .map(str::to_string);
        } else if !explicit {
            let project = self.project_snapshot();
            request.roi = project.focused_source_key.as_deref().and_then(|focused| {
                project
                    .rois
                    .iter()
                    .find(|roi| roi.source_key().as_deref() == Some(focused))
                    .map(|roi| roi.id.clone())
            });
        }
        Ok(json!({
            "url":request.to_url(),
            "request":request,
            "source":if explicit { "request" } else { "current_state" },
        }))
    }

    pub(super) fn current_deep_link_request(&self) -> Result<DeepLinkRequest, ControlError> {
        let Some(dataset) = self.dataset.as_ref() else {
            return Ok(DeepLinkRequest::default());
        };
        let viewport = &dataset.workspace.active().state;
        let active_channel = viewport
            .channels
            .get(viewport.active_channel)
            .map(|channel| channel.name.clone());
        let visible_channels = viewport
            .channel_order
            .iter()
            .filter_map(|index| viewport.channels.get(*index))
            .filter(|channel| channel.visible)
            .map(|channel| channel.name.clone())
            .collect::<Vec<_>>();
        let channel_contrasts = viewport
            .channels
            .iter()
            .filter_map(|channel| {
                channel.window.map(|(min, max)| DeepLinkChannelContrast {
                    channel: channel.name.clone(),
                    min,
                    max,
                })
            })
            .collect();
        let channel_colors = viewport
            .channels
            .iter()
            .map(|channel| DeepLinkChannelColor {
                channel: channel.name.clone(),
                color_rgb: channel.color_rgb,
            })
            .collect();
        let filter = viewport
            .objects
            .get("filter")
            .cloned()
            .unwrap_or_else(default_object_filter_model);
        let object_query = filter
            .get("query")
            .and_then(Value::as_str)
            .map(str::to_string);
        let object_filters = filter
            .get("clauses")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|clause| clause.get("enabled").and_then(Value::as_bool) != Some(false))
            .filter_map(|clause| {
                Some(DeepLinkObjectFilterClause {
                    property_key: clause.get("property")?.as_str()?.to_string(),
                    query: clause.get("query")?.as_str()?.to_string(),
                })
            })
            .filter(|clause| !clause.property_key.is_empty() && !clause.query.is_empty())
            .collect();
        Ok(DeepLinkRequest {
            channel: active_channel,
            visible_channels,
            channel_order: Some(DeepLinkChannelOrder::Listed),
            channel_contrasts,
            channel_colors,
            cell_color_by: viewport
                .objects
                .get("color_property")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            object_color_mapping: viewport
                .objects
                .get("color_mapping")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok()),
            fill_cells: viewport.objects.get("fill_cells").and_then(Value::as_bool),
            show_selection_overlay: viewport
                .objects
                .get("show_selection_overlay")
                .and_then(Value::as_bool),
            fast_object_rendering: viewport
                .objects
                .get("fast_rendering")
                .and_then(Value::as_bool),
            object_filters,
            object_filter_logic: match filter.get("logic").and_then(Value::as_str) {
                Some("any") => Some(DeepLinkObjectFilterLogic::Any),
                Some("all") => Some(DeepLinkObjectFilterLogic::All),
                _ => None,
            },
            object_query,
            center_world: Some(viewport.center),
            zoom: Some(viewport.zoom),
            ..DeepLinkRequest::default()
        })
    }

    pub(super) fn apply_deep_link_to_current_dataset(
        &mut self,
        request: &DeepLinkRequest,
        object_filter: Option<ControlObjectFilterResult>,
    ) -> Result<Vec<String>, ControlError> {
        let dataset = self.dataset_mut()?;
        let viewport_id = dataset.workspace.active_id().clone();
        let before = dataset.workspace.active().state.clone();
        let object_resource = dataset.object_resource.clone();
        let abs_max = dataset.descriptor.abs_max.max(1.0);
        let requested_label = requested_bundled_label(request);
        let object_requested = object_segmentation_requested(request);
        let suppress_labels = object_requested
            || request.load_segmentation_labels == Some(false)
            || request
                .segmentation_source
                .as_deref()
                .or(request.segmentation.as_deref())
                .is_some_and(|source| normalize_deep_link_term(source) == "none");

        if let Some(label) = requested_label.as_deref() {
            if dataset.label_loaded.as_deref() != Some(label) {
                return Err(ControlError::new(
                    ControlErrorKind::ResourceNotFound,
                    format!("labels/{label} was not loaded by the deep-link transaction"),
                ));
            }
            dataset.label_selected = label.to_string();
            dataset.label_status = format!("Loaded labels/{label} from deep link.");
        }

        let notes = {
            let viewport = &mut dataset.workspace.active_mut().state;
            let notes = apply_deep_link_viewport(
                viewport,
                request,
                object_resource.as_deref(),
                object_filter,
                abs_max,
            )?;
            if requested_label.is_some() {
                viewport.segmentation_labels_visible = true;
                viewport.native_layers.set_segmentation_labels(true, true);
                let _ = viewport.native_layers.set_active("segmentation_labels");
            } else if suppress_labels {
                viewport.segmentation_labels_visible = false;
                if viewport.native_layers.get("segmentation_labels").is_some() {
                    let _ = viewport
                        .native_layers
                        .set_visibility("segmentation_labels", false);
                }
            }
            if object_requested {
                viewport.objects["visible"] = Value::Bool(true);
                viewport.segmentation_geojson_visible = object_resource.is_some();
                viewport
                    .native_layers
                    .set_primary_objects(object_resource.is_some());
                if object_resource.is_some() {
                    let _ = viewport.native_layers.set_active("segmentation_objects");
                }
            }
            notes
        };
        let after = dataset.workspace.active().state.clone();
        if after.center != before.center || after.zoom != before.zoom {
            let _ = dataset.workspace.bump_navigation_revision(&viewport_id);
            if dataset.workspace.links().camera {
                propagate_camera(&mut dataset.workspace, &viewport_id, &after);
            }
        }
        if presentation_changed(&before, &after) {
            let _ = dataset.workspace.bump_presentation_revision(&viewport_id);
        }
        Ok(notes)
    }

    pub(super) fn sync_current_dataset_view_to_project(&mut self) -> Result<(), ControlError> {
        let dataset = self.dataset()?;
        let source_key = dataset.descriptor.source.source_key();
        let workspace = project_workspace_view_json(&dataset.workspace);
        let active = &dataset.workspace.active().state;
        let view = json!({
            "channel_order":active.channel_order,
            "channels":active.channels.iter().map(project_channel_view_json).collect::<Vec<_>>(),
            "active_channel":active.active_channel,
            "segmentation":project_segmentation_view_json(dataset, active),
            "analysis":{"show_selection_overlay":active.objects.get("show_selection_overlay").cloned().unwrap_or(Value::Bool(true))},
            "camera":{"center_world_lvl0":active.center,"zoom_screen_per_lvl0_px":active.zoom},
            "object_filter":active.objects.get("filter").cloned().unwrap_or_else(default_object_filter_model),
            "object_visible":active.objects.get("visible").cloned().unwrap_or(Value::Bool(false)),
            "object_opacity":active.objects.get("opacity").cloned().unwrap_or(json!(0.75_f32)),
            "object_width_screen_px":active.objects.get("width_screen_px").cloned().unwrap_or(json!(1.25_f32)),
            "object_color_rgb":active.objects.get("color_rgb").cloned().unwrap_or(json!([255,255,255])),
            "object_show_selection_overlay":active.objects.get("show_selection_overlay").cloned().unwrap_or(Value::Bool(true)),
            "workspace":workspace,
            "annotation_layers":self.annotations.states(),
        });
        self.project.set_roi_view_state_json(&source_key, view)
    }

    pub fn prepare_lifecycle_project_save(&mut self) -> Result<(Value, u64), ControlError> {
        if self
            .dataset
            .as_ref()
            .is_some_and(|dataset| dataset.masks.dirty())
        {
            self.sync_masks_to_project()?;
        }
        // Capture the canonical actor workspace immediately before creating the immutable save
        // payload. Persistence must never depend on a renderer panel having observed a frame.
        if self.dataset.is_some() {
            self.sync_current_dataset_view_to_project()?;
        }
        self.project_persistence_payload()
    }

    pub fn prepare_settings_set(
        &mut self,
        params: &Value,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let candidate = self.settings.patched(params).map_err(invalid)?;
        if candidate == self.settings {
            return Ok(SettingsMutationOutcome::Immediate(self.settings_snapshot()));
        }
        let path = self.settings_save_path()?;
        let response = settings_snapshot_for(
            &candidate,
            Some(&path),
            format!("Saved settings to {}.", path.display()),
            self.settings_operation_generation.wrapping_add(1).max(1),
            false,
        );
        Ok(SettingsMutationOutcome::Persist(
            self.begin_settings_save(candidate, path, response)?,
        ))
    }

    pub fn prepare_recent_project_forget(
        &mut self,
        path: PathBuf,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        let forgotten = candidate.forget_recent_project(&path);
        let response = json!({
            "forgotten":forgotten,
            "path":path.to_string_lossy(),
            "remaining":candidate.recent_projects.len(),
        });
        if !forgotten {
            return Ok(SettingsMutationOutcome::Immediate(response));
        }
        let save_path = self.settings_save_path()?;
        let operation = self.begin_settings_save(candidate, save_path, response)?;
        self.recent_project_exists
            .retain(|candidate, _| candidate != &path);
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn prepare_recent_project_record(
        &mut self,
        path: PathBuf,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        if !candidate.record_recent_project(&path) {
            return Ok(SettingsMutationOutcome::Immediate(json!({
                "recorded":false,
                "path":path.to_string_lossy(),
            })));
        }
        let recorded_path = candidate
            .recent_projects
            .first()
            .map(|item| item.path.clone());
        let Some(save_path) = self.settings_path.clone() else {
            if let Some(recorded_path) = recorded_path {
                self.recent_project_exists.insert(recorded_path, true);
            }
            self.settings = candidate;
            return Ok(SettingsMutationOutcome::Immediate(json!({
                "recorded":true,
                "path":path.to_string_lossy(),
                "persisted":false,
            })));
        };
        let operation = self.begin_settings_save(
            candidate.clone(),
            save_path,
            json!({"recorded":true,"path":path.to_string_lossy(),"persisted":true}),
        )?;
        // The recent-project entry belongs to the successful project transaction. Persisting it
        // remains asynchronous, but actor queries immediately observe the canonical new list.
        self.settings = candidate;
        if let Some(recorded_path) = recorded_path {
            self.recent_project_exists.insert(recorded_path, true);
        }
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn prepare_recent_projects_clear(
        &mut self,
    ) -> Result<SettingsMutationOutcome, ControlError> {
        let mut candidate = self.settings.clone();
        let cleared = candidate.recent_projects.len();
        if !candidate.clear_recent_projects() {
            return Ok(SettingsMutationOutcome::Immediate(json!({"cleared":0})));
        }
        let path = self.settings_save_path()?;
        let operation = self.begin_settings_save(candidate, path, json!({"cleared":cleared}))?;
        self.recent_project_exists.clear();
        Ok(SettingsMutationOutcome::Persist(operation))
    }

    pub fn install_settings_for_generation(
        &mut self,
        generation: u64,
        settings: AppSettings,
        response: Value,
    ) -> Option<Value> {
        if !self.settings_operation_pending || generation != self.settings_operation_generation {
            return None;
        }
        self.settings = settings;
        self.recent_project_exists.retain(|path, _| {
            self.settings
                .recent_projects
                .iter()
                .any(|project| &project.path == path)
        });
        self.settings_operation_pending = false;
        self.settings_status = self
            .settings_path
            .as_ref()
            .map(|path| format!("Saved settings to {}.", path.display()))
            .unwrap_or_else(|| "Saved settings.".to_string());
        self.readiness
            .finish(OperationKind::SettingsIo, generation, "Settings saved");
        Some(response)
    }

    pub fn fail_settings_for_generation(
        &mut self,
        generation: u64,
        message: impl Into<String>,
    ) -> bool {
        if !self.settings_operation_pending || generation != self.settings_operation_generation {
            return false;
        }
        self.settings_operation_pending = false;
        self.settings_status = message.into();
        self.readiness.fail(
            OperationKind::SettingsIo,
            generation,
            self.settings_status.clone(),
        );
        true
    }

    pub(super) fn settings_save_path(&self) -> Result<PathBuf, ControlError> {
        self.settings_path.clone().ok_or_else(|| {
            ControlError::new(
                ControlErrorKind::NotReady,
                "application settings path has not been bootstrapped",
            )
        })
    }

    pub(super) fn begin_settings_save(
        &mut self,
        settings: AppSettings,
        path: PathBuf,
        response: Value,
    ) -> Result<SettingsSaveOperation, ControlError> {
        if self.settings_operation_pending {
            return Err(ControlError::new(
                ControlErrorKind::NotReady,
                "another settings persistence operation is already active",
            )
            .with_data(json!({"loading":self.loading_state()["loading"]})));
        }
        self.settings_operation_generation =
            self.settings_operation_generation.wrapping_add(1).max(1);
        self.settings_operation_pending = true;
        self.settings_status = format!("Saving settings to {}...", path.display());
        self.readiness.begin(
            OperationKind::SettingsIo,
            self.settings_operation_generation,
            self.settings_status.clone(),
        );
        Ok(SettingsSaveOperation {
            generation: self.settings_operation_generation,
            path,
            settings,
            response,
        })
    }
}
