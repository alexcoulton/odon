//! Typed command outbox, actor projections, versioned persistence, and segmentation matching controls.

use super::*;

impl ProjectSpace {
    pub fn set_control_actor_owned(&mut self, owned: bool) {
        self.control_actor_owned = owned;
        if !owned {
            self.pending_control_intents.clear();
        }
    }

    pub fn take_control_intents(&mut self) -> Vec<ProjectControlIntent> {
        std::mem::take(&mut self.pending_control_intents)
    }

    pub(super) fn submit_control_intent(
        &mut self,
        method: &'static str,
        params: serde_json::Value,
    ) -> bool {
        if !self.control_actor_owned {
            return false;
        }
        self.pending_control_intents
            .push(ProjectControlIntent { method, params });
        true
    }

    pub fn submit_action_control_intent(&mut self, action: &ProjectSpaceAction) -> bool {
        let (method, params) = match action {
            ProjectSpaceAction::Open(roi) => {
                ("project.rois.open", serde_json::json!({"roi":roi.id}))
            }
            ProjectSpaceAction::OpenView(roi, spec) => {
                let mut request = spec.to_deep_link_request(None);
                request.roi = Some(roi.id.clone());
                ("deep_links.apply", serde_json::json!({"request":request}))
            }
            ProjectSpaceAction::OpenLocalPath(path) => {
                ("datasets.open_tiff", serde_json::json!({"path":path}))
            }
            ProjectSpaceAction::OpenProject(path) => {
                ("project.open", serde_json::json!({"path":path}))
            }
            ProjectSpaceAction::SaveProject(path) => {
                ("project.save_as", serde_json::json!({"path":path}))
            }
            ProjectSpaceAction::ForgetRecentProject(path) => (
                "app.recent_projects.forget",
                serde_json::json!({"path":path}),
            ),
            ProjectSpaceAction::ClearRecentProjects => {
                ("app.recent_projects.clear", serde_json::json!({}))
            }
            ProjectSpaceAction::OpenMosaic => {
                ("project.rois.open_selected_mosaic", serde_json::json!({}))
            }
            ProjectSpaceAction::PreloadObjectSegmentations(settings) => (
                "project.objects.preload.start",
                serde_json::json!({
                    "mode":match settings.mode {
                        ObjectPreloadMode::FullGeometry => "full_geometry",
                        ObjectPreloadMode::CentroidPoints => "centroid_points",
                    },
                    "lazy_properties":settings.lazy_properties,
                }),
            ),
            ProjectSpaceAction::ClearObjectCache => {
                ("project.objects.preload.clear", serde_json::json!({}))
            }
            ProjectSpaceAction::CaptureCurrentView
            | ProjectSpaceAction::OpenRemoteDialog
            | ProjectSpaceAction::ShowHelp(_) => return false,
        };
        self.submit_control_intent(method, params)
    }

    pub(super) fn browser_state(&self) -> ProjectBrowserState {
        let mut selected = self.selected.iter().cloned().collect::<Vec<_>>();
        selected.sort();
        ProjectBrowserState {
            focused: self.focused.clone(),
            selected,
        }
    }

    pub(super) fn state_for_save(&self) -> ProjectState {
        let mut state = self.state.clone();
        state.browser = self.browser_state();
        let valid_roi_keys = self
            .config
            .rois
            .iter()
            .filter_map(ProjectRoi::source_key)
            .collect::<HashSet<_>>();
        state
            .roi_views
            .retain(|source_key, _| valid_roi_keys.contains(source_key));
        state
    }

    pub fn config(&self) -> &ProjectConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut ProjectConfig {
        &mut self.config
    }

    pub fn layer_groups(&self) -> &ProjectLayerGroups {
        &self.config.layer_groups
    }

    pub fn update_layer_groups(&mut self, f: impl FnOnce(&mut ProjectLayerGroups)) {
        f(&mut self.config.layer_groups);
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
    }

    pub fn config_generation(&self) -> u64 {
        self.config_generation
    }

    pub fn control_actor_project_snapshot(&self) -> odon::model::ProjectModelSnapshot {
        let mut selected_source_keys = self.selected.iter().cloned().collect::<Vec<_>>();
        selected_source_keys.sort();
        let view_presets = self
            .state
            .view_presets
            .iter()
            .map(|preset| {
                serde_json::to_value(preset).expect("project view preset is serializable")
            })
            .collect();
        odon::model::ProjectModelSnapshot {
            config: self.config.clone(),
            state: serde_json::to_value(self.state_for_save())
                .expect("project state is serializable"),
            load_generation: self.control_actor_load_generation,
            rois: self.config.rois.clone(),
            default_dataset: self.config.default_dataset.clone(),
            secondary_dataset: self.config.secondary_dataset.clone(),
            default_threshold_marker: self.config.default_threshold_marker.clone(),
            mosaic_segmentation_search_roots: self.config.mosaic_segmentation_search_roots.clone(),
            dataset_keys: self.config.datasets.keys().cloned().collect(),
            selected_source_keys,
            focused_source_key: self.focused.clone(),
            saved_path: self.project_file_path.clone(),
            config_generation: self.config_generation,
            view_presets,
            view_count: self.state.view_presets.len(),
            dirty: self.config_json_dirty,
        }
    }

    /// Lightweight snapshot for detecting native semantic commits each frame. Complete persisted
    /// config/state are intentionally omitted; those are copied only for actor bootstrap/load.
    #[cfg(test)]
    pub fn control_actor_project_delta_snapshot(&self) -> odon::model::ProjectModelSnapshot {
        let mut selected_source_keys = self.selected.iter().cloned().collect::<Vec<_>>();
        selected_source_keys.sort();
        let view_presets = self
            .state
            .view_presets
            .iter()
            .map(|preset| {
                serde_json::to_value(preset).expect("project view preset is serializable")
            })
            .collect();
        odon::model::ProjectModelSnapshot {
            rois: self.config.rois.clone(),
            default_dataset: self.config.default_dataset.clone(),
            secondary_dataset: self.config.secondary_dataset.clone(),
            default_threshold_marker: self.config.default_threshold_marker.clone(),
            mosaic_segmentation_search_roots: self.config.mosaic_segmentation_search_roots.clone(),
            dataset_keys: self.config.datasets.keys().cloned().collect(),
            selected_source_keys,
            focused_source_key: self.focused.clone(),
            saved_path: self.project_file_path.clone(),
            config_generation: self.config_generation,
            view_presets,
            view_count: self.state.view_presets.len(),
            dirty: self.config_json_dirty,
            load_generation: self.control_actor_load_generation,
            ..odon::model::ProjectModelSnapshot::default()
        }
    }

    pub fn apply_control_actor_project_projection(
        &mut self,
        snapshot: &odon::model::ProjectModelSnapshot,
    ) {
        if snapshot.load_generation > self.control_actor_load_generation {
            match serde_json::from_value::<ProjectState>(snapshot.state.clone()) {
                Ok(state) => {
                    self.config.clone_from(&snapshot.config);
                    self.state = state;
                    self.control_actor_load_generation = snapshot.load_generation;
                }
                Err(error) => {
                    self.status = format!("Actor project state could not be materialized: {error}");
                }
            }
        }
        self.config.rois.clone_from(&snapshot.rois);
        self.config
            .default_dataset
            .clone_from(&snapshot.default_dataset);
        self.config
            .secondary_dataset
            .clone_from(&snapshot.secondary_dataset);
        self.config
            .default_threshold_marker
            .clone_from(&snapshot.default_threshold_marker);
        self.config
            .mosaic_segmentation_search_roots
            .clone_from(&snapshot.mosaic_segmentation_search_roots);
        match snapshot
            .view_presets
            .iter()
            .cloned()
            .map(serde_json::from_value::<ProjectViewPreset>)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(view_presets) => self.state.view_presets = view_presets,
            Err(error) => {
                self.status = format!("Actor project views could not be materialized: {error}")
            }
        }
        self.selected = snapshot.selected_source_keys.iter().cloned().collect();
        self.focused.clone_from(&snapshot.focused_source_key);
        self.project_file_path.clone_from(&snapshot.saved_path);
        self.save_path = snapshot
            .saved_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string())
            .unwrap_or_default();
        self.config_generation = snapshot.config_generation;
        self.config_json_dirty = snapshot.dirty;
    }

    pub fn set_object_cache_ui_state(&mut self, state: ProjectObjectCacheUiState) {
        self.object_cache_ui = state;
    }

    pub fn set_recent_projects(&mut self, recent_projects: &[RecentProject]) {
        self.recent_projects = recent_projects.to_vec();
    }

    pub fn load_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        self.load_path = path.to_string_lossy().to_string();
        self.load_from_path();
        if self.status.starts_with("Load failed:")
            || self.status.starts_with("Unsupported project version:")
        {
            anyhow::bail!("{}", self.status);
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn save_to_file(&mut self, path: &Path) -> anyhow::Result<()> {
        self.save_path = path.to_string_lossy().to_string();
        self.state = self.state_for_save();
        let file = ProjectFileV6 {
            version: 6,
            config: self.config.clone(),
            state: self.state.clone(),
        };
        let text = serde_json::to_string_pretty(&file)?;
        fs::write(path, text)?;
        self.project_file_path = Some(path.to_path_buf());
        self.status = format!("Saved: {}", path.to_string_lossy());
        self.show_save_toast(path);
        Ok(())
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    #[cfg(test)]
    pub(super) fn show_save_toast(&mut self, path: &Path) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("project");
        self.save_toast = Some(ProjectSaveToast {
            message: format!("Saved {name}"),
            created_at: Instant::now(),
        });
    }

    pub(super) fn ui_save_toast(&mut self, ctx: &egui::Context) {
        let Some(toast) = self.save_toast.as_ref() else {
            return;
        };

        let elapsed = toast.created_at.elapsed();
        let total = Duration::from_millis(2200);
        if elapsed >= total {
            self.save_toast = None;
            return;
        }
        ctx.request_repaint_after(Duration::from_millis(16));

        let fade = Duration::from_millis(300);
        let alpha = if elapsed < fade {
            elapsed.as_secs_f32() / fade.as_secs_f32()
        } else if total.saturating_sub(elapsed) < fade {
            total.saturating_sub(elapsed).as_secs_f32() / fade.as_secs_f32()
        } else {
            1.0
        }
        .clamp(0.0, 1.0);
        let bg_alpha = (220.0 * alpha).round() as u8;
        let text_alpha = (255.0 * alpha).round() as u8;
        let message = toast.message.clone();

        egui::Area::new(egui::Id::new("project-save-toast"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-22.0, -22.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgba_premultiplied(36, 38, 43, bg_alpha))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_premultiplied(110, 115, 125, bg_alpha),
                    ))
                    .corner_radius(egui::CornerRadius::same(6))
                    .inner_margin(egui::Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(message)
                                .color(egui::Color32::from_rgba_premultiplied(
                                    245, 245, 245, text_alpha,
                                ))
                                .strong(),
                        );
                    });
            });
    }

    pub(super) fn resolved_segmentation_search_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        let mut seen = HashSet::new();
        let project_dir = self.project_dir();

        for root in &self.config.mosaic_segmentation_search_roots {
            let resolved = if root.is_relative() {
                project_dir
                    .as_ref()
                    .map(|dir| dir.join(root))
                    .unwrap_or_else(|| root.clone())
            } else {
                root.clone()
            };
            let resolved = resolved.canonicalize().unwrap_or(resolved);
            if seen.insert(resolved.clone()) {
                roots.push(resolved);
            }
        }

        if let Some(dir) = project_dir {
            let dir = dir.canonicalize().unwrap_or(dir);
            if seen.insert(dir.clone()) {
                roots.push(dir);
            }
        }

        for roi in &self.config.rois {
            if let Some(parent) = roi.local_path().and_then(|path| path.parent()) {
                let parent = parent
                    .canonicalize()
                    .unwrap_or_else(|_| parent.to_path_buf());
                if seen.insert(parent.clone()) {
                    roots.push(parent);
                }
            }
        }

        roots
    }

    pub(super) fn add_segmentation_search_root(&mut self, root: PathBuf) {
        let root = root.canonicalize().unwrap_or(root);
        if self
            .config
            .mosaic_segmentation_search_roots
            .iter()
            .any(|existing| existing == &root)
        {
            return;
        }
        let mut roots = self.config.mosaic_segmentation_search_roots.clone();
        roots.push(root);
        if self.submit_control_intent(
            "project.update_metadata",
            serde_json::json!({"mosaic_segmentation_search_roots":roots}),
        ) {
            return;
        }
        self.config.mosaic_segmentation_search_roots = roots;
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub(super) fn remove_segmentation_search_root(&mut self, index: usize) {
        if index >= self.config.mosaic_segmentation_search_roots.len() {
            return;
        }
        let mut roots = self.config.mosaic_segmentation_search_roots.clone();
        roots.remove(index);
        if self.submit_control_intent(
            "project.update_metadata",
            serde_json::json!({"mosaic_segmentation_search_roots":roots}),
        ) {
            return;
        }
        self.config.mosaic_segmentation_search_roots = roots;
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub(super) fn auto_match_segmentations(&mut self, selected_only: bool) {
        let roots = self.resolved_segmentation_search_roots();
        if roots.is_empty() {
            self.status = "No segmentation search roots available.".to_string();
            return;
        }
        let candidates = collect_segmentation_candidates(&roots, 6);
        if candidates.is_empty() {
            self.status = "No segmentation candidates found in search roots.".to_string();
            return;
        }

        let mut matched = 0usize;
        let mut unmatched = 0usize;
        let mut replacements = Vec::new();
        for roi in &self.config.rois {
            if selected_only
                && !roi
                    .source_key()
                    .as_ref()
                    .is_some_and(|key| self.selected.contains(key))
            {
                continue;
            }
            if roi.local_path().is_none() {
                continue;
            }
            if let Some(best) = best_segmentation_match_for_roi(roi, &candidates) {
                let mut replacement = roi.clone();
                replacement.segpath = Some(best.path.clone());
                replacements.push((roi.id.clone(), replacement));
                matched += 1;
            } else {
                unmatched += 1;
            }
        }
        for (id, replacement) in replacements {
            if let Err(error) = self.update_roi_record(&id, replacement) {
                self.status = error;
                return;
            }
        }
        self.status = match (matched, unmatched) {
            (0, _) => "No segmentation matches found.".to_string(),
            (_, 0) => format!("Matched segmentation for {matched} ROI(s)."),
            _ => format!("Matched {matched} ROI(s); {unmatched} unmatched."),
        };
    }
}
