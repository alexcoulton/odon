//! ROI CRUD/selection/focus, saved view state, masks, and samplesheet export.

use super::*;

impl ProjectSpace {
    pub fn rois(&self) -> &[ProjectRoi] {
        &self.config.rois
    }

    pub fn roi_index_by_id(&self, id: &str) -> Result<usize, String> {
        let id = id.trim();
        if id.is_empty() {
            return Err("ROI id must not be empty".to_string());
        }
        let matches = self
            .config
            .rois
            .iter()
            .enumerate()
            .filter(|(_, roi)| roi.id == id)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [index] => Ok(*index),
            [] => Err(format!("ROI '{id}' was not found")),
            _ => Err(format!("ROI id '{id}' is ambiguous")),
        }
    }

    pub fn add_roi_record(&mut self, mut roi: ProjectRoi) -> Result<usize, String> {
        roi.id = roi.id.trim().to_string();
        if roi.id.is_empty() {
            return Err("ROI id must not be empty".to_string());
        }
        if self
            .config
            .rois
            .iter()
            .any(|existing| existing.id == roi.id)
        {
            return Err(format!("ROI '{}' already exists", roi.id));
        }
        let Some(source_key) = roi.source_key() else {
            return Err("ROI must have a dataset source".to_string());
        };
        if self
            .config
            .rois
            .iter()
            .any(|existing| existing.source_key().as_deref() == Some(source_key.as_str()))
        {
            return Err("ROI dataset source is already present in the project".to_string());
        }
        if self.submit_control_intent("project.rois.add", serde_json::json!({"replacement":roi})) {
            return Ok(self.config.rois.len());
        }
        self.config.rois.push(roi);
        let index = self.config.rois.len() - 1;
        if self.focused.is_none() {
            self.focused = Some(source_key.clone());
        }
        self.selected.insert(source_key);
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
        Ok(index)
    }

    pub fn update_roi_record(&mut self, id: &str, mut roi: ProjectRoi) -> Result<usize, String> {
        let index = self.roi_index_by_id(id)?;
        roi.id = roi.id.trim().to_string();
        if roi.id.is_empty() {
            return Err("ROI id must not be empty".to_string());
        }
        if self
            .config
            .rois
            .iter()
            .enumerate()
            .any(|(candidate, existing)| candidate != index && existing.id == roi.id)
        {
            return Err(format!("ROI '{}' already exists", roi.id));
        }
        let Some(new_key) = roi.source_key() else {
            return Err("ROI must have a dataset source".to_string());
        };
        if self
            .config
            .rois
            .iter()
            .enumerate()
            .any(|(candidate, existing)| {
                candidate != index && existing.source_key().as_deref() == Some(new_key.as_str())
            })
        {
            return Err("ROI dataset source is already present in the project".to_string());
        }
        if self.submit_control_intent(
            "project.rois.update",
            serde_json::json!({"target_id":id,"replacement":roi}),
        ) {
            return Ok(index);
        }
        let old_key = self.config.rois[index].source_key();
        self.config.rois[index] = roi;
        if let Some(old_key) = old_key
            && old_key != new_key
        {
            if self.selected.remove(old_key.as_str()) {
                self.selected.insert(new_key.clone());
            }
            if self.focused.as_deref() == Some(old_key.as_str()) {
                self.focused = Some(new_key.clone());
            }
            if let Some(view) = self.state.roi_views.remove(old_key.as_str()) {
                self.state.roi_views.insert(new_key, view);
            }
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
        Ok(index)
    }

    pub fn remove_roi_by_id(&mut self, id: &str) -> Result<ProjectRoi, String> {
        let index = self.roi_index_by_id(id)?;
        if self.submit_control_intent("project.rois.remove", serde_json::json!({"id":id})) {
            return Ok(self.config.rois[index].clone());
        }
        let removed = self.config.rois.remove(index);
        if let Some(key) = removed.source_key() {
            self.selected.remove(key.as_str());
            self.state.roi_views.remove(key.as_str());
            if self.focused.as_deref() == Some(key.as_str()) {
                self.focused = None;
            }
        }
        if self.focused.is_none() {
            self.focused = self.config.rois.first().and_then(ProjectRoi::source_key);
        }
        if self.selected.is_empty()
            && let Some(key) = self.focused.clone()
        {
            self.selected.insert(key);
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
        Ok(removed)
    }

    #[cfg(test)]
    pub fn reorder_rois(&mut self, ids: &[String]) -> Result<(), String> {
        if ids.len() != self.config.rois.len() {
            return Err("ROI order must contain every project ROI exactly once".to_string());
        }
        let mut seen = HashSet::new();
        if ids.iter().any(|id| !seen.insert(id.as_str())) {
            return Err("ROI order must not contain duplicate IDs".to_string());
        }
        let mut next = Vec::with_capacity(ids.len());
        for id in ids {
            let index = self.roi_index_by_id(id)?;
            next.push(self.config.rois[index].clone());
        }
        if next
            .iter()
            .map(|roi| &roi.id)
            .eq(self.config.rois.iter().map(|roi| &roi.id))
        {
            return Ok(());
        }
        if self.submit_control_intent("project.rois.reorder", serde_json::json!({"ids":ids})) {
            return Ok(());
        }
        self.config.rois = next;
        self.config_generation = self.config_generation.wrapping_add(1);
        self.config_json_dirty = true;
        Ok(())
    }

    pub fn select_roi_ids(&mut self, ids: &[String], mode: &str) -> Result<(), String> {
        let keys = ids
            .iter()
            .map(|id| {
                let index = self.roi_index_by_id(id)?;
                self.config.rois[index]
                    .source_key()
                    .ok_or_else(|| format!("ROI '{id}' has no dataset source"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if !matches!(mode, "replace" | "add" | "remove" | "toggle") {
            return Err("selection mode must be replace, add, remove, or toggle".to_string());
        }
        if self.submit_control_intent(
            "project.rois.select",
            serde_json::json!({"ids":ids,"mode":mode}),
        ) {
            return Ok(());
        }
        match mode {
            "replace" => {
                self.selected.clear();
                self.selected.extend(keys.iter().cloned());
            }
            "add" => self.selected.extend(keys.iter().cloned()),
            "remove" => {
                for key in &keys {
                    self.selected.remove(key.as_str());
                }
            }
            "toggle" => {
                for key in &keys {
                    if !self.selected.remove(key.as_str()) {
                        self.selected.insert(key.clone());
                    }
                }
            }
            _ => unreachable!("selection mode was validated"),
        }
        if let Some(key) = keys.last() {
            self.focused = Some(key.clone());
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        Ok(())
    }

    #[cfg(test)]
    pub fn focus_roi_id(&mut self, id: &str) -> Result<(), String> {
        self.roi_index_by_id(id)?;
        if self.submit_control_intent("project.rois.focus", serde_json::json!({"id":id})) {
            return Ok(());
        }
        let key = self
            .config
            .rois
            .iter()
            .find(|roi| roi.id == id)
            .and_then(ProjectRoi::source_key)
            .ok_or_else(|| format!("ROI '{id}' has no dataset source"))?;
        self.focused = Some(key);
        self.config_generation = self.config_generation.wrapping_add(1);
        Ok(())
    }

    #[cfg(test)]
    pub fn step_focused_roi(&mut self, step: i64, wrap: bool) -> Result<&ProjectRoi, String> {
        if self.config.rois.is_empty() {
            return Err("project has no ROIs".to_string());
        }
        let current = self
            .focused
            .as_deref()
            .and_then(|key| {
                self.config
                    .rois
                    .iter()
                    .position(|roi| roi.source_key().as_deref() == Some(key))
            })
            .unwrap_or_default();
        let len = self.config.rois.len() as i64;
        let candidate = current as i64 + step;
        let index = if wrap {
            candidate.rem_euclid(len) as usize
        } else {
            candidate.clamp(0, len - 1) as usize
        };
        let key = self.config.rois[index]
            .source_key()
            .ok_or_else(|| "focused ROI has no dataset source".to_string())?;
        self.focused = Some(key);
        self.config_generation = self.config_generation.wrapping_add(1);
        Ok(&self.config.rois[index])
    }

    pub fn roi_mask_layers(&self, roi_path: &Path) -> Option<&[ProjectMaskLayer]> {
        let key = roi_path
            .canonicalize()
            .unwrap_or_else(|_| roi_path.to_path_buf());
        let key_s = key.to_string_lossy();
        self.config
            .rois
            .iter()
            .find(|it| {
                it.local_path()
                    .is_some_and(|path| path == key.as_path() || path.to_string_lossy() == key_s)
            })
            .map(|it| it.mask_layers.as_slice())
    }

    pub(super) fn ensure_roi_for_source(&mut self, source: &DatasetSource) {
        let source_key = source.source_key();
        if self
            .config
            .rois
            .iter()
            .any(|roi| roi.source_key().as_deref() == Some(source_key.as_str()))
        {
            return;
        }

        let display_name = source.display_name();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let mut roi = ProjectRoi {
            id: display_name.clone(),
            source: None,
            path: None,
            dataset: source.is_local().then_some(default_dataset),
            display_name: Some(display_name),
            segpath: None,
            mask_layers: Vec::new(),
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(source.clone());
        self.config.rois.push(roi);
    }

    pub(super) fn roi_view_state_mut(
        &mut self,
        source: &DatasetSource,
    ) -> &mut ProjectRoiViewState {
        self.ensure_roi_for_source(source);
        self.state.roi_views.entry(source.source_key()).or_default()
    }

    pub fn roi_view_state(&self, source: &DatasetSource) -> Option<&ProjectRoiViewState> {
        self.state.roi_views.get(&source.source_key())
    }

    pub fn set_roi_view_state(&mut self, source: &DatasetSource, view: ProjectRoiViewState) {
        let dst = self.roi_view_state_mut(source);
        if *dst == view {
            return;
        }
        *dst = view;
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn save_view_preset(&mut self, name: String, spec: ProjectViewSpec) {
        let name = name.trim();
        if name.is_empty() {
            self.status = "View preset name is empty.".to_string();
            return;
        }
        let mut spec = spec;
        if let Some(active) = spec.channel_ref.as_mut()
            && let Some(visible) = spec
                .visible_channel_refs
                .iter()
                .find(|visible| visible.label == active.label)
        {
            active.alias = visible.alias.clone();
        }
        let preset = ProjectViewPreset {
            name: name.to_string(),
            description: String::new(),
            spec,
        };
        if self.submit_control_intent(
            "project.views.create",
            serde_json::json!({"name":preset.name,"spec":preset.spec}),
        ) {
            self.status = format!("Saving view preset '{name}'...");
            return;
        }
        if let Some((idx, existing)) = self
            .state
            .view_presets
            .iter_mut()
            .enumerate()
            .find(|(_, preset)| preset.name == name)
        {
            *existing = preset;
            self.selected_view_preset = idx;
            self.status = format!("Updated view preset '{name}'.");
        } else {
            self.state.view_presets.push(preset);
            self.selected_view_preset = self.state.view_presets.len().saturating_sub(1);
            self.status = format!("Saved view preset '{name}'.");
        }
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    #[cfg(test)]
    pub fn view_presets(&self) -> &[ProjectViewPreset] {
        &self.state.view_presets
    }

    pub fn delete_view_preset(&mut self, index: usize) -> Result<ProjectViewPreset, String> {
        if index >= self.state.view_presets.len() {
            return Err(format!("view preset index {index} is out of range"));
        }
        let removed = self.state.view_presets[index].clone();
        if self.submit_control_intent(
            "project.views.delete",
            serde_json::json!({"name":removed.name}),
        ) {
            self.status = format!("Deleting view preset '{}'...", removed.name);
            return Ok(removed);
        }
        let removed = self.state.view_presets.remove(index);
        self.selected_view_preset = self
            .selected_view_preset
            .min(self.state.view_presets.len().saturating_sub(1));
        self.status = format!("Deleted view preset '{}'.", removed.name);
        self.config_generation = self.config_generation.wrapping_add(1);
        Ok(removed)
    }

    #[cfg(test)]
    pub fn rename_view_preset(&mut self, index: usize, name: String) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("view preset name must not be empty".to_string());
        }
        if self
            .state
            .view_presets
            .iter()
            .enumerate()
            .any(|(candidate, preset)| candidate != index && preset.name == name)
        {
            return Err(format!("a view preset named '{name}' already exists"));
        }
        let Some(previous_name) = self
            .state
            .view_presets
            .get(index)
            .map(|preset| preset.name.clone())
        else {
            return Err(format!("view preset index {index} is out of range"));
        };
        if previous_name == name {
            return Ok(());
        }
        if self.submit_control_intent(
            "project.views.rename",
            serde_json::json!({"name":previous_name,"new_name":name}),
        ) {
            self.selected_view_preset = index;
            self.status = format!("Renaming view preset to '{name}'...");
            return Ok(());
        }
        let preset = self
            .state
            .view_presets
            .get_mut(index)
            .expect("validated view preset remains present");
        preset.name = name.to_string();
        self.selected_view_preset = index;
        self.status = format!("Renamed view preset to '{name}'.");
        self.config_generation = self.config_generation.wrapping_add(1);
        Ok(())
    }

    pub fn set_view_preset_draft(&mut self, spec: ProjectViewSpec) {
        self.view_preset_draft = Some(spec);
        self.views_dialog_open = true;
        self.status = "Captured current view. Review aliases, then save.".to_string();
    }

    pub fn mosaic_view_state(&self) -> Option<&ProjectMosaicViewState> {
        self.state.mosaic.as_ref()
    }

    pub fn set_mosaic_view_state(&mut self, view: ProjectMosaicViewState) {
        if self.state.mosaic.as_ref() == Some(&view) {
            return;
        }
        self.state.mosaic = Some(view);
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn set_roi_mask_layers(&mut self, roi_path: &Path, layers: Vec<ProjectMaskLayer>) {
        let key = roi_path
            .canonicalize()
            .unwrap_or_else(|_| roi_path.to_path_buf());
        let key_s = key.to_string_lossy();
        if let Some(it) = self.config.rois.iter_mut().find(|it| {
            it.local_path()
                .is_some_and(|path| path == key.as_path() || path.to_string_lossy() == key_s)
        }) {
            if it.mask_layers == layers {
                return;
            }
            it.mask_layers = layers;
            self.config_generation = self.config_generation.wrapping_add(1);
            return;
        }

        // If the ROI isn't part of the explicit list yet, add it (best-effort) so masks can be
        // persisted when the user saves the Project JSON.
        let display_name = key
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string());
        let id = display_name.clone().unwrap_or_else(|| "ROI".to_string());
        let mut roi = ProjectRoi {
            id,
            source: None,
            path: None,
            dataset: None,
            display_name,
            segpath: None,
            mask_layers: layers,
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(DatasetSource::Local(key));
        self.config.rois.push(roi);
        self.config_generation = self.config_generation.wrapping_add(1);
    }

    pub fn focused_roi(&self) -> Option<&ProjectRoi> {
        let key = self.focused.as_ref()?;
        self.config
            .rois
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(key.as_str()))
    }

    pub fn selected_rois(&self) -> Vec<ProjectRoi> {
        self.config
            .rois
            .iter()
            .filter(|roi| {
                roi.source_key()
                    .is_some_and(|key| self.selected.contains(key.as_str()))
            })
            .cloned()
            .collect()
    }

    pub fn rois_for_local_paths(&self, paths: &[PathBuf]) -> Vec<ProjectRoi> {
        let selected = paths.iter().collect::<HashSet<_>>();
        self.config
            .rois
            .iter()
            .filter(|roi| {
                roi.local_path()
                    .is_some_and(|path| selected.contains(&path.to_path_buf()))
            })
            .cloned()
            .collect()
    }

    #[cfg(test)]
    pub fn roi_for_link_target(
        &self,
        roi_query: Option<&str>,
        sample_query: Option<&str>,
    ) -> Result<ProjectRoi, String> {
        odon::deep_link::resolve_roi_target(&self.config.rois, roi_query, sample_query)
    }

    pub fn add_roi_source(&mut self, source: DatasetSource) {
        let source_key = source.source_key();
        if let Some(existing_id) = self
            .config
            .rois
            .iter()
            .find(|roi| roi.source_key().as_deref() == Some(source_key.as_str()))
            .map(|roi| roi.id.clone())
        {
            if let Err(error) = self.select_roi_ids(&[existing_id], "replace") {
                self.status = error;
            }
            return;
        }
        let display_name = source.display_name();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let mut roi = ProjectRoi {
            id: display_name.clone(),
            source: None,
            path: None,
            dataset: source.is_local().then_some(default_dataset),
            display_name: Some(display_name),
            segpath: None,
            mask_layers: Vec::new(),
            channel_order: Vec::new(),
            meta: Default::default(),
        };
        roi.set_dataset_source(source);
        if let Err(error) = self.add_roi_record(roi) {
            self.status = error;
            return;
        }
        self.status.clear();
    }

    pub fn project_dir(&self) -> Option<PathBuf> {
        self.project_file_path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }

    pub fn saved_project_path(&self) -> Option<PathBuf> {
        self.project_file_path.clone()
    }

    pub fn current_project_path(&self) -> Option<PathBuf> {
        self.project_file_path.clone().or_else(|| {
            (!self.save_path.trim().is_empty()).then(|| PathBuf::from(self.save_path.trim()))
        })
    }

    pub(super) fn choose_project_save_path(&self, title: &str) -> Option<PathBuf> {
        let default_name = self
            .current_project_path()
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("project.json")
            .to_string();
        let mut dialog = FileDialog::new()
            .add_filter("Project JSON", &["json"])
            .set_file_name(&default_name)
            .set_title(title);
        if let Some(parent) = self
            .current_project_path()
            .as_ref()
            .and_then(|path| path.parent())
        {
            dialog = dialog.set_directory(parent);
        }
        dialog.save_file()
    }

    pub(super) fn exportable_local_roi_count(&self) -> usize {
        self.config
            .rois
            .iter()
            .filter(|roi| roi.local_path().is_some())
            .count()
    }

    pub(super) fn default_samplesheet_export_path(&self) -> PathBuf {
        let project_path = self.current_project_path();

        let stem = project_path
            .as_ref()
            .and_then(|path| path.file_stem())
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.trim().is_empty())
            .unwrap_or("samplesheet");
        let stem = stem.strip_suffix(".project").unwrap_or(stem);
        let file_name = format!("{stem}.samplesheet.csv");

        project_path
            .as_ref()
            .and_then(|path| path.parent())
            .map(|dir| dir.join(&file_name))
            .unwrap_or_else(|| PathBuf::from(file_name))
    }

    pub fn export_samplesheet_csv(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut meta_columns = BTreeSet::new();
        let mut rows = Vec::new();
        let mut skipped_non_local = 0usize;

        for roi in &self.config.rois {
            let Some(local_path) = roi.local_path() else {
                skipped_non_local += 1;
                continue;
            };

            let mut meta = roi
                .meta
                .iter()
                .filter_map(|(key, value)| {
                    let key = key.trim();
                    (!key.is_empty()).then(|| (key.to_string(), value.clone()))
                })
                .collect::<std::collections::HashMap<_, _>>();

            if let Some(dataset) = roi.dataset.as_ref().filter(|s| !s.trim().is_empty()) {
                meta.insert("dataset".to_string(), dataset.clone());
            }

            if let Some(segpath) = roi.segpath.as_ref() {
                let segpath = segpath
                    .canonicalize()
                    .unwrap_or_else(|_| segpath.to_path_buf());
                meta.insert("segpath".to_string(), segpath.to_string_lossy().to_string());
            }

            for key in meta.keys() {
                meta_columns.insert(key.clone());
            }

            let id = if roi.id.trim().is_empty() {
                roi.display_name
                    .clone()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| {
                        local_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("ROI")
                            .to_string()
                    })
            } else {
                roi.id.trim().to_string()
            };

            rows.push(SampleRow {
                id,
                path: local_path
                    .canonicalize()
                    .unwrap_or_else(|_| local_path.to_path_buf()),
                meta,
            });
        }

        if rows.is_empty() {
            anyhow::bail!("project has no local ROIs to export");
        }

        let exported_count = rows.len();
        let sheet = SampleSheet {
            meta_columns: meta_columns.into_iter().collect(),
            rows,
        };
        write_samplesheet_csv(path, &sheet)?;

        self.status = if skipped_non_local > 0 {
            format!(
                "Exported {exported_count} ROI(s) to {}; skipped {skipped_non_local} non-local ROI(s).",
                path.to_string_lossy()
            )
        } else {
            format!(
                "Exported {exported_count} ROI(s) to {}.",
                path.to_string_lossy()
            )
        };

        Ok(())
    }

    pub fn set_current_dataset_root(&mut self, root: Option<&Path>) {
        let Some(root) = root else {
            if self.save_path.trim().is_empty() {
                self.save_path = "odon.project.json".to_string();
            }
            if self.load_path.trim().is_empty() {
                self.load_path = self.save_path.clone();
            }
            return;
        };
        let root_s = root.to_string_lossy();
        if self.save_path.is_empty() {
            self.save_path = format!("{root_s}.project.json");
        }
        if self.load_path.is_empty() {
            self.load_path = self.save_path.clone();
        }
    }

    pub fn handle_dropped_paths(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        for p in paths {
            if let Some(root) = normalize_local_dataset_path(&p) {
                self.add_roi(root);
                continue;
            }
            self.status = format!("Unsupported dataset: {}", p.to_string_lossy());
        }
    }

    pub(super) fn add_roi(&mut self, root: PathBuf) {
        let root = root.canonicalize().unwrap_or(root);
        let Some(kind) = classify_local_dataset_path(&root) else {
            self.status = format!("Not a supported dataset root: {}", root.to_string_lossy());
            return;
        };
        let mut source = DatasetSource::Local(root);
        if !matches!(kind, LocalDatasetKind::OmeZarr) {
            // TIFF/Xenium stay local but don't default into the mosaic dataset selector.
            source = DatasetSource::Local(source.local_path().unwrap().to_path_buf());
        }
        self.add_roi_source(source);
    }
}
