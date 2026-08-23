//! Samplesheet import, filtered selection, dataset discovery, and native file loading.

use super::*;

impl ProjectSpace {
    pub fn import_rois_from_csv(&mut self, path: &Path) -> anyhow::Result<()> {
        let sheet = load_samplesheet_csv(path)?;
        let base_dir = path.parent();
        self.config.rois.clear();
        self.roi_browse.clear();
        let default_dataset = self
            .config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        for row in sheet.rows {
            let meta = row.meta;
            let resolved_path = if row.path.is_relative() {
                base_dir
                    .map(|dir| dir.join(&row.path))
                    .unwrap_or_else(|| row.path.clone())
            } else {
                row.path.clone()
            };
            let resolved_path = resolved_path.canonicalize().unwrap_or(resolved_path);
            let segpath = meta
                .get("segpath")
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from)
                .map(|seg| {
                    if seg.is_relative() {
                        base_dir.map(|dir| dir.join(&seg)).unwrap_or(seg)
                    } else {
                        seg
                    }
                })
                .map(|seg| seg.canonicalize().unwrap_or(seg));
            let dataset = meta
                .get("dataset")
                .filter(|s| !s.trim().is_empty())
                .cloned()
                .or_else(|| Some(default_dataset.clone()));
            let mut roi = ProjectRoi {
                id: row.id.clone(),
                source: None,
                path: None,
                dataset,
                display_name: Some(row.id),
                segpath,
                mask_layers: Vec::new(),
                channel_order: Vec::new(),
                meta,
            };
            roi.set_dataset_source(DatasetSource::Local(resolved_path));
            self.config.rois.push(roi);
        }
        self.focused = self.config.rois.first().and_then(ProjectRoi::source_key);
        self.selected.clear();
        if let Some(p) = self.focused.clone() {
            self.selected.insert(p);
        }
        self.config_generation = self.config_generation.wrapping_add(1);
        self.status = format!(
            "Imported {} ROIs from samplesheet ({} metadata columns).",
            self.config.rois.len(),
            sheet.meta_columns.len()
        );
        Ok(())
    }

    pub(super) fn sync_filtered_selection(&mut self, visible_indices: &[usize]) {
        let visible_keys = visible_indices
            .iter()
            .filter_map(|&idx| self.config.rois.get(idx).and_then(ProjectRoi::source_key))
            .collect::<HashSet<_>>();

        if self.control_actor_owned {
            let mut selected = self
                .selected
                .iter()
                .filter(|key| visible_keys.contains(*key))
                .cloned()
                .collect::<HashSet<_>>();
            let focused = self
                .focused
                .as_ref()
                .filter(|key| visible_keys.contains(*key))
                .cloned()
                .or_else(|| {
                    visible_indices.first().and_then(|&index| {
                        self.config.rois.get(index).and_then(ProjectRoi::source_key)
                    })
                });
            if selected.is_empty()
                && let Some(key) = focused.clone()
            {
                selected.insert(key);
            }
            if selected == self.selected && focused == self.focused {
                return;
            }
            let mut ids = self
                .config
                .rois
                .iter()
                .filter(|roi| {
                    roi.source_key()
                        .is_some_and(|key| selected.contains(key.as_str()))
                })
                .map(|roi| roi.id.clone())
                .collect::<Vec<_>>();
            if let Some(focused_id) = focused.and_then(|key| {
                self.config
                    .rois
                    .iter()
                    .find(|roi| roi.source_key().as_deref() == Some(key.as_str()))
                    .map(|roi| roi.id.clone())
            }) && let Some(position) = ids.iter().position(|id| id == &focused_id)
            {
                ids.remove(position);
                ids.push(focused_id);
            }
            if let Err(error) = self.select_roi_ids(&ids, "replace") {
                self.status = error;
            }
            return;
        }

        self.selected.retain(|key| visible_keys.contains(key));

        if self
            .focused
            .as_ref()
            .is_some_and(|key| !visible_keys.contains(key))
        {
            self.focused = None;
        }
        if self.focused.is_none() {
            self.focused = visible_indices
                .first()
                .and_then(|&idx| self.config.rois.get(idx).and_then(ProjectRoi::source_key));
        }
        if self.selected.is_empty() {
            if let Some(key) = self.focused.clone() {
                self.selected.insert(key);
            }
        }
    }

    pub fn import_rois_from_root(&mut self, root: &Path) -> anyhow::Result<()> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        if !root.is_dir() {
            anyhow::bail!("not a directory: {}", root.to_string_lossy());
        }

        let before = self.config.rois.len();
        let roots = discover_omezarr_roots_under(&root);
        if roots.is_empty() {
            anyhow::bail!(
                "no OME-Zarr datasets found under {}",
                root.to_string_lossy()
            );
        }
        for roi_root in roots {
            self.add_roi(roi_root);
        }
        let added = self.config.rois.len().saturating_sub(before);
        self.status = format!(
            "Added {added} OME-Zarr ROI(s) from {}.",
            root.to_string_lossy()
        );
        Ok(())
    }

    pub(super) fn load_from_path(&mut self) {
        let path = PathBuf::from(self.load_path.trim());
        if path.as_os_str().is_empty() {
            self.status = "Load path is empty.".to_string();
            return;
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                self.status = format!("Load failed: {e}");
                return;
            }
        };
        let version = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("version").and_then(|x| x.as_u64()))
            .unwrap_or(1);

        let (mut config, mut state): (ProjectConfig, ProjectState) = match version {
            1 => {
                let file: ProjectFileV1 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let focused = file.selected.and_then(|i| {
                    file.items
                        .get(i)
                        .map(|it| DatasetSource::Local(it.path.clone()).source_key())
                });
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused,
                            selected: Vec::new(),
                        },
                        ..Default::default()
                    },
                )
            }
            2 => {
                let file: ProjectFileV2 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            3 => {
                let file: ProjectFileV3Legacy = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                let rois = file
                    .items
                    .into_iter()
                    .map(|it| {
                        let mut roi = ProjectRoi {
                            id: it
                                .display_name
                                .clone()
                                .unwrap_or_else(|| it.path.to_string_lossy().to_string()),
                            source: None,
                            path: None,
                            dataset: None,
                            display_name: it.display_name,
                            segpath: None,
                            mask_layers: Vec::new(),
                            channel_order: Vec::new(),
                            meta: Default::default(),
                        };
                        roi.set_dataset_source(DatasetSource::Local(it.path));
                        roi
                    })
                    .collect();
                (
                    ProjectConfig {
                        rois,
                        ..Default::default()
                    },
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            4 => {
                let file: ProjectFileV4 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (
                    file.config,
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file
                                .focused
                                .map(|path| DatasetSource::Local(path).source_key()),
                            selected: file
                                .selected
                                .into_iter()
                                .map(|path| DatasetSource::Local(path).source_key())
                                .collect(),
                        },
                        ..Default::default()
                    },
                )
            }
            5 => {
                let file: ProjectFileV5 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (
                    file.config,
                    ProjectState {
                        browser: ProjectBrowserState {
                            focused: file.focused,
                            selected: file.selected,
                        },
                        ..Default::default()
                    },
                )
            }
            6 => {
                let file: ProjectFileV6 = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        self.status = format!("Load failed: {e}");
                        return;
                    }
                };
                (file.config, file.state)
            }
            _ => {
                self.status = format!("Unsupported project version: {version}");
                return;
            }
        };
        self.project_file_path = Some(path.clone());
        self.save_path = path.to_string_lossy().to_string();
        self.load_path = path.to_string_lossy().to_string();
        let project_dir = path.parent().map(Path::to_path_buf);

        let mut seen: HashSet<String> = HashSet::new();
        let mut cleaned: Vec<ProjectRoi> = Vec::new();
        let default_dataset = config
            .default_dataset
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let rois = std::mem::take(&mut config.rois);
        for mut roi in rois.into_iter() {
            let Some(source) = roi.dataset_source() else {
                continue;
            };
            match source {
                DatasetSource::Local(path) => {
                    let resolved_path = resolve_project_relative_path(project_dir.as_deref(), path);
                    let p = resolved_path.canonicalize().unwrap_or(resolved_path);
                    let dedupe_key = DatasetSource::Local(p.clone()).source_key();
                    if !seen.insert(dedupe_key) {
                        continue;
                    }
                    let kind = classify_local_dataset_path(&p);
                    if let Some(segpath) = roi.segpath.take() {
                        let resolved_segpath =
                            resolve_project_relative_path(project_dir.as_deref(), segpath);
                        roi.segpath =
                            Some(resolved_segpath.canonicalize().unwrap_or(resolved_segpath));
                    }
                    if roi.display_name.is_none() {
                        roi.display_name = p
                            .file_name()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string());
                    }
                    if roi.id.trim().is_empty() {
                        roi.id = roi
                            .display_name
                            .clone()
                            .unwrap_or_else(|| p.to_string_lossy().to_string());
                    }
                    if roi
                        .dataset
                        .as_deref()
                        .map(|s| s.trim().is_empty())
                        .unwrap_or(true)
                        && matches!(kind, Some(LocalDatasetKind::OmeZarr))
                    {
                        roi.dataset = Some(default_dataset.clone());
                    }
                    roi.set_dataset_source(DatasetSource::Local(p));
                }
                other => {
                    let dedupe_key = other.source_key();
                    if !seen.insert(dedupe_key) {
                        continue;
                    }
                    if roi.display_name.is_none() {
                        roi.display_name = Some(other.display_name());
                    }
                    if roi.id.trim().is_empty() {
                        roi.id = roi
                            .display_name
                            .clone()
                            .unwrap_or_else(|| other.display_name());
                    }
                    roi.set_dataset_source(other);
                }
            }
            cleaned.push(roi);
        }

        for roi in &mut cleaned {
            let key = roi.source_key();
            if let Some(key) = key {
                if !roi.channel_order.is_empty() {
                    let view = state.roi_views.entry(key).or_default();
                    if view.channel_order.is_empty() {
                        view.channel_order = std::mem::take(&mut roi.channel_order);
                    } else {
                        roi.channel_order.clear();
                    }
                }
            }
        }

        config.rois = cleaned;
        self.config = config;
        self.state = state;
        self.config_generation = self.config_generation.wrapping_add(1);

        self.focused = self
            .state
            .browser
            .focused
            .clone()
            .filter(|key| {
                self.config
                    .rois
                    .iter()
                    .any(|it| it.source_key().as_deref() == Some(key.as_str()))
            })
            .or_else(|| self.config.rois.first().and_then(ProjectRoi::source_key));

        self.selected.clear();
        for key in self.state.browser.selected.clone() {
            if self
                .config
                .rois
                .iter()
                .any(|it| it.source_key().as_deref() == Some(key.as_str()))
            {
                self.selected.insert(key);
            }
        }
        if self.selected.is_empty() {
            if let Some(p) = self.focused.clone() {
                self.selected.insert(p);
            }
        }
        self.status = format!("Loaded: {}", path.to_string_lossy());
    }
}
