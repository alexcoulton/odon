use super::*;

impl OmeZarrViewerApp {
    pub(super) fn project_workspace_view_state(&self) -> Option<ProjectWorkspaceViewState> {
        let workspace = self.viewport_workspace.as_ref()?;
        if workspace.len() <= 1 {
            return None;
        }
        let links = workspace.links();
        Some(ProjectWorkspaceViewState {
            version: 1,
            layout: workspace.layout().as_str().to_string(),
            split_ratio: workspace.split_ratio(),
            active_viewport_id: workspace.active_id().as_str().to_string(),
            link_camera: links.camera,
            link_plane: links.plane,
            link_selection: links.selection,
            viewports: workspace
                .viewports()
                .iter()
                .map(|viewport| {
                    let state = &viewport.state;
                    ProjectViewportViewState {
                        id: viewport.id.as_str().to_string(),
                        title: viewport.title.clone(),
                        navigation_revision: viewport.navigation_revision.max(1),
                        presentation_revision: viewport.presentation_revision.max(1),
                        layer_groups: Some(state.layer_groups.clone()),
                        camera: Some(ProjectCameraState {
                            center_world_lvl0: [
                                state.camera.center_world_lvl0.x,
                                state.camera.center_world_lvl0.y,
                            ],
                            zoom_screen_per_lvl0_px: state.camera.zoom_screen_per_lvl0_px,
                        }),
                        plane_mode: Some(state.view_plane_mode.label().to_ascii_lowercase()),
                        x_level0: Some(state.current_x_level0),
                        y_level0: Some(state.current_y_level0),
                        z_level0: Some(state.current_z_level0),
                        channel_order: state.channel_layer_order.clone(),
                        channels: state
                            .channels
                            .iter()
                            .map(|channel| ProjectChannelViewState {
                                name: Some(channel.name.clone()),
                                visible: Some(channel.visible),
                                color_rgb: Some(channel.color_rgb),
                                window: channel.window.map(|(lo, hi)| [lo, hi]),
                                note: (!channel.note.is_empty()).then(|| channel.note.clone()),
                                ..Default::default()
                            })
                            .collect(),
                        active_channel: Some(state.selected_channel),
                        active_layer: Some(Self::layer_id_storage_key(state.active_layer)),
                        overlay_order: state
                            .overlay_layer_order
                            .iter()
                            .copied()
                            .map(Self::layer_id_storage_key)
                            .collect(),
                        overlay_visibility: state
                            .overlay_layer_order
                            .iter()
                            .copied()
                            .filter_map(|id| {
                                state
                                    .layer_visible(id)
                                    .map(|visible| (Self::layer_id_storage_key(id), visible))
                            })
                            .collect(),
                        segmentation: Some(ProjectSegmentationViewState {
                            label_name: (!self.seg_label_selected.is_empty())
                                .then(|| self.seg_label_selected.clone()),
                            outlines_color_rgb: Some(state.cells_outlines_color_rgb),
                            outlines_opacity: Some(state.cells_outlines_opacity),
                            outlines_width_px: Some(state.cells_outlines_width_px),
                            object_display: Some(state.object_display.clone()),
                        }),
                        object_filter: Some(state.object_filter.project_json()),
                        object_visible: Some(state.object_visible),
                        object_opacity: Some(state.object_opacity),
                        object_width_screen_px: Some(state.object_width_screen_px),
                        object_color_rgb: Some(state.object_color_rgb),
                        object_show_selection_overlay: Some(state.object_show_selection_overlay),
                        presentation: Some(state.project_presentation_json()),
                    }
                })
                .collect(),
        })
    }

    pub(super) fn restore_project_workspace(
        &mut self,
        saved: &ProjectWorkspaceViewState,
    ) -> Result<(), String> {
        if saved.version != 1 {
            return Err(format!(
                "unsupported viewport workspace version {}",
                saved.version
            ));
        }
        let layout = ViewportLayout::parse(&saved.layout)
            .ok_or_else(|| format!("unknown viewport layout '{}'", saved.layout))?;
        let base = ViewerViewportState::capture(self);
        let mut slots = Vec::with_capacity(saved.viewports.len());
        for saved_viewport in &saved.viewports {
            let id = ViewportId::new(&saved_viewport.id).map_err(|error| error.to_string())?;
            let mut state = base.clone();
            state.last_canvas_rect = None;
            if let Some(layer_groups) = saved_viewport.layer_groups.as_ref() {
                state.layer_groups.clone_from(layer_groups);
            }
            if let Some(camera) = saved_viewport.camera.as_ref() {
                state.camera.center_world_lvl0 =
                    egui::pos2(camera.center_world_lvl0[0], camera.center_world_lvl0[1]);
                state.camera.zoom_screen_per_lvl0_px = camera.zoom_screen_per_lvl0_px.max(1e-6);
            }
            if let Some(mode) = saved_viewport.plane_mode.as_deref() {
                state.view_plane_mode = match mode.to_ascii_lowercase().as_str() {
                    "xy" => ViewPlaneMode::Xy,
                    "xz" => ViewPlaneMode::Xz,
                    "yz" => ViewPlaneMode::Yz,
                    _ => return Err(format!("unknown viewport plane mode '{mode}'")),
                };
            }
            if let Some(value) = saved_viewport.x_level0 {
                state.current_x_level0 = value;
            }
            if let Some(value) = saved_viewport.y_level0 {
                state.current_y_level0 = value;
            }
            if let Some(value) = saved_viewport.z_level0 {
                state.current_z_level0 = value;
            }
            if !saved_viewport.channel_order.is_empty() {
                state.channel_layer_order = saved_viewport.channel_order.clone();
            }
            for (index, channel) in state.channels.iter_mut().enumerate() {
                let saved_channel = saved_viewport
                    .channels
                    .iter()
                    .find(|candidate| candidate.name.as_deref() == Some(channel.name.as_str()))
                    .or_else(|| saved_viewport.channels.get(index));
                let Some(saved_channel) = saved_channel else {
                    continue;
                };
                if let Some(visible) = saved_channel.visible {
                    channel.visible = visible;
                }
                if let Some(color) = saved_channel.color_rgb {
                    channel.color_rgb = color;
                }
                if let Some([lo, hi]) = saved_channel.window {
                    channel.window = Some((lo, hi));
                }
            }
            if let Some(active_channel) = saved_viewport.active_channel {
                state.selected_channel = active_channel.min(state.channels.len().saturating_sub(1));
            }
            if let Some(segmentation) = saved_viewport.segmentation.as_ref() {
                if let Some(color) = segmentation.outlines_color_rgb {
                    state.cells_outlines_color_rgb = color;
                }
                if let Some(opacity) = segmentation.outlines_opacity {
                    state.cells_outlines_opacity = opacity;
                }
                if let Some(width) = segmentation.outlines_width_px {
                    state.cells_outlines_width_px = width;
                }
                if let Some(display) = segmentation.object_display.as_ref() {
                    state.object_display = display.clone();
                }
            }
            if let Some(filter) = saved_viewport.object_filter.as_ref() {
                state.object_filter = ObjectViewportFilterState::from_project_json(filter)?;
                state.object_filter_cache = ObjectViewportFilterCacheState::empty();
            }
            if let Some(value) = saved_viewport.object_visible {
                state.object_visible = value;
            }
            if let Some(value) = saved_viewport.object_opacity {
                state.object_opacity = value.clamp(0.0, 1.0);
            }
            if let Some(value) = saved_viewport.object_width_screen_px {
                state.object_width_screen_px = value.max(0.0);
            }
            if let Some(value) = saved_viewport.object_color_rgb {
                state.object_color_rgb = value;
            }
            if let Some(value) = saved_viewport.object_show_selection_overlay {
                state.object_show_selection_overlay = value;
            }
            if let Some(presentation) = saved_viewport.presentation.as_ref() {
                state.apply_project_presentation_json(presentation)?;
            }
            if !saved_viewport.overlay_order.is_empty() {
                state.overlay_layer_order = saved_viewport
                    .overlay_order
                    .iter()
                    .filter_map(|key| self.parse_layer_id_storage_key(key))
                    .collect();
            }
            for (key, visible) in &saved_viewport.overlay_visibility {
                if let Some(layer_id) = self.parse_layer_id_storage_key(key) {
                    state.set_layer_visible(layer_id, *visible);
                }
            }
            if let Some(active_layer) = saved_viewport
                .active_layer
                .as_deref()
                .and_then(|key| self.parse_layer_id_storage_key(key))
            {
                state.active_layer = active_layer;
            }
            slots.push(ViewportSlot {
                id,
                title: saved_viewport.title.clone(),
                state,
                navigation_revision: saved_viewport.navigation_revision.max(1),
                presentation_revision: saved_viewport.presentation_revision.max(1),
            });
        }
        let active =
            ViewportId::new(&saved.active_viewport_id).map_err(|error| error.to_string())?;
        let mut workspace = ViewportWorkspace::restore(
            slots,
            active,
            layout,
            ViewportLinks {
                camera: saved.link_camera,
                plane: saved.link_plane,
                // Selection identity is document-owned in workspace v1, so a
                // stale or hand-edited false value cannot create a misleading
                // apparently-independent selection mode.
                selection: true,
            },
        )
        .map_err(|error| error.to_string())?;
        workspace
            .set_split_ratio(saved.split_ratio)
            .map_err(|error| error.to_string())?;
        workspace.active().state.apply(self);
        self.viewport_workspace = Some(workspace);
        self.bump_render_id();
        Ok(())
    }

    pub(super) fn sync_current_view_state_into_project_space(&mut self) {
        self.sync_runtime_to_active_viewport();
        self.sync_mask_layers_into_project_space();
        self.ensure_loaded_layer_offset_baselines();
        let layer_groups = Some(self.current_layer_groups());
        let overlay_order = self
            .overlay_layer_order
            .iter()
            .copied()
            .map(Self::layer_id_storage_key)
            .collect::<Vec<_>>();
        let overlay_visibility = self
            .overlay_layer_order
            .iter()
            .copied()
            .filter_map(|id| {
                self.layer_visible_value(id)
                    .map(|visible| (Self::layer_id_storage_key(id), visible))
            })
            .collect::<BTreeMap<_, _>>();
        let overlay_offsets_world = self
            .overlay_layer_order
            .iter()
            .copied()
            .filter_map(|id| {
                let off = self.layer_offset_world(id);
                ((off.x.abs() > 1e-6) || (off.y.abs() > 1e-6))
                    .then(|| (Self::layer_id_storage_key(id), [off.x, off.y]))
            })
            .collect::<BTreeMap<_, _>>();
        let overlay_original_offsets_world = self
            .overlay_layer_order
            .iter()
            .copied()
            .filter_map(|id| {
                let current = self.layer_offset_world(id);
                self.loaded_layer_offsets_world
                    .get(&id)
                    .copied()
                    .filter(|baseline| layer_offsets_differ(current, *baseline))
                    .map(|baseline| (Self::layer_id_storage_key(id), vec2_to_array(baseline)))
            })
            .collect::<BTreeMap<_, _>>();
        let workspace = self.project_workspace_view_state();
        self.project_space.set_roi_view_state(
            &self.dataset.source,
            ProjectRoiViewState {
                layer_groups,
                channel_order: self.channel_layer_order.clone(),
                channels: self
                    .channels
                    .iter()
                    .enumerate()
                    .map(|(idx, ch)| ProjectChannelViewState {
                        name: Some(ch.name.clone()),
                        visible: Some(ch.visible),
                        color_rgb: Some(ch.color_rgb),
                        window: ch.window.map(|(lo, hi)| [lo, hi]),
                        offset_world: self
                            .channel_offsets_world
                            .get(idx)
                            .map(|off| [off.x, off.y]),
                        original_offset_world: self
                            .channel_offsets_world
                            .get(idx)
                            .and_then(|current| {
                                self.loaded_layer_offsets_world
                                    .get(&LayerId::Channel(idx))
                                    .copied()
                                    .filter(|baseline| layer_offsets_differ(*current, *baseline))
                            })
                            .map(vec2_to_array),
                        scale: self.channel_scales.get(idx).map(|scale| [scale.x, scale.y]),
                        rotation_rad: self.channel_rotations_rad.get(idx).copied(),
                        note: (!ch.note.is_empty()).then(|| ch.note.clone()),
                    })
                    .collect(),
                active_channel: Some(self.selected_channel),
                active_layer: Some(Self::layer_id_storage_key(self.active_layer)),
                overlay_order,
                overlay_visibility,
                overlay_offsets_world,
                overlay_original_offsets_world,
                segmentation: Some(ProjectSegmentationViewState {
                    label_name: (!self.seg_label_selected.is_empty())
                        .then(|| self.seg_label_selected.clone()),
                    outlines_color_rgb: Some(self.cells_outlines_color_rgb),
                    outlines_opacity: Some(self.cells_outlines_opacity),
                    outlines_width_px: Some(self.cells_outlines_width_px),
                    object_display: Some(self.seg_objects.project_display_state()),
                }),
                analysis: Some(self.seg_objects.project_analysis_state()),
                camera: Some(self.project_camera_state()),
                ui: Some(self.project_ui_state()),
                annotation_layers: self
                    .annotation_layers
                    .iter()
                    .map(|layer| self.project_annotation_layer_state(layer))
                    .collect(),
                workspace,
            },
        );
    }

    pub(super) fn current_layer_groups(&self) -> ProjectLayerGroups {
        self.viewport_layer_groups.clone()
    }

    pub(super) fn set_current_layer_groups(&mut self, groups: ProjectLayerGroups) {
        self.viewport_layer_groups.clone_from(&groups);
        self.persist_current_layer_groups(groups);
    }

    pub(super) fn commit_current_channel_groups(&mut self, groups: ProjectLayerGroups) -> bool {
        if self.native_viewport_actor_owned()
            && let Some((viewport_id, revision)) = self.active_viewport_command_scope()
        {
            self.submit_native_viewport_intent(
                "viewer.viewports.channels.set_group",
                serde_json::json!({
                    "viewport_id":viewport_id,
                    "if_presentation_revision":revision,
                    "replace_all":true,
                    "groups":channel_groups_snapshot(&groups, &self.channels),
                }),
            );
            self.persist_current_layer_groups(groups);
            true
        } else {
            self.set_current_layer_groups(groups);
            false
        }
    }

    pub(super) fn persist_current_layer_groups(&mut self, groups: ProjectLayerGroups) {
        let mut view = self
            .project_space
            .roi_view_state(&self.dataset.source)
            .cloned()
            .unwrap_or_default();
        view.layer_groups = Some(groups);
        self.project_space
            .set_roi_view_state(&self.dataset.source, view);
    }

    pub(super) fn apply_view_state_from_project_space(&mut self) {
        let saved_view = self
            .project_space
            .roi_view_state(&self.dataset.source)
            .cloned();
        if let Some(view) = saved_view.as_ref() {
            self.viewport_layer_groups = view.layer_groups.clone().unwrap_or_default();
            self.channel_window_overrides.clear();
            if let Some(ui) = view.ui.as_ref() {
                self.apply_project_ui_state(ui);
            }
            if !view.channel_order.is_empty() {
                self.channel_layer_order = view.channel_order.clone();
            }
            let channel_notes_by_name = view
                .channels
                .iter()
                .filter_map(|saved| {
                    let name = saved.name.as_deref()?;
                    let note = saved.note.as_ref()?;
                    Some((name, note))
                })
                .collect::<HashMap<_, _>>();
            for (idx, saved) in view.channels.iter().enumerate() {
                let Some(ch) = self.channels.get_mut(idx) else {
                    continue;
                };
                if let Some(note) = channel_notes_by_name.get(ch.name.as_str()) {
                    ch.note = (*note).clone();
                }
                if let Some(visible) = saved.visible {
                    ch.visible = visible;
                }
                if let Some(color_rgb) = saved.color_rgb {
                    ch.color_rgb = color_rgb;
                }
                if let Some([lo, hi]) = saved.window {
                    ch.window = Some((lo, hi));
                    self.channel_window_overrides
                        .insert(ch.name.clone(), (lo, hi));
                }
                if let Some([x, y]) = saved.offset_world
                    && let Some(dst) = self.channel_offsets_world.get_mut(idx)
                {
                    *dst = egui::vec2(x, y);
                }
                if let Some([x, y]) = saved.scale
                    && let Some(dst) = self.channel_scales.get_mut(idx)
                {
                    *dst = egui::vec2(x, y);
                }
                if let Some(rotation_rad) = saved.rotation_rad
                    && let Some(dst) = self.channel_rotations_rad.get_mut(idx)
                {
                    *dst = rotation_rad;
                }
            }
            if let Some(segmentation) = view.segmentation.as_ref() {
                if let Some(label_name) = segmentation.label_name.as_ref() {
                    self.seg_label_selected = label_name.clone();
                    self.seg_label_input = self.seg_label_selected.clone();
                }
                if let Some(outlines_color_rgb) = segmentation.outlines_color_rgb {
                    self.cells_outlines_color_rgb = outlines_color_rgb;
                }
                if let Some(outlines_opacity) = segmentation.outlines_opacity {
                    self.cells_outlines_opacity = outlines_opacity;
                }
                if let Some(outlines_width_px) = segmentation.outlines_width_px {
                    self.cells_outlines_width_px = outlines_width_px;
                }
                if let Some(object_display) = segmentation.object_display.as_ref() {
                    self.seg_objects.apply_project_display_state(object_display);
                } else {
                    self.seg_objects.clear_project_display_state();
                }
                self.seg_objects.fast_rendering = self.fast_object_rendering;
            } else {
                self.seg_objects.clear_project_display_state();
                self.seg_objects.fast_rendering = self.fast_object_rendering;
            }
            if !view.overlay_order.is_empty() {
                self.overlay_layer_order = view
                    .overlay_order
                    .iter()
                    .filter_map(|id| self.parse_layer_id_storage_key(id))
                    .collect();
            }
            if let Some(active_channel) = view.active_channel {
                self.selected_channel = active_channel.min(self.channels.len().saturating_sub(1));
            }
            if let Some(analysis) = view.analysis.as_ref() {
                let active_channel_name = self
                    .channels
                    .get(self.selected_channel)
                    .map(|channel| channel.name.as_str());
                self.seg_objects
                    .apply_project_analysis_state(analysis, active_channel_name);
            }
            if let Some(camera) = view.camera.as_ref() {
                self.apply_project_camera_state(camera);
            }
            self.restore_annotation_layers(&view.annotation_layers);
        }
        self.rebuild_layer_orders();
        if let Some(view) = saved_view.as_ref() {
            for (id, visible) in &view.overlay_visibility {
                if let Some(layer_id) = self.parse_layer_id_storage_key(id)
                    && let Some(dst) = self.layer_visible_mut(layer_id)
                {
                    *dst = *visible;
                }
            }
            for (id, [x, y]) in &view.overlay_offsets_world {
                if let Some(layer_id) = self.parse_layer_id_storage_key(id)
                    && let Some(dst) = self.layer_offset_world_mut(layer_id)
                {
                    *dst = egui::vec2(*x, *y);
                }
            }
            if let Some(active_layer) = view
                .active_layer
                .as_deref()
                .and_then(|id| self.parse_layer_id_storage_key(id))
            {
                self.set_active_layer(active_layer);
            } else if !self.channels.is_empty() {
                self.set_active_layer(LayerId::Channel(
                    self.selected_channel
                        .min(self.channels.len().saturating_sub(1)),
                ));
            }
        }
        if let Some(view) = saved_view.as_ref() {
            self.restore_loaded_layer_offsets_from_project_view(view);
        } else {
            self.capture_loaded_layer_offsets();
        }
        if let Some(workspace) = saved_view.as_ref().and_then(|view| view.workspace.as_ref()) {
            if let Err(error) = self.restore_project_workspace(workspace) {
                self.viewport_workspace =
                    Some(ViewportWorkspace::new(ViewerViewportState::capture(self)));
                self.set_status(format!(
                    "Could not restore multi-viewport workspace: {error}. Loaded the active view instead."
                ));
            }
        } else {
            // Migration path for all pre-workspace project files.
            self.viewport_workspace =
                Some(ViewportWorkspace::new(ViewerViewportState::capture(self)));
        }
    }

    pub fn take_project_space(&mut self) -> ProjectSpace {
        self.sync_current_view_state_into_project_space();
        std::mem::take(&mut self.project_space)
    }

    pub fn project_space(&self) -> &ProjectSpace {
        &self.project_space
    }

    pub fn project_space_mut(&mut self) -> &mut ProjectSpace {
        &mut self.project_space
    }
}
