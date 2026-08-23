use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_viewport_workspace(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let frame_plan_started = Instant::now();
        let mut workspace = self
            .viewport_workspace
            .take()
            .unwrap_or_else(|| ViewportWorkspace::new(ViewerViewportState::capture(self)));
        let active_id = workspace.active_id().clone();
        if let Some(active) = workspace.get_mut(&active_id) {
            let state = ViewerViewportState::capture(self);
            let navigation_changed =
                state.camera_changed_from(&active.state) || state.plane_changed_from(&active.state);
            let presentation_changed = state.presentation_changed_from(&active.state);
            active.state = state;
            if navigation_changed {
                active.navigation_revision = active.navigation_revision.wrapping_add(1).max(1);
            }
            if presentation_changed {
                active.presentation_revision = active.presentation_revision.wrapping_add(1).max(1);
            }
        }

        let viewport_ids = workspace
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        let multi_view = viewport_ids.len() > 1;
        let mut remove_requested = None;
        if multi_view {
            self.viewport_raw_active_keys = Some(HashSet::new());
            self.viewport_cpu_active_keys = Some(HashSet::new());
            self.viewport_label_active_keys = Some(HashSet::new());
            self.viewport_spatial_image_active_keys = Some(HashMap::new());
            self.loader
                .set_active_render_ids(Self::workspace_cpu_render_ids(&workspace));
        }

        match workspace.layout() {
            ViewportLayout::Single => {
                if let Some(id) = viewport_ids.first() {
                    if self.ui_viewport_cell(ui, ctx, &mut workspace, id) {
                        remove_requested = Some(id.clone());
                    }
                }
            }
            ViewportLayout::Horizontal => {
                let available = ui.available_size();
                let separator_width = ui.spacing().item_spacing.x.max(1.0);
                let content_width = (available.x - separator_width).max(2.0);
                let first_width = (content_width * workspace.split_ratio()).max(1.0);
                let second_width = (content_width - first_width).max(1.0);
                ui.horizontal(|ui| {
                    ui.allocate_ui_with_layout(
                        egui::vec2(first_width, available.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            if self.ui_viewport_cell(ui, ctx, &mut workspace, &viewport_ids[0]) {
                                remove_requested = Some(viewport_ids[0].clone());
                            }
                        },
                    );
                    ui.separator();
                    ui.allocate_ui_with_layout(
                        egui::vec2(second_width, available.y),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            if self.ui_viewport_cell(ui, ctx, &mut workspace, &viewport_ids[1]) {
                                remove_requested = Some(viewport_ids[1].clone());
                            }
                        },
                    );
                });
            }
            ViewportLayout::Vertical => {
                let available = ui.available_size();
                let separator_height = ui.spacing().item_spacing.y.max(1.0);
                let content_height = (available.y - separator_height).max(2.0);
                for (index, id) in viewport_ids.iter().enumerate() {
                    let cell_height = if index == 0 {
                        content_height * workspace.split_ratio()
                    } else {
                        content_height * (1.0 - workspace.split_ratio())
                    }
                    .max(1.0);
                    ui.allocate_ui_with_layout(
                        egui::vec2(available.x, cell_height),
                        egui::Layout::top_down(egui::Align::Min),
                        |ui| {
                            if self.ui_viewport_cell(ui, ctx, &mut workspace, id) {
                                remove_requested = Some(id.clone());
                            }
                        },
                    );
                    if index + 1 < viewport_ids.len() {
                        ui.separator();
                    }
                }
            }
        }

        if let Some(viewport_id) = remove_requested
            && workspace.len() > 1
            && workspace.remove(&viewport_id).is_ok()
        {
            self.cancel_viewport_transient_gestures();
            self.screenshot_pending
                .retain(|pending| pending.viewport_id != viewport_id);
        }

        if let Some(active_keys) = self.viewport_raw_active_keys.take() {
            if let Some(loader) = self.raw_loader.as_ref() {
                loader.set_active_keys(active_keys.clone());
            }
            if let Some(tiles_gl) = self.tiles_gl.as_ref() {
                tiles_gl.prune_in_flight(&active_keys);
            }
        }
        if let Some(active_keys) = self.viewport_cpu_active_keys.take() {
            self.loader.set_active_keys(active_keys.clone());
            self.cache.prune_in_flight(&active_keys);
        }
        if let Some(active_keys) = self.viewport_label_active_keys.take()
            && let Some(labels_gl) = self.labels_gl.as_ref()
        {
            labels_gl.prune_in_flight(&active_keys);
        }
        if let Some(mut active_keys_by_layer) = self.viewport_spatial_image_active_keys.take() {
            for layer in &self.spatial_image_layers.images {
                let active_keys = active_keys_by_layer.remove(&layer.id).unwrap_or_default();
                layer.prune_in_flight(&active_keys);
            }
        }

        if multi_view {
            self.loader
                .set_active_render_ids(Self::workspace_cpu_render_ids(&workspace));
        }

        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_frame_plan_ms = frame_plan_started.elapsed().as_secs_f32() * 1_000.0;
        self.viewport_frame_plan_ema_ms = if self.viewport_frame_plan_samples == 0 {
            self.viewport_frame_plan_ms
        } else {
            self.viewport_frame_plan_ema_ms * 0.9 + self.viewport_frame_plan_ms * 0.1
        };
        self.viewport_frame_plan_samples = self.viewport_frame_plan_samples.saturating_add(1);
        self.viewport_workspace = Some(workspace);
    }

    pub(super) fn workspace_cpu_render_ids(
        workspace: &ViewportWorkspace<ViewerViewportState>,
    ) -> HashSet<u64> {
        workspace
            .viewports()
            .iter()
            .flat_map(|viewport| {
                std::iter::once(viewport.state.active_render_id)
                    .chain(viewport.state.previous_render_id)
            })
            .filter(|render_id| *render_id != 0)
            .collect()
    }

    pub(super) fn cpu_render_id_is_current(&self, render_id: u64) -> bool {
        if render_id == self.active_render_id || self.previous_render_id == Some(render_id) {
            return true;
        }
        self.viewport_workspace.as_ref().is_some_and(|workspace| {
            workspace.viewports().iter().any(|viewport| {
                viewport.state.active_render_id == render_id
                    || viewport.state.previous_render_id == Some(render_id)
            })
        })
    }

    pub(super) fn smooth_pixels_for_render_id(&self, render_id: u64) -> bool {
        if self.active_render_id == render_id {
            return self.active_render_smooth_pixels;
        }
        if self.previous_render_id == Some(render_id) {
            return self
                .previous_render_smooth_pixels
                .unwrap_or(self.active_render_smooth_pixels);
        }
        self.viewport_workspace
            .as_ref()
            .and_then(|workspace| {
                workspace.viewports().iter().find_map(|viewport| {
                    (viewport.state.active_render_id == render_id)
                        .then_some(viewport.state.active_render_smooth_pixels)
                        .or_else(|| {
                            (viewport.state.previous_render_id == Some(render_id)).then_some(
                                viewport
                                    .state
                                    .previous_render_smooth_pixels
                                    .unwrap_or(viewport.state.active_render_smooth_pixels),
                            )
                        })
                })
            })
            .unwrap_or(self.smooth_pixels)
    }

    pub(super) fn ui_viewport_cell(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        workspace: &mut ViewportWorkspace<ViewerViewportState>,
        viewport_id: &ViewportId,
    ) -> bool {
        let Some(viewport) = workspace.get(viewport_id) else {
            return false;
        };
        let before = viewport.state.clone();
        let title = viewport.title.clone();
        let is_active = workspace.active_id() == viewport_id;
        self.native_viewport_command_scope = Some(NativeViewportCommandScope {
            viewport_id: viewport_id.as_str().to_string(),
            navigation_revision: viewport.navigation_revision.max(1),
            presentation_revision: viewport.presentation_revision.max(1),
        });

        before.apply(self);
        self.bump_render_id();

        let mut activate = false;
        let mut renamed_title = None;
        let mut close_requested = false;
        let header_height = ui.spacing().interact_size.y + 2.0;
        let indicator_width = 12.0;
        let title_width = 120.0;
        let title_edit_id = egui::Id::new(("viewport-title-edit", viewport_id.as_str()));
        let title_focus_id = egui::Id::new(("viewport-title-focus", viewport_id.as_str()));
        let title_buffer_id = egui::Id::new(("viewport-title-buffer", viewport_id.as_str()));
        let editing_title = ctx.data(|data| data.get_temp::<bool>(title_edit_id).unwrap_or(false));
        let (header_rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), header_height),
            egui::Sense::hover(),
        );
        let mut header_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(header_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        let (indicator_rect, _) = header_ui.allocate_exact_size(
            egui::vec2(indicator_width, header_height),
            egui::Sense::hover(),
        );
        if is_active && header_ui.is_rect_visible(indicator_rect) {
            header_ui.painter().circle_filled(
                indicator_rect.center(),
                3.0,
                header_ui.visuals().selection.stroke.color,
            );
        }
        if editing_title {
            let mut edited = ctx
                .data(|data| data.get_temp::<String>(title_buffer_id))
                .unwrap_or_else(|| title.clone());
            let response = header_ui.add_sized(
                [title_width, header_height],
                egui::TextEdit::singleline(&mut edited).hint_text("Viewport name"),
            );
            let focus_pending =
                ctx.data(|data| data.get_temp::<bool>(title_focus_id).unwrap_or(false));
            if focus_pending {
                response.request_focus();
                ctx.data_mut(|data| data.insert_temp(title_focus_id, false));
            }
            if response.changed() {
                ctx.data_mut(|data| data.insert_temp(title_buffer_id, edited.clone()));
            }
            let enter_pressed = response.has_focus()
                && header_ui.input(|input| input.key_pressed(egui::Key::Enter));
            let escape_pressed = response.has_focus()
                && header_ui.input(|input| input.key_pressed(egui::Key::Escape));
            let edit_finished =
                icon_button(&mut header_ui, Icon::Confirm, false, egui::Sense::click())
                    .on_hover_text("Finish renaming")
                    .clicked();
            if response.lost_focus() || enter_pressed || escape_pressed || edit_finished {
                if !escape_pressed && !edited.trim().is_empty() && edited.trim() != title {
                    renamed_title = Some(edited.trim().to_string());
                }
                ctx.data_mut(|data| {
                    data.insert_temp(title_edit_id, false);
                    data.remove_temp::<String>(title_buffer_id);
                });
            }
        } else {
            let title_text = if is_active {
                egui::RichText::new(title.clone()).strong()
            } else {
                egui::RichText::new(title.clone())
            };
            let title_response = header_ui
                .add_sized(
                    [title_width, header_height],
                    egui::Label::new(title_text)
                        .truncate()
                        .sense(egui::Sense::click()),
                )
                .on_hover_text(if is_active {
                    "Double-click to rename this viewport"
                } else {
                    "Make this the active viewport"
                });
            activate |= title_response.clicked();
            let edit_requested = title_response.double_clicked()
                || icon_button(&mut header_ui, Icon::Edit, false, egui::Sense::click())
                    .on_hover_text("Rename this viewport")
                    .clicked();
            if edit_requested {
                activate = true;
                ctx.data_mut(|data| {
                    data.insert_temp(title_edit_id, true);
                    data.insert_temp(title_focus_id, true);
                    data.insert_temp(title_buffer_id, title.clone());
                });
                ctx.request_repaint();
            }
        }
        if workspace.links().camera {
            header_ui.small("camera linked");
        }
        if workspace.links().plane && self.view_plane_modes().len() > 1 {
            header_ui.small("plane linked");
        }
        if workspace.len() > 1 {
            close_requested = header_ui
                .small_button("×")
                .on_hover_text("Close this viewport")
                .clicked();
        }
        if let Some(title) = renamed_title {
            let revision = workspace
                .get(viewport_id)
                .map(|viewport| viewport.presentation_revision)
                .unwrap_or(1);
            if self.native_viewport_actor_owned() {
                self.submit_native_viewport_intent(
                    "viewer.viewports.rename",
                    serde_json::json!({
                        "viewport_id":viewport_id.as_str(),
                        "if_presentation_revision":revision,
                        "title":title,
                    }),
                );
            } else if workspace.rename(viewport_id, title).unwrap_or(false) {
                let _ = workspace.bump_presentation_revision(viewport_id);
            }
        }
        ui.separator();

        let tool_mode = self.tool_mode;
        if !is_active {
            self.tool_mode = ToolMode::Pan;
        }
        let (raw_request_budget, cpu_request_budget) =
            viewport_image_request_budgets(workspace.len() > 1, is_active);
        activate |= self.ui_canvas(
            ui,
            ctx,
            viewport_id,
            is_active,
            raw_request_budget,
            cpu_request_budget,
        );
        self.tool_mode = tool_mode;

        let after = ViewerViewportState::capture(self);
        self.native_viewport_command_scope = None;
        if let Some(viewport) = workspace.get_mut(viewport_id) {
            viewport.state = after.clone();
        }

        let links = workspace.links();
        let camera_changed = after.camera_changed_from(&before);
        let plane_changed = after.plane_changed_from(&before);
        if camera_changed || plane_changed {
            let _ = workspace.bump_navigation_revision(viewport_id);
        }
        if after.presentation_changed_from(&before) {
            let _ = workspace.bump_presentation_revision(viewport_id);
        }
        if (links.camera && camera_changed) || (links.plane && plane_changed) {
            let other_ids = workspace
                .viewports()
                .iter()
                .filter(|viewport| viewport.id != *viewport_id)
                .map(|viewport| viewport.id.clone())
                .collect::<Vec<_>>();
            for other_id in other_ids {
                if let Some(other) = workspace.get_mut(&other_id) {
                    other.state.copy_linked_navigation_from(&after, links);
                }
                let _ = workspace.bump_navigation_revision(&other_id);
            }
        }

        if activate {
            if self.native_viewport_actor_owned() {
                if !is_active {
                    self.submit_native_viewport_intent(
                        "viewer.viewports.set_active",
                        serde_json::json!({"viewport_id":viewport_id.as_str()}),
                    );
                }
            } else if workspace.set_active(viewport_id).unwrap_or(false) {
                self.cancel_viewport_transient_gestures();
            }
        }
        if close_requested && self.native_viewport_actor_owned() {
            self.submit_native_viewport_intent(
                "viewer.viewports.remove",
                serde_json::json!({"viewport_id":viewport_id.as_str()}),
            );
            false
        } else {
            close_requested
        }
    }
}
