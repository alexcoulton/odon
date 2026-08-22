use super::*;

impl OmeZarrViewerApp {
    pub fn take_request(&mut self) -> Option<ViewerRequest> {
        self.pending_request.take()
    }

    pub fn take_native_control_intents(&mut self) -> Vec<NativeControlIntent> {
        std::mem::take(&mut self.native_control_intents)
    }

    pub(super) fn record_native_viewport_intents(
        &mut self,
        before: &ViewportWorkspace<ViewerViewportState>,
        after: &ViewportWorkspace<ViewerViewportState>,
    ) {
        let before_ids = before
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        let after_ids = after
            .viewports()
            .iter()
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();

        if after.len() == before.len() + 1 {
            if let Some(created) = after
                .viewports()
                .iter()
                .find(|viewport| before.get(&viewport.id).is_none())
            {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.clone",
                    params: serde_json::json!({
                        "viewport_id": before.active_id().as_str(),
                        "title": created.title,
                        "layout": after.layout().as_str(),
                        "ratio": after.split_ratio(),
                        "activate": created.id == *after.active_id(),
                    }),
                });
            }
        } else if after.len() + 1 == before.len() {
            if let Some(removed) = before
                .viewports()
                .iter()
                .find(|viewport| after.get(&viewport.id).is_none())
            {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.remove",
                    params: serde_json::json!({"viewport_id": removed.id.as_str()}),
                });
            }
        }

        if before.len() == after.len()
            && (before.layout() != after.layout()
                || (before.split_ratio() - after.split_ratio()).abs() > f32::EPSILON)
        {
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.workspace.layout.set",
                params: serde_json::json!({
                    "layout": after.layout().as_str(),
                    "ratio": after.split_ratio(),
                }),
            });
        }
        if before.len() == 2 && after.len() == 2 && before_ids.iter().eq(after_ids.iter().rev()) {
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.workspace.swap",
                params: serde_json::json!({}),
            });
        }
        if before.links() != after.links() {
            let links = after.links();
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.viewport_links.set",
                params: serde_json::json!({
                    "camera": links.camera,
                    "plane": links.plane,
                    "selection": links.selection,
                }),
            });
        }
        if before.active_id() != after.active_id() {
            self.native_control_intents.push(NativeControlIntent {
                method: "viewer.viewports.set_active",
                params: serde_json::json!({"viewport_id": after.active_id().as_str()}),
            });
        }

        for viewport in after.viewports() {
            let Some(previous) = before.get(&viewport.id) else {
                continue;
            };
            if previous.title != viewport.title {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.rename",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "title": viewport.title,
                    }),
                });
            }
            if viewport.state.camera_changed_from(&previous.state) {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.camera.set",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "center_world_lvl0": [
                            viewport.state.camera.center_world_lvl0.x,
                            viewport.state.camera.center_world_lvl0.y,
                        ],
                        "zoom_screen_per_lvl0_px": viewport.state.camera.zoom_screen_per_lvl0_px,
                    }),
                });
            }
            if viewport.state.plane_changed_from(&previous.state) {
                let slice = match viewport.state.view_plane_mode {
                    ViewPlaneMode::Xy => viewport.state.current_z_level0,
                    ViewPlaneMode::Xz => viewport.state.current_y_level0,
                    ViewPlaneMode::Yz => viewport.state.current_x_level0,
                };
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.planes.set",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "mode": viewport.state.view_plane_mode.label().to_ascii_lowercase(),
                        "slice": slice,
                    }),
                });
            }

            let visible_before = previous
                .state
                .channels
                .iter()
                .filter(|channel| channel.visible)
                .map(|channel| channel.index)
                .collect::<Vec<_>>();
            let visible_after = viewport
                .state
                .channels
                .iter()
                .filter(|channel| channel.visible)
                .map(|channel| channel.index)
                .collect::<Vec<_>>();
            if visible_before != visible_after {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.channels.set_visible",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "channels": visible_after,
                        "mode": "only",
                    }),
                });
            }
            if previous.state.selected_channel != viewport.state.selected_channel {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.channels.set_active",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "channel": viewport.state.selected_channel,
                    }),
                });
            }
            for channel in &viewport.state.channels {
                let Some(previous_channel) = previous
                    .state
                    .channels
                    .iter()
                    .find(|candidate| candidate.index == channel.index)
                else {
                    continue;
                };
                if previous_channel.color_rgb != channel.color_rgb {
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.viewports.channels.set_color",
                        params: serde_json::json!({
                            "viewport_id": viewport.id.as_str(),
                            "channel": channel.index,
                            "color_rgb": channel.color_rgb,
                        }),
                    });
                }
                if previous_channel.window != channel.window
                    && let Some((min, max)) = channel.window
                {
                    self.native_control_intents.push(NativeControlIntent {
                        method: "viewer.viewports.channels.set_contrast",
                        params: serde_json::json!({
                            "viewport_id": viewport.id.as_str(),
                            "channel": channel.index,
                            "min": min,
                            "max": max,
                        }),
                    });
                }
            }
            if previous.state.channel_layer_order != viewport.state.channel_layer_order {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.channels.set_order",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "channels": viewport.state.channel_layer_order,
                        "mode": "exact",
                    }),
                });
            }
            if previous.state.channel_sort_mode != viewport.state.channel_sort_mode {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.channels.set_order",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "sort": viewport.state.channel_sort_mode.storage_key(),
                    }),
                });
            }
            if previous.state.layer_groups != viewport.state.layer_groups {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.channels.set_group",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "replace_all": true,
                        "groups": channel_groups_snapshot(
                            &viewport.state.layer_groups,
                            &viewport.state.channels,
                        ),
                    }),
                });
            }
            if previous.state.object_display != viewport.state.object_display
                || previous.state.object_visible != viewport.state.object_visible
                || previous.state.object_opacity != viewport.state.object_opacity
                || previous.state.object_width_screen_px != viewport.state.object_width_screen_px
                || previous.state.object_color_rgb != viewport.state.object_color_rgb
                || previous.state.object_show_selection_overlay
                    != viewport.state.object_show_selection_overlay
            {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.objects.style.set",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "visible": viewport.state.object_visible,
                        "opacity": viewport.state.object_opacity,
                        "width_screen_px": viewport.state.object_width_screen_px,
                        "color_rgb": viewport.state.object_color_rgb,
                        "fill_cells": viewport.state.object_display.fill_cells,
                        "fill_opacity": viewport.state.object_display.fill_opacity,
                        "selected_fill_opacity": viewport.state.object_display.selected_fill_opacity,
                        "show_selection_overlay": viewport.state.object_show_selection_overlay,
                        "fast_rendering": viewport.state.object_display.fast_rendering,
                        "color_property": viewport.state.object_display.color_property_key,
                    }),
                });
            }
            if previous.state.object_display.color_level_overrides
                != viewport.state.object_display.color_level_overrides
                && viewport.state.object_display.color_property_key.is_some()
            {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.objects.legend.set",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "entries": viewport.state.object_display.color_level_overrides.iter().map(|(value, style)| serde_json::json!({
                            "value": value,
                            "visible": style.visible,
                            "color_rgb": style.color_rgb,
                        })).collect::<Vec<_>>(),
                    }),
                });
            }
            if previous.state.object_filter != viewport.state.object_filter {
                let mut params = viewport.state.object_filter.project_json();
                params
                    .as_object_mut()
                    .expect("viewport object filter projection is an object")
                    .insert(
                        "viewport_id".to_string(),
                        serde_json::json!(viewport.id.as_str()),
                    );
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.objects.filter.set",
                    params,
                });
            }
            if previous.state.smooth_pixels != viewport.state.smooth_pixels
                || previous.state.show_scale_bar != viewport.state.show_scale_bar
                || previous.state.show_hud != viewport.state.show_hud
                || previous.state.show_tile_debug != viewport.state.show_tile_debug
            {
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.viewports.rendering.set",
                    params: serde_json::json!({
                        "viewport_id": viewport.id.as_str(),
                        "smooth_pixels": viewport.state.smooth_pixels,
                        "show_scale_bar": viewport.state.show_scale_bar,
                        "show_hud": viewport.state.show_hud,
                        "show_tile_debug": viewport.state.show_tile_debug,
                    }),
                });
            }
        }
    }

    pub(super) fn split_active_viewport(
        &mut self,
        layout: ViewportLayout,
        title: Option<String>,
    ) -> Result<ViewportId, String> {
        self.cancel_viewport_transient_gestures();
        let mut workspace = self
            .viewport_workspace
            .take()
            .unwrap_or_else(|| ViewportWorkspace::new(ViewerViewportState::capture(self)));
        let source_id = workspace.active_id().clone();
        if let Some(source) = workspace.get_mut(&source_id) {
            source.state = ViewerViewportState::capture(self);
        }
        let result = workspace
            .clone_viewport(&source_id, title, layout)
            .map_err(|error| error.to_string());
        if result.is_ok() {
            workspace.active().state.apply(self);
        }
        self.viewport_workspace = Some(workspace);
        result
    }

    pub(super) fn remove_viewport(&mut self, viewport_id: &ViewportId) -> Result<(), String> {
        self.cancel_viewport_transient_gestures();
        let mut workspace = self
            .viewport_workspace
            .take()
            .unwrap_or_else(|| ViewportWorkspace::new(ViewerViewportState::capture(self)));
        let active_id = workspace.active_id().clone();
        if let Some(active) = workspace.get_mut(&active_id) {
            active.state = ViewerViewportState::capture(self);
        }
        let result = workspace
            .remove(viewport_id)
            .map(|_| ())
            .map_err(|error| error.to_string());
        if result.is_ok() {
            self.screenshot_pending
                .retain(|pending| pending.viewport_id != *viewport_id);
            self.loader
                .set_active_render_ids(Self::workspace_cpu_render_ids(&workspace));
        }
        workspace.active().state.apply(self);
        self.bump_render_id();
        self.viewport_workspace = Some(workspace);
        result
    }

    pub(super) fn set_viewport_layout(&mut self, layout: ViewportLayout) -> Result<bool, String> {
        let Some(workspace) = self.viewport_workspace.as_mut() else {
            return Err("viewer workspace is not initialized".to_string());
        };
        workspace
            .set_layout(layout)
            .map_err(|error| error.to_string())
    }

    pub(super) fn set_viewport_links(&mut self, links: ViewportLinks) -> bool {
        let Some(mut workspace) = self.viewport_workspace.take() else {
            return false;
        };
        let active_id = workspace.active_id().clone();
        if let Some(active) = workspace.get_mut(&active_id) {
            active.state = ViewerViewportState::capture(self);
        }
        let active_state = workspace.active().state.clone();
        let other_ids = workspace
            .viewports()
            .iter()
            .filter(|viewport| viewport.id != active_id)
            .map(|viewport| viewport.id.clone())
            .collect::<Vec<_>>();
        for id in other_ids {
            if let Some(viewport) = workspace.get_mut(&id) {
                let before = viewport.state.clone();
                viewport
                    .state
                    .copy_linked_navigation_from(&active_state, links);
                if viewport.state.camera_changed_from(&before)
                    || viewport.state.plane_changed_from(&before)
                {
                    viewport.navigation_revision =
                        viewport.navigation_revision.wrapping_add(1).max(1);
                }
            }
        }
        let changed = workspace.set_links(links);
        self.viewport_workspace = Some(workspace);
        changed
    }

    pub(super) fn ui_viewport_controls(&mut self, ui: &mut egui::Ui) {
        let Some(workspace) = self.viewport_workspace.as_ref() else {
            return;
        };
        let viewport_count = workspace.len();
        let layout = workspace.layout();
        let split_ratio = workspace.split_ratio();
        let links = workspace.links();
        let active_id = workspace.active_id().clone();

        ui.menu_button(format!("Views ({viewport_count})"), |ui| {
            if viewport_count == 1 {
                if ui.button("Split side by side").clicked() {
                    let _ = self.split_active_viewport(ViewportLayout::Horizontal, None);
                    ui.close();
                }
                if ui.button("Split top and bottom").clicked() {
                    let _ = self.split_active_viewport(ViewportLayout::Vertical, None);
                    ui.close();
                }
                return;
            }

            ui.label("Layout");
            if ui
                .selectable_label(layout == ViewportLayout::Horizontal, "Side by side")
                .clicked()
            {
                let _ = self.set_viewport_layout(ViewportLayout::Horizontal);
            }
            if ui
                .selectable_label(layout == ViewportLayout::Vertical, "Top and bottom")
                .clicked()
            {
                let _ = self.set_viewport_layout(ViewportLayout::Vertical);
            }
            if ui.button("Swap positions").clicked() {
                if let Some(workspace) = self.viewport_workspace.as_mut() {
                    workspace.swap_order();
                }
            }
            let mut next_ratio = split_ratio;
            if ui
                .add(
                    egui::Slider::new(&mut next_ratio, 0.1..=0.9)
                        .text("Split ratio")
                        .fixed_decimals(2),
                )
                .changed()
                && let Some(workspace) = self.viewport_workspace.as_mut()
            {
                let _ = workspace.set_split_ratio(next_ratio);
            }

            ui.separator();
            ui.label("Linked state");
            let mut next_links = links;
            let mut links_changed = false;
            links_changed |= ui.checkbox(&mut next_links.camera, "Camera").changed();
            links_changed |= ui.checkbox(&mut next_links.plane, "Plane").changed();
            links_changed |= ui
                .add_enabled(
                    false,
                    egui::Checkbox::new(&mut next_links.selection, "Selection"),
                )
                .on_hover_text("Object selection is shared by the document")
                .changed();
            if links_changed {
                self.set_viewport_links(next_links);
            }

            ui.separator();
            if ui.button("Close active view").clicked() {
                let _ = self.remove_viewport(&active_id);
                ui.close();
            }
        });
    }

    pub(super) fn parse_viewport_id(params: &serde_json::Value) -> Result<ViewportId, String> {
        let value = params
            .get("viewport_id")
            .or_else(|| params.get("id"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "viewport_id is required".to_string())?;
        ViewportId::new(value).map_err(|error| error.to_string())
    }

    pub(super) fn parse_viewport_split_ratio(
        params: &serde_json::Value,
    ) -> Result<Option<f32>, String> {
        let Some(value) = params.get("ratio") else {
            return Ok(None);
        };
        let ratio = value
            .as_f64()
            .ok_or_else(|| "ratio must be a number".to_string())? as f32;
        if !ratio.is_finite() || !(0.1..=0.9).contains(&ratio) {
            return Err("ratio must be finite and between 0.1 and 0.9".to_string());
        }
        Ok(Some(ratio))
    }

    pub(super) fn sync_runtime_to_active_viewport(&mut self) {
        let state = ViewerViewportState::capture(self);
        if let Some(workspace) = self.viewport_workspace.as_mut() {
            let active_id = workspace.active_id().clone();
            let previous = workspace.active().state.clone();
            workspace.active_mut().state = state.clone();
            if state.camera_changed_from(&previous) || state.plane_changed_from(&previous) {
                let _ = workspace.bump_navigation_revision(&active_id);
            }
            if state.presentation_changed_from(&previous) {
                let _ = workspace.bump_presentation_revision(&active_id);
            }
        } else {
            self.viewport_workspace = Some(ViewportWorkspace::new(state));
        }
    }

    pub(super) fn viewport_state_snapshot(
        viewport_id: &ViewportId,
        title: &str,
        active: bool,
        navigation_revision: u64,
        presentation_revision: u64,
        state: &ViewerViewportState,
    ) -> serde_json::Value {
        let slice = match state.view_plane_mode {
            ViewPlaneMode::Xy => state.current_z_level0,
            ViewPlaneMode::Xz => state.current_y_level0,
            ViewPlaneMode::Yz => state.current_x_level0,
        };
        let channels = state
            .channels
            .iter()
            .enumerate()
            .map(|(index, channel)| {
                serde_json::json!({
                    "index": index,
                    "name": channel.name,
                    "visible": channel.visible,
                    "selected": index == state.selected_channel,
                    "color_rgb": channel.color_rgb,
                    "window": channel.window.map(|(min, max)| serde_json::json!({
                        "min": min,
                        "max": max,
                    })),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "viewport_id": viewport_id.as_str(),
            "title": title,
            "active": active,
            "navigation_revision": navigation_revision,
            "presentation_revision": presentation_revision,
            "camera": {
                "center_world_lvl0": [
                    state.camera.center_world_lvl0.x,
                    state.camera.center_world_lvl0.y,
                ],
                "zoom_screen_per_lvl0_px": state.camera.zoom_screen_per_lvl0_px,
                "viewport": state.last_canvas_rect.map(|rect| [
                    rect.min.x,
                    rect.min.y,
                    rect.max.x,
                    rect.max.y,
                ]),
            },
            "plane": {
                "mode": state.view_plane_mode.label().to_ascii_lowercase(),
                "slice": slice,
            },
            "channels": channels,
            "channel_order": state.channel_layer_order,
            "channel_sort": state.channel_sort_mode.storage_key(),
            "channel_groups": channel_groups_snapshot(&state.layer_groups, &state.channels),
            "objects": {
                "visible": state.object_visible,
                "opacity": state.object_opacity,
                "width_screen_px": state.object_width_screen_px,
                "color_rgb": state.object_color_rgb,
                "fill_cells": state.object_display.fill_cells,
                "fill_opacity": state.object_display.fill_opacity,
                "selected_fill_opacity": state.object_display.selected_fill_opacity,
                "show_selection_overlay": state.object_show_selection_overlay,
                "fast_rendering": state.object_display.fast_rendering,
                "color_property": state.object_display.color_property_key,
                "color_level_overrides": state.object_display.color_level_overrides,
                "filter": state.object_filter.project_json(),
            },
            "object_overlay_visibility": {
                "segmentation_labels": state.cells_outlines_visible,
                "segmentation_geojson": state.seg_geojson_visible,
            },
            "rendering": {
                "smooth_pixels": state.smooth_pixels,
                "show_scale_bar": state.show_scale_bar,
                "show_hud": state.show_hud,
                "show_tile_debug": state.show_tile_debug,
            },
        })
    }
}
