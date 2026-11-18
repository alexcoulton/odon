use super::*;

impl OmeZarrViewerApp {
    pub(super) fn submit_native_viewport_intent(
        &mut self,
        method: &'static str,
        params: serde_json::Value,
    ) -> bool {
        self.native_command_ingress
            .push(NativeControlIntent { method, params });
        true
    }

    pub(super) fn submit_native_active_viewport_rendering(
        &mut self,
        smooth_pixels: bool,
        show_scale_bar: bool,
        show_hud: bool,
        show_tile_debug: bool,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_command_scope() else {
            return false;
        };
        self.submit_native_viewport_intent(
            "viewer.viewports.rendering.set",
            serde_json::json!({
                "viewport_id":viewport_id,
                "if_presentation_revision":revision,
                "smooth_pixels":smooth_pixels,
                "show_scale_bar":show_scale_bar,
                "show_hud":show_hud,
                "show_tile_debug":show_tile_debug,
            }),
        )
    }

    pub(crate) fn submit_native_scale_bar_visibility(&mut self, show: bool) -> bool {
        self.submit_native_active_viewport_rendering(
            self.smooth_pixels,
            show,
            self.show_hud,
            self.show_tile_debug,
        )
    }

    pub(crate) fn scale_bar_visible(&self) -> bool {
        self.show_scale_bar
    }

    pub(super) fn submit_native_active_viewport_plane(
        &mut self,
        mode: ViewPlaneMode,
        slice: Option<u64>,
    ) -> bool {
        let Some((viewport_id, revision)) = self.active_viewport_navigation_scope() else {
            return false;
        };
        let mut params = serde_json::json!({
            "viewport_id":viewport_id,
            "if_navigation_revision":revision,
            "mode":mode.label().to_ascii_lowercase(),
        });
        if let Some(slice) = slice {
            params["slice"] = serde_json::json!(slice);
        }
        self.submit_native_viewport_intent("viewer.viewports.planes.set", params)
    }

    pub(super) fn active_viewport_navigation_scope(&self) -> Option<(String, u64)> {
        if let Some(scope) = self.native_viewport_command_scope.as_ref() {
            return Some((scope.viewport_id.clone(), scope.navigation_revision.max(1)));
        }
        let workspace = self.viewport_workspace.as_ref()?;
        let active = workspace.active();
        Some((
            active.id.as_str().to_string(),
            active.navigation_revision.max(1),
        ))
    }

    /// Submits an explicit camera target (fit, zoom-to-object). The actor's projection applies it,
    /// so any gesture hold is released first.
    pub(super) fn submit_native_camera(&mut self, camera: &Camera) -> bool {
        self.native_camera_hold = None;
        let Some((viewport_id, revision)) = self.active_viewport_navigation_scope() else {
            return false;
        };
        self.submit_native_viewport_intent(
            "viewer.viewports.camera.set",
            serde_json::json!({
                "viewport_id":viewport_id,
                "if_navigation_revision":revision,
                "center_world_lvl0":[
                    camera.center_world_lvl0.x,
                    camera.center_world_lvl0.y,
                ],
                "zoom_screen_per_lvl0_px":camera.zoom_screen_per_lvl0_px,
            }),
        )
    }

    /// Reports a pan/zoom step that was already applied to the local camera.
    ///
    /// Gesture steps are last-write-wins: they carry no revision precondition, only the newest
    /// position is kept while one command is in flight, and the local camera stays authoritative
    /// until the actor has caught up. A late projection therefore cannot reject or rewind a step.
    pub(super) fn record_native_camera_gesture(&mut self) {
        let Some((viewport_id, _)) = self.active_viewport_navigation_scope() else {
            return;
        };
        let camera = self.camera.clone();
        match self.native_camera_hold.as_mut() {
            Some(held) if held.viewport_id == viewport_id => held.hold.record_local(camera),
            _ => {
                self.native_camera_hold = Some(NativeCameraHold {
                    viewport_id,
                    hold: crate::camera::CameraIntentHold::new(camera),
                });
            }
        }
        self.flush_native_camera_hold();
    }

    /// Drops a hold that settled while the renderer was idle, so the next projection applies.
    pub(super) fn release_settled_native_camera_hold(&mut self) {
        let in_flight = self.native_camera_command_in_flight();
        if self
            .native_camera_hold
            .as_ref()
            .is_some_and(|held| held.hold.settled(in_flight) && !held.hold.needs_wake())
        {
            self.native_camera_hold = None;
        }
    }

    fn native_camera_command_in_flight(&self) -> bool {
        self.native_command_ingress
            .contains_pending("viewer.viewports.camera.set")
    }

    pub(super) fn flush_native_camera_hold(&mut self) {
        let in_flight = self.native_camera_command_in_flight();
        let Some(held) = self.native_camera_hold.as_mut() else {
            return;
        };
        let Some(camera) = held.hold.take_unsent(in_flight) else {
            return;
        };
        let params = serde_json::json!({
            "viewport_id":held.viewport_id,
            "center_world_lvl0":[
                camera.center_world_lvl0.x,
                camera.center_world_lvl0.y,
            ],
            "zoom_screen_per_lvl0_px":camera.zoom_screen_per_lvl0_px,
        });
        if !self.native_command_ingress.push(NativeControlIntent {
            method: "viewer.viewports.camera.set",
            params,
        }) && let Some(held) = self.native_camera_hold.as_mut()
        {
            held.hold.mark_unsent();
        }
    }

    /// Sends the newest unsent gesture camera and releases the hold once the actor caught up.
    pub(super) fn tick_native_camera_hold(&mut self, ctx: &egui::Context) {
        self.flush_native_camera_hold();
        let in_flight = self.native_camera_command_in_flight();
        let Some(held) = self.native_camera_hold.as_ref() else {
            return;
        };
        if !held.hold.settled(in_flight) {
            if held.hold.needs_wake() {
                ctx.request_repaint_after(crate::camera::CameraIntentHold::POLL);
            }
            return;
        }
        let Some(held) = self.native_camera_hold.take() else {
            return;
        };
        let viewport_id = held.viewport_id;
        let Some(camera) = held.hold.finish() else {
            return;
        };
        // The actor published a different camera while the gesture owned it; adopt it now.
        if let Some(workspace) = self.viewport_workspace.as_mut() {
            let linked = workspace.links().camera;
            let active = workspace.active_id().as_str() == viewport_id;
            for slot in workspace.viewports_mut() {
                if linked || slot.id.as_str() == viewport_id {
                    slot.state.camera = camera.clone();
                }
            }
            if linked || active {
                self.camera = camera;
            }
        }
        ctx.request_repaint();
    }

    pub(super) fn submit_native_object_filter_at(
        &mut self,
        viewport_id: &str,
        revision: u64,
        filter: &ObjectViewportFilterState,
    ) -> bool {
        let mut params = filter.project_json();
        let object = params
            .as_object_mut()
            .expect("viewport object filter projection is an object");
        object.insert("viewport_id".to_string(), serde_json::json!(viewport_id));
        object.insert(
            "if_presentation_revision".to_string(),
            serde_json::json!(revision.max(1)),
        );
        self.submit_native_viewport_intent("viewer.viewports.objects.filter.set", params)
    }

    pub fn take_platform_effect(&mut self) -> Option<ViewerPlatformEffect> {
        self.pending_platform_effect.take()
    }

    #[cfg(test)]
    pub fn take_native_control_intents(&mut self) -> Vec<NativeControlIntent> {
        self.native_command_ingress.take_recorded()
    }

    pub fn set_native_command_ingress(&mut self, ingress: NativeControlIngress) {
        self.native_command_ingress = ingress;
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
                    self.submit_native_viewport_intent(
                        "viewer.viewports.clone",
                        serde_json::json!({
                            "viewport_id":active_id.as_str(),
                            "layout":"horizontal",
                            "ratio":split_ratio,
                            "activate":true,
                        }),
                    );
                    ui.close();
                }
                if ui.button("Split top and bottom").clicked() {
                    self.submit_native_viewport_intent(
                        "viewer.viewports.clone",
                        serde_json::json!({
                            "viewport_id":active_id.as_str(),
                            "layout":"vertical",
                            "ratio":split_ratio,
                            "activate":true,
                        }),
                    );
                    ui.close();
                }
                return;
            }

            ui.label("Layout");
            if ui
                .selectable_label(layout == ViewportLayout::Horizontal, "Side by side")
                .clicked()
            {
                self.submit_native_viewport_intent(
                    "viewer.workspace.layout.set",
                    serde_json::json!({"layout":"horizontal","ratio":split_ratio}),
                );
            }
            if ui
                .selectable_label(layout == ViewportLayout::Vertical, "Top and bottom")
                .clicked()
            {
                self.submit_native_viewport_intent(
                    "viewer.workspace.layout.set",
                    serde_json::json!({"layout":"vertical","ratio":split_ratio}),
                );
            }
            if ui.button("Swap positions").clicked() {
                self.submit_native_viewport_intent("viewer.workspace.swap", serde_json::json!({}));
            }
            let mut next_ratio = split_ratio;
            if ui
                .add(
                    egui::Slider::new(&mut next_ratio, 0.1..=0.9)
                        .text("Split ratio")
                        .fixed_decimals(2),
                )
                .changed()
            {
                self.submit_native_viewport_intent(
                    "viewer.workspace.layout.set",
                    serde_json::json!({"layout":layout.as_str(),"ratio":next_ratio}),
                );
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
                self.submit_native_viewport_intent(
                    "viewer.viewport_links.set",
                    serde_json::json!({
                        "camera":next_links.camera,
                        "plane":next_links.plane,
                        "selection":next_links.selection,
                    }),
                );
            }

            ui.separator();
            if ui.button("Close active view").clicked() {
                self.submit_native_viewport_intent(
                    "viewer.viewports.remove",
                    serde_json::json!({"viewport_id":active_id.as_str()}),
                );
                ui.close();
            }
        });
    }

    #[cfg(test)]
    pub(super) fn parse_viewport_id(params: &serde_json::Value) -> Result<ViewportId, String> {
        let value = params
            .get("viewport_id")
            .or_else(|| params.get("id"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "viewport_id is required".to_string())?;
        ViewportId::new(value).map_err(|error| error.to_string())
    }

    pub(super) fn sync_runtime_to_active_viewport(&mut self) {
        if let Some(mut workspace) = self.viewport_workspace.take() {
            workspace.active_mut().state.capture_runtime(self);
            self.viewport_workspace = Some(workspace);
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
                "viewport": state.render.last_canvas_rect.map(|rect| [
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
                "unticked_categories": state.object_display.unticked_categories,
                "show_selection_overlay": state.object_show_selection_overlay,
                "fast_rendering": state.object_display.fast_rendering,
                "color_property": state.object_display.color_property_key,
                "color_mapping": state.object_display.color_mapping,
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
