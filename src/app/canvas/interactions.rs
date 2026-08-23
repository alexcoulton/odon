use super::*;

impl OmeZarrViewerApp {
    pub(super) fn handle_canvas_interactions(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        rect: egui::Rect,
        response: &egui::Response,
        allows_edits: bool,
    ) -> bool {
        let space_down = ctx.input(|i| i.key_down(egui::Key::Space));

        // Gesture handling happens before any drawing so the frame is rendered from a coherent
        // camera/tool snapshot. Each tool owns a separate interaction path here; later code only
        // consumes the resulting state.
        let mut closed_mask_polygon_this_frame = false;
        if response.double_clicked() {
            match self.tool_mode {
                ToolMode::Select => {}
                ToolMode::Pan => self.fit_to_rect(rect),
                ToolMode::MoveLayer => self.fit_to_rect(rect),
                ToolMode::TransformLayer => self.fit_to_rect(rect),
                ToolMode::DrawMaskPolygon => {
                    closed_mask_polygon_this_frame = self.finish_drawing_mask_polygon();
                }
                ToolMode::LassoSelect => {}
            }
        }

        // Camera navigation is global. It is intentionally independent from layer/tool logic so
        // panning and zooming behave consistently across all content types.
        let mut camera_changed = false;
        if response.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            let pinch = ui.input(|i| i.zoom_delta());
            if scroll != 0.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let factor = (scroll * 0.0015).exp();
                    self.camera.zoom_about_screen_point(rect, pos, factor);
                    camera_changed = true;
                }
            }
            if pinch != 1.0 {
                if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                    self.camera.zoom_about_screen_point(rect, pos, pinch);
                    camera_changed = true;
                }
            }
        }

        if self.tool_mode == ToolMode::DrawMaskPolygon
            && !space_down
            && !closed_mask_polygon_this_frame
            && response.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                if let Some(id) = self
                    .drawing_mask_layer
                    .or_else(|| self.ensure_editable_mask_layer())
                {
                    self.drawing_mask_layer = Some(id);
                    let off = self.layer_offset_world(LayerId::Mask(id));
                    let closes_at_first_point = self.drawing_mask_polygon.len() >= 3
                        && self.drawing_mask_polygon.first().is_some_and(|first| {
                            let first_screen = self.camera.world_to_screen(*first + off, rect);
                            pos.distance(first_screen) <= MASK_POLYGON_CLOSE_HIT_RADIUS_SCREEN_PX
                        });

                    if closes_at_first_point {
                        self.finish_drawing_mask_polygon();
                    } else {
                        self.drawing_mask_polygon.push(world - off);
                    }
                }
            }
        }

        if self.tool_mode == ToolMode::DrawMaskPolygon && !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.drawing_mask_polygon.clear();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Backspace)) {
                self.drawing_mask_polygon.pop();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.finish_drawing_mask_polygon();
            }
        }

        // Spatial selection tools operate in world coordinates, but they are only valid for a
        // subset of active layers. The drag state is kept separate from the final selection so
        // Escape can cancel the gesture without mutating layer selections.
        let mut spatial_selection_drag_consumed = false;
        let can_rect_select = self.tool_mode == ToolMode::Select
            && !space_down
            && !ctx.wants_keyboard_input()
            && self.active_layer_supports_spatial_selection();
        if can_rect_select
            && response.drag_started_by(egui::PointerButton::Primary)
            && !ui.input(|i| i.modifiers.command)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                self.selection_rect_start_world = Some(world);
                self.selection_rect_current_world = Some(world);
            }
        }
        if can_rect_select && response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                self.selection_rect_current_world = Some(self.camera.screen_to_world(pos, rect));
            }
        }
        if can_rect_select && response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(pos) = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()))
            {
                self.selection_rect_current_world = Some(self.camera.screen_to_world(pos, rect));
            }
            if let (Some(start), Some(end)) = (
                self.selection_rect_start_world,
                self.selection_rect_current_world,
            ) {
                let selection_rect = egui::Rect::from_two_pos(start, end);
                let min_drag_world = 4.0 / self.camera.zoom_screen_per_lvl0_px.max(1e-6);
                if selection_rect
                    .width()
                    .abs()
                    .max(selection_rect.height().abs())
                    >= min_drag_world
                {
                    let additive = ctx.input(|i| i.modifiers.shift);
                    spatial_selection_drag_consumed = true;
                    let _ = self.commit_rect_selection_to_active_layer(selection_rect, additive);
                }
            }
            self.clear_spatial_selection_drag();
        }

        let can_lasso_select = self.tool_mode == ToolMode::LassoSelect
            && !space_down
            && !ctx.wants_keyboard_input()
            && self.active_layer_supports_spatial_selection();
        if can_lasso_select && response.drag_started_by(egui::PointerButton::Primary) {
            self.selection_lasso_world.clear();
            if let Some(pos) = response.interact_pointer_pos() {
                self.selection_lasso_world
                    .push(self.camera.screen_to_world(pos, rect));
            }
        }
        if can_lasso_select && response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                let min_step_world = 2.0 / self.camera.zoom_screen_per_lvl0_px.max(1e-6);
                let should_push = self
                    .selection_lasso_world
                    .last()
                    .is_none_or(|last| last.distance(world) >= min_step_world);
                if should_push {
                    self.selection_lasso_world.push(world);
                }
            }
        }
        if can_lasso_select && response.drag_stopped_by(egui::PointerButton::Primary) {
            if self.selection_lasso_world.len() >= 3 {
                let additive = ctx.input(|i| i.modifiers.shift || i.modifiers.command);
                let lasso_world = self.selection_lasso_world.clone();
                spatial_selection_drag_consumed = true;
                let _ = self.commit_lasso_selection_to_active_layer(&lasso_world, additive);
            }
            self.clear_spatial_selection_drag();
        }

        let can_edit_mask_polygon = self.tool_mode == ToolMode::Select
            && !space_down
            && !ctx.wants_keyboard_input()
            && !spatial_selection_drag_consumed
            && self.selection_rect_start_world.is_none()
            && matches!(self.active_layer, LayerId::Mask(_));
        if can_edit_mask_polygon
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
            && let (LayerId::Mask(layer_id), Some(pos)) = (
                self.active_layer,
                response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.interact_pos())),
            )
        {
            let world = self.camera.screen_to_world(pos, rect);
            if let Some(hit) = self.hit_mask_polygon_at(layer_id, world, pos, rect) {
                let selection = MaskPolygonSelection {
                    layer_id,
                    polygon_idx: hit.polygon_idx,
                };
                if let Some(vertex_idx) = hit.vertex_idx {
                    self.begin_mask_vertex_drag(selection, vertex_idx);
                }
            }
        }
        if can_edit_mask_polygon
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary))
            && let (LayerId::Mask(layer_id), Some(pos)) =
                (self.active_layer, ui.input(|i| i.pointer.hover_pos()))
            && rect.contains(pos)
        {
            let world = self.camera.screen_to_world(pos, rect);
            self.select_mask_polygon_at(layer_id, world, pos, rect);
        }
        if can_edit_mask_polygon
            && response.dragged_by(egui::PointerButton::Primary)
            && let (Some(mut drag), Some(pos)) = (
                self.dragging_mask_vertex.clone(),
                response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.interact_pos())),
            )
        {
            let world = self.camera.screen_to_world(pos, rect);
            let selection = drag.selection;
            let vertex_idx = drag.vertex_idx;
            if !drag.undo_recorded {
                drag.undo_recorded = true;
                self.dragging_mask_vertex = Some(drag);
            }
            if self.move_mask_polygon_vertex(selection, vertex_idx, world) {
                self.selected_mask_polygon = Some(selection);
                self.selected_mask_vertex = Some(vertex_idx);
                self.bump_render_id();
            }
        }

        if self.tool_mode == ToolMode::Select
            && !space_down
            && !ctx.wants_keyboard_input()
            && !spatial_selection_drag_consumed
            && self.selection_rect_start_world.is_none()
            && response.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pos) = response.interact_pointer_pos() {
                let world = self.camera.screen_to_world(pos, rect);
                let mods = ctx.input(|i| i.modifiers);
                match self.active_or_spatial_selection_layer() {
                    target @ (LayerId::SegmentationObjects | LayerId::SpatialShape(_)) => {
                        self.commit_point_selection_to_layer(
                            target,
                            world,
                            mods.shift,
                            mods.command,
                        );
                    }
                    LayerId::Mask(id) => {
                        self.select_mask_polygon_at(id, world, pos, rect);
                        if self.dragging_mask_vertex.is_some() {
                            self.cancel_mask_polygon_gesture();
                        }
                    }
                    _ => {}
                }
            }
        }

        let cancel_selection_gesture = allows_edits
            && !ctx.wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::Escape))
            && matches!(self.tool_mode, ToolMode::Select | ToolMode::LassoSelect)
            && (self.selection_rect_start_world.is_some()
                || !self.selection_lasso_world.is_empty());
        if cancel_selection_gesture {
            self.clear_spatial_selection_drag();
        } else if allows_edits
            && !ctx.wants_keyboard_input()
            && ctx.input(|i| i.key_pressed(egui::Key::Escape))
        {
            if matches!(self.tool_mode, ToolMode::Select | ToolMode::LassoSelect) {
                self.clear_spatial_selection_drag();
            }
            match self.active_or_spatial_selection_layer() {
                target @ (LayerId::SegmentationObjects | LayerId::SpatialShape(_)) => {
                    self.commit_clear_object_selection(target);
                }
                LayerId::Mask(_) => self.commit_clear_mask_polygon_selection(),
                _ => {}
            }
        }

        if allows_edits
            && !ctx.wants_keyboard_input()
            && ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z))
            && self.request_native_mask_undo()
        {
            self.bump_render_id();
        }

        if !ctx.wants_keyboard_input()
            && matches!(self.active_layer, LayerId::Mask(_))
            && self.tool_mode == ToolMode::Select
            && (ctx.input(|i| i.key_pressed(egui::Key::Delete))
                || ctx.input(|i| i.key_pressed(egui::Key::Backspace)))
            && self.delete_selected_mask_polygon()
        {
            self.bump_render_id();
        }

        if self.tool_mode == ToolMode::TransformLayer && !ctx.wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.tool_mode = ToolMode::Pan;
                self.layer_transform = None;
            }
        }

        let can_pan_primary = self.tool_mode == ToolMode::Pan || space_down;
        if can_pan_primary
            && self.dragging_mask_vertex.is_none()
            && response.dragged_by(egui::PointerButton::Primary)
        {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera.pan_by_screen_delta(delta);
            camera_changed = true;
        }

        let can_move_primary = self.tool_mode == ToolMode::MoveLayer && !space_down;
        if can_move_primary && response.drag_started_by(egui::PointerButton::Primary) {
            let mut polygon_move_started = false;
            let drag_start_pos = ui
                .input(|i| i.pointer.press_origin())
                .filter(|pos| rect.contains(*pos))
                .or_else(|| response.interact_pointer_pos());
            if let (LayerId::Mask(layer_id), Some(pos)) = (self.active_layer, drag_start_pos) {
                let world = self.camera.screen_to_world(pos, rect);
                if let Some(hit) = self.hit_mask_polygon_at(layer_id, world, pos, rect) {
                    let selection = MaskPolygonSelection {
                        layer_id,
                        polygon_idx: hit.polygon_idx,
                    };
                    polygon_move_started = self.begin_mask_polygon_move(selection, world);
                }
            }

            if !polygon_move_started {
                let targets = self.current_visible_move_target_layers();
                if targets.is_empty() {
                    self.layer_move = None;
                    self.set_status("No visible movable layers selected.");
                } else {
                    self.ensure_loaded_layer_offset_baselines_for(&targets);
                    self.layer_move = Some(LayerMoveState {
                        actor_scope: self.active_viewport_command_scope(),
                        targets: targets
                            .into_iter()
                            .map(|layer| LayerOffsetEntry {
                                layer,
                                offset_world: self.layer_offset_world(layer),
                            })
                            .collect(),
                    });
                }
            } else {
                self.layer_move = None;
            }
        }
        if can_move_primary && response.dragged_by(egui::PointerButton::Primary) {
            if let (Some(state), Some(pos)) = (
                self.moving_mask_polygon.clone(),
                response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.interact_pos())),
            ) {
                let world = self.camera.screen_to_world(pos, rect);
                if self.move_mask_polygon_from_start(&state, world) {
                    self.selected_mask_polygon = Some(state.selection);
                    self.selected_mask_vertex = None;
                    self.bump_render_id();
                }
            } else {
                let z = self.camera.zoom_screen_per_lvl0_px.max(1e-6);
                if let (Some(state), Some(delta)) = (
                    self.layer_move.clone(),
                    ui.input(|i| i.pointer.total_drag_delta()),
                ) {
                    // total_drag_delta is in screen points; convert to world lvl0.
                    let offsets = state
                        .targets
                        .into_iter()
                        .map(|target| LayerOffsetEntry {
                            layer: target.layer,
                            offset_world: target.offset_world + delta / z,
                        })
                        .collect::<Vec<_>>();
                    if self.apply_layer_offsets(&offsets) {
                        self.bump_render_id();
                    }
                }
            }
        }
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let Some(state) = self.layer_move.take() {
                self.finish_native_layer_move(&state);
            }
            self.finish_mask_polygon_gesture();
        }

        // Transform tool (channels only): translate/scale/rotate with handles drawn on-canvas.
        // Use pointer-down instead of drag-start so the grab registers immediately (no drag threshold).
        let can_transform_primary = self.tool_mode == ToolMode::TransformLayer
            && !space_down
            && self.view_plane_is_xy()
            && matches!(self.active_layer, LayerId::Channel(_));

        if can_transform_primary
            && response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary))
        {
            let pointer = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()));
            if let (Some(pointer), LayerId::Channel(ch_idx0)) = (pointer, self.active_layer) {
                let ch_idx = ch_idx0.min(self.channels.len().saturating_sub(1));

                let (pivot_screen, corners, rotate_handle) =
                    self.channel_transform_gizmo_screen(rect, ch_idx);

                let hit_r = 10.0;
                let mut kind = None;
                if rotate_handle.distance(pointer) <= hit_r {
                    kind = Some(LayerTransformKind::Rotate);
                } else {
                    for &c in corners.iter() {
                        if c.distance(pointer) <= hit_r {
                            kind = Some(LayerTransformKind::Scale);
                            break;
                        }
                    }
                }
                if kind.is_none() && point_in_convex_quad(pointer, &corners) {
                    kind = Some(LayerTransformKind::Translate);
                }

                if let Some(kind) = kind {
                    let start_offset_world = self
                        .channel_offsets_world
                        .get(ch_idx)
                        .copied()
                        .unwrap_or_default();
                    let start_scale = self
                        .channel_scales
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(egui::Vec2::splat(1.0));
                    let start_rotation_rad = self
                        .channel_rotations_rad
                        .get(ch_idx)
                        .copied()
                        .unwrap_or(0.0);
                    let start_vec_screen = pointer - pivot_screen;
                    let start_angle_rad = start_vec_screen.y.atan2(start_vec_screen.x);
                    let start_len_screen = start_vec_screen.length().max(1e-6);
                    self.layer_transform = Some(LayerTransformState {
                        layer: LayerId::Channel(ch_idx),
                        kind,
                        start_offset_world,
                        start_scale,
                        start_rotation_rad,
                        pivot_screen,
                        start_pointer_screen: pointer,
                        start_angle_rad,
                        start_len_screen,
                        actor_scope: self.active_viewport_command_scope(),
                    });
                } else {
                    self.layer_transform = None;
                }
            } else {
                self.layer_transform = None;
            }
        }

        if can_transform_primary
            && response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
        {
            let z = self.camera.zoom_screen_per_lvl0_px.max(1e-6);
            let pointer = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()));
            if let (Some(state), Some(pointer)) = (self.layer_transform.clone(), pointer) {
                if let LayerId::Channel(ch_idx0) = state.layer {
                    let ch_idx = ch_idx0.min(self.channels.len().saturating_sub(1));
                    match state.kind {
                        LayerTransformKind::Translate => {
                            let delta_screen = pointer - state.start_pointer_screen;
                            if let Some(off) = self.channel_offsets_world.get_mut(ch_idx) {
                                *off = state.start_offset_world + delta_screen / z;
                            }
                        }
                        LayerTransformKind::Scale => {
                            let v = pointer - state.pivot_screen;
                            let len = v.length().max(1e-6);
                            let factor = (len / state.start_len_screen).clamp(0.01, 100.0);
                            if let Some(scale) = self.channel_scales.get_mut(ch_idx) {
                                let candidate = state.start_scale * factor;
                                scale.x = candidate.x.clamp(0.01, 100.0);
                                scale.y = candidate.y.clamp(0.01, 100.0);
                            }
                        }
                        LayerTransformKind::Rotate => {
                            let v = pointer - state.pivot_screen;
                            let angle = v.y.atan2(v.x);
                            let delta = angle - state.start_angle_rad;
                            if let Some(rot) = self.channel_rotations_rad.get_mut(ch_idx) {
                                *rot = state.start_rotation_rad + delta;
                            }
                        }
                    }
                    self.hist_dirty = true;
                } else {
                    self.layer_transform = None;
                }
            }
        }

        if can_transform_primary
            && ui.input(|i| i.pointer.button_released(egui::PointerButton::Primary))
        {
            if let Some(
                state @ LayerTransformState {
                    layer: LayerId::Channel(channel),
                    ..
                },
            ) = self.layer_transform.take()
            {
                let offset = self
                    .channel_offsets_world
                    .get(channel)
                    .copied()
                    .unwrap_or_default();
                let scale = self
                    .channel_scales
                    .get(channel)
                    .copied()
                    .unwrap_or(egui::Vec2::splat(1.0));
                let rotation = self
                    .channel_rotations_rad
                    .get(channel)
                    .copied()
                    .unwrap_or_default();
                if let Some((viewport_id, revision)) = state.actor_scope.as_ref() {
                    if let Some(value) = self.channel_offsets_world.get_mut(channel) {
                        *value = state.start_offset_world;
                    }
                    if let Some(value) = self.channel_scales.get_mut(channel) {
                        *value = state.start_scale;
                    }
                    if let Some(value) = self.channel_rotations_rad.get_mut(channel) {
                        *value = state.start_rotation_rad;
                    }
                    self.submit_native_channel_transform_at(
                        viewport_id,
                        *revision,
                        channel,
                        Some(offset),
                        Some(scale),
                        Some(rotation),
                    );
                }
            } else {
                self.layer_transform = None;
            }
        }

        if camera_changed {
            let desired = self.camera.clone();
            self.submit_native_camera(&desired);
            self.hist_dirty = true;
            self.hist_navigation_dirty_since = Some(Instant::now());
        }

        camera_changed
    }
}
