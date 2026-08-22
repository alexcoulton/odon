use super::*;

impl OmeZarrViewerApp {
    pub(super) fn ui_canvas(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        viewport_id: &ViewportId,
        allows_edits: bool,
        raw_request_budget: usize,
        cpu_request_budget: usize,
    ) -> bool {
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        let activate_viewport = response.clicked() || response.drag_started();
        self.last_canvas_rect = Some(rect);
        self.mask_draw_debug_stats = MaskDrawDebugStats::default();
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(10));

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
                let id = self
                    .drawing_mask_layer
                    .unwrap_or_else(|| self.ensure_editable_mask_layer());
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
                    let _ = self.apply_rect_selection_to_active_layer(selection_rect, additive);
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
                let _ = self.apply_lasso_selection_to_active_layer(&lasso_world, additive);
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
                self.selected_mask_polygon = Some(selection);
                self.selected_mask_vertex = hit.vertex_idx;
                if let Some(vertex_idx) = hit.vertex_idx {
                    self.dragging_mask_vertex = Some(MaskVertexDrag {
                        selection,
                        vertex_idx,
                        undo_recorded: false,
                    });
                } else {
                    self.dragging_mask_vertex = None;
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
            if let Some(hit) = self.hit_mask_polygon_at(layer_id, world, pos, rect) {
                self.selected_mask_polygon = Some(MaskPolygonSelection {
                    layer_id,
                    polygon_idx: hit.polygon_idx,
                });
                self.selected_mask_vertex = hit.vertex_idx;
            }
        }
        if can_edit_mask_polygon
            && response.dragged_by(egui::PointerButton::Primary)
            && let (Some(mut drag), Some(pos)) = (
                self.dragging_mask_vertex,
                response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.interact_pos())),
            )
        {
            let world = self.camera.screen_to_world(pos, rect);
            if !drag.undo_recorded {
                self.push_mask_undo_snapshot();
                drag.undo_recorded = true;
                self.dragging_mask_vertex = Some(drag);
            }
            if self.move_mask_polygon_vertex(drag.selection, drag.vertex_idx, world) {
                self.selected_mask_polygon = Some(drag.selection);
                self.selected_mask_vertex = Some(drag.vertex_idx);
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
                    LayerId::SegmentationObjects => {
                        let off = self.layer_offset_world(LayerId::SegmentationObjects);
                        self.seg_objects.select_at(
                            world,
                            off,
                            &self.camera,
                            mods.shift,
                            mods.command,
                        );
                    }
                    LayerId::SpatialShape(id) => {
                        if let Some(layer) =
                            self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                        {
                            layer.select_at(world, mods.shift, mods.command, &self.camera);
                        }
                    }
                    LayerId::Mask(id) => {
                        self.select_mask_polygon_at(id, world, pos, rect);
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
                LayerId::SegmentationObjects => self.seg_objects.clear_selection(),
                LayerId::SpatialShape(id) => {
                    if let Some(layer) = self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                    {
                        layer.clear_selection();
                    }
                }
                LayerId::Mask(_) => self.clear_mask_polygon_selection(),
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
                    self.push_layer_offsets_undo_snapshot(&targets);
                    self.layer_move = Some(LayerMoveState {
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
            self.layer_move = None;
            self.dragging_mask_vertex = None;
            self.moving_mask_polygon = None;
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
            if let Some(LayerTransformState {
                layer: LayerId::Channel(channel),
                ..
            }) = self.layer_transform.take()
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
                self.native_control_intents.push(NativeControlIntent {
                    method: "viewer.channels.set_transform",
                    params: serde_json::json!({
                        "channel": channel,
                        "offset_world": [offset.x, offset.y],
                        "scale": [scale.x, scale.y],
                        "rotation_rad": rotation,
                    }),
                });
            } else {
                self.layer_transform = None;
            }
        }

        if camera_changed {
            self.hist_dirty = true;
            self.hist_navigation_dirty_since = Some(Instant::now());
        }

        let visible_world = self.visible_world_rect(rect);
        let visible_world_tiles_world =
            if self.view_plane_is_xy() && self.any_visible_channel_affine() {
                self.union_visible_world_for_visible_channels_xform(visible_world)
            } else if self.view_plane_is_xy() && self.any_visible_channel_offset() {
                self.union_visible_world_for_visible_channels(visible_world)
            } else {
                visible_world
            };
        let visible_world_tiles = self.primary_image_world_rect_to_local(visible_world_tiles_world);
        let prev_visible_world_tiles = self.last_visible_world_tiles.unwrap_or(visible_world_tiles);
        let target_level = self.choose_level();

        // Short-lived "zoom-out floor": when zooming out, keep drawing the previous (finer) target
        // level over the previously-visible region for a moment. This avoids sudden blur jumps
        // while the new coarser target is still loading.
        const ZOOM_OUT_FLOOR_MS: u64 = 400;
        let now = Instant::now();
        if let Some(until) = self.zoom_out_floor_until {
            if now > until {
                self.zoom_out_floor_level = None;
                self.zoom_out_floor_until = None;
                self.zoom_out_floor_visible_world_tiles = None;
            }
        }
        if let Some(prev_target) = self.last_target_level {
            if target_level > prev_target {
                self.zoom_out_floor_level = Some(prev_target);
                self.zoom_out_floor_until = Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_MS));
                self.zoom_out_floor_visible_world_tiles = Some(prev_visible_world_tiles);
            } else if target_level < prev_target {
                self.zoom_out_floor_level = None;
                self.zoom_out_floor_until = None;
                self.zoom_out_floor_visible_world_tiles = None;
            }
        }

        // Sticky "fallback ceiling": when zooming in, keep requesting/drawing intermediate levels
        // between the current target and the last coarser target we came from. This avoids a
        // situation where we fall back all the way to the coarsest level once the target settles.
        let coarsest = self.dataset.levels.len().saturating_sub(1);
        let mut ceiling = self.fallback_ceiling_level.unwrap_or(target_level);
        if let Some(prev_target) = self.last_target_level {
            if target_level < prev_target {
                ceiling = ceiling.max(prev_target);
            } else if target_level > prev_target {
                // Zooming out: reset the ceiling to match the new coarser target.
                ceiling = target_level;
            }
        } else {
            ceiling = target_level;
        }
        ceiling = ceiling.min(coarsest);
        self.fallback_ceiling_level = Some(ceiling);

        // Request levels: normal coarse/mid/target plus (when zooming in) the intermediate ladder
        // up to the sticky ceiling. Zoom-out uses a separate "floor" draw-only overlay.
        let mut levels_to_draw = levels_to_draw(self.dataset.levels.len(), target_level);
        if ceiling > target_level {
            for l in target_level..=ceiling {
                levels_to_draw.push(l);
            }
        }
        levels_to_draw.sort_unstable_by(|a, b| b.cmp(a)); // coarse -> fine
        levels_to_draw.dedup();

        let Some(level0) = self.dataset.levels.first() else {
            return activate_viewport;
        };
        let Some(display_axes) = self.display_axes() else {
            return activate_viewport;
        };
        let mut needed_per_level: Vec<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> =
            Vec::with_capacity(levels_to_draw.len());
        let active_view = self.displayed_view_selection();
        let fallback_view = self.fallback_view_selection();
        for &level in &levels_to_draw {
            let level_info = self.dataset.levels[level].clone();
            let coords: Vec<TileCoord> = tiles_needed_lvl0_rect_for_axes(
                visible_world_tiles,
                level0,
                &level_info,
                display_axes,
                1,
            );
            let mut needed: Vec<TileKey> = coords
                .into_iter()
                .map(|c| TileKey {
                    render_id: self.active_render_id,
                    view: active_view,
                    level,
                    tile_y: c.tile_y,
                    tile_x: c.tile_x,
                })
                .collect();
            self.sort_tile_keys_near_center(&level_info, &mut needed);
            needed_per_level.push((level, level_info, needed));
        }
        let render_channels = self.render_channels_for_request(target_level);
        let visible_target_raw_request_count = needed_per_level
            .iter()
            .find(|(level, _, _)| *level == target_level)
            .map(|(_, _, needed)| needed.len())
            .unwrap_or_default()
            .saturating_mul(render_channels.len());
        let raw_tile_cache_capacity = if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            self.maybe_grow_raw_tile_cache(tiles_gl, visible_target_raw_request_count);
            tiles_gl.capacity()
        } else {
            0
        };
        let visible_raw_request_count = needed_per_level
            .iter()
            .map(|(_, _, needed)| needed.len())
            .sum::<usize>()
            .saturating_mul(render_channels.len());
        let high_fanout_raw_request_mode =
            self.tiles_gl.is_some() && render_channels.len() >= RAW_TILE_ADAPTIVE_CHANNEL_THRESHOLD;
        let adaptive_raw_request_mode = self.tiles_gl.is_some()
            && (high_fanout_raw_request_mode
                || visible_raw_request_count > raw_tile_cache_capacity);

        let mut prefetch_needed_per_level: Vec<(usize, Vec<TileKey>)> = Vec::new();
        if !adaptive_raw_request_mode && !self.pinned_levels.has_loading() {
            let target_level_prefetch_needed = needed_per_level
                .iter()
                .find(|(level, _, _)| *level == target_level)
                .map(|(level, level_info, needed)| {
                    self.prefetch_keys_for_level(*level, level_info, visible_world_tiles, needed)
                })
                .unwrap_or_default();
            if !target_level_prefetch_needed.is_empty() {
                prefetch_needed_per_level.push((target_level, target_level_prefetch_needed));
            }
            if self.tile_prefetch_mode == TilePrefetchMode::TargetAndFinerHalo && target_level > 0 {
                let finer_level = target_level - 1;
                if let Some(level_info) = self.dataset.levels.get(finer_level) {
                    let finer_visible: Vec<TileKey> = tiles_needed_lvl0_rect_for_axes(
                        visible_world_tiles,
                        level0,
                        level_info,
                        display_axes,
                        1,
                    )
                    .into_iter()
                    .map(|c| TileKey {
                        render_id: self.active_render_id,
                        view: active_view,
                        level: finer_level,
                        tile_y: c.tile_y,
                        tile_x: c.tile_x,
                    })
                    .collect();
                    let finer_prefetch = self.prefetch_keys_for_level(
                        finer_level,
                        level_info,
                        visible_world_tiles,
                        &finer_visible,
                    );
                    if !finer_prefetch.is_empty() {
                        prefetch_needed_per_level.push((finer_level, finer_prefetch));
                    }
                }
            }
        }

        self.last_target_level = Some(target_level);
        self.last_visible_world_tiles = Some(visible_world_tiles);

        if let (Some(tiles_gl), Some(raw_loader)) =
            (self.tiles_gl.clone(), self.raw_loader.as_ref())
        {
            let raw_tx = raw_loader.tx.clone();
            // If the coarser target level is already "ready enough" over the previous visible
            // region, drop the zoom-out floor early. If not, extend it a bit past the nominal
            // timeout so we don't get a sudden blur jump when IO is slower than expected.
            const ZOOM_OUT_FLOOR_EXTEND_MS: u64 = 200;
            if let Some(floor_level) = self.zoom_out_floor_level {
                if floor_level >= self.dataset.levels.len() || floor_level >= target_level {
                    self.zoom_out_floor_level = None;
                    self.zoom_out_floor_until = None;
                    self.zoom_out_floor_visible_world_tiles = None;
                } else {
                    let floor_rect = self
                        .zoom_out_floor_visible_world_tiles
                        .unwrap_or(prev_visible_world_tiles);
                    if let Some(level_info_tgt) = self.dataset.levels.get(target_level) {
                        let coords_tgt = tiles_needed_lvl0_rect_for_axes(
                            floor_rect,
                            level0,
                            level_info_tgt,
                            display_axes,
                            0,
                        );
                        let sample_max = 16usize;
                        let stride = (coords_tgt.len() / sample_max).max(1);
                        let mut total = 0usize;
                        let mut ready = 0usize;
                        if !coords_tgt.is_empty() && !render_channels.is_empty() {
                            for c in coords_tgt.iter().step_by(stride).take(sample_max) {
                                for ch in &render_channels {
                                    total += 1;
                                    let k = RawTileKey {
                                        view: active_view,
                                        level: target_level,
                                        tile_y: c.tile_y,
                                        tile_x: c.tile_x,
                                        channel: ch.index,
                                    };
                                    if tiles_gl.contains(&k) {
                                        ready += 1;
                                    }
                                }
                            }
                        }
                        let ready_enough = total == 0 || ready * 10 >= total * 8; // >=80%
                        if ready_enough {
                            self.zoom_out_floor_level = None;
                            self.zoom_out_floor_until = None;
                            self.zoom_out_floor_visible_world_tiles = None;
                        } else if self.zoom_out_floor_until.map(|u| now > u).unwrap_or(true) {
                            self.zoom_out_floor_until =
                                Some(now + Duration::from_millis(ZOOM_OUT_FLOOR_EXTEND_MS));
                        }
                    }
                }
            }

            let fallback_floor: Option<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> =
                (|| -> Option<(usize, crate::data::ome::LevelInfo, Vec<TileKey>)> {
                    let floor_level = self.zoom_out_floor_level?;
                    if floor_level >= self.dataset.levels.len() || floor_level >= target_level {
                        return None;
                    }
                    if let Some(until) = self.zoom_out_floor_until {
                        if now > until {
                            return None;
                        }
                    }
                    let floor_rect = self
                        .zoom_out_floor_visible_world_tiles
                        .unwrap_or(prev_visible_world_tiles);
                    if floor_rect.width() <= 0.0 || floor_rect.height() <= 0.0 {
                        return None;
                    }
                    let level_info = self.dataset.levels[floor_level].clone();
                    let coords: Vec<TileCoord> = tiles_needed_lvl0_rect_for_axes(
                        floor_rect,
                        level0,
                        &level_info,
                        display_axes,
                        1,
                    );
                    if coords.is_empty() {
                        return None;
                    }
                    // Keep the draw-only floor lightweight even if the previous visible area was large.
                    let mut needed: Vec<TileKey> = coords
                        .into_iter()
                        .take(1024)
                        .map(|c| TileKey {
                            render_id: self.active_render_id,
                            view: active_view,
                            level: floor_level,
                            tile_y: c.tile_y,
                            tile_x: c.tile_x,
                        })
                        .collect();
                    self.sort_tile_keys_near_center(&level_info, &mut needed);
                    Some((floor_level, level_info, needed))
                })();

            let mut requested_this_frame = 0usize;
            let max_requests_per_frame = raw_request_budget;
            let mut request_levels: Vec<usize> = Vec::new();
            if adaptive_raw_request_mode {
                let coarsest_level = needed_per_level.first().map(|(level, _, _)| *level);
                let bridge_level = target_level.checked_add(1).and_then(|level| {
                    needed_per_level
                        .iter()
                        .any(|(l, _, _)| *l == level)
                        .then_some(level)
                });

                if let Some(level) = bridge_level {
                    request_levels.push(level);
                }
                if let Some(level) = coarsest_level {
                    if high_fanout_raw_request_mode {
                        // With many visible channels, requesting the full coarse ladder causes
                        // the same fallback tiles to dominate the queue. Keep only the bridge
                        // level plus the target level in this mode.
                    } else if Some(level) != bridge_level && level != target_level {
                        request_levels.push(level);
                    }
                }
                request_levels.push(target_level);
            } else {
                request_levels.extend(needed_per_level.iter().map(|(level, _, _)| *level));
            }

            request_levels.sort_unstable();
            request_levels.dedup();

            if adaptive_raw_request_mode {
                crate::log_debug!(
                    "raw request mode: adaptive={} high_fanout={} target={} bridge={:?} levels={:?} channels={} visible_raw={} cache_cap={}",
                    adaptive_raw_request_mode,
                    high_fanout_raw_request_mode,
                    target_level,
                    target_level.checked_add(1).and_then(|level| {
                        needed_per_level
                            .iter()
                            .any(|(l, _, _)| *l == level)
                            .then_some(level)
                    }),
                    request_levels,
                    render_channels.len(),
                    visible_raw_request_count,
                    raw_tile_cache_capacity
                );
            }

            let keep_levels: Vec<usize> = if adaptive_raw_request_mode {
                request_levels.clone()
            } else {
                needed_per_level
                    .iter()
                    .map(|(level, _, _)| *level)
                    .collect()
            };
            let mut raw_active_keys: HashSet<RawTileKey> = HashSet::new();
            for (level, _level_info, needed) in needed_per_level.iter() {
                if !keep_levels.contains(level) {
                    continue;
                }
                for key in needed {
                    for ch in &render_channels {
                        raw_active_keys.insert(RawTileKey {
                            view: active_view,
                            level: *level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            channel: ch.index,
                        });
                    }
                }
            }
            for (level, needed) in &prefetch_needed_per_level {
                if !keep_levels.contains(level) {
                    continue;
                }
                for key in needed {
                    for ch in &render_channels {
                        raw_active_keys.insert(RawTileKey {
                            view: active_view,
                            level: *level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            channel: ch.index,
                        });
                    }
                }
            }
            if let Some((level, _level_info, needed)) = fallback_floor.as_ref() {
                for key in needed {
                    for ch in &render_channels {
                        raw_active_keys.insert(RawTileKey {
                            view: active_view,
                            level: *level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            channel: ch.index,
                        });
                    }
                }
            }
            let aggregate_raw_keys = self.viewport_raw_active_keys.is_some();
            if let Some(active_keys) = self.viewport_raw_active_keys.as_mut() {
                merge_viewport_active_keys(active_keys, raw_active_keys.iter().copied());
                // Publish the growing union before work is submitted. A fast
                // loader must not compare new requests with the preceding
                // frame's active-key set and discard them.
                raw_loader.set_active_keys(active_keys.clone());
            } else {
                raw_loader.set_active_keys(raw_active_keys.clone());
            }

            for level in &request_levels {
                let Some((_, _, needed)) = needed_per_level.iter().find(|(l, _, _)| l == level)
                else {
                    continue;
                };
                let phase_max = if adaptive_raw_request_mode && *level != target_level {
                    let tiles_per_phase = if *level == target_level.saturating_add(1) {
                        RAW_TILE_ADAPTIVE_BRIDGE_TILES_PER_FRAME
                    } else {
                        RAW_TILE_ADAPTIVE_COARSE_TILES_PER_FRAME
                    };
                    (requested_this_frame + render_channels.len().saturating_mul(tiles_per_phase))
                        .min(max_requests_per_frame)
                } else {
                    max_requests_per_frame
                };
                self.request_raw_tiles_with_budget(
                    &tiles_gl,
                    &raw_tx,
                    *level,
                    needed,
                    &render_channels,
                    &mut requested_this_frame,
                    phase_max,
                );
            }
            if !adaptive_raw_request_mode {
                for (level, needed) in &prefetch_needed_per_level {
                    if requested_this_frame >= max_requests_per_frame {
                        break;
                    }
                    self.request_raw_tiles_with_budget(
                        &tiles_gl,
                        &raw_tx,
                        *level,
                        needed,
                        &render_channels,
                        &mut requested_this_frame,
                        max_requests_per_frame,
                    );
                }
            }

            // Prune stale in-flight requests so the app can go idle immediately after a fast pan/zoom.
            if !aggregate_raw_keys && let Some(tiles_gl_ref) = self.tiles_gl.as_ref() {
                tiles_gl_ref.prune_in_flight(&raw_active_keys);
            }

            // Build draw list coarse -> fine.
            let mut draws: Vec<TileDraw> = Vec::new();
            draws.reserve(512);
            let any_affine_visible = self.view_plane_is_xy() && self.any_visible_channel_affine();
            let mut max_abs_off_screen = egui::Vec2::ZERO;
            if self.view_plane_is_xy() && !any_affine_visible && self.any_visible_channel_offset() {
                let z = self.camera.zoom_screen_per_lvl0_px;
                for (i, ch) in self.channels.iter().enumerate() {
                    if !ch.visible {
                        continue;
                    }
                    let off = self
                        .channel_offsets_world
                        .get(i)
                        .copied()
                        .unwrap_or_default()
                        * z;
                    max_abs_off_screen.x = max_abs_off_screen.x.max(off.x.abs());
                    max_abs_off_screen.y = max_abs_off_screen.y.max(off.y.abs());
                }
            }
            let draw_rect = if max_abs_off_screen.x > 0.0 || max_abs_off_screen.y > 0.0 {
                rect.expand2(max_abs_off_screen + egui::vec2(2.0, 2.0))
            } else {
                rect
            };
            for (level, level_info, needed) in needed_per_level {
                for key in needed {
                    let (_tile_world_rect, tile_screen_rect) =
                        self.tile_rects(&key, rect, &level_info);
                    if any_affine_visible || tile_screen_rect.intersects(draw_rect) {
                        draws.push(TileDraw {
                            view: key.view,
                            fallback_view,
                            level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            screen_rect: tile_screen_rect,
                        });
                    }
                }
            }
            // Draw-only zoom-out floor overlay last (finer than the current target).
            if let Some((level, level_info, needed)) = fallback_floor {
                for key in needed {
                    let (_tile_world_rect, tile_screen_rect) =
                        self.tile_rects(&key, rect, &level_info);
                    if any_affine_visible || tile_screen_rect.intersects(draw_rect) {
                        draws.push(TileDraw {
                            view: key.view,
                            fallback_view,
                            level,
                            tile_y: key.tile_y,
                            tile_x: key.tile_x,
                            screen_rect: tile_screen_rect,
                        });
                    }
                }
            }

            let zoom = self.camera.zoom_screen_per_lvl0_px;
            let pivot_screen = self
                .camera
                .world_to_screen(self.image_world_rect_lvl0().center(), rect);
            let mut channel_offsets_world: Vec<egui::Vec2> =
                Vec::with_capacity(render_channels.len());
            let mut channel_xforms_screen: Vec<crate::render::tiles_gl::ChannelScreenTransform> =
                Vec::with_capacity(render_channels.len());
            let mut any_offset = false;
            let mut any_affine = false;
            for ch in &render_channels {
                let idx = self
                    .channels
                    .iter()
                    .position(|c| c.index as u64 == ch.index)
                    .unwrap_or(0);
                let (off_world, scale, rot) = if self.view_plane_is_xy() {
                    (
                        self.channel_offsets_world
                            .get(idx)
                            .copied()
                            .unwrap_or_default(),
                        self.channel_scales
                            .get(idx)
                            .copied()
                            .unwrap_or(egui::Vec2::splat(1.0)),
                        self.channel_rotations_rad.get(idx).copied().unwrap_or(0.0),
                    )
                } else {
                    (egui::Vec2::ZERO, egui::Vec2::splat(1.0), 0.0)
                };
                any_offset |= off_world.x.abs() > 1e-6 || off_world.y.abs() > 1e-6;
                any_affine |= (scale.x - 1.0).abs() > 1e-6
                    || (scale.y - 1.0).abs() > 1e-6
                    || rot.abs() > 1e-6;
                channel_offsets_world.push(off_world);
                channel_xforms_screen.push(crate::render::tiles_gl::ChannelScreenTransform {
                    pivot_screen,
                    translation_screen: off_world * zoom,
                    scale,
                    rotation_rad: rot,
                });
            }

            let channels: Vec<ChannelDraw> =
                render_channels.into_iter().map(ChannelDraw::from).collect();
            let smooth_pixels = self.smooth_pixels;
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                tiles_gl.set_smooth_pixels(smooth_pixels);
                if any_affine || any_offset {
                    if any_affine {
                        tiles_gl.paint_with_channel_transforms_screen(
                            info,
                            painter,
                            &draws,
                            &channels,
                            &channel_xforms_screen,
                        );
                        return;
                    }
                    tiles_gl.paint_with_channel_offsets(
                        info,
                        painter,
                        &draws,
                        &channels,
                        &channel_offsets_world,
                        zoom,
                    );
                } else {
                    tiles_gl.paint(info, painter, &draws, &channels);
                }
            });
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(cb),
            });
        } else {
            // CPU (RGBA) path.
            let mut keep: HashSet<TileKey> = HashSet::new();
            for (_level, _level_info, needed) in needed_per_level.iter() {
                for key in needed {
                    keep.insert(*key);
                }
            }
            for (_level, needed) in &prefetch_needed_per_level {
                for key in needed {
                    keep.insert(*key);
                }
            }
            let aggregate_cpu_keys = self.viewport_cpu_active_keys.is_some();
            if let Some(active_keys) = self.viewport_cpu_active_keys.as_mut() {
                merge_viewport_active_keys(active_keys, keep.iter().copied());
                self.loader.set_active_keys(active_keys.clone());
            } else {
                self.loader.set_active_keys(keep.clone());
            }

            // Request tiles fine -> coarse so zoom-in upgrades quickly.
            let mut requested_this_frame = 0usize;
            let max_requests_per_frame = cpu_request_budget;
            for (level, _level_info, needed) in needed_per_level.iter().rev() {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                self.request_tiles_with_budget(
                    *level,
                    needed,
                    &mut requested_this_frame,
                    max_requests_per_frame,
                );
            }
            for (level, needed) in &prefetch_needed_per_level {
                if requested_this_frame >= max_requests_per_frame {
                    break;
                }
                self.request_tiles_with_budget(
                    *level,
                    needed,
                    &mut requested_this_frame,
                    max_requests_per_frame,
                );
            }

            // Prune stale in-flight requests so we don't keep repainting while the loader finishes
            // work that is no longer visible.
            if !aggregate_cpu_keys {
                self.cache.prune_in_flight(&keep);
            }

            // Draw tiles coarse -> fine (fallback first, then refine).
            for (level, level_info, needed) in needed_per_level {
                for key in needed {
                    if let Some(tex) = self.get_tile_texture(&key) {
                        let (_tile_world_rect, tile_screen_rect) =
                            self.tile_rects(&key, rect, &level_info);
                        if tile_screen_rect.intersects(rect) {
                            ui.painter().image(
                                tex.id(),
                                tile_screen_rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                egui::Color32::WHITE,
                            );
                        }
                    } else if level == target_level {
                        let (_tile_world_rect, tile_screen_rect) =
                            self.tile_rects(&key, rect, &level_info);
                        ui.painter().rect_stroke(
                            tile_screen_rect,
                            0.0,
                            egui::Stroke::new(1.0, egui::Color32::from_gray(30)),
                            egui::StrokeKind::Inside,
                        );
                    }
                }
            }
        }

        // If segmentation is hidden, clear any in-flight label tile requests so we don't keep
        // repainting while the background loader drains work we won't display.
        if (!self.view_plane_is_xy() || !self.cells_outlines_visible)
            && self.viewport_label_active_keys.is_none()
        {
            if let Some(labels_gl) = self.labels_gl.as_ref() {
                let keep: HashSet<LabelTileKey> = HashSet::new();
                labels_gl.prune_in_flight(&keep);
            }
        }

        // Overlays in the user-controlled layer order (bottom -> top).
        if self.view_plane_is_xy() {
            self.rebuild_layer_orders();
            let overlay_order = self.overlay_layer_order.clone();
            for layer in overlay_order.into_iter().rev() {
                match layer {
                    LayerId::Channel(_) => {}
                    LayerId::SpatialImage(id) => {
                        if let Some(layer) = self
                            .spatial_image_layers
                            .images
                            .iter_mut()
                            .find(|l| l.id == id)
                        {
                            let active_keys = layer.draw(
                                ui,
                                &self.camera,
                                rect,
                                visible_world,
                                self.smooth_pixels,
                            );
                            if let Some(active_keys_by_layer) =
                                self.viewport_spatial_image_active_keys.as_mut()
                            {
                                active_keys_by_layer
                                    .entry(id)
                                    .or_default()
                                    .extend(active_keys);
                            } else {
                                layer.prune_in_flight(&active_keys);
                            }
                        }
                    }
                    LayerId::SegmentationLabels => {
                        self.draw_cells_segmentation_overlay(ui, rect, visible_world, target_level);
                    }
                    LayerId::SegmentationGeoJson => {
                        let off = self.layer_offset_world(LayerId::SegmentationGeoJson);
                        let mut cam = self.camera.clone();
                        cam.center_world_lvl0 -= off;
                        self.seg_geojson.draw(
                            ui,
                            &cam,
                            rect,
                            visible_world.translate(-off),
                            self.tiles_gl.is_some(),
                        );
                    }
                    LayerId::SegmentationObjects => {
                        let off = self.layer_offset_world(LayerId::SegmentationObjects);
                        self.seg_objects.draw(
                            ui,
                            &self.camera,
                            rect,
                            visible_world,
                            off,
                            self.tiles_gl.is_some(),
                        );
                    }
                    LayerId::Mask(id) => self.draw_mask_layer_overlay(ui, rect, id),
                    LayerId::Points => self.draw_points_overlay(ui, rect, visible_world),
                    LayerId::Annotation(id) => {
                        let Some(local_root) = self.current_local_dataset_root() else {
                            continue;
                        };
                        let roi_id = self
                            .project_space
                            .rois()
                            .iter()
                            .find(|r| r.local_path().is_some_and(|path| path == local_root))
                            .map(|r| r.id.clone())
                            .or_else(|| {
                                local_root
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_else(|| "ROI".to_string());
                        let off = self.layer_offset_world(LayerId::Annotation(id));
                        let current_groups = self.current_layer_groups();
                        if let Some(layer) = self.annotation_layers.iter_mut().find(|l| l.id == id)
                        {
                            let group_tint =
                                layer_groups::effective_annotation_tint(&current_groups, id);
                            layer.offset_world = off;
                            layer.draw_single(
                                ui,
                                rect,
                                self.camera.center_world_lvl0,
                                self.camera.zoom_screen_per_lvl0_px,
                                roi_id.as_str(),
                                group_tint,
                                self.tiles_gl.is_some(),
                            );
                            if self.active_layer == LayerId::Annotation(id) {
                                if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                                    if rect.contains(pointer) {
                                        let world = self.camera.screen_to_world(pointer, rect);
                                        layer.maybe_hover_tooltip(
                                            ui.ctx(),
                                            rect,
                                            world,
                                            self.camera.zoom_screen_per_lvl0_px,
                                            roi_id.as_str(),
                                            egui::Vec2::ZERO,
                                            1.0,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    LayerId::SpatialShape(id) => {
                        let off = self.layer_offset_world(LayerId::SpatialShape(id));
                        if let Some(layer) =
                            self.spatial_layers.shapes.iter_mut().find(|s| s.id == id)
                        {
                            layer.draw(
                                ui,
                                &self.camera,
                                rect,
                                visible_world,
                                self.tiles_gl.is_some(),
                                off,
                            );
                        }
                    }
                    LayerId::SpatialPoints => {
                        let off = self.layer_offset_world(LayerId::SpatialPoints);
                        if let Some(layer) = self.spatial_layers.points.as_ref() {
                            layer.draw(ui, rect, &self.camera, off, self.tiles_gl.is_some());
                        }
                    }
                    LayerId::XeniumCells => {
                        let off = self.layer_offset_world(LayerId::XeniumCells);
                        if let Some(layer) = self.xenium_layers.cells.as_ref() {
                            layer.draw(
                                ui,
                                &self.camera,
                                rect,
                                visible_world,
                                self.tiles_gl.is_some(),
                                off,
                            );
                        }
                    }
                    LayerId::XeniumTranscripts => {
                        let off = self.layer_offset_world(LayerId::XeniumTranscripts);
                        if let Some(layer) = self.xenium_layers.transcripts.as_ref() {
                            layer.draw(ui, rect, &self.camera, off, self.tiles_gl.is_some());
                        }
                    }
                }
            }
        }

        if self.view_plane_is_xy() {
            self.draw_threshold_region_preview(ui, rect);
        }

        // In-progress polygon preview (Draw mask tool).
        if self.tool_mode == ToolMode::DrawMaskPolygon && !self.drawing_mask_polygon.is_empty() {
            let mask_id = self.drawing_mask_layer.or_else(|| {
                if let LayerId::Mask(id) = self.active_layer {
                    Some(id)
                } else {
                    None
                }
            });
            let (c, off, opacity, display_mode) = mask_id
                .and_then(|id| {
                    self.mask_layers
                        .iter()
                        .find(|l| l.id == id)
                        .map(|l| (l.color_rgb, l.offset_world, l.opacity, l.display_mode))
                })
                .unwrap_or((
                    [255, 210, 60],
                    egui::Vec2::ZERO,
                    0.9,
                    MaskDisplayMode::default_new_layer(),
                ));

            let color = egui::Color32::from_rgb(c[0], c[1], c[2]);
            let stroke = egui::Stroke::new(2.0, color);

            let pts = self
                .drawing_mask_polygon
                .iter()
                .copied()
                .map(|p| self.camera.world_to_screen(p + off, rect))
                .collect::<Vec<_>>();

            if pts.len() >= 3 {
                if let Some(fill_color) = mask_fill_color(c, opacity, display_mode) {
                    let mut fill_pts = pts.clone();
                    if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                        fill_pts.push(cursor);
                    }
                    paint_filled_polygon(ui, &fill_pts, fill_color);
                }
            }

            if pts.len() >= 2 {
                ui.painter().add(egui::Shape::line(pts.clone(), stroke));
            }

            let mut cursor_closes_at_first_point = false;
            if let Some(cursor) = ui.input(|i| i.pointer.hover_pos()) {
                if let Some(last) = pts.last().copied() {
                    cursor_closes_at_first_point = pts.len() >= 3
                        && pts.first().is_some_and(|first| {
                            cursor.distance(*first) <= MASK_POLYGON_CLOSE_HIT_RADIUS_SCREEN_PX
                        });
                    let preview_end = if cursor_closes_at_first_point {
                        pts[0]
                    } else {
                        cursor
                    };
                    let preview_stroke = if cursor_closes_at_first_point {
                        egui::Stroke::new(2.0, egui::Color32::WHITE)
                    } else {
                        egui::Stroke::new(1.0, color)
                    };
                    ui.painter()
                        .line_segment([last, preview_end], preview_stroke);
                }
            }

            for (i, p) in pts.iter().copied().enumerate() {
                if i == 0 && cursor_closes_at_first_point {
                    ui.painter()
                        .circle_filled(p, 6.0, egui::Color32::from_rgb(80, 220, 140));
                    ui.painter().circle_stroke(
                        p,
                        8.0,
                        egui::Stroke::new(2.0, egui::Color32::WHITE),
                    );
                } else {
                    let r = if i == 0 { 4.0 } else { 3.0 };
                    ui.painter().circle_filled(p, r, color);
                }
            }
        }

        let selection_color = egui::Color32::from_rgba_unmultiplied(255, 210, 80, 180);
        let selection_stroke = egui::Stroke::new(2.0, selection_color);
        if self.tool_mode == ToolMode::Select
            && let (Some(start), Some(end)) = (
                self.selection_rect_start_world,
                self.selection_rect_current_world,
            )
        {
            let rect_screen = egui::Rect::from_two_pos(
                self.camera.world_to_screen(start, rect),
                self.camera.world_to_screen(end, rect),
            );
            ui.painter().rect_filled(
                rect_screen,
                0.0,
                egui::Color32::from_rgba_unmultiplied(255, 210, 80, 36),
            );
            ui.painter()
                .rect_stroke(rect_screen, 0.0, selection_stroke, egui::StrokeKind::Inside);
        }
        if self.tool_mode == ToolMode::LassoSelect && self.selection_lasso_world.len() >= 2 {
            let lasso_screen = self
                .selection_lasso_world
                .iter()
                .copied()
                .map(|point| self.camera.world_to_screen(point, rect))
                .collect::<Vec<_>>();
            ui.painter()
                .add(egui::Shape::line(lasso_screen.clone(), selection_stroke));
            if let (Some(first), Some(last)) =
                (lasso_screen.first().copied(), lasso_screen.last().copied())
            {
                ui.painter().line_segment(
                    [last, first],
                    egui::Stroke::new(1.0, selection_color.gamma_multiply(0.7)),
                );
            }
        }

        let selection_count = self.active_object_selection_count();
        let selection_elements = self.active_object_selection_elements_snapshot();
        response.context_menu(|ui| {
            if let LayerId::Mask(id) = self.active_layer {
                self.validate_mask_polygon_selection();
                let selected_polygon_idx = self
                    .selected_mask_polygon
                    .filter(|selection| selection.layer_id == id)
                    .map(|selection| selection.polygon_idx);
                if let Some(polygon_idx) = selected_polygon_idx {
                    ui.label(format!("Selected polygon {}", polygon_idx + 1));
                    if ui.button("Delete polygon").clicked() {
                        if self.delete_selected_mask_polygon() {
                            self.bump_render_id();
                        }
                        ui.close();
                    }
                } else {
                    ui.label("No polygon selected.");
                }
                if ui
                    .add_enabled(
                        self.mask_undo_available(),
                        egui::Button::new("Undo last edit"),
                    )
                    .clicked()
                {
                    if self.request_native_mask_undo() {
                        self.bump_render_id();
                    }
                    ui.close();
                }
                return;
            }

            if selection_count == 0 || !self.active_layer_supports_spatial_selection() {
                ui.label("No selected cells.");
                return;
            }

            ui.label(format!("Selected cells: {selection_count}"));
            if ui
                .button("New selection element from selected cells")
                .clicked()
            {
                let _ = self.create_selection_element_from_active_selection();
                ui.close();
            }
            ui.menu_button("Add selected cells to element", |ui| {
                if selection_elements.is_empty() {
                    ui.label("No selection elements.");
                    return;
                }
                for (idx, name, count) in &selection_elements {
                    if ui.button(format!("{name} ({count})")).clicked() {
                        let _ = self.add_active_selection_to_element(*idx);
                        ui.close();
                    }
                }
            });
            if ui.button("Clear selection").clicked() {
                match self.active_layer {
                    LayerId::SegmentationObjects => self.seg_objects.clear_selection(),
                    LayerId::SpatialShape(id) => {
                        if let Some(layer) = self
                            .spatial_layers
                            .shapes
                            .iter_mut()
                            .find(|layer| layer.id == id)
                        {
                            layer.clear_selection();
                        }
                    }
                    _ => {}
                }
                ui.close();
            }
        });

        let screenshot = self
            .screenshot_pending
            .iter()
            .position(|pending| pending.viewport_id == *viewport_id)
            .and_then(|index| self.screenshot_pending.remove(index))
            .map(|pending| pending.request);
        if let Some(request) = screenshot.as_ref() {
            self.screenshot_in_flight
                .insert(request.id, viewport_id.clone());
        }
        let screenshot_active = screenshot.is_some();

        // HUD (disabled while capturing screenshots).
        if !screenshot_active && self.show_hud {
            let hud = format!(
                "level {target_level} zoom {:.3}  center ({:.0}, {:.0})",
                self.camera.zoom_screen_per_lvl0_px,
                self.camera.center_world_lvl0.x,
                self.camera.center_world_lvl0.y
            );
            canvas_overlays::paint_hud(ui, rect, hud);
        }

        // Scale bar (bottom-left). Uses microns if the dataset encodes physical units;
        // otherwise falls back to pixels.
        let draw_scale_bar = screenshot
            .as_ref()
            .map(|s| s.settings.include_scale_bar)
            .unwrap_or(self.show_scale_bar);
        if draw_scale_bar && self.view_plane_is_xy() {
            canvas_overlays::paint_scale_bar(
                ui,
                rect,
                canvas_overlays::ScaleBarParams {
                    zoom_screen_per_lvl0_px: self.camera.zoom_screen_per_lvl0_px,
                    um_per_lvl0_px: self.dataset_pixel_size_um(),
                    scale: screenshot
                        .as_ref()
                        .map(|s| s.settings.scale_bar_scale)
                        .unwrap_or(1.0),
                },
            );
        }

        // Legend (bottom-right) for visible channels (screenshot-only for now).
        if screenshot
            .as_ref()
            .is_some_and(|s| s.settings.include_legend)
        {
            let groups = self.current_layer_groups();
            let order = if self.channel_layer_order.len() == self.channels.len() {
                self.channel_layer_order.clone()
            } else {
                (0..self.channels.len()).collect()
            };
            let mut entries: Vec<(egui::Color32, String)> = Vec::new();
            for idx in order {
                let Some(ch) = self.channels.get(idx) else {
                    continue;
                };
                if !ch.visible {
                    continue;
                }
                let rgb = layer_groups::effective_channel_color_rgb(
                    &groups,
                    ch.name.as_str(),
                    ch.color_rgb,
                );
                entries.push((
                    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]),
                    ch.name.clone(),
                ));
            }
            canvas_overlays::paint_marker_legend(
                ui,
                rect,
                &entries,
                screenshot
                    .as_ref()
                    .map(|s| s.settings.legend_scale)
                    .unwrap_or(1.0),
            );
        }

        // Transform gizmo overlay (for channels only).
        if self.view_plane_is_xy() && self.tool_mode == ToolMode::TransformLayer {
            if let LayerId::Channel(ch_idx) = self.active_layer {
                self.draw_channel_transform_gizmo(ui, rect, ch_idx);
            }
        }

        // Loading indicator (top-right). Avoid capturing transient spinners in screenshots.
        if !screenshot_active {
            if self.show_tile_debug {
                canvas_overlays::paint_hud(
                    ui,
                    rect.translate(egui::vec2(0.0, 18.0)),
                    self.tile_debug_overlay_text(),
                );
            }
            let loading_text = self.loading_indicator_text();
            let tile_loading_count = self.image_tile_request_count();
            let spinner_text = if self.show_tile_debug && tile_loading_count > 0 {
                Some(format!("{tile_loading_count} tiles"))
            } else {
                None
            };
            canvas_overlays::paint_spinner(
                ui,
                rect,
                loading_text.is_some(),
                spinner_text.as_deref(),
            );
            if let Some(text) = loading_text {
                canvas_overlays::paint_loading_badge(ui, rect, text);
            }
        }

        // Always-on hover tooltip (active layer only). Avoid capturing tooltips in screenshots.
        if !screenshot_active {
            // Important: when Segmentation Objects is the active layer, hover picking can be expensive
            // at low zoom. Avoid doing that work while the camera is actively moving.
            self.ui_active_layer_tooltip(ui, ctx, rect, &response, camera_changed);
        }

        // Screenshot capture: read back the canvas pixels after overlays have been drawn.
        if let Some(spec) = screenshot {
            let tx = self.screenshot_worker.tx.clone();
            let id = spec.id;
            let path = spec.path.clone();
            let capture_rect = rect;
            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                let viewport = info.viewport;
                let ppp = info.pixels_per_point.max(1e-6);
                let viewport_w_px = (viewport.width().max(1.0) * ppp).round().max(1.0) as i32;
                let viewport_h_px = (viewport.height().max(1.0) * ppp).round().max(1.0) as i32;

                let x_px = ((capture_rect.min.x - viewport.min.x) * ppp)
                    .round()
                    .max(0.0) as i32;
                let y_px = ((viewport.max.y - capture_rect.max.y) * ppp)
                    .round()
                    .max(0.0) as i32;
                let mut w_px = (capture_rect.width() * ppp).round().max(1.0) as i32;
                let mut h_px = (capture_rect.height() * ppp).round().max(1.0) as i32;

                if x_px >= viewport_w_px || y_px >= viewport_h_px {
                    return;
                }
                if x_px + w_px > viewport_w_px {
                    w_px = (viewport_w_px - x_px).max(1);
                }
                if y_px + h_px > viewport_h_px {
                    h_px = (viewport_h_px - y_px).max(1);
                }
                if w_px <= 0 || h_px <= 0 {
                    return;
                }

                let gl = painter.gl();
                let mut rgba = vec![0u8; (w_px as usize) * (h_px as usize) * 4];
                unsafe {
                    let gl_ref = gl.as_ref();
                    gl_ref.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
                    gl_ref.read_pixels(
                        x_px,
                        y_px,
                        w_px,
                        h_px,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::Slice(Some(rgba.as_mut_slice())),
                    );
                }
                let _ = tx.send(ScreenshotWorkerMsg::SavePng {
                    id,
                    path: path.clone(),
                    width: w_px as usize,
                    height: h_px as usize,
                    rgba_bottom_up: rgba,
                });
            });
            ui.painter().add(egui::PaintCallback {
                rect,
                callback: Arc::new(cb),
            });
        }
        activate_viewport
    }
}
