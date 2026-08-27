//! Offscreen, world-aligned object-ID fill tiles.

use super::*;

const OBJECT_FILL_ID_TILE_SIZE_PX: i32 = 512;
const MAX_ID_TILES_COMPLETED_PER_FRAME: usize = 2;
const MAX_ID_TILE_VERTICES_PER_FRAME: usize = 300_000;
pub(super) const MAX_PENDING_ID_TILES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectFillTileKey {
    pub resource_cache_id: u64,
    pub geometry_generation: u64,
    pub level: u8,
    pub tile_x: i32,
    pub tile_y: i32,
}

#[derive(Debug, Clone)]
pub struct ObjectFillTileGeometry {
    pub cache_id: u64,
    pub generation: u64,
    pub bounds_local: egui::Rect,
    pub vertices_local: Arc<Vec<[f32; 3]>>,
}

#[derive(Debug, Clone)]
pub struct ObjectFillTileDrawItem {
    pub key: ObjectFillTileKey,
    pub bounds_local: egui::Rect,
    pub geometry: Vec<ObjectFillTileGeometry>,
}

#[derive(Debug, Clone)]
pub struct ObjectFillTileStyle {
    pub style_cache_id: u64,
    pub state_cache_id: u64,
    pub object_count: usize,
    pub state_generation: u64,
    pub object_state: Arc<Vec<u8>>,
    pub color_cache_id: u64,
    pub color_generation: u64,
    pub object_colors_rgba: Option<Arc<Vec<[u8; 4]>>>,
    pub selected_color: egui::Color32,
    pub primary_color: egui::Color32,
    pub object_color_opacity: f32,
    pub selection_overlay: Option<ObjectFillTileSelectionStyle>,
}

#[derive(Debug, Clone)]
pub struct ObjectFillTileSelectionStyle {
    pub state_cache_id: u64,
    pub state_generation: u64,
    pub object_state: Arc<Vec<u8>>,
    pub selected_color: egui::Color32,
    pub primary_color: egui::Color32,
}

#[derive(Debug, Clone)]
pub struct ObjectFillTileGlParams {
    pub frame_generation: u64,
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub visible: bool,
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectFillTilePaintResult {
    pub supported: bool,
    pub requested: usize,
    pub ready: usize,
    pub generated: usize,
    pub pending: usize,
    pub discarded: usize,
    pub raster_vertices: usize,
}

pub(super) struct ObjectFillIdTileGpu {
    pub(super) texture: glow::Texture,
    pub(super) bytes: usize,
}

pub(super) struct ObjectFillPendingIdTileGpu {
    texture: glow::Texture,
    bytes: usize,
    next_geometry: usize,
    next_vertex: usize,
}

#[derive(Clone)]
pub(super) struct ObjectFillTileGlObjects {
    id_program: glow::Program,
    compose_program: glow::Program,
    vao: glow::VertexArray,
    quad_vbo: glow::Buffer,
    framebuffer: glow::Framebuffer,
    id_u_tile_min: Option<glow::UniformLocation>,
    id_u_tile_size: Option<glow::UniformLocation>,
    compose_u_id_tex: Option<glow::UniformLocation>,
    compose_u_state_tex: Option<glow::UniformLocation>,
    compose_u_state_tex_size: Option<glow::UniformLocation>,
    compose_u_color_tex: Option<glow::UniformLocation>,
    compose_u_color_tex_size: Option<glow::UniformLocation>,
    compose_u_use_object_colors: Option<glow::UniformLocation>,
    compose_u_object_color_opacity: Option<glow::UniformLocation>,
    compose_u_selected_color: Option<glow::UniformLocation>,
    compose_u_primary_color: Option<glow::UniformLocation>,
    compose_u_selection_tex: Option<glow::UniformLocation>,
    compose_u_selection_tex_size: Option<glow::UniformLocation>,
    compose_u_use_selection_overlay: Option<glow::UniformLocation>,
    compose_u_selection_selected_color: Option<glow::UniformLocation>,
    compose_u_selection_primary_color: Option<glow::UniformLocation>,
}

impl ObjectFillTileGlObjects {
    pub(super) fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let id_program = compile_program_with_attributes(
            gl,
            ID_TILE_VERT_330,
            ID_TILE_FRAG_330,
            &[(0, "a_pos"), (1, "a_object_id")],
        )?;
        let compose_program = compile_program_with_attributes(
            gl,
            ID_TILE_COMPOSE_VERT_330,
            ID_TILE_COMPOSE_FRAG_330,
            &[(0, "a_pos_ndc"), (1, "a_uv")],
        )?;
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|error| anyhow!("create object fill tile VAO failed: {error}"))?
        };
        let quad_vbo = unsafe {
            gl.create_buffer()
                .map_err(|error| anyhow!("create object fill tile quad VBO failed: {error}"))?
        };
        let framebuffer = unsafe {
            gl.create_framebuffer()
                .map_err(|error| anyhow!("create object fill tile framebuffer failed: {error}"))?
        };
        Ok(Self {
            id_program,
            compose_program,
            vao,
            quad_vbo,
            framebuffer,
            id_u_tile_min: unsafe { gl.get_uniform_location(id_program, "u_tile_min") },
            id_u_tile_size: unsafe { gl.get_uniform_location(id_program, "u_tile_size") },
            compose_u_id_tex: unsafe { gl.get_uniform_location(compose_program, "u_id_tex") },
            compose_u_state_tex: unsafe { gl.get_uniform_location(compose_program, "u_state_tex") },
            compose_u_state_tex_size: unsafe {
                gl.get_uniform_location(compose_program, "u_state_tex_size")
            },
            compose_u_color_tex: unsafe { gl.get_uniform_location(compose_program, "u_color_tex") },
            compose_u_color_tex_size: unsafe {
                gl.get_uniform_location(compose_program, "u_color_tex_size")
            },
            compose_u_use_object_colors: unsafe {
                gl.get_uniform_location(compose_program, "u_use_object_colors")
            },
            compose_u_object_color_opacity: unsafe {
                gl.get_uniform_location(compose_program, "u_object_color_opacity")
            },
            compose_u_selected_color: unsafe {
                gl.get_uniform_location(compose_program, "u_selected_color")
            },
            compose_u_primary_color: unsafe {
                gl.get_uniform_location(compose_program, "u_primary_color")
            },
            compose_u_selection_tex: unsafe {
                gl.get_uniform_location(compose_program, "u_selection_tex")
            },
            compose_u_selection_tex_size: unsafe {
                gl.get_uniform_location(compose_program, "u_selection_tex_size")
            },
            compose_u_use_selection_overlay: unsafe {
                gl.get_uniform_location(compose_program, "u_use_selection_overlay")
            },
            compose_u_selection_selected_color: unsafe {
                gl.get_uniform_location(compose_program, "u_selection_selected_color")
            },
            compose_u_selection_primary_color: unsafe {
                gl.get_uniform_location(compose_program, "u_selection_primary_color")
            },
        })
    }
}

impl ObjectFillGlRenderer {
    pub fn id_tiles_have_coverage(&self, keys: &[ObjectFillTileKey]) -> bool {
        let inner = self.inner.lock();
        inner.tile_gl_objects.is_some()
            && !keys.is_empty()
            && keys.iter().all(|key| inner.id_tile_has_coverage(*key))
    }

    pub fn id_tiles_have_any_coverage(&self, keys: &[ObjectFillTileKey]) -> bool {
        let inner = self.inner.lock();
        inner.tile_gl_objects.is_some() && keys.iter().any(|key| inner.id_tile_has_coverage(*key))
    }

    pub fn paint_id_tiles(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        request_items: &[ObjectFillTileDrawItem],
        draw_items: &[ObjectFillTileDrawItem],
        styles: &[ObjectFillTileStyle],
        params: &ObjectFillTileGlParams,
        compose: bool,
    ) -> ObjectFillTilePaintResult {
        let mut result = ObjectFillTilePaintResult {
            requested: draw_items.len(),
            ..ObjectFillTilePaintResult::default()
        };
        if !params.visible || request_items.is_empty() || draw_items.is_empty() || styles.is_empty()
        {
            return result;
        }
        let gl = painter.gl();
        if gl.version().major < 3 {
            return result;
        }

        let mut inner = self.inner.lock();
        if !inner.tile_gl_init_attempted {
            inner.tile_gl_init_attempted = true;
            inner.tile_gl_objects = ObjectFillTileGlObjects::new(gl).ok();
        }
        let Some(tile_gl) = inner.tile_gl_objects.clone() else {
            return result;
        };
        result.supported = true;
        inner.delete_queued(gl);
        inner.begin_tile_frame(params.frame_generation);
        inner.tile_visible = inner.tile_visible.saturating_add(draw_items.len());
        inner.tile_requests = inner
            .tile_requests
            .saturating_add(request_items.len() as u64);

        let mut previous_viewport = [0i32; 4];
        let mut previous_scissor = [0i32; 4];
        let previous_scissor_enabled;
        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.get_parameter_i32_slice(glow::VIEWPORT, &mut previous_viewport);
            gl_ref.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut previous_scissor);
            previous_scissor_enabled = gl_ref.get_parameter_bool(glow::SCISSOR_TEST);
        }

        let raster_started = Instant::now();
        let mut generated = 0usize;
        let mut raster_vertices = 0usize;
        let discarded_before = inner.tile_discarded;
        for item in request_items {
            if inner.id_tiles.contains(&item.key) {
                inner.tile_hits = inner.tile_hits.saturating_add(1);
                continue;
            }
            if inner.tile_frame_generated >= MAX_ID_TILES_COMPLETED_PER_FRAME
                || inner.tile_frame_raster_vertices >= MAX_ID_TILE_VERTICES_PER_FRAME
            {
                break;
            }
            let remaining =
                MAX_ID_TILE_VERTICES_PER_FRAME.saturating_sub(inner.tile_frame_raster_vertices);
            let Some((tile, consumed)) = inner.advance_id_tile(gl, &tile_gl, item, remaining)
            else {
                continue;
            };
            raster_vertices = raster_vertices.saturating_add(consumed);
            inner.tile_frame_raster_vertices =
                inner.tile_frame_raster_vertices.saturating_add(consumed);
            if let Some(tile) = tile {
                inner.insert_id_tile(item.key, tile);
                generated += 1;
                inner.tile_frame_generated = inner.tile_frame_generated.saturating_add(1);
            }
        }
        result.generated = generated;
        result.raster_vertices = raster_vertices;
        result.discarded = inner.tile_discarded.saturating_sub(discarded_before) as usize;
        inner.last_tile_raster_vertices = inner
            .last_tile_raster_vertices
            .saturating_add(raster_vertices as u64);
        inner.total_tile_raster_vertices = inner
            .total_tile_raster_vertices
            .saturating_add(raster_vertices as u64);
        inner.last_tile_raster_ms += raster_started.elapsed().as_secs_f64() * 1_000.0;
        inner.tile_peak_pending = inner.tile_peak_pending.max(inner.pending_id_tiles.len());

        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl_ref.viewport(
                previous_viewport[0],
                previous_viewport[1],
                previous_viewport[2],
                previous_viewport[3],
            );
        }

        result.ready = draw_items
            .iter()
            .filter(|item| inner.id_tiles.contains(&item.key))
            .count();
        result.pending = result.requested.saturating_sub(result.ready);
        inner.tile_pending = inner.pending_id_tiles.len();

        let compose_started = Instant::now();
        if compose {
            inner.compose_id_tiles(gl, &tile_gl, &info, draw_items, styles, params);
        }
        inner.last_tile_compose_ms += compose_started.elapsed().as_secs_f64() * 1_000.0;

        unsafe {
            let gl_ref = gl.as_ref();
            if previous_scissor_enabled {
                gl_ref.enable(glow::SCISSOR_TEST);
            } else {
                gl_ref.disable(glow::SCISSOR_TEST);
            }
            gl_ref.scissor(
                previous_scissor[0],
                previous_scissor[1],
                previous_scissor[2],
                previous_scissor[3],
            );
        }
        inner.delete_queued(gl);
        result
    }
}

impl ObjectFillInner {
    fn begin_tile_frame(&mut self, frame_generation: u64) {
        if self.tile_frame_generation == frame_generation {
            return;
        }
        self.tile_frame_generation = frame_generation;
        self.tile_frame_generated = 0;
        self.tile_frame_raster_vertices = 0;
        self.tile_request_generation = self.tile_request_generation.wrapping_add(1).max(1);
        self.tile_visible = 0;
        self.last_tile_raster_vertices = 0;
        self.last_tile_raster_draw_calls = 0;
        self.last_tile_compose_draw_calls = 0;
        self.last_tile_selection_compose_draw_calls = 0;
        self.last_tile_raster_ms = 0.0;
        self.last_tile_compose_ms = 0.0;
    }

    fn id_tile_has_coverage(&self, key: ObjectFillTileKey) -> bool {
        if self.id_tiles.contains(&key) {
            return true;
        }
        (1..=8).any(|level_delta| {
            object_fill_tile_ancestor_key(key, level_delta)
                .is_some_and(|ancestor| self.id_tiles.contains(&ancestor))
        })
    }

    fn id_tile_for_draw(&mut self, key: ObjectFillTileKey) -> Option<(glow::Texture, [f32; 4])> {
        if let Some(texture) = self.id_tiles.get(&key).map(|tile| tile.texture) {
            return Some((texture, [0.0, 1.0, 1.0, 0.0]));
        }
        for level_delta in 1..=8 {
            let Some(ancestor) = object_fill_tile_ancestor_key(key, level_delta) else {
                break;
            };
            let Some(texture) = self.id_tiles.get(&ancestor).map(|tile| tile.texture) else {
                continue;
            };
            let factor = 1i32.checked_shl(level_delta.into()).unwrap_or(i32::MAX);
            let relative_x = key.tile_x.rem_euclid(factor) as f32;
            let relative_y = key.tile_y.rem_euclid(factor) as f32;
            let factor = factor as f32;
            let u_min = relative_x / factor;
            let u_max = (relative_x + 1.0) / factor;
            let v_top = 1.0 - relative_y / factor;
            let v_bottom = 1.0 - (relative_y + 1.0) / factor;
            return Some((texture, [u_min, u_max, v_top, v_bottom]));
        }
        None
    }

    fn advance_id_tile(
        &mut self,
        gl: &Arc<glow::Context>,
        tile_gl: &ObjectFillTileGlObjects,
        item: &ObjectFillTileDrawItem,
        vertex_budget: usize,
    ) -> Option<(Option<ObjectFillIdTileGpu>, usize)> {
        if vertex_budget < 3 {
            return Some((None, 0));
        }
        let mut pending = if let Some(pending) = self.pending_id_tiles.pop(&item.key) {
            pending
        } else {
            self.create_pending_id_tile(gl, tile_gl)?
        };

        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(tile_gl.framebuffer));
            gl_ref.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(pending.texture),
                0,
            );
            if gl_ref.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
                self.tile_pending_bytes = self.tile_pending_bytes.saturating_sub(pending.bytes);
                self.tile_discarded = self.tile_discarded.saturating_add(1);
                self.textures_to_delete.push(pending.texture);
                return None;
            }
            gl_ref.viewport(
                0,
                0,
                OBJECT_FILL_ID_TILE_SIZE_PX,
                OBJECT_FILL_ID_TILE_SIZE_PX,
            );
            gl_ref.disable(glow::SCISSOR_TEST);
            gl_ref.disable(glow::BLEND);
            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);
            gl_ref.use_program(Some(tile_gl.id_program));
            gl_ref.bind_vertex_array(Some(tile_gl.vao));
            gl_ref.uniform_2_f32(
                tile_gl.id_u_tile_min.as_ref(),
                item.bounds_local.min.x,
                item.bounds_local.min.y,
            );
            gl_ref.uniform_2_f32(
                tile_gl.id_u_tile_size.as_ref(),
                item.bounds_local.width().max(1.0e-6),
                item.bounds_local.height().max(1.0e-6),
            );
        }

        let mut consumed = 0usize;
        let mut failed = false;
        while pending.next_geometry < item.geometry.len() && consumed + 3 <= vertex_budget {
            let geometry = &item.geometry[pending.next_geometry];
            let Some((vbo, count)) = self
                .ensure_object_mesh_uploaded(
                    gl,
                    item.key.resource_cache_id,
                    geometry.cache_id,
                    geometry.generation,
                    geometry.vertices_local.as_slice(),
                )
                .map(|mesh| (mesh.vbo, mesh.count))
            else {
                failed = true;
                break;
            };
            if pending.next_vertex >= count {
                pending.next_geometry += 1;
                pending.next_vertex = 0;
                continue;
            }
            let available = vertex_budget.saturating_sub(consumed);
            let draw_count = (count - pending.next_vertex).min(available) / 3 * 3;
            if draw_count == 0 {
                break;
            }
            let scissor = id_tile_scissor_box(geometry.bounds_local, item.bounds_local);
            if scissor[2] <= 0 || scissor[3] <= 0 {
                pending.next_geometry += 1;
                pending.next_vertex = 0;
                continue;
            }
            unsafe {
                let gl_ref = gl.as_ref();
                gl_ref.enable(glow::SCISSOR_TEST);
                gl_ref.scissor(scissor[0], scissor[1], scissor[2], scissor[3]);
                gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                gl_ref.enable_vertex_attrib_array(0);
                gl_ref.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 12, 0);
                gl_ref.enable_vertex_attrib_array(1);
                gl_ref.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 12, 8);
                gl_ref.draw_arrays(
                    glow::TRIANGLES,
                    pending.next_vertex as i32,
                    draw_count as i32,
                );
            }
            self.last_tile_raster_draw_calls = self.last_tile_raster_draw_calls.saturating_add(1);
            pending.next_vertex += draw_count;
            consumed += draw_count;
            if pending.next_vertex == count {
                pending.next_geometry += 1;
                pending.next_vertex = 0;
            }
        }

        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, None);
            gl_ref.bind_vertex_array(None);
            gl_ref.use_program(None);
            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
        }

        if failed {
            self.tile_pending_bytes = self.tile_pending_bytes.saturating_sub(pending.bytes);
            self.tile_discarded = self.tile_discarded.saturating_add(1);
            self.textures_to_delete.push(pending.texture);
            return None;
        }
        if pending.next_geometry >= item.geometry.len() {
            self.tile_pending_bytes = self.tile_pending_bytes.saturating_sub(pending.bytes);
            self.tile_generations = self.tile_generations.saturating_add(1);
            return Some((
                Some(ObjectFillIdTileGpu {
                    texture: pending.texture,
                    bytes: pending.bytes,
                }),
                consumed,
            ));
        }
        if let Some((_key, evicted)) = self.pending_id_tiles.push(item.key, pending) {
            self.tile_pending_bytes = self.tile_pending_bytes.saturating_sub(evicted.bytes);
            self.tile_discarded = self.tile_discarded.saturating_add(1);
            self.textures_to_delete.push(evicted.texture);
        }
        Some((None, consumed))
    }

    fn create_pending_id_tile(
        &mut self,
        gl: &Arc<glow::Context>,
        tile_gl: &ObjectFillTileGlObjects,
    ) -> Option<ObjectFillPendingIdTileGpu> {
        let bytes = (OBJECT_FILL_ID_TILE_SIZE_PX as usize)
            .saturating_mul(OBJECT_FILL_ID_TILE_SIZE_PX as usize)
            .saturating_mul(std::mem::size_of::<u32>());
        if bytes > self.tile_budget_bytes {
            return None;
        }
        while self
            .tile_bytes
            .saturating_add(self.tile_pending_bytes)
            .saturating_add(bytes)
            > self.tile_budget_bytes
        {
            if let Some((_key, evicted)) = self.id_tiles.pop_lru() {
                self.tile_bytes = self.tile_bytes.saturating_sub(evicted.bytes);
                self.tile_evictions = self.tile_evictions.saturating_add(1);
                self.textures_to_delete.push(evicted.texture);
                continue;
            }
            let (_key, evicted) = self.pending_id_tiles.pop_lru()?;
            self.tile_pending_bytes = self.tile_pending_bytes.saturating_sub(evicted.bytes);
            self.tile_discarded = self.tile_discarded.saturating_add(1);
            self.textures_to_delete.push(evicted.texture);
        }

        let texture = unsafe { gl.as_ref().create_texture().ok()? };
        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl_ref.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl_ref.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl_ref.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl_ref.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl_ref.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R32UI as i32,
                OBJECT_FILL_ID_TILE_SIZE_PX,
                OBJECT_FILL_ID_TILE_SIZE_PX,
                0,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(None),
            );
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(tile_gl.framebuffer));
            gl_ref.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            if gl_ref.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
                gl_ref.delete_texture(texture);
                return None;
            }
            gl_ref.viewport(
                0,
                0,
                OBJECT_FILL_ID_TILE_SIZE_PX,
                OBJECT_FILL_ID_TILE_SIZE_PX,
            );
            gl_ref.disable(glow::SCISSOR_TEST);
            gl_ref.disable(glow::BLEND);
            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);
            gl_ref.clear_buffer_u32_slice(glow::COLOR, 0, &[0, 0, 0, 0]);
            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
        self.tile_pending_bytes = self.tile_pending_bytes.saturating_add(bytes);
        Some(ObjectFillPendingIdTileGpu {
            texture,
            bytes,
            next_geometry: 0,
            next_vertex: 0,
        })
    }

    fn insert_id_tile(&mut self, key: ObjectFillTileKey, tile: ObjectFillIdTileGpu) {
        let bytes = tile.bytes;
        if let Some((_old_key, evicted)) = self.id_tiles.push(key, tile) {
            self.tile_bytes = self.tile_bytes.saturating_sub(evicted.bytes);
            self.tile_evictions = self.tile_evictions.saturating_add(1);
            self.textures_to_delete.push(evicted.texture);
        }
        self.tile_bytes = self.tile_bytes.saturating_add(bytes);
        while self.tile_bytes.saturating_add(self.tile_pending_bytes) > self.tile_budget_bytes {
            let Some((_old_key, evicted)) = self.id_tiles.pop_lru() else {
                break;
            };
            self.tile_bytes = self.tile_bytes.saturating_sub(evicted.bytes);
            self.tile_evictions = self.tile_evictions.saturating_add(1);
            self.textures_to_delete.push(evicted.texture);
        }
    }

    fn compose_id_tiles(
        &mut self,
        gl: &Arc<glow::Context>,
        tile_gl: &ObjectFillTileGlObjects,
        info: &egui::PaintCallbackInfo,
        items: &[ObjectFillTileDrawItem],
        styles: &[ObjectFillTileStyle],
        params: &ObjectFillTileGlParams,
    ) {
        let viewport = info.viewport;
        let ppp = info.pixels_per_point.max(1.0e-6);
        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);
            gl_ref.enable(glow::BLEND);
            gl_ref.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl_ref.use_program(Some(tile_gl.compose_program));
            gl_ref.bind_vertex_array(Some(tile_gl.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(tile_gl.quad_vbo));
            gl_ref.uniform_1_i32(tile_gl.compose_u_id_tex.as_ref(), 0);
            gl_ref.uniform_1_i32(tile_gl.compose_u_state_tex.as_ref(), 1);
            gl_ref.uniform_1_i32(tile_gl.compose_u_color_tex.as_ref(), 2);
            gl_ref.uniform_1_i32(tile_gl.compose_u_selection_tex.as_ref(), 3);
        }

        for style in styles {
            unsafe {
                gl.as_ref().active_texture(glow::TEXTURE0);
            }
            let Some((state_texture, state_width, state_height)) = self
                .ensure_state_uploaded(
                    gl,
                    style.style_cache_id,
                    style.state_cache_id,
                    style.state_generation,
                    style.object_count,
                    style.object_state.as_slice(),
                )
                .map(|state| (state.texture, state.width, state.height))
            else {
                continue;
            };
            let color_texture = style.object_colors_rgba.as_ref().and_then(|colors| {
                self.ensure_color_uploaded(
                    gl,
                    style.style_cache_id,
                    style.color_cache_id,
                    style.color_generation,
                    style.object_count,
                    colors.as_slice(),
                )
                .map(|color| (color.texture, color.width, color.height))
            });
            let selection_texture = style.selection_overlay.as_ref().and_then(|selection| {
                self.ensure_state_uploaded(
                    gl,
                    style.style_cache_id,
                    selection.state_cache_id,
                    selection.state_generation,
                    style.object_count,
                    selection.object_state.as_slice(),
                )
                .map(|state| (state.texture, state.width, state.height))
            });
            let selected = color_f32(style.selected_color);
            let primary = color_f32(style.primary_color);
            let selection_selected = style
                .selection_overlay
                .as_ref()
                .map_or([0.0; 4], |selection| color_f32(selection.selected_color));
            let selection_primary = style
                .selection_overlay
                .as_ref()
                .map_or([0.0; 4], |selection| color_f32(selection.primary_color));
            unsafe {
                let gl_ref = gl.as_ref();
                gl_ref.use_program(Some(tile_gl.compose_program));
                gl_ref.uniform_2_i32(
                    tile_gl.compose_u_state_tex_size.as_ref(),
                    state_width,
                    state_height,
                );
                gl_ref.uniform_1_i32(
                    tile_gl.compose_u_use_object_colors.as_ref(),
                    i32::from(color_texture.is_some()),
                );
                gl_ref.uniform_1_f32(
                    tile_gl.compose_u_object_color_opacity.as_ref(),
                    style.object_color_opacity.clamp(0.0, 1.0),
                );
                gl_ref.uniform_4_f32_slice(tile_gl.compose_u_selected_color.as_ref(), &selected);
                gl_ref.uniform_4_f32_slice(tile_gl.compose_u_primary_color.as_ref(), &primary);
                gl_ref.uniform_1_i32(
                    tile_gl.compose_u_use_selection_overlay.as_ref(),
                    i32::from(selection_texture.is_some()),
                );
                gl_ref.uniform_4_f32_slice(
                    tile_gl.compose_u_selection_selected_color.as_ref(),
                    &selection_selected,
                );
                gl_ref.uniform_4_f32_slice(
                    tile_gl.compose_u_selection_primary_color.as_ref(),
                    &selection_primary,
                );
                gl_ref.active_texture(glow::TEXTURE1);
                gl_ref.bind_texture(glow::TEXTURE_2D, Some(state_texture));
                if let Some((texture, width, height)) = color_texture {
                    gl_ref.uniform_2_i32(tile_gl.compose_u_color_tex_size.as_ref(), width, height);
                    gl_ref.active_texture(glow::TEXTURE2);
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(texture));
                } else {
                    gl_ref.uniform_2_i32(tile_gl.compose_u_color_tex_size.as_ref(), 0, 0);
                }
                if let Some((texture, width, height)) = selection_texture {
                    gl_ref.uniform_2_i32(
                        tile_gl.compose_u_selection_tex_size.as_ref(),
                        width,
                        height,
                    );
                    gl_ref.active_texture(glow::TEXTURE3);
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(texture));
                } else {
                    gl_ref.uniform_2_i32(tile_gl.compose_u_selection_tex_size.as_ref(), 0, 0);
                }
            }

            for item in items {
                let Some((tile_texture, uv)) = self.id_tile_for_draw(item.key) else {
                    continue;
                };
                let screen_rect = tile_screen_rect(item.bounds_local, params, viewport);
                let clip = screen_rect.intersect(viewport).intersect(info.clip_rect);
                if !clip.is_positive() {
                    continue;
                }
                let scissor = object_fill_scissor_box(clip, ppp, info.screen_size_px);
                if scissor[2] <= 0 || scissor[3] <= 0 {
                    continue;
                }
                let vertices = tile_quad_vertices(screen_rect, viewport, ppp, uv);
                unsafe {
                    let gl_ref = gl.as_ref();
                    gl_ref.enable(glow::SCISSOR_TEST);
                    gl_ref.scissor(scissor[0], scissor[1], scissor[2], scissor[3]);
                    gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(tile_gl.quad_vbo));
                    gl_ref.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&vertices),
                        glow::STREAM_DRAW,
                    );
                    gl_ref.enable_vertex_attrib_array(0);
                    gl_ref.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
                    gl_ref.enable_vertex_attrib_array(1);
                    gl_ref.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);
                    gl_ref.active_texture(glow::TEXTURE0);
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(tile_texture));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
                self.last_tile_compose_draw_calls =
                    self.last_tile_compose_draw_calls.saturating_add(1);
                if style.selection_overlay.is_some() {
                    self.last_tile_selection_compose_draw_calls = self
                        .last_tile_selection_compose_draw_calls
                        .saturating_add(1);
                }
            }
        }

        unsafe {
            let gl_ref = gl.as_ref();
            for unit in [
                glow::TEXTURE3,
                glow::TEXTURE2,
                glow::TEXTURE1,
                glow::TEXTURE0,
            ] {
                gl_ref.active_texture(unit);
                gl_ref.bind_texture(glow::TEXTURE_2D, None);
            }
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, None);
            gl_ref.bind_vertex_array(None);
            gl_ref.use_program(None);
        }
    }
}

fn color_f32(color: egui::Color32) -> [f32; 4] {
    [
        color.r() as f32 / 255.0,
        color.g() as f32 / 255.0,
        color.b() as f32 / 255.0,
        color.a() as f32 / 255.0,
    ]
}

fn tile_screen_rect(
    local_rect: egui::Rect,
    params: &ObjectFillTileGlParams,
    viewport: egui::Rect,
) -> egui::Rect {
    let to_screen = |local: egui::Pos2| {
        let world = egui::pos2(
            params.local_to_world_offset.x + local.x * params.local_to_world_scale.x,
            params.local_to_world_offset.y + local.y * params.local_to_world_scale.y,
        );
        viewport.center() + (world - params.center_world) * params.zoom_screen_per_world.max(1.0e-6)
    };
    egui::Rect::from_two_pos(to_screen(local_rect.min), to_screen(local_rect.max))
}

fn id_tile_scissor_box(geometry: egui::Rect, tile: egui::Rect) -> [i32; 4] {
    let size = OBJECT_FILL_ID_TILE_SIZE_PX;
    let width = tile.width().max(1.0e-6);
    let height = tile.height().max(1.0e-6);
    let to_x = |x: f32| (((x - tile.min.x) / width) * size as f32).round() as i32;
    let to_y_from_top = |y: f32| (((y - tile.min.y) / height) * size as f32).round() as i32;
    let left = to_x(geometry.min.x).clamp(0, size);
    let right = to_x(geometry.max.x).clamp(left, size);
    let top = to_y_from_top(geometry.min.y).clamp(0, size);
    let bottom = to_y_from_top(geometry.max.y).clamp(top, size);
    [left, size - bottom, right - left, bottom - top]
}

fn tile_quad_vertices(
    screen_rect: egui::Rect,
    viewport: egui::Rect,
    pixels_per_point: f32,
    uv: [f32; 4],
) -> [f32; 24] {
    let ppp = pixels_per_point.max(1.0e-6);
    let snap = |value: f32| (value * ppp).round() / ppp;
    let x0 = ((snap(screen_rect.min.x) - viewport.min.x) / viewport.width().max(1.0)) * 2.0 - 1.0;
    let x1 = ((snap(screen_rect.max.x) - viewport.min.x) / viewport.width().max(1.0)) * 2.0 - 1.0;
    let y0 = 1.0 - ((snap(screen_rect.min.y) - viewport.min.y) / viewport.height().max(1.0)) * 2.0;
    let y1 = 1.0 - ((snap(screen_rect.max.y) - viewport.min.y) / viewport.height().max(1.0)) * 2.0;
    let [u_min, u_max, v_top, v_bottom] = uv;
    [
        x0, y0, u_min, v_top, x1, y0, u_max, v_top, x1, y1, u_max, v_bottom, x0, y0, u_min, v_top,
        x1, y1, u_max, v_bottom, x0, y1, u_min, v_bottom,
    ]
}

fn object_fill_tile_ancestor_key(
    key: ObjectFillTileKey,
    level_delta: u8,
) -> Option<ObjectFillTileKey> {
    let factor = 1i32.checked_shl(level_delta.into())?;
    Some(ObjectFillTileKey {
        level: key.level.checked_add(level_delta)?,
        tile_x: key.tile_x.div_euclid(factor),
        tile_y: key.tile_y.div_euclid(factor),
        ..key
    })
}

const ID_TILE_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_object_id;

uniform vec2 u_tile_min;
uniform vec2 u_tile_size;

flat out uint v_object_id;

void main() {
    vec2 rel = (a_pos - u_tile_min) / max(u_tile_size, vec2(1e-6));
    gl_Position = vec4(rel.x * 2.0 - 1.0, 1.0 - rel.y * 2.0, 0.0, 1.0);
    v_object_id = uint(a_object_id + 0.5) + 1u;
}"#;

const ID_TILE_FRAG_330: &str = r#"#version 330 core
flat in uint v_object_id;
layout(location = 0) out uint out_object_id;
void main() {
    out_object_id = v_object_id;
}"#;

const ID_TILE_COMPOSE_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos_ndc;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}"#;

const ID_TILE_COMPOSE_FRAG_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform usampler2D u_id_tex;
uniform sampler2D u_state_tex;
uniform ivec2 u_state_tex_size;
uniform sampler2D u_color_tex;
uniform ivec2 u_color_tex_size;
uniform int u_use_object_colors;
uniform float u_object_color_opacity;
uniform vec4 u_selected_color;
uniform vec4 u_primary_color;
uniform sampler2D u_selection_tex;
uniform ivec2 u_selection_tex_size;
uniform int u_use_selection_overlay;
uniform vec4 u_selection_selected_color;
uniform vec4 u_selection_primary_color;

out vec4 out_color;

void main() {
    uint stored_id = texture(u_id_tex, v_uv).r;
    if (stored_id == 0u || u_state_tex_size.x <= 0 || u_state_tex_size.y <= 0) {
        discard;
    }
    int object_id = int(stored_id - 1u);
    int state_x = object_id % u_state_tex_size.x;
    int state_y = object_id / u_state_tex_size.x;
    if (state_y < 0 || state_y >= u_state_tex_size.y) {
        discard;
    }
    float state = texelFetch(u_state_tex, ivec2(state_x, state_y), 0).r;
    if (state < 0.001) {
        discard;
    }
    if (u_use_selection_overlay != 0) {
        if (u_selection_tex_size.x <= 0 || u_selection_tex_size.y <= 0) {
            discard;
        }
        int selection_x = object_id % u_selection_tex_size.x;
        int selection_y = object_id / u_selection_tex_size.x;
        if (selection_y < 0 || selection_y >= u_selection_tex_size.y) {
            discard;
        }
        float selection_state = texelFetch(
            u_selection_tex,
            ivec2(selection_x, selection_y),
            0
        ).r;
        if (selection_state >= 0.001) {
            out_color = selection_state > 0.75
                ? u_selection_primary_color
                : u_selection_selected_color;
            return;
        }
    }
    if (u_use_object_colors != 0) {
        if (u_color_tex_size.x <= 0 || u_color_tex_size.y <= 0) {
            discard;
        }
        int color_x = object_id % u_color_tex_size.x;
        int color_y = object_id / u_color_tex_size.x;
        if (color_y < 0 || color_y >= u_color_tex_size.y) {
            discard;
        }
        vec4 object_color = texelFetch(u_color_tex, ivec2(color_x, color_y), 0);
        object_color.a *= u_object_color_opacity;
        if (object_color.a <= 0.0) {
            discard;
        }
        out_color = object_color;
        return;
    }
    out_color = state > 0.75 ? u_primary_color : u_selected_color;
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_key_excludes_camera_and_style_state() {
        let key = ObjectFillTileKey {
            resource_cache_id: 7,
            geometry_generation: 9,
            level: 3,
            tile_x: -2,
            tile_y: 4,
        };
        let same_geometry = ObjectFillTileKey { ..key };
        assert_eq!(key, same_geometry);
    }

    #[test]
    fn shared_tile_work_budget_resets_once_per_ui_frame() {
        let renderer = ObjectFillGlRenderer::new(4, 4);
        let mut inner = renderer.inner.lock();

        inner.begin_tile_frame(41);
        inner.tile_frame_generated = MAX_ID_TILES_COMPLETED_PER_FRAME;
        inner.tile_frame_raster_vertices = MAX_ID_TILE_VERTICES_PER_FRAME;
        inner.begin_tile_frame(41);

        assert_eq!(inner.tile_frame_generated, MAX_ID_TILES_COMPLETED_PER_FRAME);
        assert_eq!(
            inner.tile_frame_raster_vertices,
            MAX_ID_TILE_VERTICES_PER_FRAME
        );

        inner.begin_tile_frame(42);
        assert_eq!(inner.tile_frame_generated, 0);
        assert_eq!(inner.tile_frame_raster_vertices, 0);
    }

    #[test]
    fn quad_maps_local_top_to_texture_top() {
        let viewport = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(100.0, 100.0));
        let vertices = tile_quad_vertices(viewport, viewport, 1.0, [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(&vertices[0..4], &[-1.0, 1.0, 0.0, 1.0]);
        assert_eq!(&vertices[8..12], &[1.0, -1.0, 1.0, 0.0]);
    }

    #[test]
    fn ancestor_keys_use_euclidean_division_for_negative_world_tiles() {
        let key = ObjectFillTileKey {
            resource_cache_id: 1,
            geometry_generation: 2,
            level: 3,
            tile_x: -1,
            tile_y: -2,
        };
        let ancestor = object_fill_tile_ancestor_key(key, 2).expect("ancestor");
        assert_eq!(ancestor.level, 5);
        assert_eq!(ancestor.tile_x, -1);
        assert_eq!(ancestor.tile_y, -1);
    }

    #[test]
    fn spatial_bin_scissors_partition_an_id_tile_without_overlap() {
        let tile = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(4096.0, 4096.0));
        let top = id_tile_scissor_box(
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(4096.0, 2048.0)),
            tile,
        );
        let bottom = id_tile_scissor_box(
            egui::Rect::from_min_max(egui::pos2(0.0, 2048.0), egui::pos2(4096.0, 4096.0)),
            tile,
        );

        assert_eq!(top, [0, 256, 512, 256]);
        assert_eq!(bottom, [0, 0, 512, 256]);
    }
}
