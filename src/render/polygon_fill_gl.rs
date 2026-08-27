use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use anyhow::anyhow;
use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

mod tiles;
pub use tiles::{
    ObjectFillTileDrawItem, ObjectFillTileGeometry, ObjectFillTileGlParams, ObjectFillTileKey,
    ObjectFillTileSelectionStyle, ObjectFillTileStyle,
};

#[derive(Debug, Clone)]
pub struct PolygonFillGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub color: egui::Color32,
    pub visible: bool,
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
}

#[derive(Debug, Clone)]
pub struct PolygonFillGlDrawData {
    pub cache_id: u64,
    pub generation: u64,
    pub vertices_local: Arc<Vec<[f32; 2]>>,
}

#[derive(Debug, Clone)]
pub struct PolygonFillGlDrawItem {
    pub data: PolygonFillGlDrawData,
    pub params: PolygonFillGlDrawParams,
    pub visible_world: egui::Rect,
}

#[derive(Clone)]
pub struct PolygonFillGlRenderer {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for PolygonFillGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolygonFillGlRenderer")
            .finish_non_exhaustive()
    }
}

impl PolygonFillGlRenderer {
    pub fn new(max_meshes: usize) -> Self {
        let cap = NonZeroUsize::new(max_meshes.max(1)).unwrap();
        Self {
            inner: Arc::new(Mutex::new(Inner::new(cap))),
        }
    }

    pub fn paint_many(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        items: &[PolygonFillGlDrawItem],
    ) {
        if items.is_empty() {
            return;
        }

        let gl = painter.gl();
        if gl.version().major < 3 {
            return;
        }

        let mut inner = self.inner.lock();
        if inner.gl_objects.is_none() {
            inner.gl_objects = GlObjects::new(gl).ok();
        }
        inner.delete_queued(gl);

        let Some(objects) = inner.gl_objects.as_ref() else {
            return;
        };
        let program = objects.program;
        let vao = objects.vao;
        let u_center_world = objects.u_center_world.clone();
        let u_zoom_px = objects.u_zoom_px.clone();
        let u_viewport_min_px = objects.u_viewport_min_px.clone();
        let u_viewport_size_px = objects.u_viewport_size_px.clone();
        let u_color = objects.u_color.clone();
        let u_local_to_world_offset = objects.u_local_to_world_offset.clone();
        let u_local_to_world_scale = objects.u_local_to_world_scale.clone();

        let viewport_pt = info.viewport;
        let ppp = info.pixels_per_point.max(1e-6);
        let viewport_min_px = viewport_pt.min * ppp;
        let viewport_size_px = viewport_pt.size() * ppp;

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));
            gl.uniform_2_f32(
                u_viewport_min_px.as_ref(),
                viewport_min_px.x,
                viewport_min_px.y,
            );
            gl.uniform_2_f32(
                u_viewport_size_px.as_ref(),
                viewport_size_px.x.max(1.0),
                viewport_size_px.y.max(1.0),
            );
        }

        for item in items {
            if !item.params.visible || !item.visible_world.is_positive() {
                continue;
            }
            let Some(gpu) = inner.ensure_mesh_uploaded(
                gl,
                item.data.cache_id,
                item.data.generation,
                item.data.vertices_local.as_slice(),
            ) else {
                continue;
            };
            let c = item.params.color;
            let color = [
                c.r() as f32 / 255.0,
                c.g() as f32 / 255.0,
                c.b() as f32 / 255.0,
                c.a() as f32 / 255.0,
            ];
            unsafe {
                let gl = gl.as_ref();
                gl.uniform_2_f32(
                    u_center_world.as_ref(),
                    item.params.center_world.x,
                    item.params.center_world.y,
                );
                gl.uniform_1_f32(
                    u_zoom_px.as_ref(),
                    (item.params.zoom_screen_per_world.max(1e-6) * ppp).max(1e-6),
                );
                gl.uniform_4_f32_slice(u_color.as_ref(), &color);
                gl.uniform_2_f32(
                    u_local_to_world_offset.as_ref(),
                    item.params.local_to_world_offset.x,
                    item.params.local_to_world_offset.y,
                );
                gl.uniform_2_f32(
                    u_local_to_world_scale.as_ref(),
                    item.params.local_to_world_scale.x.max(1e-9),
                    item.params.local_to_world_scale.y.max(1e-9),
                );
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu.vbo));
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
                gl.draw_arrays(glow::TRIANGLES, 0, gpu.count as i32);
            }
        }

        unsafe {
            let gl = gl.as_ref();
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

struct MeshGpu {
    vbo: glow::Buffer,
    count: usize,
}

struct Inner {
    gl_objects: Option<GlObjects>,
    meshes: LruCache<(u64, u64), MeshGpu>,
    buffers_to_delete: Vec<glow::Buffer>,
}

impl Inner {
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            gl_objects: None,
            meshes: LruCache::new(cap),
            buffers_to_delete: Vec::new(),
        }
    }

    fn delete_queued(&mut self, gl: &Arc<glow::Context>) {
        if self.buffers_to_delete.is_empty() {
            return;
        }
        unsafe {
            let gl = gl.as_ref();
            for b in self.buffers_to_delete.drain(..) {
                gl.delete_buffer(b);
            }
        }
    }

    fn ensure_mesh_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        vertices_local: &[[f32; 2]],
    ) -> Option<&MeshGpu> {
        let key = (cache_id, generation);
        if self.meshes.contains(&key) {
            return self.meshes.get(&key);
        }
        if vertices_local.is_empty() {
            return None;
        }

        let vbo = unsafe { gl.as_ref().create_buffer().map_err(|_| ()).ok()? };
        unsafe {
            let gl = gl.as_ref();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(vertices_local),
                glow::STATIC_DRAW,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }

        if let Some((_k, ev)) = self.meshes.push(
            key,
            MeshGpu {
                vbo,
                count: vertices_local.len(),
            },
        ) {
            self.buffers_to_delete.push(ev.vbo);
        }
        self.delete_queued(gl);
        self.meshes.get(&key)
    }
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    u_center_world: Option<glow::UniformLocation>,
    u_zoom_px: Option<glow::UniformLocation>,
    u_viewport_min_px: Option<glow::UniformLocation>,
    u_viewport_size_px: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_local_to_world_offset: Option<glow::UniformLocation>,
    u_local_to_world_scale: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let program = compile_program(gl, VERT_330, FRAG_330)?;
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| anyhow!("create_vertex_array failed: {e}"))?
        };
        unsafe {
            gl.bind_vertex_array(Some(vao));
            gl.bind_vertex_array(None);
        }
        Ok(Self {
            program,
            vao,
            u_center_world: unsafe { gl.get_uniform_location(program, "u_center_world") },
            u_zoom_px: unsafe { gl.get_uniform_location(program, "u_zoom_px") },
            u_viewport_min_px: unsafe { gl.get_uniform_location(program, "u_viewport_min_px") },
            u_viewport_size_px: unsafe { gl.get_uniform_location(program, "u_viewport_size_px") },
            u_color: unsafe { gl.get_uniform_location(program, "u_color") },
            u_local_to_world_offset: unsafe {
                gl.get_uniform_location(program, "u_local_to_world_offset")
            },
            u_local_to_world_scale: unsafe {
                gl.get_uniform_location(program, "u_local_to_world_scale")
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct ObjectFillGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub selected_color: egui::Color32,
    pub primary_color: egui::Color32,
    pub visible: bool,
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
    pub object_color_opacity: f32,
}

#[derive(Debug, Clone)]
pub struct ObjectFillGlDrawData {
    pub cache_id: u64,
    pub state_cache_id: u64,
    pub generation: u64,
    pub vertices_local: Arc<Vec<[f32; 3]>>,
    pub object_count: usize,
    pub selection_generation: u64,
    pub selection_state: Arc<Vec<u8>>,
    pub color_cache_id: u64,
    pub color_generation: u64,
    pub object_colors_rgba: Option<Arc<Vec<[u8; 4]>>>,
}

#[derive(Debug, Clone)]
pub struct ObjectFillGlDrawItem {
    pub data: ObjectFillGlDrawData,
    pub params: ObjectFillGlDrawParams,
    pub visible_world: egui::Rect,
}

const DEFAULT_OBJECT_FILL_MESH_BUDGET_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_OBJECT_FILL_TEXTURE_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_OBJECT_FILL_TILE_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObjectFillGlStats {
    pub mesh_entries: usize,
    pub state_entries: usize,
    pub color_entries: usize,
    pub mesh_bytes: usize,
    pub state_bytes: usize,
    pub color_bytes: usize,
    pub tile_entries: usize,
    pub tile_bytes: usize,
    pub tile_pending_bytes: usize,
    pub mesh_budget_bytes: usize,
    pub texture_budget_bytes: usize,
    pub tile_budget_bytes: usize,
    pub mesh_uploads: u64,
    pub state_uploads: u64,
    pub color_uploads: u64,
    pub mesh_evictions: u64,
    pub texture_evictions: u64,
    pub tile_requests: u64,
    pub tile_request_generation: u64,
    pub tile_visible: usize,
    pub tile_hits: u64,
    pub tile_generations: u64,
    pub tile_discarded: u64,
    pub tile_evictions: u64,
    pub tile_pending: usize,
    pub tile_peak_pending: usize,
    pub last_tile_raster_vertices: u64,
    pub last_tile_raster_draw_calls: u64,
    pub last_tile_compose_draw_calls: u64,
    pub last_tile_selection_compose_draw_calls: u64,
    pub total_tile_raster_vertices: u64,
    pub last_tile_raster_ms: f64,
    pub last_tile_compose_ms: f64,
    pub tile_supported: Option<bool>,
    pub last_draw_calls: u64,
    pub last_triangles: u64,
    pub total_draw_calls: u64,
    pub total_triangles: u64,
    pub last_paint_ms: f64,
}

fn object_fill_screen_rect(
    local_rect: egui::Rect,
    params: &ObjectFillGlDrawParams,
    viewport: egui::Rect,
) -> egui::Rect {
    let local_to_screen = |local: egui::Pos2| {
        let world = egui::pos2(
            params.local_to_world_offset.x + local.x * params.local_to_world_scale.x,
            params.local_to_world_offset.y + local.y * params.local_to_world_scale.y,
        );
        viewport.center() + (world - params.center_world) * params.zoom_screen_per_world.max(1.0e-6)
    };
    egui::Rect::from_two_pos(
        local_to_screen(local_rect.min),
        local_to_screen(local_rect.max),
    )
}

fn object_fill_scissor_box(
    rect: egui::Rect,
    pixels_per_point: f32,
    screen_size_px: [u32; 2],
) -> [i32; 4] {
    let ppp = pixels_per_point.max(1.0e-6);
    let screen_width = screen_size_px[0] as i32;
    let screen_height = screen_size_px[1] as i32;
    let left = (rect.min.x * ppp).round() as i32;
    let top = (rect.min.y * ppp).round() as i32;
    let right = (rect.max.x * ppp).round() as i32;
    let bottom = (rect.max.y * ppp).round() as i32;
    let left = left.clamp(0, screen_width);
    let right = right.clamp(left, screen_width);
    let top = top.clamp(0, screen_height);
    let bottom = bottom.clamp(top, screen_height);
    [
        left,
        screen_height.saturating_sub(bottom),
        right.saturating_sub(left),
        bottom.saturating_sub(top),
    ]
}

#[derive(Clone)]
pub struct ObjectFillGlRenderer {
    inner: Arc<Mutex<ObjectFillInner>>,
}

impl std::fmt::Debug for ObjectFillGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectFillGlRenderer")
            .finish_non_exhaustive()
    }
}

impl ObjectFillGlRenderer {
    pub fn new(max_meshes: usize, max_state_textures: usize) -> Self {
        Self::with_byte_budgets(
            max_meshes,
            max_state_textures,
            DEFAULT_OBJECT_FILL_MESH_BUDGET_BYTES,
            DEFAULT_OBJECT_FILL_TEXTURE_BUDGET_BYTES,
        )
    }

    pub fn with_byte_budgets(
        max_meshes: usize,
        max_state_textures: usize,
        mesh_budget_bytes: usize,
        texture_budget_bytes: usize,
    ) -> Self {
        Self::with_all_byte_budgets(
            max_meshes,
            max_state_textures,
            mesh_budget_bytes,
            texture_budget_bytes,
            DEFAULT_OBJECT_FILL_TILE_BUDGET_BYTES,
        )
    }

    pub fn with_all_byte_budgets(
        max_meshes: usize,
        max_state_textures: usize,
        mesh_budget_bytes: usize,
        texture_budget_bytes: usize,
        tile_budget_bytes: usize,
    ) -> Self {
        let mesh_cap = NonZeroUsize::new(max_meshes.max(1)).unwrap();
        let state_cap = NonZeroUsize::new(max_state_textures.max(1)).unwrap();
        Self {
            inner: Arc::new(Mutex::new(ObjectFillInner::new(
                mesh_cap,
                state_cap,
                mesh_budget_bytes.max(1),
                texture_budget_bytes.max(1),
                tile_budget_bytes.max(1),
            ))),
        }
    }

    pub fn stats(&self) -> ObjectFillGlStats {
        self.inner.lock().stats()
    }

    pub fn paint_many(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        items: &[ObjectFillGlDrawItem],
    ) {
        let paint_started = Instant::now();
        if items.is_empty() {
            return;
        }

        let gl = painter.gl();
        if gl.version().major < 3 {
            return;
        }

        let mut inner = self.inner.lock();
        inner.last_draw_calls = 0;
        inner.last_triangles = 0;
        if inner.gl_objects.is_none() {
            inner.gl_objects = ObjectFillGlObjects::new(gl).ok();
        }
        inner.delete_queued(gl);

        let Some(objects) = inner.gl_objects.as_ref() else {
            return;
        };
        let program = objects.program;
        let vao = objects.vao;
        let u_center_world = objects.u_center_world.clone();
        let u_zoom_px = objects.u_zoom_px.clone();
        let u_viewport_min_px = objects.u_viewport_min_px.clone();
        let u_viewport_size_px = objects.u_viewport_size_px.clone();
        let u_selected_color = objects.u_selected_color.clone();
        let u_primary_color = objects.u_primary_color.clone();
        let u_local_to_world_offset = objects.u_local_to_world_offset.clone();
        let u_local_to_world_scale = objects.u_local_to_world_scale.clone();
        let u_state_tex = objects.u_state_tex.clone();
        let u_state_tex_size = objects.u_state_tex_size.clone();
        let u_color_tex = objects.u_color_tex.clone();
        let u_color_tex_size = objects.u_color_tex_size.clone();
        let u_use_object_colors = objects.u_use_object_colors.clone();
        let u_object_color_opacity = objects.u_object_color_opacity.clone();

        let viewport_pt = info.viewport;
        let ppp = info.pixels_per_point.max(1e-6);
        let viewport_min_px = viewport_pt.min * ppp;
        let viewport_size_px = viewport_pt.size() * ppp;
        let mut previous_scissor = [0i32; 4];
        let previous_scissor_enabled;

        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut previous_scissor);
            previous_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
        }

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.active_texture(glow::TEXTURE0);
            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));
            gl.uniform_2_f32(
                u_viewport_min_px.as_ref(),
                viewport_min_px.x,
                viewport_min_px.y,
            );
            gl.uniform_2_f32(
                u_viewport_size_px.as_ref(),
                viewport_size_px.x.max(1.0),
                viewport_size_px.y.max(1.0),
            );
            gl.uniform_1_i32(u_state_tex.as_ref(), 0);
            gl.uniform_1_i32(u_color_tex.as_ref(), 1);
        }

        for item in items {
            if !item.params.visible || !item.visible_world.is_positive() {
                continue;
            }
            let item_clip = object_fill_screen_rect(item.visible_world, &item.params, viewport_pt)
                .intersect(info.clip_rect)
                .intersect(viewport_pt);
            if !item_clip.is_positive() {
                continue;
            }
            let scissor = object_fill_scissor_box(item_clip, ppp, info.screen_size_px);
            if scissor[2] <= 0 || scissor[3] <= 0 {
                continue;
            }
            let Some((mesh_vbo, mesh_count)) = inner
                .ensure_object_mesh_uploaded(
                    gl,
                    item.data.cache_id,
                    item.data.generation,
                    item.data.vertices_local.as_slice(),
                )
                .map(|mesh| (mesh.vbo, mesh.count))
            else {
                continue;
            };
            let Some((state_texture, state_width, state_height)) = inner
                .ensure_state_uploaded(
                    gl,
                    item.data.state_cache_id,
                    item.data.selection_generation,
                    item.data.object_count,
                    item.data.selection_state.as_slice(),
                )
                .map(|state| (state.texture, state.width, state.height))
            else {
                continue;
            };
            let selected = item.params.selected_color;
            let primary = item.params.primary_color;
            let selected_color = [
                selected.r() as f32 / 255.0,
                selected.g() as f32 / 255.0,
                selected.b() as f32 / 255.0,
                selected.a() as f32 / 255.0,
            ];
            let primary_color = [
                primary.r() as f32 / 255.0,
                primary.g() as f32 / 255.0,
                primary.b() as f32 / 255.0,
                primary.a() as f32 / 255.0,
            ];
            let color_texture = item.data.object_colors_rgba.as_ref().and_then(|colors| {
                inner
                    .ensure_color_uploaded(
                        gl,
                        item.data.color_cache_id,
                        item.data.color_generation,
                        item.data.object_count,
                        colors.as_slice(),
                    )
                    .map(|color| (color.texture, color.width, color.height))
            });

            unsafe {
                let gl = gl.as_ref();
                gl.enable(glow::SCISSOR_TEST);
                gl.scissor(scissor[0], scissor[1], scissor[2], scissor[3]);
                gl.uniform_2_f32(
                    u_center_world.as_ref(),
                    item.params.center_world.x,
                    item.params.center_world.y,
                );
                gl.uniform_1_f32(
                    u_zoom_px.as_ref(),
                    (item.params.zoom_screen_per_world.max(1e-6) * ppp).max(1e-6),
                );
                gl.uniform_4_f32_slice(u_selected_color.as_ref(), &selected_color);
                gl.uniform_4_f32_slice(u_primary_color.as_ref(), &primary_color);
                gl.uniform_2_f32(
                    u_local_to_world_offset.as_ref(),
                    item.params.local_to_world_offset.x,
                    item.params.local_to_world_offset.y,
                );
                gl.uniform_2_f32(
                    u_local_to_world_scale.as_ref(),
                    item.params.local_to_world_scale.x.max(1e-9),
                    item.params.local_to_world_scale.y.max(1e-9),
                );
                gl.uniform_2_i32(u_state_tex_size.as_ref(), state_width, state_height);
                gl.uniform_1_i32(
                    u_use_object_colors.as_ref(),
                    i32::from(color_texture.is_some()),
                );
                gl.uniform_1_f32(
                    u_object_color_opacity.as_ref(),
                    item.params.object_color_opacity.clamp(0.0, 1.0),
                );
                gl.bind_texture(glow::TEXTURE_2D, Some(state_texture));
                if let Some((texture, width, height)) = color_texture {
                    gl.uniform_2_i32(u_color_tex_size.as_ref(), width, height);
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.active_texture(glow::TEXTURE0);
                } else {
                    gl.uniform_2_i32(u_color_tex_size.as_ref(), 0, 0);
                }
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(mesh_vbo));
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 12, 0);
                gl.enable_vertex_attrib_array(1);
                gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 12, 8);
                gl.draw_arrays(glow::TRIANGLES, 0, mesh_count as i32);
            }
            let triangles = (mesh_count / 3) as u64;
            inner.last_draw_calls = inner.last_draw_calls.saturating_add(1);
            inner.last_triangles = inner.last_triangles.saturating_add(triangles);
            inner.total_draw_calls = inner.total_draw_calls.saturating_add(1);
            inner.total_triangles = inner.total_triangles.saturating_add(triangles);
        }

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            if previous_scissor_enabled {
                gl.enable(glow::SCISSOR_TEST);
            } else {
                gl.disable(glow::SCISSOR_TEST);
            }
            gl.scissor(
                previous_scissor[0],
                previous_scissor[1],
                previous_scissor[2],
                previous_scissor[3],
            );
        }
        inner.last_paint_ms = paint_started.elapsed().as_secs_f64() * 1_000.0;
    }
}

struct ObjectFillMeshGpu {
    vbo: glow::Buffer,
    count: usize,
    bytes: usize,
}

struct ObjectFillStateGpu {
    texture: glow::Texture,
    width: i32,
    height: i32,
    generation: u64,
    bytes: usize,
}

struct ObjectFillColorGpu {
    texture: glow::Texture,
    width: i32,
    height: i32,
    generation: u64,
    bytes: usize,
}

struct ObjectFillInner {
    gl_objects: Option<ObjectFillGlObjects>,
    meshes: LruCache<(u64, u64), ObjectFillMeshGpu>,
    states: LruCache<u64, ObjectFillStateGpu>,
    colors: LruCache<u64, ObjectFillColorGpu>,
    id_tiles: LruCache<tiles::ObjectFillTileKey, tiles::ObjectFillIdTileGpu>,
    pending_id_tiles: LruCache<tiles::ObjectFillTileKey, tiles::ObjectFillPendingIdTileGpu>,
    tile_gl_objects: Option<tiles::ObjectFillTileGlObjects>,
    tile_gl_init_attempted: bool,
    buffers_to_delete: Vec<glow::Buffer>,
    textures_to_delete: Vec<glow::Texture>,
    mesh_budget_bytes: usize,
    texture_budget_bytes: usize,
    tile_budget_bytes: usize,
    mesh_bytes: usize,
    state_bytes: usize,
    color_bytes: usize,
    tile_bytes: usize,
    tile_pending_bytes: usize,
    mesh_uploads: u64,
    state_uploads: u64,
    color_uploads: u64,
    mesh_evictions: u64,
    texture_evictions: u64,
    tile_requests: u64,
    tile_request_generation: u64,
    tile_visible: usize,
    tile_hits: u64,
    tile_generations: u64,
    tile_discarded: u64,
    tile_evictions: u64,
    tile_pending: usize,
    tile_peak_pending: usize,
    last_tile_raster_vertices: u64,
    last_tile_raster_draw_calls: u64,
    last_tile_compose_draw_calls: u64,
    last_tile_selection_compose_draw_calls: u64,
    total_tile_raster_vertices: u64,
    last_tile_raster_ms: f64,
    last_tile_compose_ms: f64,
    last_draw_calls: u64,
    last_triangles: u64,
    total_draw_calls: u64,
    total_triangles: u64,
    last_paint_ms: f64,
}

impl ObjectFillInner {
    fn new(
        mesh_cap: NonZeroUsize,
        state_cap: NonZeroUsize,
        mesh_budget_bytes: usize,
        texture_budget_bytes: usize,
        tile_budget_bytes: usize,
    ) -> Self {
        Self {
            gl_objects: None,
            meshes: LruCache::new(mesh_cap),
            states: LruCache::new(state_cap),
            colors: LruCache::new(state_cap),
            id_tiles: LruCache::new(NonZeroUsize::new(1024).unwrap()),
            pending_id_tiles: LruCache::new(
                NonZeroUsize::new(tiles::MAX_PENDING_ID_TILES).unwrap(),
            ),
            tile_gl_objects: None,
            tile_gl_init_attempted: false,
            buffers_to_delete: Vec::new(),
            textures_to_delete: Vec::new(),
            mesh_budget_bytes,
            texture_budget_bytes,
            tile_budget_bytes,
            mesh_bytes: 0,
            state_bytes: 0,
            color_bytes: 0,
            tile_bytes: 0,
            tile_pending_bytes: 0,
            mesh_uploads: 0,
            state_uploads: 0,
            color_uploads: 0,
            mesh_evictions: 0,
            texture_evictions: 0,
            tile_requests: 0,
            tile_request_generation: 0,
            tile_visible: 0,
            tile_hits: 0,
            tile_generations: 0,
            tile_discarded: 0,
            tile_evictions: 0,
            tile_pending: 0,
            tile_peak_pending: 0,
            last_tile_raster_vertices: 0,
            last_tile_raster_draw_calls: 0,
            last_tile_compose_draw_calls: 0,
            last_tile_selection_compose_draw_calls: 0,
            total_tile_raster_vertices: 0,
            last_tile_raster_ms: 0.0,
            last_tile_compose_ms: 0.0,
            last_draw_calls: 0,
            last_triangles: 0,
            total_draw_calls: 0,
            total_triangles: 0,
            last_paint_ms: 0.0,
        }
    }

    fn stats(&self) -> ObjectFillGlStats {
        ObjectFillGlStats {
            mesh_entries: self.meshes.len(),
            state_entries: self.states.len(),
            color_entries: self.colors.len(),
            tile_entries: self.id_tiles.len(),
            mesh_bytes: self.mesh_bytes,
            state_bytes: self.state_bytes,
            color_bytes: self.color_bytes,
            tile_bytes: self.tile_bytes,
            tile_pending_bytes: self.tile_pending_bytes,
            mesh_budget_bytes: self.mesh_budget_bytes,
            texture_budget_bytes: self.texture_budget_bytes,
            tile_budget_bytes: self.tile_budget_bytes,
            mesh_uploads: self.mesh_uploads,
            state_uploads: self.state_uploads,
            color_uploads: self.color_uploads,
            mesh_evictions: self.mesh_evictions,
            texture_evictions: self.texture_evictions,
            tile_requests: self.tile_requests,
            tile_request_generation: self.tile_request_generation,
            tile_visible: self.tile_visible,
            tile_hits: self.tile_hits,
            tile_generations: self.tile_generations,
            tile_discarded: self.tile_discarded,
            tile_evictions: self.tile_evictions,
            tile_pending: self.tile_pending,
            tile_peak_pending: self.tile_peak_pending,
            last_tile_raster_vertices: self.last_tile_raster_vertices,
            last_tile_raster_draw_calls: self.last_tile_raster_draw_calls,
            last_tile_compose_draw_calls: self.last_tile_compose_draw_calls,
            last_tile_selection_compose_draw_calls: self.last_tile_selection_compose_draw_calls,
            total_tile_raster_vertices: self.total_tile_raster_vertices,
            last_tile_raster_ms: self.last_tile_raster_ms,
            last_tile_compose_ms: self.last_tile_compose_ms,
            tile_supported: self
                .tile_gl_init_attempted
                .then_some(self.tile_gl_objects.is_some()),
            last_draw_calls: self.last_draw_calls,
            last_triangles: self.last_triangles,
            total_draw_calls: self.total_draw_calls,
            total_triangles: self.total_triangles,
            last_paint_ms: self.last_paint_ms,
        }
    }

    fn evict_meshes_to_budget(&mut self) {
        while self.mesh_bytes > self.mesh_budget_bytes {
            let Some((_key, evicted)) = self.meshes.pop_lru() else {
                break;
            };
            self.mesh_bytes = self.mesh_bytes.saturating_sub(evicted.bytes);
            self.mesh_evictions = self.mesh_evictions.saturating_add(1);
            self.buffers_to_delete.push(evicted.vbo);
        }
    }

    fn evict_textures_to_budget(&mut self) {
        while self.state_bytes.saturating_add(self.color_bytes) > self.texture_budget_bytes {
            let evict_state = match (self.states.peek_lru(), self.colors.peek_lru()) {
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (Some(_), Some(_)) => self.state_bytes >= self.color_bytes,
                (None, None) => break,
            };
            if evict_state {
                let Some((_key, evicted)) = self.states.pop_lru() else {
                    break;
                };
                self.state_bytes = self.state_bytes.saturating_sub(evicted.bytes);
                self.textures_to_delete.push(evicted.texture);
            } else {
                let Some((_key, evicted)) = self.colors.pop_lru() else {
                    break;
                };
                self.color_bytes = self.color_bytes.saturating_sub(evicted.bytes);
                self.textures_to_delete.push(evicted.texture);
            }
            self.texture_evictions = self.texture_evictions.saturating_add(1);
        }
    }

    fn delete_queued(&mut self, gl: &Arc<glow::Context>) {
        unsafe {
            let gl = gl.as_ref();
            for b in self.buffers_to_delete.drain(..) {
                gl.delete_buffer(b);
            }
            for t in self.textures_to_delete.drain(..) {
                gl.delete_texture(t);
            }
        }
    }

    fn ensure_object_mesh_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        vertices_local: &[[f32; 3]],
    ) -> Option<&ObjectFillMeshGpu> {
        let key = (cache_id, generation);
        if self.meshes.contains(&key) {
            return self.meshes.get(&key);
        }
        if vertices_local.is_empty() {
            return None;
        }
        let bytes = vertices_local
            .len()
            .saturating_mul(std::mem::size_of::<[f32; 3]>());
        if bytes > self.mesh_budget_bytes {
            return None;
        }

        let vbo = unsafe { gl.as_ref().create_buffer().map_err(|_| ()).ok()? };
        unsafe {
            let gl = gl.as_ref();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(vertices_local),
                glow::STATIC_DRAW,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }

        if let Some((_k, evicted)) = self.meshes.push(
            key,
            ObjectFillMeshGpu {
                vbo,
                count: vertices_local.len(),
                bytes,
            },
        ) {
            self.mesh_bytes = self.mesh_bytes.saturating_sub(evicted.bytes);
            self.mesh_evictions = self.mesh_evictions.saturating_add(1);
            self.buffers_to_delete.push(evicted.vbo);
        }
        self.mesh_bytes = self.mesh_bytes.saturating_add(bytes);
        self.mesh_uploads = self.mesh_uploads.saturating_add(1);
        self.evict_meshes_to_budget();
        self.delete_queued(gl);
        self.meshes.get(&key)
    }

    fn ensure_state_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        object_count: usize,
        selection_state: &[u8],
    ) -> Option<&ObjectFillStateGpu> {
        let padded_len = object_count.max(1);
        let width = padded_len.min(4096) as i32;
        let height = ((padded_len + width as usize - 1) / width as usize).max(1) as i32;

        let state = self.states.get(&cache_id);
        if state.is_some_and(|state| state.generation == generation) {
            return self.states.get(&cache_id);
        }

        let texels_len = (width as usize).saturating_mul(height as usize);
        let bytes = texels_len;
        if bytes > self.texture_budget_bytes {
            return None;
        }
        let mut texels = vec![0u8; texels_len];
        let copy_len = selection_state.len().min(object_count).min(texels.len());
        texels[..copy_len].copy_from_slice(&selection_state[..copy_len]);

        let texture = if let Some(existing) = self.states.get(&cache_id) {
            existing.texture
        } else {
            unsafe { gl.as_ref().create_texture().map_err(|_| ()).ok()? }
        };

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R8 as i32,
                width,
                height,
                0,
                glow::RED,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(texels.as_slice())),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
        }

        if let Some((_k, evicted)) = self.states.push(
            cache_id,
            ObjectFillStateGpu {
                texture,
                width,
                height,
                generation,
                bytes,
            },
        ) {
            self.state_bytes = self.state_bytes.saturating_sub(evicted.bytes);
            if evicted.texture != texture {
                self.textures_to_delete.push(evicted.texture);
                self.texture_evictions = self.texture_evictions.saturating_add(1);
            }
        }

        self.state_bytes = self.state_bytes.saturating_add(bytes);
        self.state_uploads = self.state_uploads.saturating_add(1);
        self.evict_textures_to_budget();

        self.delete_queued(gl);
        self.states.get(&cache_id)
    }

    fn ensure_color_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        object_count: usize,
        colors_rgba: &[[u8; 4]],
    ) -> Option<&ObjectFillColorGpu> {
        let padded_len = object_count.max(1);
        let width = padded_len.min(4096) as i32;
        let height = ((padded_len + width as usize - 1) / width as usize).max(1) as i32;
        if self
            .colors
            .get(&cache_id)
            .is_some_and(|color| color.generation == generation)
        {
            return self.colors.get(&cache_id);
        }
        let mut texels = vec![0u8; width as usize * height as usize * 4];
        let bytes = texels.len();
        if bytes > self.texture_budget_bytes {
            return None;
        }
        let copy_len = colors_rgba.len().min(object_count);
        texels[..copy_len * 4].copy_from_slice(bytemuck::cast_slice(&colors_rgba[..copy_len]));
        let texture = if let Some(existing) = self.colors.get(&cache_id) {
            existing.texture
        } else {
            unsafe { gl.as_ref().create_texture().map_err(|_| ()).ok()? }
        };
        unsafe {
            let gl = gl.as_ref();
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width,
                height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(texels.as_slice())),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
        }
        if let Some((_key, evicted)) = self.colors.push(
            cache_id,
            ObjectFillColorGpu {
                texture,
                width,
                height,
                generation,
                bytes,
            },
        ) {
            self.color_bytes = self.color_bytes.saturating_sub(evicted.bytes);
            if evicted.texture != texture {
                self.textures_to_delete.push(evicted.texture);
                self.texture_evictions = self.texture_evictions.saturating_add(1);
            }
        }
        self.color_bytes = self.color_bytes.saturating_add(bytes);
        self.color_uploads = self.color_uploads.saturating_add(1);
        self.evict_textures_to_budget();
        self.delete_queued(gl);
        self.colors.get(&cache_id)
    }
}

struct ObjectFillGlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    u_center_world: Option<glow::UniformLocation>,
    u_zoom_px: Option<glow::UniformLocation>,
    u_viewport_min_px: Option<glow::UniformLocation>,
    u_viewport_size_px: Option<glow::UniformLocation>,
    u_selected_color: Option<glow::UniformLocation>,
    u_primary_color: Option<glow::UniformLocation>,
    u_local_to_world_offset: Option<glow::UniformLocation>,
    u_local_to_world_scale: Option<glow::UniformLocation>,
    u_state_tex: Option<glow::UniformLocation>,
    u_state_tex_size: Option<glow::UniformLocation>,
    u_color_tex: Option<glow::UniformLocation>,
    u_color_tex_size: Option<glow::UniformLocation>,
    u_use_object_colors: Option<glow::UniformLocation>,
    u_object_color_opacity: Option<glow::UniformLocation>,
}

impl ObjectFillGlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let program = compile_program_with_attributes(
            gl,
            OBJECT_FILL_VERT_330,
            OBJECT_FILL_FRAG_330,
            &[(0, "a_pos"), (1, "a_object_id")],
        )?;
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| anyhow!("create_vertex_array failed: {e}"))?
        };
        unsafe {
            gl.bind_vertex_array(Some(vao));
            gl.bind_vertex_array(None);
        }
        Ok(Self {
            program,
            vao,
            u_center_world: unsafe { gl.get_uniform_location(program, "u_center_world") },
            u_zoom_px: unsafe { gl.get_uniform_location(program, "u_zoom_px") },
            u_viewport_min_px: unsafe { gl.get_uniform_location(program, "u_viewport_min_px") },
            u_viewport_size_px: unsafe { gl.get_uniform_location(program, "u_viewport_size_px") },
            u_selected_color: unsafe { gl.get_uniform_location(program, "u_selected_color") },
            u_primary_color: unsafe { gl.get_uniform_location(program, "u_primary_color") },
            u_local_to_world_offset: unsafe {
                gl.get_uniform_location(program, "u_local_to_world_offset")
            },
            u_local_to_world_scale: unsafe {
                gl.get_uniform_location(program, "u_local_to_world_scale")
            },
            u_state_tex: unsafe { gl.get_uniform_location(program, "u_state_tex") },
            u_state_tex_size: unsafe { gl.get_uniform_location(program, "u_state_tex_size") },
            u_color_tex: unsafe { gl.get_uniform_location(program, "u_color_tex") },
            u_color_tex_size: unsafe { gl.get_uniform_location(program, "u_color_tex_size") },
            u_use_object_colors: unsafe { gl.get_uniform_location(program, "u_use_object_colors") },
            u_object_color_opacity: unsafe {
                gl.get_uniform_location(program, "u_object_color_opacity")
            },
        })
    }
}

fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
) -> anyhow::Result<glow::Program> {
    compile_program_with_attributes(gl, vs_src, fs_src, &[(0, "a_pos")])
}

fn compile_program_with_attributes(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    attributes: &[(u32, &str)],
) -> anyhow::Result<glow::Program> {
    unsafe {
        let vs = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow!("create vertex shader failed: {e}"))?;
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            gl.delete_shader(vs);
            return Err(anyhow!("vertex shader compile failed: {log}"));
        }

        let fs = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(|e| anyhow!("create fragment shader failed: {e}"))?;
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            return Err(anyhow!("fragment shader compile failed: {log}"));
        }

        let program = gl
            .create_program()
            .map_err(|e| anyhow!("create_program failed: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        for (location, name) in attributes {
            gl.bind_attrib_location(program, *location, name);
        }
        gl.link_program(program);
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(anyhow!("program link failed: {log}"));
        }
        Ok(program)
    }
}

const VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos; // local coordinates

uniform vec2 u_center_world;
uniform float u_zoom_px;
uniform vec2 u_viewport_min_px;
uniform vec2 u_viewport_size_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;

void main() {
    vec2 world = u_local_to_world_offset + a_pos * u_local_to_world_scale;
    vec2 screen = (world - u_center_world) * u_zoom_px + u_viewport_min_px + 0.5 * u_viewport_size_px;
    vec2 rel = (screen - u_viewport_min_px) / u_viewport_size_px;
    vec2 ndc = vec2(rel.x * 2.0 - 1.0, 1.0 - rel.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}"#;

const FRAG_330: &str = r#"#version 330 core
uniform vec4 u_color;
out vec4 out_color;
void main() {
    out_color = u_color;
}"#;

const OBJECT_FILL_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_object_id;

uniform vec2 u_center_world;
uniform float u_zoom_px;
uniform vec2 u_viewport_min_px;
uniform vec2 u_viewport_size_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;

flat out int v_object_id;

void main() {
    vec2 world = u_local_to_world_offset + a_pos * u_local_to_world_scale;
    vec2 screen = (world - u_center_world) * u_zoom_px + u_viewport_min_px + 0.5 * u_viewport_size_px;
    vec2 rel = (screen - u_viewport_min_px) / u_viewport_size_px;
    vec2 ndc = vec2(rel.x * 2.0 - 1.0, 1.0 - rel.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_object_id = int(a_object_id + 0.5);
}"#;

const OBJECT_FILL_FRAG_330: &str = r#"#version 330 core
uniform sampler2D u_state_tex;
uniform ivec2 u_state_tex_size;
uniform sampler2D u_color_tex;
uniform ivec2 u_color_tex_size;
uniform int u_use_object_colors;
uniform float u_object_color_opacity;
uniform vec4 u_selected_color;
uniform vec4 u_primary_color;

flat in int v_object_id;
out vec4 out_color;

void main() {
    if (u_state_tex_size.x <= 0 || u_state_tex_size.y <= 0 || v_object_id < 0) {
        discard;
    }
    int x = v_object_id % u_state_tex_size.x;
    int y = v_object_id / u_state_tex_size.x;
    if (y < 0 || y >= u_state_tex_size.y) {
        discard;
    }
    float state = texelFetch(u_state_tex, ivec2(x, y), 0).r;
    if (state < 0.001) {
        discard;
    }
    if (u_use_object_colors != 0) {
        if (u_color_tex_size.x <= 0 || u_color_tex_size.y <= 0) {
            discard;
        }
        int color_x = v_object_id % u_color_tex_size.x;
        int color_y = v_object_id / u_color_tex_size.x;
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
mod object_fill_tests {
    use super::*;

    fn params() -> ObjectFillGlDrawParams {
        ObjectFillGlDrawParams {
            center_world: egui::pos2(50.0, 100.0),
            zoom_screen_per_world: 2.0,
            selected_color: egui::Color32::WHITE,
            primary_color: egui::Color32::WHITE,
            visible: true,
            local_to_world_offset: egui::vec2(10.0, 20.0),
            local_to_world_scale: egui::vec2(2.0, 3.0),
            object_color_opacity: 1.0,
        }
    }

    #[test]
    fn object_fill_screen_rect_applies_transform_camera_and_zoom() {
        let viewport = egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(800.0, 600.0));
        let local = egui::Rect::from_min_max(egui::pos2(20.0, 30.0), egui::pos2(40.0, 50.0));
        let screen = object_fill_screen_rect(local, &params(), viewport);

        assert_eq!(screen.min, egui::pos2(500.0, 370.0));
        assert_eq!(screen.max, egui::pos2(580.0, 490.0));
    }

    #[test]
    fn object_fill_scissor_box_uses_bottom_left_gl_origin() {
        let rect = egui::Rect::from_min_max(egui::pos2(100.0, 50.0), egui::pos2(300.0, 150.0));
        assert_eq!(
            object_fill_scissor_box(rect, 1.0, [1000, 800]),
            [100, 650, 200, 100]
        );
    }

    #[test]
    fn object_fill_cache_reports_configured_byte_budgets_before_gl_initializes() {
        let renderer = ObjectFillGlRenderer::with_byte_budgets(3, 2, 1234, 5678);
        let stats = renderer.stats();
        assert_eq!(stats.mesh_budget_bytes, 1234);
        assert_eq!(stats.texture_budget_bytes, 5678);
        assert_eq!(stats.mesh_bytes, 0);
        assert_eq!(stats.state_bytes, 0);
        assert_eq!(stats.color_bytes, 0);
    }
}
