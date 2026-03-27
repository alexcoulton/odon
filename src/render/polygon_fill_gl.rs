use std::num::NonZeroUsize;
use std::sync::Arc;

use anyhow::anyhow;
use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

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
}

#[derive(Debug, Clone)]
pub struct ObjectFillGlDrawData {
    pub cache_id: u64,
    pub generation: u64,
    pub vertices_local: Arc<Vec<[f32; 3]>>,
    pub object_count: usize,
    pub selection_generation: u64,
    pub selection_state: Arc<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct ObjectFillGlDrawItem {
    pub data: ObjectFillGlDrawData,
    pub params: ObjectFillGlDrawParams,
    pub visible_world: egui::Rect,
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
        let mesh_cap = NonZeroUsize::new(max_meshes.max(1)).unwrap();
        let state_cap = NonZeroUsize::new(max_state_textures.max(1)).unwrap();
        Self {
            inner: Arc::new(Mutex::new(ObjectFillInner::new(mesh_cap, state_cap))),
        }
    }

    pub fn paint_many(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        items: &[ObjectFillGlDrawItem],
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
        }

        for item in items {
            if !item.params.visible || !item.visible_world.is_positive() {
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
                    item.data.cache_id,
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
                gl.bind_texture(glow::TEXTURE_2D, Some(state_texture));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(mesh_vbo));
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 12, 0);
                gl.enable_vertex_attrib_array(1);
                gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 12, 8);
                gl.draw_arrays(glow::TRIANGLES, 0, mesh_count as i32);
            }
        }

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

struct ObjectFillMeshGpu {
    vbo: glow::Buffer,
    count: usize,
}

struct ObjectFillStateGpu {
    texture: glow::Texture,
    width: i32,
    height: i32,
    generation: u64,
}

struct ObjectFillInner {
    gl_objects: Option<ObjectFillGlObjects>,
    meshes: LruCache<(u64, u64), ObjectFillMeshGpu>,
    states: LruCache<u64, ObjectFillStateGpu>,
    buffers_to_delete: Vec<glow::Buffer>,
    textures_to_delete: Vec<glow::Texture>,
}

impl ObjectFillInner {
    fn new(mesh_cap: NonZeroUsize, state_cap: NonZeroUsize) -> Self {
        Self {
            gl_objects: None,
            meshes: LruCache::new(mesh_cap),
            states: LruCache::new(state_cap),
            buffers_to_delete: Vec::new(),
            textures_to_delete: Vec::new(),
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
            },
        ) {
            self.buffers_to_delete.push(evicted.vbo);
        }
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
            },
        ) && evicted.texture != texture
        {
            self.textures_to_delete.push(evicted.texture);
        }

        self.delete_queued(gl);
        self.states.get(&cache_id)
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
}

impl ObjectFillGlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let program = compile_program(gl, OBJECT_FILL_VERT_330, OBJECT_FILL_FRAG_330)?;
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
        })
    }
}

fn compile_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
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
        gl.bind_attrib_location(program, 0, "a_pos");
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
    out_color = state > 0.75 ? u_primary_color : u_selected_color;
}"#;
