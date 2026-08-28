use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

use crate::render::line_bins::ObjectLineSegmentsBins;

use super::program::compile_program_with_attributes;

mod color_texture;
mod shaders;
mod stats;
use color_texture::ObjectColorGpu;
use shaders::OBJECT_LINE_FRAG_330;
pub use stats::ObjectLineBinsGlStats;

const OBJECT_LINE_VERT_330: &str = shaders::OBJECT_LINE_VERT_330;

#[derive(Debug, Clone)]
pub struct ObjectLineBinsGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub base_width_points: f32,
    pub selected_width_points: f32,
    pub primary_width_points: f32,
    pub base_color: egui::Color32,
    pub selected_color: egui::Color32,
    pub primary_color: egui::Color32,
    pub draw_unselected: bool,
    pub visible: bool,
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
    pub object_color_opacity: f32,
}

#[derive(Debug, Clone)]
pub struct ObjectLineBinsGlDrawData {
    pub cache_id: u64,
    pub state_cache_id: u64,
    /// Changes only when the uploaded outline geometry changes. Presentation/style generations
    /// must not participate in the geometry-buffer cache key.
    pub geometry_generation: u64,
    pub bins: Arc<ObjectLineSegmentsBins>,
    pub selection_generation: u64,
    pub selection_state: Arc<Vec<u8>>,
    pub object_count: usize,
    pub color_cache_id: u64,
    pub color_generation: u64,
    pub object_colors_rgba: Option<Arc<Vec<[u8; 4]>>>,
}

#[derive(Debug, Clone)]
pub struct ObjectLineBinsGlDrawItem {
    pub data: ObjectLineBinsGlDrawData,
    pub params: ObjectLineBinsGlDrawParams,
    pub visible_world: egui::Rect,
}

#[derive(Clone)]
pub struct ObjectLineBinsGlRenderer {
    inner: Arc<Mutex<ObjectLineInner>>,
}

impl std::fmt::Debug for ObjectLineBinsGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectLineBinsGlRenderer")
            .finish_non_exhaustive()
    }
}

impl ObjectLineBinsGlRenderer {
    pub fn new(max_uploaded_bins: usize, max_state_textures: usize) -> Self {
        let bin_cap = NonZeroUsize::new(max_uploaded_bins.max(8)).unwrap();
        let state_cap = NonZeroUsize::new(max_state_textures.max(1)).unwrap();
        Self {
            inner: Arc::new(Mutex::new(ObjectLineInner::new(bin_cap, state_cap))),
        }
    }

    pub fn stats(&self) -> ObjectLineBinsGlStats {
        self.inner.lock().stats()
    }

    pub fn paint_many(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        items: &[ObjectLineBinsGlDrawItem],
    ) {
        if items.is_empty() {
            return;
        }

        let gl = painter.gl();
        if gl.version().major < 3 {
            return;
        }

        let mut inner = self.inner.lock();
        inner.last_draw_calls = 0;
        inner.last_records = 0;
        inner.last_paint_ms = 0.0;
        let paint_started = Instant::now();
        if inner.gl_objects.is_none() {
            inner.gl_objects = ObjectLineGlObjects::new(gl).ok();
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
        let u_base_width_px = objects.u_base_width_px.clone();
        let u_selected_width_px = objects.u_selected_width_px.clone();
        let u_primary_width_px = objects.u_primary_width_px.clone();
        let u_base_color = objects.u_base_color.clone();
        let u_selected_color = objects.u_selected_color.clone();
        let u_primary_color = objects.u_primary_color.clone();
        let u_draw_unselected = objects.u_draw_unselected.clone();
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

        let mut uploaded_this_frame = 0usize;
        let mut missing_this_frame = 0usize;
        const MAX_BIN_UPLOADS_PER_FRAME: usize = 64;

        for it in items {
            if !it.params.visible || it.data.bins.segments.is_empty() {
                continue;
            }

            let base = it.params.base_color;
            let selected = it.params.selected_color;
            let primary = it.params.primary_color;
            let base_color = [
                base.r() as f32 / 255.0,
                base.g() as f32 / 255.0,
                base.b() as f32 / 255.0,
                base.a() as f32 / 255.0,
            ];
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
            let base_width_px = (it.params.base_width_points.max(0.0) * ppp).max(0.5);
            let selected_width_px = (it.params.selected_width_points.max(0.0) * ppp).max(0.5);
            let primary_width_px = (it.params.primary_width_points.max(0.0) * ppp).max(0.5);

            let Some((state_texture, state_width, state_height)) = inner
                .ensure_state_uploaded(
                    gl,
                    it.data.state_cache_id,
                    it.data.selection_generation,
                    it.data.object_count,
                    it.data.selection_state.as_slice(),
                )
                .map(|state| (state.texture, state.width, state.height))
            else {
                continue;
            };
            let color_texture = it.data.object_colors_rgba.as_ref().and_then(|colors| {
                inner
                    .ensure_color_uploaded(
                        gl,
                        it.data.color_cache_id,
                        it.data.color_generation,
                        it.data.object_count,
                        colors.as_slice(),
                    )
                    .map(|color| (color.texture, color.width, color.height))
            });

            unsafe {
                let gl = gl.as_ref();
                gl.uniform_2_f32(
                    u_center_world.as_ref(),
                    it.params.center_world.x,
                    it.params.center_world.y,
                );
                gl.uniform_1_f32(
                    u_zoom_px.as_ref(),
                    (it.params.zoom_screen_per_world.max(1e-6) * ppp).max(1e-6),
                );
                gl.uniform_1_f32(u_base_width_px.as_ref(), base_width_px);
                gl.uniform_1_f32(u_selected_width_px.as_ref(), selected_width_px);
                gl.uniform_1_f32(u_primary_width_px.as_ref(), primary_width_px);
                gl.uniform_4_f32_slice(u_base_color.as_ref(), &base_color);
                gl.uniform_4_f32_slice(u_selected_color.as_ref(), &selected_color);
                gl.uniform_4_f32_slice(u_primary_color.as_ref(), &primary_color);
                gl.uniform_1_i32(
                    u_draw_unselected.as_ref(),
                    if it.params.draw_unselected { 1 } else { 0 },
                );
                gl.uniform_2_f32(
                    u_local_to_world_offset.as_ref(),
                    it.params.local_to_world_offset.x,
                    it.params.local_to_world_offset.y,
                );
                gl.uniform_2_f32(
                    u_local_to_world_scale.as_ref(),
                    it.params.local_to_world_scale.x.max(1e-9),
                    it.params.local_to_world_scale.y.max(1e-9),
                );
                gl.uniform_2_i32(u_state_tex_size.as_ref(), state_width, state_height);
                gl.bind_texture(glow::TEXTURE_2D, Some(state_texture));
                gl.uniform_1_i32(
                    u_use_object_colors.as_ref(),
                    i32::from(color_texture.is_some()),
                );
                gl.uniform_1_f32(
                    u_object_color_opacity.as_ref(),
                    it.params.object_color_opacity.clamp(0.0, 1.0),
                );
                if let Some((texture, width, height)) = color_texture {
                    gl.uniform_2_i32(u_color_tex_size.as_ref(), width, height);
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.active_texture(glow::TEXTURE0);
                } else {
                    gl.uniform_2_i32(u_color_tex_size.as_ref(), 0, 0);
                }
            }

            let (bx0, by0, bx1, by1) = it.data.bins.bin_range_for_world_rect(it.visible_world);
            for by in by0..=by1 {
                for bx in bx0..=bx1 {
                    let bin_index = by * it.data.bins.bins_w + bx;
                    let slice = it.data.bins.bin_slice(bin_index);
                    if slice.is_empty() {
                        continue;
                    }

                    let allow_upload = uploaded_this_frame < MAX_BIN_UPLOADS_PER_FRAME;
                    let Some((bin_vbo, bin_count, uploaded)) = inner
                        .ensure_bin_uploaded(
                            gl,
                            it.data.cache_id,
                            it.data.geometry_generation,
                            bin_index,
                            slice,
                            allow_upload,
                        )
                        .map(|gpu| (gpu.handle.vbo, gpu.handle.count, gpu.uploaded))
                    else {
                        missing_this_frame += 1;
                        continue;
                    };
                    if uploaded {
                        uploaded_this_frame += 1;
                    }

                    unsafe {
                        let gl = gl.as_ref();
                        gl.bind_buffer(glow::ARRAY_BUFFER, Some(bin_vbo));
                        gl.enable_vertex_attrib_array(0);
                        gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 20, 0);
                        gl.vertex_attrib_divisor(0, 1);
                        gl.enable_vertex_attrib_array(1);
                        gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 20, 16);
                        gl.vertex_attrib_divisor(1, 1);
                        gl.draw_arrays_instanced(glow::TRIANGLES, 0, 6, bin_count as i32);
                    }
                    inner.last_draw_calls = inner.last_draw_calls.saturating_add(1);
                    inner.last_records = inner.last_records.saturating_add(bin_count as u64);
                }
            }
        }

        inner.last_frame_missing_bins = missing_this_frame;

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
        inner.last_paint_ms = paint_started.elapsed().as_secs_f64() * 1_000.0;
    }
}

#[derive(Clone, Copy)]
struct ObjectBinGpuHandle {
    vbo: glow::Buffer,
    count: usize,
}

#[derive(Clone, Copy)]
struct ObjectBinUpload {
    handle: ObjectBinGpuHandle,
    uploaded: bool,
}

struct ObjectBinGpu {
    vbo: glow::Buffer,
    count: usize,
    bytes: usize,
}

struct ObjectStateGpu {
    texture: glow::Texture,
    width: i32,
    height: i32,
    generation: u64,
}

struct ObjectTextureDelete {
    texture: glow::Texture,
    bytes: usize,
}

struct ObjectLineInner {
    gl_objects: Option<ObjectLineGlObjects>,
    bins: LruCache<(u64, u64, usize), ObjectBinGpu>,
    states: LruCache<u64, ObjectStateGpu>,
    colors: LruCache<u64, ObjectColorGpu>,
    buffers_to_delete: Vec<ObjectBinGpu>,
    textures_to_delete: Vec<ObjectTextureDelete>,
    last_frame_missing_bins: usize,
    bin_uploads: u64,
    state_uploads: u64,
    color_uploads: u64,
    bin_evictions: u64,
    texture_evictions: u64,
    buffer_deletions: u64,
    texture_deletions: u64,
    last_draw_calls: u64,
    last_records: u64,
    last_paint_ms: f64,
}

impl ObjectLineInner {
    fn new(bin_cap: NonZeroUsize, state_cap: NonZeroUsize) -> Self {
        Self {
            gl_objects: None,
            bins: LruCache::new(bin_cap),
            states: LruCache::new(state_cap),
            colors: LruCache::new(state_cap),
            buffers_to_delete: Vec::new(),
            textures_to_delete: Vec::new(),
            last_frame_missing_bins: 0,
            bin_uploads: 0,
            state_uploads: 0,
            color_uploads: 0,
            bin_evictions: 0,
            texture_evictions: 0,
            buffer_deletions: 0,
            texture_deletions: 0,
            last_draw_calls: 0,
            last_records: 0,
            last_paint_ms: 0.0,
        }
    }

    fn delete_queued(&mut self, gl: &Arc<glow::Context>) {
        unsafe {
            let gl = gl.as_ref();
            for buffer in self.buffers_to_delete.drain(..) {
                gl.delete_buffer(buffer.vbo);
                self.buffer_deletions = self.buffer_deletions.saturating_add(1);
            }
            for texture in self.textures_to_delete.drain(..) {
                gl.delete_texture(texture.texture);
                self.texture_deletions = self.texture_deletions.saturating_add(1);
            }
        }
    }

    fn ensure_bin_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        bin_index: usize,
        segments: &[[f32; 5]],
        allow_upload: bool,
    ) -> Option<ObjectBinUpload> {
        let key = (cache_id, generation, bin_index);
        if let Some(v) = self.bins.get(&key) {
            return Some(ObjectBinUpload {
                handle: ObjectBinGpuHandle {
                    vbo: v.vbo,
                    count: v.count,
                },
                uploaded: false,
            });
        }
        if !allow_upload {
            return None;
        }

        let vbo = unsafe { gl.as_ref().create_buffer().map_err(|_| ()).ok()? };
        let bytes = segments
            .len()
            .saturating_mul(std::mem::size_of::<[f32; 5]>());
        unsafe {
            let gl = gl.as_ref();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(segments),
                glow::STATIC_DRAW,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }

        if let Some((_ek, ev)) = self.bins.push(
            key,
            ObjectBinGpu {
                vbo,
                count: segments.len(),
                bytes,
            },
        ) {
            self.buffers_to_delete.push(ev);
            self.bin_evictions = self.bin_evictions.saturating_add(1);
        }
        self.bin_uploads = self.bin_uploads.saturating_add(1);
        self.delete_queued(gl);

        let v = self.bins.get(&key)?;
        Some(ObjectBinUpload {
            handle: ObjectBinGpuHandle {
                vbo: v.vbo,
                count: v.count,
            },
            uploaded: true,
        })
    }

    fn ensure_state_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        object_count: usize,
        selection_state: &[u8],
    ) -> Option<&ObjectStateGpu> {
        let state = self.states.get(&cache_id);
        if state.is_some_and(|state| state.generation == generation) {
            return self.states.get(&cache_id);
        }

        let padded_len = object_count.max(1);
        let width = padded_len.min(4096) as i32;
        let height = ((padded_len + width as usize - 1) / width as usize).max(1) as i32;

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

        if let Some((_ek, ev)) = self.states.push(
            cache_id,
            ObjectStateGpu {
                texture,
                width,
                height,
                generation,
            },
        ) && ev.texture != texture
        {
            self.textures_to_delete.push(ObjectTextureDelete {
                texture: ev.texture,
                bytes: (ev.width.max(0) as usize).saturating_mul(ev.height.max(0) as usize),
            });
            self.texture_evictions = self.texture_evictions.saturating_add(1);
        }
        self.state_uploads = self.state_uploads.saturating_add(1);
        self.delete_queued(gl);
        self.states.get(&cache_id)
    }
}

struct ObjectLineGlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    u_center_world: Option<glow::UniformLocation>,
    u_zoom_px: Option<glow::UniformLocation>,
    u_viewport_min_px: Option<glow::UniformLocation>,
    u_viewport_size_px: Option<glow::UniformLocation>,
    u_base_width_px: Option<glow::UniformLocation>,
    u_selected_width_px: Option<glow::UniformLocation>,
    u_primary_width_px: Option<glow::UniformLocation>,
    u_base_color: Option<glow::UniformLocation>,
    u_selected_color: Option<glow::UniformLocation>,
    u_primary_color: Option<glow::UniformLocation>,
    u_draw_unselected: Option<glow::UniformLocation>,
    u_local_to_world_offset: Option<glow::UniformLocation>,
    u_local_to_world_scale: Option<glow::UniformLocation>,
    u_state_tex: Option<glow::UniformLocation>,
    u_state_tex_size: Option<glow::UniformLocation>,
    u_color_tex: Option<glow::UniformLocation>,
    u_color_tex_size: Option<glow::UniformLocation>,
    u_use_object_colors: Option<glow::UniformLocation>,
    u_object_color_opacity: Option<glow::UniformLocation>,
}

impl ObjectLineGlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let program = compile_program_with_attributes(
            gl,
            OBJECT_LINE_VERT_330,
            OBJECT_LINE_FRAG_330,
            &[(0, "a_seg"), (1, "a_object_id")],
        )?;
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?
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
            u_base_width_px: unsafe { gl.get_uniform_location(program, "u_base_width_px") },
            u_selected_width_px: unsafe { gl.get_uniform_location(program, "u_selected_width_px") },
            u_primary_width_px: unsafe { gl.get_uniform_location(program, "u_primary_width_px") },
            u_base_color: unsafe { gl.get_uniform_location(program, "u_base_color") },
            u_selected_color: unsafe { gl.get_uniform_location(program, "u_selected_color") },
            u_primary_color: unsafe { gl.get_uniform_location(program, "u_primary_color") },
            u_draw_unselected: unsafe { gl.get_uniform_location(program, "u_draw_unselected") },
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
