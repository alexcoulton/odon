use std::num::NonZeroUsize;
use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

use crate::render::line_bins::LineSegmentsBins;

#[derive(Debug, Clone)]
pub struct LineBinsGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub width_points: f32,
    pub color: egui::Color32,
    pub visible: bool,
    /// Maps local segment coordinates to the camera's world coordinate system:
    /// `world = local_to_world_offset + local * local_to_world_scale`.
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
}

#[derive(Debug, Clone)]
pub struct LineBinsGlDrawData {
    /// Cache namespace (use a stable per-layer/per-item id).
    pub cache_id: u64,
    pub generation: u64,
    pub bins: Arc<LineSegmentsBins>,
}

#[derive(Debug, Clone)]
pub struct LineBinsGlDrawItem {
    pub data: LineBinsGlDrawData,
    pub params: LineBinsGlDrawParams,
    /// Visible rect in the same coordinate system as `data.bins` (usually local coordinates).
    pub visible_world: egui::Rect,
}

#[derive(Clone)]
pub struct LineBinsGlRenderer {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for LineBinsGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LineBinsGlRenderer").finish_non_exhaustive()
    }
}

impl Default for LineBinsGlRenderer {
    fn default() -> Self {
        Self::new(512)
    }
}

impl LineBinsGlRenderer {
    pub fn new(max_uploaded_bins: usize) -> Self {
        let cap = NonZeroUsize::new(max_uploaded_bins.max(8)).unwrap();
        Self {
            inner: Arc::new(Mutex::new(Inner::new(cap))),
        }
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        data: &LineBinsGlDrawData,
        params: &LineBinsGlDrawParams,
        visible_world: egui::Rect,
    ) {
        self.paint_many(
            info,
            painter,
            &[LineBinsGlDrawItem {
                data: data.clone(),
                params: params.clone(),
                visible_world,
            }],
        );
    }

    pub fn paint_many(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        items: &[LineBinsGlDrawItem],
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
        let u_width_px = objects.u_width_px.clone();
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

        let mut uploaded_this_frame = 0usize;
        let mut missing_this_frame = 0usize;
        const MAX_BIN_UPLOADS_PER_FRAME: usize = 64;

        for it in items {
            if !it.params.visible {
                continue;
            }
            if it.data.bins.segments.is_empty() {
                continue;
            }

            let width_px = (it.params.width_points.max(0.0) * ppp).max(0.5);
            let c = it.params.color;
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
                    it.params.center_world.x,
                    it.params.center_world.y,
                );
                gl.uniform_1_f32(
                    u_zoom_px.as_ref(),
                    (it.params.zoom_screen_per_world.max(1e-6) * ppp).max(1e-6),
                );
                gl.uniform_1_f32(u_width_px.as_ref(), width_px);
                gl.uniform_4_f32_slice(u_color.as_ref(), &color);
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
                    let gpu = inner.ensure_bin_uploaded(
                        gl,
                        it.data.cache_id,
                        it.data.generation,
                        bin_index,
                        slice,
                        allow_upload,
                    );
                    let Some(gpu) = gpu else {
                        // Non-empty bin not yet uploaded and we hit the per-frame upload budget.
                        // This should drive repaint scheduling (via `last_frame_missing_bins`).
                        missing_this_frame += 1;
                        continue;
                    };
                    if gpu.uploaded {
                        uploaded_this_frame += 1;
                    }

                    unsafe {
                        let gl = gl.as_ref();
                        gl.bind_buffer(glow::ARRAY_BUFFER, Some(gpu.handle.vbo));

                        // a_seg (vec4) with divisor 1.
                        gl.enable_vertex_attrib_array(0);
                        gl.vertex_attrib_pointer_f32(0, 4, glow::FLOAT, false, 16, 0);
                        gl.vertex_attrib_divisor(0, 1);

                        gl.draw_arrays_instanced(glow::TRIANGLES, 0, 6, gpu.handle.count as i32);
                    }
                }
            }
        }

        inner.last_frame_missing_bins = missing_this_frame;

        unsafe {
            let gl = gl.as_ref();
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

#[derive(Clone, Copy)]
struct BinGpuHandle {
    vbo: glow::Buffer,
    count: usize,
}

#[derive(Clone, Copy)]
struct BinUpload {
    handle: BinGpuHandle,
    uploaded: bool,
}

struct BinGpu {
    vbo: glow::Buffer,
    count: usize,
}

struct Inner {
    gl_objects: Option<GlObjects>,
    bins: LruCache<(u64, u64, usize), BinGpu>,
    buffers_to_delete: Vec<glow::Buffer>,
    last_frame_missing_bins: usize,
}

impl Inner {
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            gl_objects: None,
            bins: LruCache::new(cap),
            buffers_to_delete: Vec::new(),
            last_frame_missing_bins: 0,
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

    fn ensure_bin_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        bin_index: usize,
        segments: &[[f32; 4]],
        allow_upload: bool,
    ) -> Option<BinUpload> {
        let key = (cache_id, generation, bin_index);
        if let Some(v) = self.bins.get(&key) {
            return Some(BinUpload {
                handle: BinGpuHandle {
                    vbo: v.vbo,
                    count: v.count,
                },
                uploaded: false,
            });
        }
        if !allow_upload {
            return None;
        }

        let vbo = unsafe { gl.as_ref().create_buffer().map_err(|_e| ()).ok()? };
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

        // Insert + evict.
        if let Some((_ek, ev)) = self.bins.push(
            key,
            BinGpu {
                vbo,
                count: segments.len(),
            },
        ) {
            self.buffers_to_delete.push(ev.vbo);
        }
        self.delete_queued(gl);

        let v = self.bins.get(&key)?;
        Some(BinUpload {
            handle: BinGpuHandle {
                vbo: v.vbo,
                count: v.count,
            },
            uploaded: true,
        })
    }
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,

    u_center_world: Option<glow::UniformLocation>,
    u_zoom_px: Option<glow::UniformLocation>,
    u_viewport_min_px: Option<glow::UniformLocation>,
    u_viewport_size_px: Option<glow::UniformLocation>,
    u_width_px: Option<glow::UniformLocation>,
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
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?
        };

        // No per-vertex attributes: we use gl_VertexID to expand each instance into a quad.
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
            u_width_px: unsafe { gl.get_uniform_location(program, "u_width_px") },
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

use super::program::compile_program;

const VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec4 a_seg; // (x0, y0, x1, y1) local coords

uniform vec2 u_center_world;
uniform float u_zoom_px; // pixels per world unit
uniform vec2 u_viewport_min_px;
uniform vec2 u_viewport_size_px;
uniform float u_width_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;

out vec2 v_screen_px;
out vec2 v_a_px;
out vec2 v_b_px;
out float v_half_w;

void main() {
    vec2 a_world = u_local_to_world_offset + a_seg.xy * u_local_to_world_scale;
    vec2 b_world = u_local_to_world_offset + a_seg.zw * u_local_to_world_scale;

    vec2 viewport_center_px = u_viewport_min_px + 0.5 * u_viewport_size_px;
    vec2 a_px = (a_world - u_center_world) * u_zoom_px + viewport_center_px;
    vec2 b_px = (b_world - u_center_world) * u_zoom_px + viewport_center_px;

    vec2 d = b_px - a_px;
    float len2 = max(dot(d, d), 1e-6);
    vec2 dir = d * inversesqrt(len2);
    vec2 n = vec2(-dir.y, dir.x);

    float half_w = 0.5 * max(u_width_px, 0.5);
    // Extend segment endpoints to overlap joins a bit (reduces cracks).
    vec2 a2 = a_px - dir * half_w;
    vec2 b2 = b_px + dir * half_w;

    // 6 vertices per instance (two triangles). gl_VertexID in [0..5].
    int vid = gl_VertexID;
    float t = 0.0;
    float side = -1.0;
    if (vid == 0) { t = 0.0; side = -1.0; }
    else if (vid == 1) { t = 1.0; side = -1.0; }
    else if (vid == 2) { t = 1.0; side = 1.0; }
    else if (vid == 3) { t = 0.0; side = -1.0; }
    else if (vid == 4) { t = 1.0; side = 1.0; }
    else { t = 0.0; side = 1.0; }

    vec2 base = mix(a2, b2, t) + n * side * half_w;

    vec2 local = base - u_viewport_min_px;
    vec2 ndc = vec2(
        (local.x / u_viewport_size_px.x) * 2.0 - 1.0,
        1.0 - (local.y / u_viewport_size_px.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_screen_px = base;
    v_a_px = a2;
    v_b_px = b2;
    v_half_w = half_w;
}
"#;

const FRAG_330: &str = r#"#version 330 core
in vec2 v_screen_px;
in vec2 v_a_px;
in vec2 v_b_px;
in float v_half_w;

uniform vec4 u_color;

out vec4 out_color;

float segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float dist = segment_distance(v_screen_px, v_a_px, v_b_px);
    // Simple 1px-ish AA band.
    float aa = 1.0;
    float alpha = 1.0 - smoothstep(v_half_w - aa, v_half_w + aa, dist);
    out_color = vec4(u_color.rgb, u_color.a * alpha);
    if (out_color.a <= 0.0) {
        discard;
    }
}
"#;
