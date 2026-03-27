use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use super::{AnnotationShape, AnnotationValueMode};

#[derive(Debug, Clone)]
pub struct AnnotationGlDraw {
    pub generation: u64,
    pub positions_local: Arc<Vec<egui::Pos2>>,
    pub values: Arc<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct AnnotationGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub roi_offset_world: egui::Vec2,
    pub roi_scale: f32,
    pub layer_offset_world: egui::Vec2,

    pub radius_screen_px: f32,
    pub opacity: f32,
    pub stroke: egui::Stroke,

    pub mode: AnnotationValueMode,
    pub cat_colors: Arc<Vec<[f32; 4]>>,
    pub cat_shapes: Arc<Vec<i32>>,
    pub cat_visible: Arc<Vec<i32>>,

    pub value_min: f32,
    pub value_max: f32,
    pub continuous_shape: AnnotationShape,
}

#[derive(Clone)]
pub struct AnnotationGlRenderer {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for AnnotationGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnnotationGlRenderer")
            .finish_non_exhaustive()
    }
}

impl Default for AnnotationGlRenderer {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }
}

impl AnnotationGlRenderer {
    fn effective_radius_points(base_radius_points: f32, zoom_screen_per_world_px: f32) -> f32 {
        let zoom = zoom_screen_per_world_px.max(1e-6);
        (base_radius_points.max(0.0) * zoom.sqrt()).clamp(0.75, 40.0)
    }

    pub fn paint(
        &self,
        info: &egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        data: &AnnotationGlDraw,
        params: &AnnotationGlDrawParams,
    ) {
        let count = data.positions_local.len().min(data.values.len());
        if count == 0 {
            return;
        }

        let gl = painter.gl();
        let gl_major = gl.version().major;

        // Upload / cache management.
        let key = BufferKey::new(&data.positions_local, &data.values);
        let (program, uniforms, vao, vbo, draw_count, cache_cap) = {
            let mut inner = self.inner.lock();
            if inner.gl_objects.is_none() {
                inner.gl_objects = GlObjects::new(gl).ok();
            }
            if inner.gl_objects.is_none() {
                return;
            }

            let needs_upload = match inner.buffers.get(&key) {
                Some(buf) => buf.uploaded_generation != data.generation || buf.count != count,
                None => true,
            };
            if needs_upload {
                if let Ok(interleaved) =
                    interleave_positions_values(&data.positions_local, &data.values)
                {
                    inner.ensure_buffer(gl, key);
                    if let Some(buf) = inner.buffers.get_mut(&key) {
                        let old = buf.count;
                        buf.upload(gl, &interleaved);
                        buf.uploaded_generation = data.generation;
                        buf.count = count;
                        if count >= old {
                            inner.points_cached = inner.points_cached.saturating_add(count - old);
                        } else {
                            inner.points_cached = inner.points_cached.saturating_sub(old - count);
                        }
                    }
                    inner.touch(key);
                }
            } else {
                inner.touch(key);
            }

            let Some(buf) = inner.buffers.get(&key) else {
                return;
            };
            if buf.count == 0 {
                return;
            }

            let cache_cap = inner.max_points_cached;
            let objects = inner.gl_objects.as_ref().expect("gl_objects checked");
            (
                objects.program,
                objects.uniforms_snapshot(),
                buf.vao,
                buf.vbo,
                buf.count,
                cache_cap,
            )
        };

        let viewport = info.viewport;
        let pixels_per_point = info.pixels_per_point.max(1e-6);
        let radius_points =
            Self::effective_radius_points(params.radius_screen_px, params.zoom_screen_per_world);
        let radius_px = (radius_points * pixels_per_point).max(1.0);
        let point_size_px = radius_px * 2.0;

        let stroke_px = (params.stroke.width.max(0.0) * pixels_per_point).min(radius_px);
        let stroke_frac = if radius_px > 0.0 {
            stroke_px / radius_px
        } else {
            0.0
        };

        let opacity = params.opacity.clamp(0.0, 1.0);

        unsafe {
            let gl = gl.as_ref();

            gl.disable(glow::DEPTH_TEST);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            if gl_major >= 3 {
                gl.enable(glow::PROGRAM_POINT_SIZE);
            } else {
                gl.enable(glow::VERTEX_PROGRAM_POINT_SIZE);
            }

            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            gl.uniform_2_f32(
                uniforms.u_center_world.as_ref(),
                params.center_world.x,
                params.center_world.y,
            );
            gl.uniform_1_f32(
                uniforms.u_zoom.as_ref(),
                params.zoom_screen_per_world.max(1e-6),
            );
            gl.uniform_2_f32(
                uniforms.u_viewport_min.as_ref(),
                viewport.min.x,
                viewport.min.y,
            );
            gl.uniform_2_f32(
                uniforms.u_viewport_size.as_ref(),
                viewport.width().max(1.0),
                viewport.height().max(1.0),
            );
            gl.uniform_1_f32(uniforms.u_point_size_px.as_ref(), point_size_px);

            gl.uniform_2_f32(
                uniforms.u_roi_offset.as_ref(),
                params.roi_offset_world.x,
                params.roi_offset_world.y,
            );
            gl.uniform_1_f32(uniforms.u_roi_scale.as_ref(), params.roi_scale.max(1e-6));
            gl.uniform_2_f32(
                uniforms.u_layer_offset.as_ref(),
                params.layer_offset_world.x,
                params.layer_offset_world.y,
            );

            gl.uniform_1_i32(
                uniforms.u_mode.as_ref(),
                match params.mode {
                    AnnotationValueMode::Categorical => 0,
                    AnnotationValueMode::Continuous => 1,
                },
            );
            gl.uniform_1_f32(uniforms.u_opacity.as_ref(), opacity);

            gl.uniform_4_f32_slice(
                uniforms.u_stroke_color.as_ref(),
                &color_to_vec4(params.stroke.color),
            );
            gl.uniform_1_f32(uniforms.u_stroke_frac.as_ref(), stroke_frac.clamp(0.0, 1.0));

            if matches!(params.mode, AnnotationValueMode::Categorical) {
                let cat_len = params
                    .cat_colors
                    .len()
                    .min(params.cat_shapes.len())
                    .min(params.cat_visible.len())
                    .min(256);
                gl.uniform_1_i32(uniforms.u_cat_len.as_ref(), cat_len as i32);
                if cat_len > 0 {
                    let mut flat: Vec<f32> = Vec::with_capacity(cat_len * 4);
                    for c in params.cat_colors.iter().take(cat_len) {
                        flat.extend_from_slice(c);
                    }
                    gl.uniform_4_f32_slice(uniforms.u_cat_color.as_ref(), &flat);
                    gl.uniform_1_i32_slice(
                        uniforms.u_cat_shape.as_ref(),
                        &params.cat_shapes[..cat_len],
                    );
                    gl.uniform_1_i32_slice(
                        uniforms.u_cat_visible.as_ref(),
                        &params.cat_visible[..cat_len],
                    );
                }
            } else {
                gl.uniform_1_f32(uniforms.u_value_min.as_ref(), params.value_min);
                gl.uniform_1_f32(uniforms.u_value_max.as_ref(), params.value_max);
                gl.uniform_1_i32(
                    uniforms.u_cont_shape.as_ref(),
                    params.continuous_shape as i32,
                );
            }

            gl.draw_arrays(glow::POINTS, 0, draw_count as i32);

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
        }

        // Cap GL cache after any uploads this frame.
        let mut inner = self.inner.lock();
        let cap = cache_cap;
        inner.gc(gl, cap);
    }
}

#[derive(Default)]
struct Inner {
    gl_objects: Option<GlObjects>,
    buffers: HashMap<BufferKey, BufferObjects>,
    lru: VecDeque<BufferKey>,
    points_cached: usize,
    max_points_cached: usize,
    buffers_to_delete: Vec<BufferObjects>,
}

impl Inner {
    fn ensure_buffer(&mut self, gl: &Arc<glow::Context>, key: BufferKey) {
        if self.buffers.contains_key(&key) {
            return;
        }
        if let Some(objects) = BufferObjects::new(gl) {
            self.buffers.insert(key, objects);
            self.lru.push_back(key);
        }
    }

    fn touch(&mut self, key: BufferKey) {
        // Update LRU order (cheap linear scan; LRU size should be small).
        if let Some(pos) = self.lru.iter().position(|k| *k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key);
    }

    fn gc(&mut self, gl: &Arc<glow::Context>, max_points: usize) {
        let cap = if max_points == 0 {
            10_000_000
        } else {
            max_points
        };
        self.max_points_cached = cap;
        while self.points_cached > cap && !self.lru.is_empty() {
            let key = self.lru.pop_front().unwrap();
            if let Some(buf) = self.buffers.remove(&key) {
                self.points_cached = self.points_cached.saturating_sub(buf.count);
                self.buffers_to_delete.push(buf);
            }
        }
        if !self.buffers_to_delete.is_empty() {
            unsafe {
                let gl = gl.as_ref();
                for b in self.buffers_to_delete.drain(..) {
                    gl.delete_vertex_array(b.vao);
                    gl.delete_buffer(b.vbo);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BufferKey {
    pos_ptr: usize,
    val_ptr: usize,
}

impl Hash for BufferKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos_ptr.hash(state);
        self.val_ptr.hash(state);
    }
}

impl BufferKey {
    fn new(pos: &Arc<Vec<egui::Pos2>>, val: &Arc<Vec<f32>>) -> Self {
        Self {
            pos_ptr: Arc::as_ptr(pos) as usize,
            val_ptr: Arc::as_ptr(val) as usize,
        }
    }
}

struct GlObjects {
    program: glow::Program,

    u_center_world: Option<glow::UniformLocation>,
    u_zoom: Option<glow::UniformLocation>,
    u_viewport_min: Option<glow::UniformLocation>,
    u_viewport_size: Option<glow::UniformLocation>,
    u_point_size_px: Option<glow::UniformLocation>,

    u_roi_offset: Option<glow::UniformLocation>,
    u_roi_scale: Option<glow::UniformLocation>,
    u_layer_offset: Option<glow::UniformLocation>,

    u_mode: Option<glow::UniformLocation>,
    u_opacity: Option<glow::UniformLocation>,

    u_cat_len: Option<glow::UniformLocation>,
    u_cat_color: Option<glow::UniformLocation>,
    u_cat_shape: Option<glow::UniformLocation>,
    u_cat_visible: Option<glow::UniformLocation>,

    u_value_min: Option<glow::UniformLocation>,
    u_value_max: Option<glow::UniformLocation>,
    u_cont_shape: Option<glow::UniformLocation>,

    u_stroke_color: Option<glow::UniformLocation>,
    u_stroke_frac: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn uniforms_snapshot(&self) -> UniformsSnapshot {
        UniformsSnapshot {
            u_center_world: self.u_center_world.clone(),
            u_zoom: self.u_zoom.clone(),
            u_viewport_min: self.u_viewport_min.clone(),
            u_viewport_size: self.u_viewport_size.clone(),
            u_point_size_px: self.u_point_size_px.clone(),
            u_roi_offset: self.u_roi_offset.clone(),
            u_roi_scale: self.u_roi_scale.clone(),
            u_layer_offset: self.u_layer_offset.clone(),
            u_mode: self.u_mode.clone(),
            u_opacity: self.u_opacity.clone(),
            u_cat_len: self.u_cat_len.clone(),
            u_cat_color: self.u_cat_color.clone(),
            u_cat_shape: self.u_cat_shape.clone(),
            u_cat_visible: self.u_cat_visible.clone(),
            u_value_min: self.u_value_min.clone(),
            u_value_max: self.u_value_max.clone(),
            u_cont_shape: self.u_cont_shape.clone(),
            u_stroke_color: self.u_stroke_color.clone(),
            u_stroke_frac: self.u_stroke_frac.clone(),
        }
    }

    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let (vs_src, fs_src) = shader_sources(gl.version().major);
        let program =
            compile_program(gl, vs_src, fs_src).context("compile annotations program failed")?;

        let uniforms = unsafe {
            (
                gl.get_uniform_location(program, "u_center_world"),
                gl.get_uniform_location(program, "u_zoom"),
                gl.get_uniform_location(program, "u_viewport_min"),
                gl.get_uniform_location(program, "u_viewport_size"),
                gl.get_uniform_location(program, "u_point_size_px"),
                gl.get_uniform_location(program, "u_roi_offset"),
                gl.get_uniform_location(program, "u_roi_scale"),
                gl.get_uniform_location(program, "u_layer_offset"),
                gl.get_uniform_location(program, "u_mode"),
                gl.get_uniform_location(program, "u_opacity"),
                gl.get_uniform_location(program, "u_cat_len"),
                gl.get_uniform_location(program, "u_cat_color"),
                gl.get_uniform_location(program, "u_cat_shape"),
                gl.get_uniform_location(program, "u_cat_visible"),
                gl.get_uniform_location(program, "u_value_min"),
                gl.get_uniform_location(program, "u_value_max"),
                gl.get_uniform_location(program, "u_cont_shape"),
                gl.get_uniform_location(program, "u_stroke_color"),
                gl.get_uniform_location(program, "u_stroke_frac"),
            )
        };

        Ok(Self {
            program,
            u_center_world: uniforms.0,
            u_zoom: uniforms.1,
            u_viewport_min: uniforms.2,
            u_viewport_size: uniforms.3,
            u_point_size_px: uniforms.4,
            u_roi_offset: uniforms.5,
            u_roi_scale: uniforms.6,
            u_layer_offset: uniforms.7,
            u_mode: uniforms.8,
            u_opacity: uniforms.9,
            u_cat_len: uniforms.10,
            u_cat_color: uniforms.11,
            u_cat_shape: uniforms.12,
            u_cat_visible: uniforms.13,
            u_value_min: uniforms.14,
            u_value_max: uniforms.15,
            u_cont_shape: uniforms.16,
            u_stroke_color: uniforms.17,
            u_stroke_frac: uniforms.18,
        })
    }
}

#[derive(Clone)]
struct UniformsSnapshot {
    u_center_world: Option<glow::UniformLocation>,
    u_zoom: Option<glow::UniformLocation>,
    u_viewport_min: Option<glow::UniformLocation>,
    u_viewport_size: Option<glow::UniformLocation>,
    u_point_size_px: Option<glow::UniformLocation>,

    u_roi_offset: Option<glow::UniformLocation>,
    u_roi_scale: Option<glow::UniformLocation>,
    u_layer_offset: Option<glow::UniformLocation>,

    u_mode: Option<glow::UniformLocation>,
    u_opacity: Option<glow::UniformLocation>,

    u_cat_len: Option<glow::UniformLocation>,
    u_cat_color: Option<glow::UniformLocation>,
    u_cat_shape: Option<glow::UniformLocation>,
    u_cat_visible: Option<glow::UniformLocation>,

    u_value_min: Option<glow::UniformLocation>,
    u_value_max: Option<glow::UniformLocation>,
    u_cont_shape: Option<glow::UniformLocation>,

    u_stroke_color: Option<glow::UniformLocation>,
    u_stroke_frac: Option<glow::UniformLocation>,
}

struct BufferObjects {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    uploaded_generation: u64,
    count: usize,
}

impl BufferObjects {
    fn new(gl: &Arc<glow::Context>) -> Option<Self> {
        let gl = gl.as_ref();
        let (vao, vbo) = unsafe {
            let vao = gl.create_vertex_array().ok()?;
            let vbo = gl.create_buffer().ok()?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            let stride = (3 * std::mem::size_of::<f32>()) as i32;
            // location 0: a_local
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            // location 1: a_value
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(
                1,
                1,
                glow::FLOAT,
                false,
                stride,
                (2 * std::mem::size_of::<f32>()) as i32,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            (vao, vbo)
        };
        Some(Self {
            vao,
            vbo,
            uploaded_generation: 0,
            count: 0,
        })
    }

    fn upload(&mut self, gl: &Arc<glow::Context>, interleaved: &[f32]) {
        let gl = gl.as_ref();
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(interleaved),
                glow::DYNAMIC_DRAW,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }
}

fn interleave_positions_values(
    positions: &[egui::Pos2],
    values: &[f32],
) -> anyhow::Result<Vec<f32>> {
    let n = positions.len().min(values.len());
    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let p = positions[i];
        out.push(p.x);
        out.push(p.y);
        out.push(values[i]);
    }
    Ok(out)
}

fn color_to_vec4(c: egui::Color32) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn shader_sources(gl_major: u32) -> (&'static str, &'static str) {
    if gl_major >= 3 {
        (VERT_330, FRAG_330)
    } else {
        (VERT_120, FRAG_120)
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
            .map_err(|e| anyhow::anyhow!("create vertex shader failed: {e}"))?;
        gl.shader_source(vs, vs_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            let log = gl.get_shader_info_log(vs);
            gl.delete_shader(vs);
            return Err(anyhow::anyhow!("vertex shader compile failed: {log}"));
        }

        let fs = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(|e| anyhow::anyhow!("create fragment shader failed: {e}"))?;
        gl.shader_source(fs, fs_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            let log = gl.get_shader_info_log(fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);
            return Err(anyhow::anyhow!("fragment shader compile failed: {log}"));
        }

        let program = gl
            .create_program()
            .map_err(|e| anyhow::anyhow!("create_program failed: {e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);

        gl.bind_attrib_location(program, 0, "a_local");
        gl.bind_attrib_location(program, 1, "a_value");

        gl.link_program(program);
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(anyhow::anyhow!("program link failed: {log}"));
        }
        Ok(program)
    }
}

const VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_local;
layout(location = 1) in float a_value;

uniform vec2 u_center_world;
uniform float u_zoom;
uniform vec2 u_viewport_min;
uniform vec2 u_viewport_size;
uniform float u_point_size_px;

uniform vec2 u_roi_offset;
uniform float u_roi_scale;
uniform vec2 u_layer_offset;

out float v_value;

void main() {
    vec2 world = u_roi_offset + a_local * u_roi_scale + u_layer_offset;
    vec2 viewport_center = u_viewport_min + 0.5 * u_viewport_size;
    vec2 screen = (world - u_center_world) * u_zoom + viewport_center;
    vec2 local = screen - u_viewport_min;
    vec2 ndc = vec2(
        (local.x / u_viewport_size.x) * 2.0 - 1.0,
        1.0 - (local.y / u_viewport_size.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_point_size_px;
    v_value = a_value;
}
"#;

const FRAG_330: &str = r#"#version 330 core
in float v_value;

uniform int u_mode; // 0 categorical, 1 continuous
uniform float u_opacity;

uniform int u_cat_len;
uniform vec4 u_cat_color[256];
uniform int u_cat_shape[256];
uniform int u_cat_visible[256];

uniform float u_value_min;
uniform float u_value_max;
uniform int u_cont_shape;

uniform vec4 u_stroke_color;
uniform float u_stroke_frac;

out vec4 out_color;

vec3 turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    vec4 v = vec4(1.0, x, x*x, x*x*x);
    vec3 r = vec3(
        dot(v, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)),
        dot(v, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)),
        dot(v, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771))
    );
    return clamp(r, 0.0, 1.0);
}

float shape_dist(vec2 c, int shape_id) {
    vec2 a = abs(c);
    if (shape_id == 0) { // circle
        return length(c);
    } else if (shape_id == 1) { // square
        return max(a.x, a.y);
    } else if (shape_id == 2) { // diamond
        return a.x + a.y;
    } else { // cross
        float bar = 0.35;
        bool in_cross = (a.x < bar && a.y < 1.0) || (a.y < bar && a.x < 1.0);
        return in_cross ? max(a.x, a.y) : 2.0;
    }
}

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0; // [-1, 1]

    vec4 fill = vec4(1.0);
    int shape_id = 0;
    if (u_mode == 0) {
        if (u_cat_len <= 0) {
            discard;
        }
        int idx = int(floor(v_value + 0.5));
        idx = idx % u_cat_len;
        if (idx < 0) { idx += u_cat_len; }
        if (u_cat_visible[idx] == 0) {
            discard;
        }
        fill = u_cat_color[idx];
        shape_id = u_cat_shape[idx];
    } else {
        float denom = max(u_value_max - u_value_min, 1e-6);
        float t = (v_value - u_value_min) / denom;
        fill = vec4(turbo(t), 1.0);
        shape_id = u_cont_shape;
    }

    float dist = shape_dist(c, shape_id);
    if (dist > 1.0) {
        discard;
    }

    float aa = fwidth(dist);
    float edge_alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);

    vec4 col = fill;
    if (u_stroke_frac > 0.0 && u_stroke_color.a > 0.0) {
        float inner = clamp(1.0 - u_stroke_frac, 0.0, 1.0);
        float stroke_mix = smoothstep(inner - aa, inner + aa, dist);
        col = mix(fill, u_stroke_color, stroke_mix);
    }
    col.a *= (edge_alpha * u_opacity);
    out_color = col;
}
"#;

const VERT_120: &str = r#"#version 120
attribute vec2 a_local;
attribute float a_value;

uniform vec2 u_center_world;
uniform float u_zoom;
uniform vec2 u_viewport_min;
uniform vec2 u_viewport_size;
uniform float u_point_size_px;

uniform vec2 u_roi_offset;
uniform float u_roi_scale;
uniform vec2 u_layer_offset;

varying float v_value;

void main() {
    vec2 world = u_roi_offset + a_local * u_roi_scale + u_layer_offset;
    vec2 viewport_center = u_viewport_min + 0.5 * u_viewport_size;
    vec2 screen = (world - u_center_world) * u_zoom + viewport_center;
    vec2 local = screen - u_viewport_min;
    vec2 ndc = vec2(
        (local.x / u_viewport_size.x) * 2.0 - 1.0,
        1.0 - (local.y / u_viewport_size.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_point_size_px;
    v_value = a_value;
}
"#;

const FRAG_120: &str = r#"#version 120
varying float v_value;

uniform int u_mode;
uniform float u_opacity;

uniform int u_cat_len;
uniform vec4 u_cat_color[256];
uniform int u_cat_shape[256];
uniform int u_cat_visible[256];

uniform float u_value_min;
uniform float u_value_max;
uniform int u_cont_shape;

uniform vec4 u_stroke_color;
uniform float u_stroke_frac;

vec3 turbo(float x) {
    x = clamp(x, 0.0, 1.0);
    vec4 v = vec4(1.0, x, x*x, x*x*x);
    vec3 r = vec3(
        dot(v, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)),
        dot(v, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)),
        dot(v, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771))
    );
    return clamp(r, 0.0, 1.0);
}

float shape_dist(vec2 c, int shape_id) {
    vec2 a = abs(c);
    if (shape_id == 0) { return length(c); }
    if (shape_id == 1) { return max(a.x, a.y); }
    if (shape_id == 2) { return a.x + a.y; }
    float bar = 0.35;
    bool in_cross = (a.x < bar && a.y < 1.0) || (a.y < bar && a.x < 1.0);
    return in_cross ? max(a.x, a.y) : 2.0;
}

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;

    vec4 fill = vec4(1.0);
    int shape_id = 0;
    if (u_mode == 0) {
        if (u_cat_len <= 0) { discard; }
        int idx = int(floor(v_value + 0.5));
        idx = idx % u_cat_len;
        if (idx < 0) { idx += u_cat_len; }
        if (u_cat_visible[idx] == 0) { discard; }
        fill = u_cat_color[idx];
        shape_id = u_cat_shape[idx];
    } else {
        float denom = max(u_value_max - u_value_min, 1e-6);
        float t = (v_value - u_value_min) / denom;
        fill = vec4(turbo(t), 1.0);
        shape_id = u_cont_shape;
    }

    float dist = shape_dist(c, shape_id);
    if (dist > 1.0) { discard; }

    float aa = fwidth(dist);
    float edge_alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);

    vec4 col = fill;
    if (u_stroke_frac > 0.0 && u_stroke_color.a > 0.0) {
        float inner = clamp(1.0 - u_stroke_frac, 0.0, 1.0);
        float stroke_mix = smoothstep(inner - aa, inner + aa, dist);
        col = mix(fill, u_stroke_color, stroke_mix);
    }
    col.a *= (edge_alpha * u_opacity);
    gl_FragColor = col;
}
"#;
