use std::sync::Arc;

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use crate::render::points::PointsStyle;

#[derive(Debug, Clone)]
pub struct PointsGlDrawData {
    pub generation: u64,
    pub positions_world: Arc<Vec<egui::Pos2>>,
    pub values: Arc<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct PointsGlDrawParams {
    pub center_world: egui::Pos2,
    pub zoom_screen_per_world: f32,
    pub threshold: f32,
    pub style: PointsStyle,
    pub visible: bool,
    pub local_to_world_offset: egui::Vec2,
    pub local_to_world_scale: egui::Vec2,
}

#[derive(Clone)]
pub struct PointsGlRenderer {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for PointsGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointsGlRenderer").finish_non_exhaustive()
    }
}

impl Default for PointsGlRenderer {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }
}

impl PointsGlRenderer {
    fn effective_radius_points(base_radius_points: f32, zoom_screen_per_world_px: f32) -> f32 {
        // Make points smaller when zoomed out, larger when zoomed in.
        // The sqrt keeps it from growing/shrinking too aggressively.
        let zoom = zoom_screen_per_world_px.max(1e-6);
        (base_radius_points.max(0.0) * zoom.sqrt()).clamp(0.75, 40.0)
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        data: &PointsGlDrawData,
        params: &PointsGlDrawParams,
    ) {
        if !params.visible {
            return;
        }
        let count = data.positions_world.len().min(data.values.len());
        if count == 0 {
            return;
        }

        let gl = painter.gl();
        let gl_major = gl.version().major;
        let mut inner = self.inner.lock();
        if inner.gl_objects.is_none() {
            inner.gl_objects = GlObjects::new(gl).ok();
        }

        let needs_upload =
            inner.uploaded_generation != data.generation || inner.uploaded_count != count;
        if needs_upload {
            if let Ok(interleaved) =
                interleave_positions_values(&data.positions_world, &data.values)
            {
                if let Some(objects) = inner.gl_objects.as_mut() {
                    objects.upload(gl, &interleaved);
                } else {
                    return;
                }
                inner.uploaded_generation = data.generation;
                inner.uploaded_count = count;
            }
        }
        let Some(objects) = inner.gl_objects.as_ref() else {
            return;
        };
        let uploaded_count = inner.uploaded_count;

        let viewport = info.viewport;
        let pixels_per_point = info.pixels_per_point.max(1e-6);
        let radius_points = Self::effective_radius_points(
            params.style.radius_screen_px,
            params.zoom_screen_per_world,
        );
        let radius_px = (radius_points * pixels_per_point).max(1.0);
        let point_size_px = radius_px * 2.0;
        let stroke_px_pos =
            (params.style.stroke_positive.width.max(0.0) * pixels_per_point).min(radius_px);
        let stroke_frac_pos = if radius_px > 0.0 {
            stroke_px_pos / radius_px
        } else {
            0.0
        };
        let stroke_px_neg =
            (params.style.stroke_negative.width.max(0.0) * pixels_per_point).min(radius_px);
        let stroke_frac_neg = if radius_px > 0.0 {
            stroke_px_neg / radius_px
        } else {
            0.0
        };

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

            gl.use_program(Some(objects.program));
            gl.bind_vertex_array(Some(objects.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(objects.vbo));

            gl.uniform_2_f32(
                objects.u_center_world.as_ref(),
                params.center_world.x,
                params.center_world.y,
            );
            gl.uniform_1_f32(
                objects.u_zoom.as_ref(),
                params.zoom_screen_per_world.max(1e-6),
            );
            gl.uniform_2_f32(
                objects.u_viewport_min.as_ref(),
                viewport.min.x,
                viewport.min.y,
            );
            gl.uniform_2_f32(
                objects.u_viewport_size.as_ref(),
                viewport.width().max(1.0),
                viewport.height().max(1.0),
            );
            gl.uniform_1_f32(objects.u_point_size_px.as_ref(), point_size_px);
            gl.uniform_1_f32(objects.u_threshold.as_ref(), params.threshold);
            gl.uniform_2_f32(
                objects.u_local_to_world_offset.as_ref(),
                params.local_to_world_offset.x,
                params.local_to_world_offset.y,
            );
            gl.uniform_2_f32(
                objects.u_local_to_world_scale.as_ref(),
                params.local_to_world_scale.x,
                params.local_to_world_scale.y,
            );
            gl.uniform_4_f32_slice(
                objects.u_fill_pos.as_ref(),
                &color_to_vec4(params.style.fill_positive),
            );
            gl.uniform_4_f32_slice(
                objects.u_fill_neg.as_ref(),
                &color_to_vec4(params.style.fill_negative),
            );
            gl.uniform_4_f32_slice(
                objects.u_stroke_pos.as_ref(),
                &color_to_vec4(params.style.stroke_positive.color),
            );
            gl.uniform_4_f32_slice(
                objects.u_stroke_neg.as_ref(),
                &color_to_vec4(params.style.stroke_negative.color),
            );
            gl.uniform_1_f32(
                objects.u_stroke_frac_pos.as_ref(),
                stroke_frac_pos.clamp(0.0, 1.0),
            );
            gl.uniform_1_f32(
                objects.u_stroke_frac_neg.as_ref(),
                stroke_frac_neg.clamp(0.0, 1.0),
            );

            gl.draw_arrays(glow::POINTS, 0, uploaded_count as i32);

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
        }
    }
}

#[derive(Default)]
struct Inner {
    gl_objects: Option<GlObjects>,
    uploaded_generation: u64,
    uploaded_count: usize,
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,

    u_center_world: Option<glow::UniformLocation>,
    u_zoom: Option<glow::UniformLocation>,
    u_viewport_min: Option<glow::UniformLocation>,
    u_viewport_size: Option<glow::UniformLocation>,
    u_point_size_px: Option<glow::UniformLocation>,
    u_threshold: Option<glow::UniformLocation>,
    u_local_to_world_offset: Option<glow::UniformLocation>,
    u_local_to_world_scale: Option<glow::UniformLocation>,
    u_fill_pos: Option<glow::UniformLocation>,
    u_fill_neg: Option<glow::UniformLocation>,
    u_stroke_pos: Option<glow::UniformLocation>,
    u_stroke_neg: Option<glow::UniformLocation>,
    u_stroke_frac_pos: Option<glow::UniformLocation>,
    u_stroke_frac_neg: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();

        let (vs_src, fs_src) = shader_sources(gl.version().major);
        let program =
            compile_program(gl, vs_src, fs_src).context("compile points program failed")?;

        let (vao, vbo, uniforms) = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?;
            let vbo = gl
                .create_buffer()
                .map_err(|e| anyhow::anyhow!("create_buffer failed: {e}"))?;

            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            // Interleaved: vec2 world + float value
            let stride = (3 * std::mem::size_of::<f32>()) as i32;
            let Some(loc_world) = gl.get_attrib_location(program, "a_world") else {
                return Err(anyhow::anyhow!("missing attribute location a_world"));
            };
            let Some(loc_value) = gl.get_attrib_location(program, "a_value") else {
                return Err(anyhow::anyhow!("missing attribute location a_value"));
            };

            gl.enable_vertex_attrib_array(loc_world);
            gl.vertex_attrib_pointer_f32(loc_world, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(loc_value);
            gl.vertex_attrib_pointer_f32(
                loc_value,
                1,
                glow::FLOAT,
                false,
                stride,
                (2 * std::mem::size_of::<f32>()) as i32,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            let uniforms = (
                gl.get_uniform_location(program, "u_center_world"),
                gl.get_uniform_location(program, "u_zoom"),
                gl.get_uniform_location(program, "u_viewport_min"),
                gl.get_uniform_location(program, "u_viewport_size"),
                gl.get_uniform_location(program, "u_point_size_px"),
                gl.get_uniform_location(program, "u_threshold"),
                gl.get_uniform_location(program, "u_local_to_world_offset"),
                gl.get_uniform_location(program, "u_local_to_world_scale"),
                gl.get_uniform_location(program, "u_fill_pos"),
                gl.get_uniform_location(program, "u_fill_neg"),
                gl.get_uniform_location(program, "u_stroke_pos"),
                gl.get_uniform_location(program, "u_stroke_neg"),
                gl.get_uniform_location(program, "u_stroke_frac_pos"),
                gl.get_uniform_location(program, "u_stroke_frac_neg"),
            );

            Ok::<_, anyhow::Error>((vao, vbo, uniforms))?
        };

        Ok(Self {
            program,
            vao,
            vbo,
            u_center_world: uniforms.0,
            u_zoom: uniforms.1,
            u_viewport_min: uniforms.2,
            u_viewport_size: uniforms.3,
            u_point_size_px: uniforms.4,
            u_threshold: uniforms.5,
            u_local_to_world_offset: uniforms.6,
            u_local_to_world_scale: uniforms.7,
            u_fill_pos: uniforms.8,
            u_fill_neg: uniforms.9,
            u_stroke_pos: uniforms.10,
            u_stroke_neg: uniforms.11,
            u_stroke_frac_pos: uniforms.12,
            u_stroke_frac_neg: uniforms.13,
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

        // Works for both 120 and 330 (ignored if already specified).
        gl.bind_attrib_location(program, 0, "a_world");
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
layout(location = 0) in vec2 a_world;
layout(location = 1) in float a_value;

uniform vec2 u_center_world;
uniform float u_zoom;
uniform vec2 u_viewport_min;
uniform vec2 u_viewport_size;
uniform float u_point_size_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;

out float v_value;

void main() {
    vec2 viewport_center = u_viewport_min + 0.5 * u_viewport_size;
    vec2 world = a_world * u_local_to_world_scale + u_local_to_world_offset;
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

uniform float u_threshold;
uniform vec4 u_fill_pos;
uniform vec4 u_fill_neg;
uniform vec4 u_stroke_pos;
uniform vec4 u_stroke_neg;
uniform float u_stroke_frac_pos; // stroke_width_px / radius_px
uniform float u_stroke_frac_neg;

out vec4 out_color;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0; // [-1, 1]
    float dist = length(c);            // 0 center, 1 edge
    if (dist > 1.0) {
        discard;
    }

    float aa = fwidth(dist);
    float edge_alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);

    bool pos = (v_value >= u_threshold);
    vec4 fill = pos ? u_fill_pos : u_fill_neg;
    vec4 stroke = pos ? u_stroke_pos : u_stroke_neg;
    float stroke_frac = pos ? u_stroke_frac_pos : u_stroke_frac_neg;
    vec4 col = fill;
    if (stroke_frac > 0.0 && stroke.a > 0.0) {
        float inner = clamp(1.0 - stroke_frac, 0.0, 1.0);
        float stroke_mix = smoothstep(inner - aa, inner + aa, dist);
        col = mix(fill, stroke, stroke_mix);
    }
    col.a *= edge_alpha;
    out_color = col;
}
"#;

const VERT_120: &str = r#"#version 120
attribute vec2 a_world;
attribute float a_value;

uniform vec2 u_center_world;
uniform float u_zoom;
uniform vec2 u_viewport_min;
uniform vec2 u_viewport_size;
uniform float u_point_size_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;

varying float v_value;

void main() {
    vec2 viewport_center = u_viewport_min + 0.5 * u_viewport_size;
    vec2 world = a_world * u_local_to_world_scale + u_local_to_world_offset;
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

uniform float u_threshold;
uniform vec4 u_fill_pos;
uniform vec4 u_fill_neg;
uniform vec4 u_stroke_pos;
uniform vec4 u_stroke_neg;
uniform float u_stroke_frac_pos;
uniform float u_stroke_frac_neg;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float dist = length(c);
    if (dist > 1.0) {
        discard;
    }

    float aa = fwidth(dist);
    float edge_alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);

    bool pos = (v_value >= u_threshold);
    vec4 fill = pos ? u_fill_pos : u_fill_neg;
    vec4 stroke = pos ? u_stroke_pos : u_stroke_neg;
    float stroke_frac = pos ? u_stroke_frac_pos : u_stroke_frac_neg;
    vec4 col = fill;
    if (stroke_frac > 0.0 && stroke.a > 0.0) {
        float inner = clamp(1.0 - stroke_frac, 0.0, 1.0);
        float stroke_mix = smoothstep(inner - aa, inner + aa, dist);
        col = mix(fill, stroke, stroke_mix);
    }
    col.a *= edge_alpha;
    gl_FragColor = col;
}
"#;
