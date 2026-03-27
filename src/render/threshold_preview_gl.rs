use std::sync::Arc;

use anyhow::Context;
use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

#[derive(Debug, Clone)]
pub struct ThresholdPreviewGlDrawData {
    pub generation: u64,
    pub width: usize,
    pub height: usize,
    pub values: Arc<Vec<u16>>,
}

#[derive(Debug, Clone)]
pub struct ThresholdPreviewGlDrawParams {
    pub visible: bool,
    pub quad_screen: [egui::Pos2; 4],
    pub threshold_u16: u16,
    pub tint: egui::Color32,
}

#[derive(Clone)]
pub struct ThresholdPreviewGlRenderer {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for ThresholdPreviewGlRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThresholdPreviewGlRenderer")
            .finish_non_exhaustive()
    }
}

impl Default for ThresholdPreviewGlRenderer {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }
}

impl ThresholdPreviewGlRenderer {
    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        data: &ThresholdPreviewGlDrawData,
        params: &ThresholdPreviewGlDrawParams,
    ) {
        if !params.visible || data.width == 0 || data.height == 0 || data.values.is_empty() {
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
        if inner.uploaded_generation != data.generation
            || inner.uploaded_size != (data.width, data.height)
        {
            if inner.texture.is_none() {
                inner.texture = unsafe { gl.as_ref().create_texture().ok() };
            }
            let Some(texture) = inner.texture else {
                return;
            };
            unsafe {
                let gl = gl.as_ref();
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
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
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::R16 as i32,
                    data.width as i32,
                    data.height as i32,
                    0,
                    glow::RED,
                    glow::UNSIGNED_SHORT,
                    glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(
                        data.values.as_slice(),
                    ))),
                );
                gl.bind_texture(glow::TEXTURE_2D, None);
            }
            inner.uploaded_generation = data.generation;
            inner.uploaded_size = (data.width, data.height);
        }

        let Some(objects) = inner.gl_objects.as_ref() else {
            return;
        };
        let Some(texture) = inner.texture else {
            return;
        };

        let viewport = info.viewport;
        let viewport_w = viewport.width().max(1.0);
        let viewport_h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);
        let verts = quad_vertices_ndc(params.quad_screen, viewport, viewport_w, viewport_h, ppp);
        let threshold_norm = params.threshold_u16 as f32 / u16::MAX as f32;
        let tint = color_to_vec4(params.tint);

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(objects.program));
            gl.bind_vertex_array(Some(objects.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(objects.vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&verts),
                glow::STREAM_DRAW,
            );
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, 16, 8);

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.uniform_1_i32(objects.u_tex.as_ref(), 0);
            gl.uniform_1_f32(objects.u_threshold_norm.as_ref(), threshold_norm);
            gl.uniform_4_f32_slice(objects.u_tint.as_ref(), &tint);

            gl.draw_arrays(glow::TRIANGLES, 0, 6);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

#[derive(Default)]
struct Inner {
    gl_objects: Option<GlObjects>,
    texture: Option<glow::Texture>,
    uploaded_generation: u64,
    uploaded_size: (usize, usize),
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_threshold_norm: Option<glow::UniformLocation>,
    u_tint: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let program = compile_program(gl, VERT_330, FRAG_330)
            .context("compile threshold preview program failed")?;
        let vao = unsafe {
            gl.create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?
        };
        let vbo = unsafe {
            gl.create_buffer()
                .map_err(|e| anyhow::anyhow!("create_buffer failed: {e}"))?
        };
        let u_tex = unsafe { gl.get_uniform_location(program, "u_tex") };
        let u_threshold_norm = unsafe { gl.get_uniform_location(program, "u_threshold_norm") };
        let u_tint = unsafe { gl.get_uniform_location(program, "u_tint") };
        Ok(Self {
            program,
            vao,
            vbo,
            u_tex,
            u_threshold_norm,
            u_tint,
        })
    }
}

fn compile_program(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> anyhow::Result<glow::Program> {
    unsafe {
        let program = gl
            .create_program()
            .map_err(|e| anyhow::anyhow!("create_program failed: {e}"))?;
        let shaders = [
            (glow::VERTEX_SHADER, vert_src),
            (glow::FRAGMENT_SHADER, frag_src),
        ];
        let mut compiled = Vec::new();
        for (kind, src) in shaders {
            let shader = gl
                .create_shader(kind)
                .map_err(|e| anyhow::anyhow!("create_shader failed: {e}"))?;
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                gl.delete_program(program);
                return Err(anyhow::anyhow!("shader compile failed: {log}"));
            }
            gl.attach_shader(program, shader);
            compiled.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            for shader in compiled {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            gl.delete_program(program);
            return Err(anyhow::anyhow!("program link failed: {log}"));
        }

        for shader in compiled {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        Ok(program)
    }
}

fn quad_vertices_ndc(
    quad: [egui::Pos2; 4],
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
) -> [f32; 6 * 4] {
    let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
    let p0 = egui::pos2(snap(quad[0].x), snap(quad[0].y));
    let p1 = egui::pos2(snap(quad[1].x), snap(quad[1].y));
    let p2 = egui::pos2(snap(quad[2].x), snap(quad[2].y));
    let p3 = egui::pos2(snap(quad[3].x), snap(quad[3].y));

    let to_ndc = |p: egui::Pos2| -> (f32, f32) {
        let x = ((p.x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
        let y = 1.0 - ((p.y - viewport.min.y) / viewport_h) * 2.0;
        (x, y)
    };
    let (x0, y0) = to_ndc(p0);
    let (x1, y1) = to_ndc(p1);
    let (x2, y2) = to_ndc(p2);
    let (x3, y3) = to_ndc(p3);

    [
        x0, y0, 0.0, 0.0, x1, y1, 1.0, 0.0, x2, y2, 1.0, 1.0, x0, y0, 0.0, 0.0, x2, y2, 1.0, 1.0,
        x3, y3, 0.0, 1.0,
    ]
}

fn color_to_vec4(color: egui::Color32) -> [f32; 4] {
    [
        color.r() as f32 / 255.0,
        color.g() as f32 / 255.0,
        color.b() as f32 / 255.0,
        color.a() as f32 / 255.0,
    ]
}

const VERT_330: &str = r#"#version 330 core
layout (location = 0) in vec2 a_pos_ndc;
layout (location = 1) in vec2 a_uv;
out vec2 v_uv;

void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
}
"#;

const FRAG_330: &str = r#"#version 330 core
uniform sampler2D u_tex;
uniform float u_threshold_norm;
uniform vec4 u_tint;
in vec2 v_uv;
out vec4 out_color;

void main() {
    float value = texture(u_tex, v_uv).r;
    if (value < u_threshold_norm) {
        discard;
    }
    out_color = u_tint;
}
"#;
