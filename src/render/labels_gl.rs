use std::collections::HashSet;
use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use crate::render::labels_raw::{LabelTileCache, LabelTileKey, LabelTileResponse};

#[derive(Debug, Clone, Copy)]
pub struct LabelDraw {
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
    pub screen_rect: egui::Rect,
}

#[derive(Debug, Clone, Copy)]
pub struct OutlinesParams {
    pub visible: bool,
    pub color_rgb: [f32; 3],
    pub opacity: f32,
    pub width_screen_px: f32,
}

#[derive(Clone)]
pub struct LabelsGl {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for LabelsGl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabelsGl").finish_non_exhaustive()
    }
}

impl LabelsGl {
    pub fn new(capacity_tiles: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(capacity_tiles))),
        }
    }

    pub fn mark_in_flight(&self, key: LabelTileKey) -> bool {
        self.inner.lock().cache.mark_in_flight(key)
    }

    pub fn cancel_in_flight(&self, key: &LabelTileKey) {
        self.inner.lock().cache.cancel_in_flight(key)
    }

    pub fn insert_pending(&self, resp: LabelTileResponse) {
        self.inner.lock().insert_pending(resp);
    }

    pub fn reset(&self) {
        let mut inner = self.inner.lock();
        for (_k, state) in inner.cache.drain() {
            if let TileState::Uploaded { tex, .. } = state {
                inner.textures_to_delete.push(tex);
            }
        }
    }

    pub fn prune_in_flight(&self, keep: &HashSet<LabelTileKey>) {
        self.inner.lock().cache.prune_in_flight(keep);
    }

    pub fn is_busy(&self) -> bool {
        self.inner.lock().cache.is_busy()
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        tiles: &[LabelDraw],
        params: OutlinesParams,
    ) {
        if !params.visible || tiles.is_empty() {
            return;
        }

        let gl = painter.gl();
        let mut inner = self.inner.lock();
        inner.ensure_gl(gl);
        inner.delete_queued_textures(gl);

        let Some(bindings) = inner.bindings() else {
            return;
        };

        let viewport = info.viewport;
        let ppp = info.pixels_per_point.max(1e-6);

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(bindings.program_outline));
            gl.bind_vertex_array(Some(bindings.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.uniform_1_i32(bindings.u_label.as_ref(), 0);
            gl.uniform_3_f32(
                bindings.u_color.as_ref(),
                params.color_rgb[0],
                params.color_rgb[1],
                params.color_rgb[2],
            );
            gl.uniform_1_f32(bindings.u_opacity.as_ref(), params.opacity.clamp(0.0, 1.0));
            gl.uniform_1_f32(
                bindings.u_outline_px.as_ref(),
                params.width_screen_px.clamp(0.0, 4.0),
            );

            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            for td in tiles {
                if !td.screen_rect.intersects(viewport) {
                    continue;
                }

                let key = LabelTileKey {
                    level: td.level,
                    tile_y: td.tile_y,
                    tile_x: td.tile_x,
                };
                let Some((tex, tex_w, tex_h)) = inner.ensure_uploaded(gl, &key) else {
                    continue;
                };

                let inset_u = 1.0 / (tex_w.max(1) as f32);
                let inset_v = 1.0 / (tex_h.max(1) as f32);
                gl.uniform_2_f32(bindings.u_texel.as_ref(), inset_u, inset_v);

                let verts = tile_vertices_ndc_uvrange(
                    td.screen_rect,
                    viewport,
                    viewport.width().max(1.0),
                    viewport.height().max(1.0),
                    ppp,
                    inset_u,
                    1.0 - inset_u,
                    inset_v,
                    1.0 - inset_v,
                );
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&verts),
                    glow::STREAM_DRAW,
                );

                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
        }
    }
}

enum TileState {
    Pending {
        tex_width: usize,
        tex_height: usize,
        data_u32: Vec<u32>,
    },
    Uploaded {
        tex_width: usize,
        tex_height: usize,
        tex: glow::Texture,
    },
}

#[derive(Clone)]
struct GlBindings {
    program_outline: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_label: Option<glow::UniformLocation>,
    u_texel: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_opacity: Option<glow::UniformLocation>,
    u_outline_px: Option<glow::UniformLocation>,
}

struct Inner {
    cache: LabelTileCache<TileState>,
    textures_to_delete: Vec<glow::Texture>,
    globj: Option<GlObjects>,
}

impl Inner {
    fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: LabelTileCache::new(capacity_tiles),
            textures_to_delete: Vec::new(),
            globj: None,
        }
    }

    fn insert_pending(&mut self, resp: LabelTileResponse) {
        let evicted = self.cache.push(
            resp.key,
            TileState::Pending {
                tex_width: resp.tex_width,
                tex_height: resp.tex_height,
                data_u32: resp.data_u32,
            },
        );
        if let Some((_k, TileState::Uploaded { tex, .. })) = evicted {
            self.textures_to_delete.push(tex);
        }
    }

    fn ensure_gl(&mut self, gl: &Arc<glow::Context>) {
        if self.globj.is_some() {
            return;
        }
        self.globj = GlObjects::new(gl).ok();
    }

    fn bindings(&self) -> Option<GlBindings> {
        let g = self.globj.as_ref()?;
        Some(GlBindings {
            program_outline: g.program_outline,
            vao: g.vao,
            vbo: g.vbo,
            u_label: g.u_label.clone(),
            u_texel: g.u_texel.clone(),
            u_color: g.u_color.clone(),
            u_opacity: g.u_opacity.clone(),
            u_outline_px: g.u_outline_px.clone(),
        })
    }

    fn delete_queued_textures(&mut self, gl: &Arc<glow::Context>) {
        if self.textures_to_delete.is_empty() {
            return;
        }
        let gl = gl.as_ref();
        unsafe {
            for tex in self.textures_to_delete.drain(..) {
                gl.delete_texture(tex);
            }
        }
    }

    fn ensure_uploaded(
        &mut self,
        gl: &glow::Context,
        key: &LabelTileKey,
    ) -> Option<(glow::Texture, usize, usize)> {
        let state = self.cache.get_mut(key)?;
        match state {
            TileState::Uploaded {
                tex,
                tex_width,
                tex_height,
            } => Some((*tex, *tex_width, *tex_height)),
            TileState::Pending {
                tex_width,
                tex_height,
                data_u32,
            } => {
                let use_int = self
                    .globj
                    .as_ref()
                    .map(|g| g.use_integer_texture)
                    .unwrap_or(false);
                let tex = if use_int {
                    upload_r32ui_texture(gl, *tex_width, *tex_height, data_u32)?
                } else {
                    let mut rgba8 = Vec::with_capacity(data_u32.len() * 4);
                    for v in data_u32.iter().copied() {
                        rgba8.extend_from_slice(&v.to_le_bytes());
                    }
                    upload_rgba8_texture(gl, *tex_width, *tex_height, &rgba8)?
                };
                let out = (tex, *tex_width, *tex_height);
                *state = TileState::Uploaded {
                    tex_width: *tex_width,
                    tex_height: *tex_height,
                    tex,
                };
                Some(out)
            }
        }
    }
}

struct GlObjects {
    program_outline: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_label: Option<glow::UniformLocation>,
    u_texel: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_opacity: Option<glow::UniformLocation>,
    u_outline_px: Option<glow::UniformLocation>,
    use_integer_texture: bool,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let gl_major = gl.version().major;
        let (vs, fs_outline, use_integer_texture) = shader_sources(gl_major);
        let program_outline = compile_program(gl, vs, fs_outline)?;

        let (vao, vbo, uniforms) = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?;
            let vbo = gl
                .create_buffer()
                .map_err(|e| anyhow::anyhow!("create_buffer failed: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            let stride = (4 * std::mem::size_of::<f32>()) as i32;
            let Some(loc_pos) = gl.get_attrib_location(program_outline, "a_pos_ndc") else {
                return Err(anyhow::anyhow!("missing attribute a_pos_ndc"));
            };
            let Some(loc_uv) = gl.get_attrib_location(program_outline, "a_uv") else {
                return Err(anyhow::anyhow!("missing attribute a_uv"));
            };
            gl.enable_vertex_attrib_array(loc_pos);
            gl.vertex_attrib_pointer_f32(loc_pos, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(loc_uv);
            gl.vertex_attrib_pointer_f32(
                loc_uv,
                2,
                glow::FLOAT,
                false,
                stride,
                (2 * std::mem::size_of::<f32>()) as i32,
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            let u_label = gl.get_uniform_location(program_outline, "u_label");
            let u_texel = gl.get_uniform_location(program_outline, "u_texel");
            let u_color = gl.get_uniform_location(program_outline, "u_color");
            let u_opacity = gl.get_uniform_location(program_outline, "u_opacity");
            let u_outline_px = gl.get_uniform_location(program_outline, "u_outline_px");

            Ok::<_, anyhow::Error>((
                vao,
                vbo,
                (u_label, u_texel, u_color, u_opacity, u_outline_px),
            ))?
        };

        Ok(Self {
            program_outline,
            vao,
            vbo,
            u_label: uniforms.0,
            u_texel: uniforms.1,
            u_color: uniforms.2,
            u_opacity: uniforms.3,
            u_outline_px: uniforms.4,
            use_integer_texture,
        })
    }
}

fn upload_rgba8_texture(
    gl: &glow::Context,
    width: usize,
    height: usize,
    data: &[u8],
) -> Option<glow::Texture> {
    if width == 0 || height == 0 || data.len() != width * height * 4 {
        return None;
    }
    unsafe {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
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
            glow::RGBA as i32,
            width as i32,
            height as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            glow::PixelUnpackData::Slice(Some(data)),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        Some(tex)
    }
}

fn upload_r32ui_texture(
    gl: &glow::Context,
    width: usize,
    height: usize,
    data: &[u32],
) -> Option<glow::Texture> {
    if width == 0 || height == 0 || data.len() != width * height {
        return None;
    }
    unsafe {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
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
            glow::R32UI as i32,
            width as i32,
            height as i32,
            0,
            glow::RED_INTEGER,
            glow::UNSIGNED_INT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(data))),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        Some(tex)
    }
}

fn tile_vertices_ndc_uvrange(
    screen_rect: egui::Rect,
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
    u0: f32,
    u1: f32,
    v0: f32,
    v1: f32,
) -> [f32; 6 * 4] {
    let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
    let min_x = snap(screen_rect.min.x);
    let max_x = snap(screen_rect.max.x);
    let min_y = snap(screen_rect.min.y);
    let max_y = snap(screen_rect.max.y);

    let x0 = ((min_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let x1 = ((max_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let y0 = 1.0 - ((min_y - viewport.min.y) / viewport_h) * 2.0;
    let y1 = 1.0 - ((max_y - viewport.min.y) / viewport_h) * 2.0;

    [
        x0, y0, u0, v0, //
        x1, y0, u1, v0, //
        x1, y1, u1, v1, //
        x0, y0, u0, v0, //
        x1, y1, u1, v1, //
        x0, y1, u0, v1, //
    ]
}

fn shader_sources(gl_major: u32) -> (&'static str, &'static str, bool) {
    if gl_major >= 3 {
        (VERT_330, FRAG_OUTLINE_330, true)
    } else {
        (VERT_120, FRAG_OUTLINE_120, false)
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

        gl.bind_attrib_location(program, 0, "a_pos_ndc");
        gl.bind_attrib_location(program, 1, "a_uv");

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
layout(location = 0) in vec2 a_pos_ndc;
layout(location = 1) in vec2 a_uv;

out vec2 v_uv;

void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

const FRAG_OUTLINE_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform usampler2D u_label;
uniform vec2 u_texel;
uniform vec3 u_color;
uniform float u_opacity;
uniform float u_outline_px;

out vec4 out_color;

uint id_at(ivec2 p) {
    return texelFetch(u_label, p, 0).r;
}

float edge_at_p(ivec2 p) {
    ivec2 sz = textureSize(u_label, 0);
    ivec2 pmin = ivec2(0, 0);
    ivec2 pmax = max(sz - ivec2(1, 1), ivec2(0, 0));

    uint c = id_at(clamp(p, pmin, pmax));
    uint up = id_at(clamp(p + ivec2(0, -1), pmin, pmax));
    uint dn = id_at(clamp(p + ivec2(0,  1), pmin, pmax));
    uint lf = id_at(clamp(p + ivec2(-1, 0), pmin, pmax));
    uint rt = id_at(clamp(p + ivec2( 1, 0), pmin, pmax));

    float any_label = ((c | up | dn | lf | rt) != 0u) ? 1.0 : 0.0;
    float diff = 0.0;
    diff = max(diff, (c != up) ? 1.0 : 0.0);
    diff = max(diff, (c != dn) ? 1.0 : 0.0);
    diff = max(diff, (c != lf) ? 1.0 : 0.0);
    diff = max(diff, (c != rt) ? 1.0 : 0.0);
    return any_label * diff;
}

void main() {
    ivec2 sz = textureSize(u_label, 0);
    ivec2 p = ivec2(v_uv * vec2(sz));
    // Keep p in the inner region so neighbors can safely read the halo border.
    p = clamp(p, ivec2(1, 1), max(sz - ivec2(2, 2), ivec2(1, 1)));

    int n = int(clamp(floor(u_outline_px + 0.5), 0.0, 4.0));
    float e = edge_at_p(p);
    for (int i = 1; i <= 4; ++i) {
        if (i > n) { break; }
        e = max(e, edge_at_p(p + ivec2( i, 0)));
        e = max(e, edge_at_p(p + ivec2(-i, 0)));
        e = max(e, edge_at_p(p + ivec2(0,  i)));
        e = max(e, edge_at_p(p + ivec2(0, -i)));
    }

    float a = u_opacity * clamp(e, 0.0, 1.0);
    out_color = vec4(u_color, a);
}
"#;

const VERT_120: &str = r#"#version 120
attribute vec2 a_pos_ndc;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}
"#;

const FRAG_OUTLINE_120: &str = r#"#version 120
varying vec2 v_uv;

uniform sampler2D u_label;
uniform vec2 u_texel;
uniform vec3 u_color;
uniform float u_opacity;
uniform float u_outline_px;

float decode_id(vec2 uv) {
    vec4 c = texture2D(u_label, uv);
    vec4 b = floor(c * 255.0 + 0.5);
    return dot(b, vec4(1.0, 256.0, 65536.0, 16777216.0));
}

float edge_at(vec2 uv) {
    float c = decode_id(uv);
    float up = decode_id(uv + vec2(0.0, -u_texel.y));
    float dn = decode_id(uv + vec2(0.0,  u_texel.y));
    float lf = decode_id(uv + vec2(-u_texel.x, 0.0));
    float rt = decode_id(uv + vec2( u_texel.x, 0.0));

    float any_label = step(0.5, max(c, max(max(up, dn), max(lf, rt))));
    float diff = 0.0;
    diff = max(diff, step(0.5, abs(c - up)));
    diff = max(diff, step(0.5, abs(c - dn)));
    diff = max(diff, step(0.5, abs(c - lf)));
    diff = max(diff, step(0.5, abs(c - rt)));
    return any_label * diff;
}

void main() {
    float e = edge_at(v_uv);
    float w1 = step(0.5, u_outline_px);
    float w2 = step(1.5, u_outline_px);
    float w3 = step(2.5, u_outline_px);
    float w4 = step(3.5, u_outline_px);

    // Thicken in label-pixel space (not screen space) to avoid filling the view when zoomed out.
    e = max(e, w1 * edge_at(v_uv + vec2( u_texel.x * 1.0, 0.0)));
    e = max(e, w1 * edge_at(v_uv + vec2(-u_texel.x * 1.0, 0.0)));
    e = max(e, w1 * edge_at(v_uv + vec2(0.0,  u_texel.y * 1.0)));
    e = max(e, w1 * edge_at(v_uv + vec2(0.0, -u_texel.y * 1.0)));

    e = max(e, w2 * edge_at(v_uv + vec2( u_texel.x * 2.0, 0.0)));
    e = max(e, w2 * edge_at(v_uv + vec2(-u_texel.x * 2.0, 0.0)));
    e = max(e, w2 * edge_at(v_uv + vec2(0.0,  u_texel.y * 2.0)));
    e = max(e, w2 * edge_at(v_uv + vec2(0.0, -u_texel.y * 2.0)));

    e = max(e, w3 * edge_at(v_uv + vec2( u_texel.x * 3.0, 0.0)));
    e = max(e, w3 * edge_at(v_uv + vec2(-u_texel.x * 3.0, 0.0)));
    e = max(e, w3 * edge_at(v_uv + vec2(0.0,  u_texel.y * 3.0)));
    e = max(e, w3 * edge_at(v_uv + vec2(0.0, -u_texel.y * 3.0)));

    e = max(e, w4 * edge_at(v_uv + vec2( u_texel.x * 4.0, 0.0)));
    e = max(e, w4 * edge_at(v_uv + vec2(-u_texel.x * 4.0, 0.0)));
    e = max(e, w4 * edge_at(v_uv + vec2(0.0,  u_texel.y * 4.0)));
    e = max(e, w4 * edge_at(v_uv + vec2(0.0, -u_texel.y * 4.0)));

    float a = u_opacity * clamp(e, 0.0, 1.0);
    gl_FragColor = vec4(u_color, a);
}
"#;
