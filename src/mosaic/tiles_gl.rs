use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use lru::LruCache;
use parking_lot::Mutex;

use super::io::{MosaicRawTileKey, MosaicRawTileResponse, MosaicSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextureFilter {
    Linear,
    Nearest,
}

impl TextureFilter {
    fn as_gl(self) -> i32 {
        match self {
            Self::Linear => glow::LINEAR as i32,
            Self::Nearest => glow::NEAREST as i32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MosaicTileDraw {
    pub dataset_id: usize,
    pub level: usize,
    pub tile_y: u64,
    pub tile_x: u64,
    pub screen_rect: egui::Rect,
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelDraw {
    pub index: u64,
    pub color_rgb: [f32; 3],
    pub window: (f32, f32),
}

#[derive(Clone)]
pub struct MosaicTilesGl {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for MosaicTilesGl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MosaicTilesGl").finish_non_exhaustive()
    }
}

impl MosaicTilesGl {
    pub fn new(capacity_textures: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(capacity_textures))),
        }
    }

    pub fn set_smooth_pixels(&self, smooth: bool) {
        let mut inner = self.inner.lock();
        inner.desired_filter = if smooth {
            TextureFilter::Linear
        } else {
            TextureFilter::Nearest
        };
    }

    pub fn mark_in_flight(&self, key: MosaicRawTileKey) -> bool {
        self.inner.lock().mark_in_flight(key)
    }

    pub fn contains(&self, key: &MosaicRawTileKey) -> bool {
        self.inner.lock().cache.contains(key)
    }

    pub fn cancel_in_flight(&self, key: &MosaicRawTileKey) {
        self.inner.lock().in_flight.remove(key);
    }

    pub fn insert_pending(&self, resp: MosaicRawTileResponse) {
        self.inner.lock().insert_pending(resp);
    }

    pub fn prune_in_flight(&self, keep: &HashSet<MosaicRawTileKey>) {
        let mut inner = self.inner.lock();
        inner.in_flight.retain(|k| keep.contains(k));
    }

    pub fn is_busy(&self) -> bool {
        !self.inner.lock().in_flight.is_empty()
    }

    pub fn loading_tile_count_for(&self, keep: &HashSet<MosaicRawTileKey>) -> usize {
        self.inner.lock().loading_count_for(keep)
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        sources: &[MosaicSource],
        tiles: &[MosaicTileDraw],
        channels: &[ChannelDraw],
    ) {
        if tiles.is_empty() || channels.is_empty() {
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
        let w = viewport.width().max(1.0);
        let h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);

        unsafe {
            let gl = gl.as_ref();
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(bindings.program));
            gl.bind_vertex_array(Some(bindings.vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl.active_texture(glow::TEXTURE0);
            gl.uniform_1_i32(bindings.u_tex.as_ref(), 0);
        }

        // Only draw a tile at this level when all (dataset-available) visible channels have the
        // tile, so that we don't replace a coarse composite with a partial high-res composite.
        //
        // In mosaic mode, some ROIs may not contain every globally-visible channel. Those missing
        // channels should not blank the ROI, so we skip them per-dataset.
        let mut complete_tiles: Vec<(MosaicTileDraw, Vec<(ChannelDraw, glow::Texture)>)> =
            Vec::new();
        complete_tiles.reserve(tiles.len().min(1024));
        for td in tiles {
            if !td.screen_rect.intersects(viewport) {
                continue;
            }
            let Some(src) = sources.get(td.dataset_id) else {
                continue;
            };
            let mut texs: Vec<(ChannelDraw, glow::Texture)> = Vec::with_capacity(channels.len());
            let mut all_present = true;
            for ch in channels.iter() {
                let gid = ch.index as usize;
                if src.channel_map.get(gid).copied().flatten().is_none() {
                    continue;
                }
                let key = MosaicRawTileKey {
                    dataset_id: td.dataset_id,
                    level: td.level,
                    tile_y: td.tile_y,
                    tile_x: td.tile_x,
                    channel: ch.index,
                };
                if let Some(tex) = inner.ensure_uploaded(gl, &key) {
                    texs.push((*ch, tex));
                } else {
                    all_present = false;
                    break;
                }
            }
            if all_present && !texs.is_empty() {
                complete_tiles.push((*td, texs));
            }
        }

        for (td, texs) in complete_tiles {
            let verts = tile_vertices_ndc(td.screen_rect, viewport, w, h, ppp);
            unsafe {
                let gl = gl.as_ref();
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck::cast_slice(&verts),
                    glow::STREAM_DRAW,
                );
            }

            let (base, base_tex) = texs[0];
            unsafe {
                let gl = gl.as_ref();
                gl.disable(glow::BLEND);
                set_channel_uniforms(gl, &bindings, base.window, base.color_rgb);
                gl.bind_texture(glow::TEXTURE_2D, Some(base_tex));
                gl.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            if texs.len() > 1 {
                unsafe {
                    let gl = gl.as_ref();
                    gl.enable(glow::BLEND);
                    gl.blend_func(glow::ONE, glow::ONE);
                }
                for (ch, tex) in texs.into_iter().skip(1) {
                    unsafe {
                        let gl = gl.as_ref();
                        set_channel_uniforms(gl, &bindings, ch.window, ch.color_rgb);
                        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                        gl.draw_arrays(glow::TRIANGLES, 0, 6);
                    }
                }
            }
        }

        unsafe {
            let gl = gl.as_ref();
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.use_program(None);
        }
    }
}

enum TileState {
    Pending {
        width: usize,
        height: usize,
        data: Vec<u16>,
    },
    Uploaded {
        tex: glow::Texture,
        filter: TextureFilter,
    },
}

#[derive(Clone)]
struct GlBindings {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_window: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
}

struct Inner {
    cache: LruCache<MosaicRawTileKey, TileState>,
    in_flight: HashSet<MosaicRawTileKey>,
    pending_count: usize,
    textures_to_delete: Vec<glow::Texture>,
    globj: Option<GlObjects>,
    desired_filter: TextureFilter,
}

impl Inner {
    fn new(capacity_textures: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity_textures.max(1)).unwrap()),
            in_flight: HashSet::new(),
            pending_count: 0,
            textures_to_delete: Vec::new(),
            globj: None,
            desired_filter: TextureFilter::Linear,
        }
    }

    fn mark_in_flight(&mut self, key: MosaicRawTileKey) -> bool {
        if self.cache.contains(&key) || self.in_flight.contains(&key) {
            return false;
        }
        self.in_flight.insert(key);
        true
    }

    fn insert_pending(&mut self, resp: MosaicRawTileResponse) {
        let evicted = self.cache.push(
            resp.key,
            TileState::Pending {
                width: resp.width,
                height: resp.height,
                data: resp.data_u16,
            },
        );
        self.in_flight.remove(&resp.key);
        self.pending_count = self.pending_count.saturating_add(1);
        if let Some((_k, TileState::Uploaded { tex, .. })) = evicted {
            self.textures_to_delete.push(tex);
        }
        if let Some((_k, TileState::Pending { .. })) = evicted {
            self.pending_count = self.pending_count.saturating_sub(1);
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
            program: g.program,
            vao: g.vao,
            vbo: g.vbo,
            u_tex: g.u_tex.clone(),
            u_window: g.u_window.clone(),
            u_color: g.u_color.clone(),
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
        gl: &Arc<glow::Context>,
        key: &MosaicRawTileKey,
    ) -> Option<glow::Texture> {
        let state = self.cache.get_mut(key)?;
        match state {
            TileState::Uploaded { tex, filter, .. } => {
                if *filter != self.desired_filter {
                    set_texture_filter(gl, *tex, self.desired_filter);
                    *filter = self.desired_filter;
                }
                Some(*tex)
            }
            TileState::Pending {
                width,
                height,
                data,
            } => {
                let tex = upload_r16_texture(gl, *width, *height, data, self.desired_filter)?;
                *state = TileState::Uploaded {
                    tex,
                    filter: self.desired_filter,
                };
                self.pending_count = self.pending_count.saturating_sub(1);
                Some(tex)
            }
        }
    }

    fn loading_count_for(&self, keep: &HashSet<MosaicRawTileKey>) -> usize {
        keep.iter()
            .filter(|key| {
                self.in_flight.contains(*key)
                    || matches!(self.cache.peek(*key), Some(TileState::Pending { .. }))
            })
            .count()
    }
}

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_window: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let (vs, fs) = shader_sources(gl.version().major);
        let program = compile_program(gl, vs, fs)?;

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
            let Some(loc_pos) = gl.get_attrib_location(program, "a_pos_ndc") else {
                return Err(anyhow::anyhow!("missing attribute a_pos_ndc"));
            };
            let Some(loc_uv) = gl.get_attrib_location(program, "a_uv") else {
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

            let u_tex = gl.get_uniform_location(program, "u_tex");
            let u_window = gl.get_uniform_location(program, "u_window");
            let u_color = gl.get_uniform_location(program, "u_color");
            Ok::<_, anyhow::Error>((vao, vbo, (u_tex, u_window, u_color)))?
        };

        Ok(Self {
            program,
            vao,
            vbo,
            u_tex: uniforms.0,
            u_window: uniforms.1,
            u_color: uniforms.2,
        })
    }
}

fn set_channel_uniforms(
    gl: &glow::Context,
    bindings: &GlBindings,
    window: (f32, f32),
    color: [f32; 3],
) {
    let (w0, w1) = window;
    unsafe {
        gl.uniform_2_f32(bindings.u_window.as_ref(), w0, w1);
        gl.uniform_3_f32(bindings.u_color.as_ref(), color[0], color[1], color[2]);
    }
}

fn tile_vertices_ndc(
    screen_rect: egui::Rect,
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
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

    let u0 = 0.0f32;
    let u1 = 1.0f32;
    let v0 = 0.0f32;
    let v1 = 1.0f32;

    [
        x0, y0, u0, v0, //
        x1, y0, u1, v0, //
        x1, y1, u1, v1, //
        x0, y0, u0, v0, //
        x1, y1, u1, v1, //
        x0, y1, u0, v1, //
    ]
}

fn upload_r16_texture(
    gl: &Arc<glow::Context>,
    width: usize,
    height: usize,
    data: &[u16],
    filter: TextureFilter,
) -> Option<glow::Texture> {
    if width == 0 || height == 0 || data.len() != width * height {
        return None;
    }
    let gl = gl.as_ref();
    unsafe {
        let tex = gl.create_texture().ok()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter.as_gl());
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter.as_gl());
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
            width as i32,
            height as i32,
            0,
            glow::RED,
            glow::UNSIGNED_SHORT,
            glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(data))),
        );
        gl.bind_texture(glow::TEXTURE_2D, None);
        Some(tex)
    }
}

fn set_texture_filter(gl: &Arc<glow::Context>, tex: glow::Texture, filter: TextureFilter) {
    let gl = gl.as_ref();
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter.as_gl());
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter.as_gl());
        gl.bind_texture(glow::TEXTURE_2D, None);
    }
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

const FRAG_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_window;
uniform vec3 u_color;

out vec4 out_color;

void main() {
    float raw = texture(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    out_color = vec4(rgb, 1.0);
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

const FRAG_120: &str = r#"#version 120
varying vec2 v_uv;

uniform sampler2D u_tex;
uniform vec2 u_window;
uniform vec3 u_color;

void main() {
    float raw = texture2D(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    gl_FragColor = vec4(rgb, 1.0);
}
"#;
