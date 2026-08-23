use std::sync::Arc;

use glow::HasContext;

use super::TextureFilter;
use super::shaders::{blit_shader_sources, compile_program, shader_sources};
use super::upload::{set_texture_filter, upload_r16_texture};
use crate::render::tiles_raw::{RawTileCache, RawTileKey, RawTileResponse};

pub(super) enum TileState {
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
pub(super) struct GlBindings {
    pub(super) program: glow::Program,
    pub(super) program_blit: glow::Program,
    pub(super) vao: glow::VertexArray,
    pub(super) vbo: glow::Buffer,
    pub(super) u_tex: Option<glow::UniformLocation>,
    pub(super) u_window: Option<glow::UniformLocation>,
    pub(super) u_color: Option<glow::UniformLocation>,
    pub(super) u_alpha_scale: Option<glow::UniformLocation>,
    pub(super) u_blit_tex: Option<glow::UniformLocation>,
    pub(super) u_blit_alpha_scale: Option<glow::UniformLocation>,
}

pub(super) struct Inner {
    pub(super) cache: RawTileCache<TileState>,
    pub(super) textures_to_delete: Vec<glow::Texture>,
    globj: Option<GlObjects>,
    pub(super) desired_filter: TextureFilter,
    pub(super) offscreen: Option<OffscreenTargets>,
}

pub(super) struct OffscreenTargets {
    pub(super) w_px: i32,
    pub(super) h_px: i32,
    pub(super) fbos: Vec<glow::Framebuffer>,
    pub(super) texs: Vec<glow::Texture>,
}

impl Inner {
    pub(super) fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: RawTileCache::new(capacity_tiles),
            textures_to_delete: Vec::new(),
            globj: None,
            desired_filter: TextureFilter::Linear,
            offscreen: None,
        }
    }

    pub(super) fn insert_pending(&mut self, resp: RawTileResponse) {
        let evicted = self.cache.push(
            resp.key,
            TileState::Pending {
                width: resp.width,
                height: resp.height,
                data: resp.data_u16,
            },
        );
        if let Some((_k, TileState::Uploaded { tex, .. })) = evicted {
            self.textures_to_delete.push(tex);
        }
    }

    pub(super) fn ensure_gl(&mut self, gl: &Arc<glow::Context>) {
        if self.globj.is_some() {
            return;
        }
        self.globj = GlObjects::new(gl).ok();
    }

    pub(super) fn bindings(&self) -> Option<GlBindings> {
        let g = self.globj.as_ref()?;
        Some(GlBindings {
            program: g.program,
            program_blit: g.program_blit,
            vao: g.vao,
            vbo: g.vbo,
            u_tex: g.u_tex.clone(),
            u_window: g.u_window.clone(),
            u_color: g.u_color.clone(),
            u_alpha_scale: g.u_alpha_scale.clone(),
            u_blit_tex: g.u_blit_tex.clone(),
            u_blit_alpha_scale: g.u_blit_alpha_scale.clone(),
        })
    }

    pub(super) fn delete_queued_textures(&mut self, gl: &Arc<glow::Context>) {
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

    pub(super) fn ensure_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        key: &RawTileKey,
    ) -> Option<glow::Texture> {
        // Touch to keep in LRU.
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
                Some(tex)
            }
        }
    }

    pub(super) fn ensure_offscreen(
        &mut self,
        gl: &Arc<glow::Context>,
        targets: usize,
        w_px: i32,
        h_px: i32,
    ) -> bool {
        if targets == 0 || w_px <= 0 || h_px <= 0 {
            self.offscreen = None;
            return false;
        }

        let needs_rebuild = match self.offscreen.as_ref() {
            None => true,
            Some(o) => o.w_px != w_px || o.h_px != h_px || o.texs.len() != targets,
        };
        if !needs_rebuild {
            if let Some(o) = self.offscreen.as_ref() {
                for &tex in &o.texs {
                    set_texture_filter(gl, tex, self.desired_filter);
                }
            }
            return true;
        }

        // Drop existing targets.
        if let Some(old) = self.offscreen.take() {
            let gl = gl.as_ref();
            unsafe {
                for f in old.fbos {
                    gl.delete_framebuffer(f);
                }
                for t in old.texs {
                    gl.delete_texture(t);
                }
            }
        }

        let gl_ref = gl.as_ref();
        let mut fbos = Vec::with_capacity(targets);
        let mut texs = Vec::with_capacity(targets);

        unsafe {
            for _ in 0..targets {
                let tex = match gl_ref.create_texture() {
                    Ok(t) => t,
                    Err(_) => return false,
                };
                gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex));
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
                gl_ref.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MIN_FILTER,
                    self.desired_filter.as_gl(),
                );
                gl_ref.tex_parameter_i32(
                    glow::TEXTURE_2D,
                    glow::TEXTURE_MAG_FILTER,
                    self.desired_filter.as_gl(),
                );
                gl_ref.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA as i32,
                    w_px,
                    h_px,
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );

                let fbo = match gl_ref.create_framebuffer() {
                    Ok(f) => f,
                    Err(_) => {
                        gl_ref.delete_texture(tex);
                        return false;
                    }
                };
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                gl_ref.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(tex),
                    0,
                );

                fbos.push(fbo);
                texs.push(tex);
            }

            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
        }

        self.offscreen = Some(OffscreenTargets {
            w_px,
            h_px,
            fbos,
            texs,
        });
        true
    }
}

struct GlObjects {
    program: glow::Program,
    program_blit: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_tex: Option<glow::UniformLocation>,
    u_window: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_alpha_scale: Option<glow::UniformLocation>,
    u_blit_tex: Option<glow::UniformLocation>,
    u_blit_alpha_scale: Option<glow::UniformLocation>,
}

impl GlObjects {
    fn new(gl: &Arc<glow::Context>) -> anyhow::Result<Self> {
        let gl = gl.as_ref();
        let (vs, fs) = shader_sources(gl.version().major);
        let program = compile_program(gl, vs, fs)?;
        let (vs_blit, fs_blit) = blit_shader_sources(gl.version().major);
        let program_blit = compile_program(gl, vs_blit, fs_blit)?;

        let (vao, vbo, uniforms) = unsafe {
            let vao = gl
                .create_vertex_array()
                .map_err(|e| anyhow::anyhow!("create_vertex_array failed: {e}"))?;
            let vbo = gl
                .create_buffer()
                .map_err(|e| anyhow::anyhow!("create_buffer failed: {e}"))?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            // a_pos_ndc (vec2), a_uv (vec2)
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
            let u_alpha_scale = gl.get_uniform_location(program, "u_alpha_scale");
            let u_blit_tex = gl.get_uniform_location(program_blit, "u_tex");
            let u_blit_alpha_scale = gl.get_uniform_location(program_blit, "u_alpha_scale");
            Ok::<_, anyhow::Error>((
                vao,
                vbo,
                (
                    u_tex,
                    u_window,
                    u_color,
                    u_alpha_scale,
                    u_blit_tex,
                    u_blit_alpha_scale,
                ),
            ))?
        };

        Ok(Self {
            program,
            program_blit,
            vao,
            vbo,
            u_tex: uniforms.0,
            u_window: uniforms.1,
            u_color: uniforms.2,
            u_alpha_scale: uniforms.3,
            u_blit_tex: uniforms.4,
            u_blit_alpha_scale: uniforms.5,
        })
    }
}
