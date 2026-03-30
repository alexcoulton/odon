use std::collections::HashSet;
use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use crate::render::tiles::RenderChannel;
use crate::render::tiles_raw::{RawTileCache, RawTileKey, RawTileResponse};

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
pub struct TileDraw {
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
    pub alpha_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelScreenTransform {
    pub pivot_screen: egui::Pos2,
    pub translation_screen: egui::Vec2,
    pub scale: egui::Vec2,
    pub rotation_rad: f32,
}

#[derive(Clone)]
pub struct TilesGl {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for TilesGl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TilesGl").finish_non_exhaustive()
    }
}

impl TilesGl {
    pub fn new(capacity_tiles: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(capacity_tiles))),
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

    pub fn mark_in_flight(&self, key: RawTileKey) -> bool {
        self.inner.lock().cache.mark_in_flight(key)
    }

    pub fn contains(&self, key: &RawTileKey) -> bool {
        self.inner.lock().cache.contains(key)
    }

    pub fn cancel_in_flight(&self, key: &RawTileKey) {
        self.inner.lock().cache.cancel_in_flight(key)
    }

    pub fn insert_pending(&self, resp: RawTileResponse) {
        self.inner.lock().insert_pending(resp);
    }

    pub fn reset(&self) {
        let mut inner = self.inner.lock();
        for (_k, state) in inner.cache.drain() {
            if let TileState::Uploaded { tex, .. } = state {
                inner.textures_to_delete.push(tex);
            }
        }
        inner.offscreen = None;
    }

    pub fn prune_in_flight(&self, keep: &HashSet<RawTileKey>) {
        self.inner.lock().cache.prune_in_flight(keep);
    }

    pub fn is_busy(&self) -> bool {
        self.inner.lock().cache.is_busy()
    }

    pub fn in_flight_len(&self) -> usize {
        self.inner.lock().cache.in_flight_len()
    }

    pub fn capacity(&self) -> usize {
        self.inner.lock().cache.capacity()
    }

    pub fn grow_capacity(&self, capacity_tiles: usize) {
        self.inner.lock().cache.grow_capacity(capacity_tiles);
    }

    pub fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        tiles: &[TileDraw],
        channels: &[ChannelDraw],
    ) {
        if tiles.is_empty() || channels.is_empty() {
            return;
        }
        // Use the same per-channel offscreen composition path as the offset/affine renderers.
        // This keeps cross-level fallback channel-specific: if channel 5 only has a coarse tile
        // while channel 0 already has a finer tile, we preserve the coarse channel 5 contribution
        // until its finer tile arrives.
        let zero_offsets = vec![egui::Vec2::ZERO; channels.len()];
        self.paint_with_channel_offsets(info, painter, tiles, channels, &zero_offsets, 1.0);
    }

    pub fn paint_with_channel_offsets(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        tiles: &[TileDraw],
        channels: &[ChannelDraw],
        channel_offsets_world: &[egui::Vec2],
        zoom_screen_per_world: f32,
    ) {
        if tiles.is_empty() || channels.is_empty() {
            return;
        }
        if channels.len() != channel_offsets_world.len() {
            // Defensive: call site should keep these in sync.
            self.paint(info, painter, tiles, channels);
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
        let viewport_w = viewport.width().max(1.0);
        let viewport_h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);

        let w_px = (viewport_w * ppp).round().max(1.0) as i32;
        let h_px = (viewport_h * ppp).round().max(1.0) as i32;
        if !inner.ensure_offscreen(gl, channels.len(), w_px, h_px) {
            return;
        }
        let (off_w_px, off_h_px, off_fbos, off_texs) = match inner.offscreen.as_ref() {
            None => return,
            Some(o) => (o.w_px, o.h_px, o.fbos.clone(), o.texs.clone()),
        };

        // Save viewport/scissor so we don't break other egui painting.
        let mut prev_viewport = [0i32; 4];
        let mut prev_scissor = [0i32; 4];
        let prev_scissor_enabled;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
        }

        unsafe {
            let gl_ref = gl.as_ref();

            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);

            // Render each channel into its own offscreen texture.
            gl_ref.use_program(Some(bindings.program));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_tex.as_ref(), 0);

            // Disable scissor while drawing to FBOs.
            gl_ref.disable(glow::SCISSOR_TEST);

            let mut available_tiles: Vec<(TileDraw, Vec<(usize, glow::Texture)>)> = Vec::new();
            available_tiles.reserve(tiles.len().min(512));
            for td in tiles {
                let mut texs: Vec<(usize, glow::Texture)> = Vec::with_capacity(channels.len());
                for (ci, ch) in channels.iter().enumerate() {
                    let key = RawTileKey {
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key) {
                        texs.push((ci, tex));
                    }
                }
                if !texs.is_empty() {
                    available_tiles.push((*td, texs));
                }
            }

            for (ci, ch) in channels.iter().enumerate() {
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(off_fbos[ci]));
                gl_ref.viewport(0, 0, off_w_px, off_h_px);
                gl_ref.disable(glow::BLEND);
                gl_ref.clear_color(0.0, 0.0, 0.0, 0.0);
                gl_ref.clear(glow::COLOR_BUFFER_BIT);

                set_channel_uniforms(gl_ref, &bindings, ch.window, ch.color_rgb, ch.alpha_scale);

                let off_screen = channel_offsets_world[ci] * zoom_screen_per_world;
                for (td, texs) in &available_tiles {
                    // Fast reject when shifted off-screen.
                    let screen_rect = td.screen_rect.translate(off_screen);
                    if !screen_rect.intersects(viewport) {
                        continue;
                    }

                    let Some((_, tex)) = texs.iter().find(|(tile_ci, _)| *tile_ci == ci) else {
                        continue;
                    };

                    let verts =
                        tile_vertices_ndc(screen_rect, viewport, viewport_w, viewport_h, ppp);
                    gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
                    gl_ref.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&verts),
                        glow::STREAM_DRAW,
                    );
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(*tex));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
            }

            // Restore drawing to the main framebuffer.
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl_ref.viewport(
                prev_viewport[0],
                prev_viewport[1],
                prev_viewport[2],
                prev_viewport[3],
            );
            if prev_scissor_enabled {
                gl_ref.enable(glow::SCISSOR_TEST);
            } else {
                gl_ref.disable(glow::SCISSOR_TEST);
            }
            gl_ref.scissor(
                prev_scissor[0],
                prev_scissor[1],
                prev_scissor[2],
                prev_scissor[3],
            );

            // Composite channel textures back to the main framebuffer.
            gl_ref.use_program(Some(bindings.program_blit));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_blit_tex.as_ref(), 0);
            gl_ref.uniform_1_f32(bindings.u_blit_alpha_scale.as_ref(), 1.0);

            let full = tile_vertices_ndc(viewport, viewport, viewport_w, viewport_h, ppp);
            gl_ref.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&full),
                glow::STREAM_DRAW,
            );

            // First channel overwrites (establishes base + alpha), then the rest additively blend.
            gl_ref.disable(glow::BLEND);
            if let Some(tex0) = off_texs.first().copied() {
                gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex0));
                gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            if off_texs.len() > 1 {
                gl_ref.enable(glow::BLEND);
                gl_ref.blend_func_separate(glow::ONE, glow::ONE, glow::ZERO, glow::ONE);
                for tex in off_texs.iter().copied().skip(1) {
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
            }

            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_vertex_array(None);
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, None);
            gl_ref.use_program(None);
        }
    }

    pub fn paint_overlay(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        tiles: &[TileDraw],
        channels: &[ChannelDraw],
        opacity: f32,
    ) {
        if tiles.is_empty() || channels.is_empty() || opacity <= 0.0 {
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
        let viewport_w = viewport.width().max(1.0);
        let viewport_h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);

        let w_px = (viewport_w * ppp).round().max(1.0) as i32;
        let h_px = (viewport_h * ppp).round().max(1.0) as i32;
        if !inner.ensure_offscreen(gl, channels.len(), w_px, h_px) {
            return;
        }
        let (off_w_px, off_h_px, off_fbos, off_texs) = match inner.offscreen.as_ref() {
            None => return,
            Some(o) => (o.w_px, o.h_px, o.fbos.clone(), o.texs.clone()),
        };

        let mut prev_viewport = [0i32; 4];
        let mut prev_scissor = [0i32; 4];
        let prev_scissor_enabled;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
        }

        unsafe {
            let gl_ref = gl.as_ref();
            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);
            gl_ref.use_program(Some(bindings.program));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_tex.as_ref(), 0);
            gl_ref.disable(glow::SCISSOR_TEST);

            let mut available_tiles: Vec<(TileDraw, Vec<(usize, glow::Texture)>)> = Vec::new();
            available_tiles.reserve(tiles.len().min(512));
            for td in tiles {
                let mut texs: Vec<(usize, glow::Texture)> = Vec::with_capacity(channels.len());
                for (ci, ch) in channels.iter().enumerate() {
                    let key = RawTileKey {
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key) {
                        texs.push((ci, tex));
                    }
                }
                if !texs.is_empty() {
                    available_tiles.push((*td, texs));
                }
            }

            for (ci, ch) in channels.iter().enumerate() {
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(off_fbos[ci]));
                gl_ref.viewport(0, 0, off_w_px, off_h_px);
                gl_ref.disable(glow::BLEND);
                gl_ref.clear_color(0.0, 0.0, 0.0, 0.0);
                gl_ref.clear(glow::COLOR_BUFFER_BIT);

                set_channel_uniforms(gl_ref, &bindings, ch.window, ch.color_rgb, ch.alpha_scale);

                for (td, texs) in &available_tiles {
                    if !td.screen_rect.intersects(viewport) {
                        continue;
                    }
                    let Some((_, tex)) = texs.iter().find(|(tile_ci, _)| *tile_ci == ci) else {
                        continue;
                    };
                    let verts =
                        tile_vertices_ndc(td.screen_rect, viewport, viewport_w, viewport_h, ppp);
                    gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
                    gl_ref.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&verts),
                        glow::STREAM_DRAW,
                    );
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(*tex));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
            }

            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl_ref.viewport(
                prev_viewport[0],
                prev_viewport[1],
                prev_viewport[2],
                prev_viewport[3],
            );
            if prev_scissor_enabled {
                gl_ref.enable(glow::SCISSOR_TEST);
            } else {
                gl_ref.disable(glow::SCISSOR_TEST);
            }
            gl_ref.scissor(
                prev_scissor[0],
                prev_scissor[1],
                prev_scissor[2],
                prev_scissor[3],
            );

            gl_ref.use_program(Some(bindings.program_blit));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_blit_tex.as_ref(), 0);
            gl_ref.uniform_1_f32(bindings.u_blit_alpha_scale.as_ref(), 1.0);
            gl_ref.uniform_1_f32(bindings.u_blit_alpha_scale.as_ref(), opacity);

            let full = tile_vertices_ndc(viewport, viewport, viewport_w, viewport_h, ppp);
            gl_ref.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&full),
                glow::STREAM_DRAW,
            );

            gl_ref.enable(glow::BLEND);
            gl_ref.blend_func_separate(
                glow::SRC_ALPHA,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
            );
            for tex in off_texs.iter().copied().take(channels.len()) {
                gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex));
                gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            gl_ref.disable(glow::BLEND);
            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_vertex_array(None);
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, None);
            gl_ref.use_program(None);
        }
    }

    pub fn paint_with_channel_transforms_screen(
        &self,
        info: egui::PaintCallbackInfo,
        painter: &egui_glow::Painter,
        tiles: &[TileDraw],
        channels: &[ChannelDraw],
        channel_xforms: &[ChannelScreenTransform],
    ) {
        if tiles.is_empty() || channels.is_empty() {
            return;
        }
        if channels.len() != channel_xforms.len() {
            // Defensive: call site should keep these in sync.
            self.paint(info, painter, tiles, channels);
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
        let viewport_w = viewport.width().max(1.0);
        let viewport_h = viewport.height().max(1.0);
        let ppp = info.pixels_per_point.max(1e-6);

        let w_px = (viewport_w * ppp).round().max(1.0) as i32;
        let h_px = (viewport_h * ppp).round().max(1.0) as i32;
        if !inner.ensure_offscreen(gl, channels.len(), w_px, h_px) {
            return;
        }
        let (off_w_px, off_h_px, off_fbos, off_texs) = match inner.offscreen.as_ref() {
            None => return,
            Some(o) => (o.w_px, o.h_px, o.fbos.clone(), o.texs.clone()),
        };

        // Save viewport/scissor so we don't break other egui painting.
        let mut prev_viewport = [0i32; 4];
        let mut prev_scissor = [0i32; 4];
        let prev_scissor_enabled;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
        }

        unsafe {
            let gl_ref = gl.as_ref();

            gl_ref.disable(glow::DEPTH_TEST);
            gl_ref.disable(glow::CULL_FACE);

            // Render each channel into its own offscreen texture.
            gl_ref.use_program(Some(bindings.program));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_tex.as_ref(), 0);

            // Disable scissor while drawing to FBOs.
            gl_ref.disable(glow::SCISSOR_TEST);

            let mut available_tiles: Vec<(TileDraw, Vec<(usize, glow::Texture)>)> = Vec::new();
            available_tiles.reserve(tiles.len().min(1024));
            for td in tiles {
                let mut texs: Vec<(usize, glow::Texture)> = Vec::with_capacity(channels.len());
                for (ci, ch) in channels.iter().enumerate() {
                    let key = RawTileKey {
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key) {
                        texs.push((ci, tex));
                    }
                }
                if !texs.is_empty() {
                    available_tiles.push((*td, texs));
                }
            }

            for (ci, ch) in channels.iter().enumerate() {
                let xf = channel_xforms[ci];
                gl_ref.bind_framebuffer(glow::FRAMEBUFFER, Some(off_fbos[ci]));
                gl_ref.viewport(0, 0, off_w_px, off_h_px);
                gl_ref.disable(glow::BLEND);
                gl_ref.clear_color(0.0, 0.0, 0.0, 0.0);
                gl_ref.clear(glow::COLOR_BUFFER_BIT);

                set_channel_uniforms(gl_ref, &bindings, ch.window, ch.color_rgb, ch.alpha_scale);

                for (td, texs) in &available_tiles {
                    let quad = xform_screen_rect_to_quad(td.screen_rect, xf);
                    let aabb = aabb_of_quad(&quad);
                    if !aabb.intersects(viewport) {
                        continue;
                    }

                    let Some((_, tex)) = texs.iter().find(|(tile_ci, _)| *tile_ci == ci) else {
                        continue;
                    };

                    let verts = tile_quad_vertices_ndc(quad, viewport, viewport_w, viewport_h, ppp);
                    gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
                    gl_ref.buffer_data_u8_slice(
                        glow::ARRAY_BUFFER,
                        bytemuck::cast_slice(&verts),
                        glow::STREAM_DRAW,
                    );
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(*tex));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
            }

            // Restore drawing to the main framebuffer.
            gl_ref.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl_ref.viewport(
                prev_viewport[0],
                prev_viewport[1],
                prev_viewport[2],
                prev_viewport[3],
            );
            if prev_scissor_enabled {
                gl_ref.enable(glow::SCISSOR_TEST);
            } else {
                gl_ref.disable(glow::SCISSOR_TEST);
            }
            gl_ref.scissor(
                prev_scissor[0],
                prev_scissor[1],
                prev_scissor[2],
                prev_scissor[3],
            );

            // Composite channel textures back to the main framebuffer.
            gl_ref.use_program(Some(bindings.program_blit));
            gl_ref.bind_vertex_array(Some(bindings.vao));
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, Some(bindings.vbo));
            gl_ref.active_texture(glow::TEXTURE0);
            gl_ref.uniform_1_i32(bindings.u_blit_tex.as_ref(), 0);

            let full = tile_vertices_ndc(viewport, viewport, viewport_w, viewport_h, ppp);
            gl_ref.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&full),
                glow::STREAM_DRAW,
            );

            // First channel overwrites (establishes base + alpha), then the rest additively blend.
            gl_ref.disable(glow::BLEND);
            if let Some(tex0) = off_texs.first().copied() {
                gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex0));
                gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
            }

            if off_texs.len() > 1 {
                gl_ref.enable(glow::BLEND);
                gl_ref.blend_func_separate(glow::ONE, glow::ONE, glow::ZERO, glow::ONE);
                for tex in off_texs.iter().copied().skip(1) {
                    gl_ref.bind_texture(glow::TEXTURE_2D, Some(tex));
                    gl_ref.draw_arrays(glow::TRIANGLES, 0, 6);
                }
            }

            gl_ref.bind_texture(glow::TEXTURE_2D, None);
            gl_ref.bind_vertex_array(None);
            gl_ref.bind_buffer(glow::ARRAY_BUFFER, None);
            gl_ref.use_program(None);
        }
    }
}

impl From<RenderChannel> for ChannelDraw {
    fn from(c: RenderChannel) -> Self {
        Self {
            index: c.index,
            color_rgb: c.color_rgb,
            window: c.window,
            alpha_scale: 1.0,
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

struct Inner {
    cache: RawTileCache<TileState>,
    textures_to_delete: Vec<glow::Texture>,
    globj: Option<GlObjects>,
    desired_filter: TextureFilter,
    offscreen: Option<OffscreenTargets>,
}

struct OffscreenTargets {
    w_px: i32,
    h_px: i32,
    fbos: Vec<glow::Framebuffer>,
    texs: Vec<glow::Texture>,
}

impl Inner {
    fn new(capacity_tiles: usize) -> Self {
        Self {
            cache: RawTileCache::new(capacity_tiles),
            textures_to_delete: Vec::new(),
            globj: None,
            desired_filter: TextureFilter::Linear,
            offscreen: None,
        }
    }

    fn insert_pending(&mut self, resp: RawTileResponse) {
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

    fn ensure_offscreen(
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

fn set_channel_uniforms(
    gl: &glow::Context,
    bindings: &GlBindings,
    window: (f32, f32),
    color: [f32; 3],
    alpha_scale: f32,
) {
    let (w0, w1) = window;
    unsafe {
        gl.uniform_2_f32(bindings.u_window.as_ref(), w0, w1);
        gl.uniform_3_f32(bindings.u_color.as_ref(), color[0], color[1], color[2]);
        gl.uniform_1_f32(bindings.u_alpha_scale.as_ref(), alpha_scale);
    }
}

fn tile_vertices_ndc(
    screen_rect: egui::Rect,
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
) -> [f32; 6 * 4] {
    // Snap tile edges to physical pixels to avoid thin gaps/black bars at some zoom levels
    // due to float precision and fractional egui points.
    let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
    let min_x = snap(screen_rect.min.x);
    let max_x = snap(screen_rect.max.x);
    let min_y = snap(screen_rect.min.y);
    let max_y = snap(screen_rect.max.y);

    let x0 = ((min_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let x1 = ((max_x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
    let y0 = 1.0 - ((min_y - viewport.min.y) / viewport_h) * 2.0;
    let y1 = 1.0 - ((max_y - viewport.min.y) / viewport_h) * 2.0;

    // (x0,y0) is top-left in NDC, but triangles need correct winding; we don't cull.
    // UVs: match egui's convention where (0,0) corresponds to the first row of the uploaded data.
    let u0 = 0.0f32;
    let u1 = 1.0f32;
    let v0 = 0.0f32;
    let v1 = 1.0f32;

    [
        // tri 1
        x0, y0, u0, v0, // tl
        x1, y0, u1, v0, // tr
        x1, y1, u1, v1, // br
        // tri 2
        x0, y0, u0, v0, // tl
        x1, y1, u1, v1, // br
        x0, y1, u0, v1, // bl
    ]
}

fn tile_quad_vertices_ndc(
    quad: [egui::Pos2; 4],
    viewport: egui::Rect,
    viewport_w: f32,
    viewport_h: f32,
    pixels_per_point: f32,
) -> [f32; 6 * 4] {
    let snap = |v: f32| (v * pixels_per_point).round() / pixels_per_point;
    let p0 = egui::pos2(snap(quad[0].x), snap(quad[0].y)); // tl
    let p1 = egui::pos2(snap(quad[1].x), snap(quad[1].y)); // tr
    let p2 = egui::pos2(snap(quad[2].x), snap(quad[2].y)); // br
    let p3 = egui::pos2(snap(quad[3].x), snap(quad[3].y)); // bl

    let to_ndc = |p: egui::Pos2| -> (f32, f32) {
        let x = ((p.x - viewport.min.x) / viewport_w) * 2.0 - 1.0;
        let y = 1.0 - ((p.y - viewport.min.y) / viewport_h) * 2.0;
        (x, y)
    };
    let (x0, y0) = to_ndc(p0);
    let (x1, y1) = to_ndc(p1);
    let (x2, y2) = to_ndc(p2);
    let (x3, y3) = to_ndc(p3);

    let u0 = 0.0f32;
    let u1 = 1.0f32;
    let v0 = 0.0f32;
    let v1 = 1.0f32;

    [
        // tri 1: tl, tr, br
        x0, y0, u0, v0, // tl
        x1, y1, u1, v0, // tr
        x2, y2, u1, v1, // br
        // tri 2: tl, br, bl
        x0, y0, u0, v0, // tl
        x2, y2, u1, v1, // br
        x3, y3, u0, v1, // bl
    ]
}

fn xform_screen_rect_to_quad(rect: egui::Rect, xf: ChannelScreenTransform) -> [egui::Pos2; 4] {
    let tl = rect.left_top();
    let tr = egui::pos2(rect.right(), rect.top());
    let br = rect.right_bottom();
    let bl = egui::pos2(rect.left(), rect.bottom());

    [
        xform_screen_point(tl, xf),
        xform_screen_point(tr, xf),
        xform_screen_point(br, xf),
        xform_screen_point(bl, xf),
    ]
}

fn xform_screen_point(p: egui::Pos2, xf: ChannelScreenTransform) -> egui::Pos2 {
    let v = p - xf.pivot_screen;
    let v = egui::vec2(v.x * xf.scale.x, v.y * xf.scale.y);
    let v = rotate_vec2(v, xf.rotation_rad);
    xf.pivot_screen + xf.translation_screen + v
}

fn rotate_vec2(v: egui::Vec2, rotation_rad: f32) -> egui::Vec2 {
    let (s, c) = rotation_rad.sin_cos();
    egui::vec2(v.x * c - v.y * s, v.x * s + v.y * c)
}

fn aabb_of_quad(quad: &[egui::Pos2; 4]) -> egui::Rect {
    let mut min_x = quad[0].x;
    let mut max_x = quad[0].x;
    let mut min_y = quad[0].y;
    let mut max_y = quad[0].y;
    for p in quad.iter().copied().skip(1) {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y))
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

fn blit_shader_sources(gl_major: u32) -> (&'static str, &'static str) {
    if gl_major >= 3 {
        (VERT_330, BLIT_FRAG_330)
    } else {
        (VERT_120, BLIT_FRAG_120)
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
uniform float u_alpha_scale;

out vec4 out_color;

void main() {
    float raw = texture(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    out_color = vec4(rgb, t * u_alpha_scale);
}
"#;

const BLIT_FRAG_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform sampler2D u_tex;
uniform float u_alpha_scale;

out vec4 out_color;

void main() {
    // Texture attached to an FBO is addressed with (0,0) at the bottom-left in UV space.
    // The rest of the viewer uses the convention that v=0 corresponds to the first row of data,
    // so flip v here to match the non-offscreen rendering path.
    vec4 c = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
    out_color = vec4(c.rgb, c.a * u_alpha_scale);
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
uniform float u_alpha_scale;

void main() {
    float raw = texture2D(u_tex, v_uv).r * 65535.0;
    float denom = max(u_window.y - u_window.x, 1.0);
    float t = clamp((raw - u_window.x) / denom, 0.0, 1.0);
    vec3 rgb = t * u_color;
    gl_FragColor = vec4(rgb, t * u_alpha_scale);
}
"#;

const BLIT_FRAG_120: &str = r#"#version 120
varying vec2 v_uv;

uniform sampler2D u_tex;
uniform float u_alpha_scale;

void main() {
    vec4 c = texture2D(u_tex, vec2(v_uv.x, 1.0 - v_uv.y));
    gl_FragColor = vec4(c.rgb, c.a * u_alpha_scale);
}
"#;
