use std::collections::HashSet;
use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use super::backend::{Inner, TileState};
use super::geometry::{
    aabb_of_quad, set_channel_uniforms, tile_quad_vertices_ndc, tile_vertices_ndc,
    xform_screen_rect_to_quad,
};
use super::{
    ChannelDraw, ChannelScreenTransform, TextureFilter, TileDraw, TilesGl, capture_gl_capabilities,
    restore_gl_capabilities,
};
use crate::render::tiles_raw::{RawTileKey, RawTileResponse};

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

    pub fn len(&self) -> usize {
        self.inner.lock().cache.len()
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
        let previous_capabilities;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
            previous_capabilities = capture_gl_capabilities(gl);
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
                        view: td.view,
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key).or_else(|| {
                        td.fallback_view.and_then(|view| {
                            inner.ensure_uploaded(
                                gl,
                                &RawTileKey {
                                    view,
                                    level: td.level,
                                    tile_y: td.tile_y,
                                    tile_x: td.tile_x,
                                    channel: ch.index,
                                },
                            )
                        })
                    }) {
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
            restore_gl_capabilities(gl_ref, previous_capabilities);
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
        let previous_capabilities;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
            previous_capabilities = capture_gl_capabilities(gl);
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
                        view: td.view,
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key).or_else(|| {
                        td.fallback_view.and_then(|view| {
                            inner.ensure_uploaded(
                                gl,
                                &RawTileKey {
                                    view,
                                    level: td.level,
                                    tile_y: td.tile_y,
                                    tile_x: td.tile_x,
                                    channel: ch.index,
                                },
                            )
                        })
                    }) {
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
            restore_gl_capabilities(gl_ref, previous_capabilities);
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
        let previous_capabilities;
        unsafe {
            let gl = gl.as_ref();
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut prev_viewport);
            gl.get_parameter_i32_slice(glow::SCISSOR_BOX, &mut prev_scissor);
            prev_scissor_enabled = gl.get_parameter_bool(glow::SCISSOR_TEST);
            previous_capabilities = capture_gl_capabilities(gl);
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
                        view: td.view,
                        level: td.level,
                        tile_y: td.tile_y,
                        tile_x: td.tile_x,
                        channel: ch.index,
                    };
                    if let Some(tex) = inner.ensure_uploaded(gl, &key).or_else(|| {
                        td.fallback_view.and_then(|view| {
                            inner.ensure_uploaded(
                                gl,
                                &RawTileKey {
                                    view,
                                    level: td.level,
                                    tile_y: td.tile_y,
                                    tile_x: td.tile_x,
                                    channel: ch.index,
                                },
                            )
                        })
                    }) {
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
            restore_gl_capabilities(gl_ref, previous_capabilities);
        }
    }
}
