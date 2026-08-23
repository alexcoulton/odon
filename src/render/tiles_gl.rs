mod backend;
mod geometry;
mod orchestration;
mod shaders;
mod upload;

use std::sync::Arc;

use eframe::egui;
use glow::HasContext;
use parking_lot::Mutex;

use crate::imaging::view_plane::ViewPlaneSelection;
use crate::render::tiles::RenderChannel;

use backend::Inner;

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
struct GlCapabilityState {
    blend: bool,
    depth_test: bool,
    cull_face: bool,
    blend_src_rgb: u32,
    blend_dst_rgb: u32,
    blend_src_alpha: u32,
    blend_dst_alpha: u32,
}

unsafe fn capture_gl_capabilities(gl: &glow::Context) -> GlCapabilityState {
    GlCapabilityState {
        blend: unsafe { gl.get_parameter_bool(glow::BLEND) },
        depth_test: unsafe { gl.get_parameter_bool(glow::DEPTH_TEST) },
        cull_face: unsafe { gl.get_parameter_bool(glow::CULL_FACE) },
        blend_src_rgb: unsafe { gl.get_parameter_i32(glow::BLEND_SRC_RGB) } as u32,
        blend_dst_rgb: unsafe { gl.get_parameter_i32(glow::BLEND_DST_RGB) } as u32,
        blend_src_alpha: unsafe { gl.get_parameter_i32(glow::BLEND_SRC_ALPHA) } as u32,
        blend_dst_alpha: unsafe { gl.get_parameter_i32(glow::BLEND_DST_ALPHA) } as u32,
    }
}

unsafe fn restore_gl_capabilities(gl: &glow::Context, state: GlCapabilityState) {
    unsafe {
        if state.blend {
            gl.enable(glow::BLEND);
        } else {
            gl.disable(glow::BLEND);
        }
        if state.depth_test {
            gl.enable(glow::DEPTH_TEST);
        } else {
            gl.disable(glow::DEPTH_TEST);
        }
        if state.cull_face {
            gl.enable(glow::CULL_FACE);
        } else {
            gl.disable(glow::CULL_FACE);
        }
        gl.blend_func_separate(
            state.blend_src_rgb,
            state.blend_dst_rgb,
            state.blend_src_alpha,
            state.blend_dst_alpha,
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TileDraw {
    pub view: ViewPlaneSelection,
    pub fallback_view: Option<ViewPlaneSelection>,
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
