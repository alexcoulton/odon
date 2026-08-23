use std::sync::Arc;

use glow::HasContext;

use super::TextureFilter;

pub(super) fn upload_r16_texture(
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

pub(super) fn set_texture_filter(
    gl: &Arc<glow::Context>,
    tex: glow::Texture,
    filter: TextureFilter,
) {
    let gl = gl.as_ref();
    unsafe {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter.as_gl());
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter.as_gl());
        gl.bind_texture(glow::TEXTURE_2D, None);
    }
}
