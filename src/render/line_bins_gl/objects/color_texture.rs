use super::*;

pub(super) struct ObjectColorGpu {
    pub(super) texture: glow::Texture,
    pub(super) width: i32,
    pub(super) height: i32,
    pub(super) generation: u64,
}

impl ObjectLineInner {
    pub(super) fn ensure_color_uploaded(
        &mut self,
        gl: &Arc<glow::Context>,
        cache_id: u64,
        generation: u64,
        object_count: usize,
        colors_rgba: &[[u8; 4]],
    ) -> Option<&ObjectColorGpu> {
        if self
            .colors
            .get(&cache_id)
            .is_some_and(|color| color.generation == generation)
        {
            return self.colors.get(&cache_id);
        }
        let padded_len = object_count.max(1);
        let width = padded_len.min(4096) as i32;
        let height = ((padded_len + width as usize - 1) / width as usize).max(1) as i32;
        let mut texels = vec![0u8; width as usize * height as usize * 4];
        let copy_len = colors_rgba.len().min(object_count);
        texels[..copy_len * 4].copy_from_slice(bytemuck::cast_slice(&colors_rgba[..copy_len]));
        let texture = if let Some(existing) = self.colors.get(&cache_id) {
            existing.texture
        } else {
            unsafe { gl.as_ref().create_texture().map_err(|_| ()).ok()? }
        };
        unsafe {
            let gl = gl.as_ref();
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
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
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                width,
                height,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(texels.as_slice())),
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
        }
        if let Some((_key, evicted)) = self.colors.push(
            cache_id,
            ObjectColorGpu {
                texture,
                width,
                height,
                generation,
            },
        ) && evicted.texture != texture
        {
            self.textures_to_delete.push(evicted.texture);
        }
        self.delete_queued(gl);
        self.colors.get(&cache_id)
    }
}
