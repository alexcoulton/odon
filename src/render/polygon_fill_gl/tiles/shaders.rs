//! GLSL programs for rasterizing and composing object-ID tiles.

pub(super) const ID_TILE_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in float a_object_id;

uniform vec2 u_tile_min;
uniform vec2 u_tile_size;

flat out uint v_object_id;

void main() {
    vec2 rel = (a_pos - u_tile_min) / max(u_tile_size, vec2(1e-6));
    gl_Position = vec4(rel.x * 2.0 - 1.0, 1.0 - rel.y * 2.0, 0.0, 1.0);
    v_object_id = uint(a_object_id + 0.5) + 1u;
}"#;

pub(super) const ID_TILE_FRAG_330: &str = r#"#version 330 core
flat in uint v_object_id;
layout(location = 0) out uint out_object_id;
void main() {
    out_object_id = v_object_id;
}"#;

pub(super) const ID_TILE_COMPOSE_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec2 a_pos_ndc;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos_ndc, 0.0, 1.0);
    v_uv = a_uv;
}"#;

pub(super) const ID_TILE_COMPOSE_FRAG_330: &str = r#"#version 330 core
in vec2 v_uv;

uniform usampler2D u_id_tex;
uniform sampler2D u_state_tex;
uniform ivec2 u_state_tex_size;
uniform sampler2D u_color_tex;
uniform ivec2 u_color_tex_size;
uniform int u_use_object_colors;
uniform float u_object_color_opacity;
uniform vec4 u_selected_color;
uniform vec4 u_primary_color;
uniform sampler2D u_selection_tex;
uniform ivec2 u_selection_tex_size;
uniform int u_use_selection_overlay;
uniform vec4 u_selection_selected_color;
uniform vec4 u_selection_primary_color;
uniform int u_border_enabled;
uniform int u_border_radius_texels;
uniform vec4 u_border_color;
uniform int u_border_use_object_colors;
uniform float u_border_object_color_opacity;
uniform vec4 u_border_selected_color;
uniform vec4 u_border_primary_color;
uniform vec2 u_texels_per_fragment;
uniform int u_multisample_resolve;

out vec4 out_color;

bool id_is_boundary(ivec2 coord, uint stored_id) {
    ivec2 size = textureSize(u_id_tex, 0);
    int radius = clamp(u_border_radius_texels, 0, 2);
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            if ((dx == 0 && dy == 0) || max(abs(dx), abs(dy)) > radius) {
                continue;
            }
            ivec2 neighbour_coord = clamp(coord + ivec2(dx, dy), ivec2(0), size - ivec2(1));
            if (texelFetch(u_id_tex, neighbour_coord, 0).r != stored_id) {
                return true;
            }
        }
    }
    return false;
}

bool id_is_overview_boundary(ivec2 coord, uint stored_id) {
    ivec2 id_size = textureSize(u_id_tex, 0);
    int radius = clamp(u_border_radius_texels, 0, 2);
    if (radius <= 0) {
        return false;
    }
    ivec2 low = ivec2(0);
    ivec2 high = id_size - ivec2(1);
    return texelFetch(u_id_tex, clamp(coord + ivec2(-radius, 0), low, high), 0).r != stored_id
        || texelFetch(u_id_tex, clamp(coord + ivec2(radius, 0), low, high), 0).r != stored_id
        || texelFetch(u_id_tex, clamp(coord + ivec2(0, -radius), low, high), 0).r != stored_id
        || texelFetch(u_id_tex, clamp(coord + ivec2(0, radius), low, high), 0).r != stored_id;
}

ivec2 id_coord_for_position(vec2 texel_position) {
    ivec2 id_size = textureSize(u_id_tex, 0);
    return clamp(ivec2(floor(texel_position)), ivec2(0), id_size - ivec2(1));
}

bool resolve_sample(ivec2 id_coord, bool overview_resolve, out vec4 resolved_color) {
    resolved_color = vec4(0.0);
    uint stored_id = texelFetch(u_id_tex, id_coord, 0).r;
    if (stored_id == 0u || u_state_tex_size.x <= 0 || u_state_tex_size.y <= 0) {
        return false;
    }
    int object_id = int(stored_id - 1u);
    int state_x = object_id % u_state_tex_size.x;
    int state_y = object_id / u_state_tex_size.x;
    if (state_y < 0 || state_y >= u_state_tex_size.y) {
        return false;
    }
    float state = texelFetch(u_state_tex, ivec2(state_x, state_y), 0).r;
    if (state < 0.001) {
        return false;
    }

    float selection_state = 0.0;
    if (u_use_selection_overlay != 0) {
        if (u_selection_tex_size.x <= 0 || u_selection_tex_size.y <= 0) {
            return false;
        }
        int selection_x = object_id % u_selection_tex_size.x;
        int selection_y = object_id / u_selection_tex_size.x;
        if (selection_y < 0 || selection_y >= u_selection_tex_size.y) {
            return false;
        }
        selection_state = texelFetch(
            u_selection_tex,
            ivec2(selection_x, selection_y),
            0
        ).r;
    }

    vec4 object_color = vec4(0.0);
    if (u_use_object_colors != 0 || u_border_use_object_colors != 0) {
        if (u_color_tex_size.x <= 0 || u_color_tex_size.y <= 0) {
            return false;
        }
        int color_x = object_id % u_color_tex_size.x;
        int color_y = object_id / u_color_tex_size.x;
        if (color_y < 0 || color_y >= u_color_tex_size.y) {
            return false;
        }
        object_color = texelFetch(u_color_tex, ivec2(color_x, color_y), 0);
    }

    if (selection_state >= 0.001) {
        resolved_color = selection_state > 0.75
            ? u_selection_primary_color
            : u_selection_selected_color;
    } else if (u_use_object_colors != 0) {
        vec4 fill_object_color = object_color;
        fill_object_color.a *= u_object_color_opacity;
        if (fill_object_color.a <= 0.0) {
            return false;
        }
        resolved_color = fill_object_color;
    } else {
        resolved_color = state > 0.75 ? u_primary_color : u_selected_color;
    }

    bool boundary = false;
    if (u_border_enabled != 0) {
        boundary = overview_resolve
            ? id_is_overview_boundary(id_coord, stored_id)
            : id_is_boundary(id_coord, stored_id);
    }
    if (u_border_enabled != 0 && boundary) {
        if (selection_state >= 0.001) {
            resolved_color = selection_state > 0.75
                ? u_border_primary_color
                : u_border_selected_color;
        } else if (u_border_use_object_colors != 0) {
            object_color.a *= u_border_object_color_opacity;
            resolved_color = object_color;
        } else {
            resolved_color = u_border_color;
        }
    }
    return resolved_color.a > 0.0;
}

void accumulate_sample(
    vec2 texel_position,
    bool overview_resolve,
    inout vec3 premultiplied_rgb,
    inout float alpha_sum
) {
    vec4 sample_color;
    if (!resolve_sample(
        id_coord_for_position(texel_position),
        overview_resolve,
        sample_color
    )) {
        return;
    }
    premultiplied_rgb += sample_color.rgb * sample_color.a;
    alpha_sum += sample_color.a;
}

void main() {
    ivec2 id_size = textureSize(u_id_tex, 0);
    vec2 texel_position = v_uv * vec2(id_size);
    bool overview_resolve = u_multisample_resolve != 0;
    if (!overview_resolve) {
        if (!resolve_sample(id_coord_for_position(texel_position), false, out_color)) {
            discard;
        }
        return;
    }

    vec2 quarter_footprint = max(u_texels_per_fragment, vec2(0.0)) * 0.25;
    vec3 premultiplied_rgb = vec3(0.0);
    float alpha_sum = 0.0;
    accumulate_sample(
        texel_position + vec2(-quarter_footprint.x, -quarter_footprint.y),
        true,
        premultiplied_rgb,
        alpha_sum
    );
    accumulate_sample(
        texel_position + vec2(quarter_footprint.x, -quarter_footprint.y),
        true,
        premultiplied_rgb,
        alpha_sum
    );
    accumulate_sample(
        texel_position + vec2(-quarter_footprint.x, quarter_footprint.y),
        true,
        premultiplied_rgb,
        alpha_sum
    );
    accumulate_sample(
        texel_position + vec2(quarter_footprint.x, quarter_footprint.y),
        true,
        premultiplied_rgb,
        alpha_sum
    );
    if (alpha_sum <= 0.0) {
        discard;
    }
    out_color = vec4(premultiplied_rgb / alpha_sum, alpha_sum * 0.25);
}"#;
