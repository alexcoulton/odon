pub(super) const OBJECT_LINE_VERT_330: &str = r#"#version 330 core
layout(location = 0) in vec4 a_seg;
layout(location = 1) in float a_object_id;

uniform vec2 u_center_world;
uniform float u_zoom_px;
uniform vec2 u_viewport_min_px;
uniform vec2 u_viewport_size_px;
uniform float u_base_width_px;
uniform float u_selected_width_px;
uniform float u_primary_width_px;
uniform vec2 u_local_to_world_offset;
uniform vec2 u_local_to_world_scale;
uniform sampler2D u_state_tex;
uniform ivec2 u_state_tex_size;

out vec2 v_screen_px;
out vec2 v_a_px;
out vec2 v_b_px;
out float v_half_w;
flat out float v_state;
flat out int v_object_id;

float selection_state_for_object(int object_id) {
    if (u_state_tex_size.x <= 0 || u_state_tex_size.y <= 0 || object_id < 0) {
        return 0.0;
    }
    int x = object_id % u_state_tex_size.x;
    int y = object_id / u_state_tex_size.x;
    if (y < 0 || y >= u_state_tex_size.y) {
        return 0.0;
    }
    return texelFetch(u_state_tex, ivec2(x, y), 0).r;
}

void main() {
    int object_id = int(a_object_id + 0.5);
    float state = selection_state_for_object(object_id);
    v_state = state;
    v_object_id = object_id;

    vec2 a_world = u_local_to_world_offset + a_seg.xy * u_local_to_world_scale;
    vec2 b_world = u_local_to_world_offset + a_seg.zw * u_local_to_world_scale;

    vec2 viewport_center_px = u_viewport_min_px + 0.5 * u_viewport_size_px;
    vec2 a_px = (a_world - u_center_world) * u_zoom_px + viewport_center_px;
    vec2 b_px = (b_world - u_center_world) * u_zoom_px + viewport_center_px;

    vec2 d = b_px - a_px;
    float len2 = max(dot(d, d), 1e-6);
    vec2 dir = d * inversesqrt(len2);
    vec2 n = vec2(-dir.y, dir.x);

    float width_px = state > 0.75 ? max(u_primary_width_px, 0.5) :
        (state > 0.49 ? max(u_selected_width_px, 0.5) : max(u_base_width_px, 0.5));
    float half_w = 0.5 * width_px;
    vec2 a2 = a_px - dir * half_w;
    vec2 b2 = b_px + dir * half_w;

    int vid = gl_VertexID;
    float t = 0.0;
    float side = -1.0;
    if (vid == 0) { t = 0.0; side = -1.0; }
    else if (vid == 1) { t = 1.0; side = -1.0; }
    else if (vid == 2) { t = 1.0; side = 1.0; }
    else if (vid == 3) { t = 0.0; side = -1.0; }
    else if (vid == 4) { t = 1.0; side = 1.0; }
    else { t = 0.0; side = 1.0; }

    vec2 base = mix(a2, b2, t) + n * side * half_w;
    vec2 local = base - u_viewport_min_px;
    vec2 ndc = vec2(
        (local.x / u_viewport_size_px.x) * 2.0 - 1.0,
        1.0 - (local.y / u_viewport_size_px.y) * 2.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_screen_px = base;
    v_a_px = a2;
    v_b_px = b2;
    v_half_w = half_w;
}
"#;

pub(super) const OBJECT_LINE_FRAG_330: &str = r#"#version 330 core
in vec2 v_screen_px;
in vec2 v_a_px;
in vec2 v_b_px;
in float v_half_w;
flat in float v_state;
flat in int v_object_id;

uniform vec4 u_base_color;
uniform vec4 u_selected_color;
uniform vec4 u_primary_color;
uniform bool u_draw_unselected;
uniform sampler2D u_color_tex;
uniform ivec2 u_color_tex_size;
uniform int u_use_object_colors;
uniform float u_object_color_opacity;

out vec4 out_color;

float segment_distance(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    if (v_state < 0.001 && !u_draw_unselected) {
        discard;
    }
    float dist = segment_distance(v_screen_px, v_a_px, v_b_px);
    float aa = 1.0;
    float alpha = 1.0 - smoothstep(v_half_w - aa, v_half_w + aa, dist);
    vec4 base_color = u_base_color;
    if (u_use_object_colors != 0) {
        if (u_color_tex_size.x <= 0 || u_color_tex_size.y <= 0 || v_object_id < 0) {
            discard;
        }
        int color_x = v_object_id % u_color_tex_size.x;
        int color_y = v_object_id / u_color_tex_size.x;
        if (color_y < 0 || color_y >= u_color_tex_size.y) {
            discard;
        }
        base_color = texelFetch(u_color_tex, ivec2(color_x, color_y), 0);
        base_color.a *= u_object_color_opacity;
    }
    vec4 color = v_state > 0.75 ? u_primary_color :
        (v_state > 0.49 ? u_selected_color : base_color);
    out_color = vec4(color.rgb, color.a * alpha);
    if (out_color.a <= 0.0) {
        discard;
    }
}
"#;
