use glow::HasContext;

pub(super) fn shader_sources(gl_major: u32) -> (&'static str, &'static str) {
    if gl_major >= 3 {
        (VERT_330, FRAG_330)
    } else {
        (VERT_120, FRAG_120)
    }
}

pub(super) fn blit_shader_sources(gl_major: u32) -> (&'static str, &'static str) {
    if gl_major >= 3 {
        (VERT_330, BLIT_FRAG_330)
    } else {
        (VERT_120, BLIT_FRAG_120)
    }
}

pub(super) fn compile_program(
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
