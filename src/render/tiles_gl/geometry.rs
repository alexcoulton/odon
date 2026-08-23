use eframe::egui;
use glow::HasContext;

use super::ChannelScreenTransform;
use super::backend::GlBindings;

pub(super) fn set_channel_uniforms(
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

pub(super) fn tile_vertices_ndc(
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

pub(super) fn tile_quad_vertices_ndc(
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

pub(super) fn xform_screen_rect_to_quad(
    rect: egui::Rect,
    xf: ChannelScreenTransform,
) -> [egui::Pos2; 4] {
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

pub(super) fn aabb_of_quad(quad: &[egui::Pos2; 4]) -> egui::Rect {
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
