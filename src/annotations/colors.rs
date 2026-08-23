use eframe::egui;

use super::{AnnotationCategoryStyle, AnnotationShape};

pub fn default_category_styles(categories: &[String]) -> Vec<AnnotationCategoryStyle> {
    let mut out = Vec::new();
    out.reserve(categories.len());
    for (i, name) in categories.iter().enumerate() {
        let (r, g, b) = categorical_color_u8(i);
        let shape = AnnotationShape::ALL[i % AnnotationShape::ALL.len()];
        out.push(AnnotationCategoryStyle {
            name: name.clone(),
            visible: true,
            color: egui::Color32::from_rgba_unmultiplied(r, g, b, 230),
            shape,
        });
    }
    out
}

pub fn build_category_luts(
    styles: &[AnnotationCategoryStyle],
    group_tint: Option<([u8; 3], f32)>,
) -> (Vec<[f32; 4]>, Vec<i32>, Vec<i32>) {
    let max = styles.len().min(256);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(max);
    let mut shapes: Vec<i32> = Vec::with_capacity(max);
    let mut vis: Vec<i32> = Vec::with_capacity(max);
    for s in styles.iter().take(max) {
        let mut c = s.color;
        if let Some((rgb, strength)) = group_tint {
            c = tint_color32(c, rgb, strength);
        }
        colors.push([
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
            c.a() as f32 / 255.0,
        ]);
        shapes.push(s.shape as i32);
        vis.push(if s.visible { 1 } else { 0 });
    }
    (colors, shapes, vis)
}

pub(super) fn tint_color32(c: egui::Color32, tint_rgb: [u8; 3], strength: f32) -> egui::Color32 {
    let t = strength.clamp(0.0, 1.0);
    if t <= 0.0 {
        return c;
    }
    if t >= 1.0 {
        return egui::Color32::from_rgba_unmultiplied(tint_rgb[0], tint_rgb[1], tint_rgb[2], c.a());
    }
    let r = (c.r() as f32 * (1.0 - t) + tint_rgb[0] as f32 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    let g = (c.g() as f32 * (1.0 - t) + tint_rgb[1] as f32 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    let b = (c.b() as f32 * (1.0 - t) + tint_rgb[2] as f32 * t)
        .round()
        .clamp(0.0, 255.0) as u8;
    egui::Color32::from_rgba_unmultiplied(r, g, b, c.a())
}

fn categorical_color_u8(i: usize) -> (u8, u8, u8) {
    // Cycle through a high-contrast palette (20 colors).
    const P: &[(u8, u8, u8)] = &[
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ];
    P[i % P.len()]
}

pub(super) fn turbo_rgb_u8(t: f32) -> (u8, u8, u8) {
    // "Turbo" colormap approximation (Google). Input t in [0,1].
    let t = t.clamp(0.0, 1.0);
    let r =
        34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05))));
    let g = 23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1850.0 + t * 0.0))));
    let b = 27.2 + t * (3211.1 + t * (-15327.97 + t * (27814.0 + t * (-22569.18 + t * 6838.66))));
    let r = r.clamp(0.0, 255.0) as u8;
    let g = g.clamp(0.0, 255.0) as u8;
    let b = b.clamp(0.0, 255.0) as u8;
    (r, g, b)
}
