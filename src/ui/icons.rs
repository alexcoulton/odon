use std::collections::HashMap;
use std::path::PathBuf;

use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Icon {
    Pan,
    Move,
    Transform,
    Polygon,
    RectSelect,
    LassoSelect,
    Image,
    Points,
    Text,
}

const ICONS_FAMILY: &str = "icons";

#[derive(Debug, Clone, Default)]
struct FontAwesomeState {
    loaded: bool,
    glyphs: HashMap<Icon, char>,
}

fn fa_state_id() -> egui::Id {
    egui::Id::new("odon.fontawesome.state")
}

pub fn try_install_fontawesome(ctx: &egui::Context) -> bool {
    // Avoid re-reading files if `apply_napari_like_dark` is called multiple times.
    if let Some(state) = ctx.data(|d| d.get_temp::<FontAwesomeState>(fa_state_id())) {
        if state.loaded {
            return true;
        }
        // Allow a retry if the user added the font files after the app started.
        if fontawesome_ttf_search_paths()
            .iter()
            .any(|p| std::fs::metadata(p).is_ok())
        {
            ctx.data_mut(|d| {
                let _ = d.remove_temp::<FontAwesomeState>(fa_state_id());
            });
        } else {
            return false;
        }
    }

    let ttf_paths = fontawesome_ttf_search_paths();
    let css_paths = fontawesome_css_search_paths();

    let ttf_bytes = ttf_paths.iter().find_map(|p| std::fs::read(p).ok());

    let Some(ttf_bytes) = ttf_bytes else {
        ctx.data_mut(|d| {
            d.insert_temp(
                fa_state_id(),
                FontAwesomeState {
                    loaded: false,
                    glyphs: HashMap::new(),
                },
            )
        });
        return false;
    };

    let css = css_paths
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    if css.is_empty() {
        eprintln!(
            "Font Awesome: found TTF but no CSS. Put `all.min.css` (or `all.css`) under `assets/fontawesome/css/` so icons can be resolved."
        );
    }

    let mut glyphs: HashMap<Icon, char> = HashMap::new();
    for icon in [
        Icon::Pan,
        Icon::Move,
        Icon::Transform,
        Icon::Polygon,
        Icon::RectSelect,
        Icon::LassoSelect,
        Icon::Image,
        Icon::Points,
        Icon::Text,
    ] {
        if let Some(ch) = resolve_fontawesome_glyph(&css, icon) {
            glyphs.insert(icon, ch);
        }
    }
    if glyphs.is_empty() {
        eprintln!(
            "Font Awesome: no icon glyphs resolved (CSS missing or unsupported). Falling back to built-in vector icons."
        );
    }

    // Install fonts without clobbering other font definitions.
    ctx.add_font(egui::epaint::text::FontInsert::new(
        "fa-solid",
        egui::FontData::from_owned(ttf_bytes),
        vec![egui::epaint::text::InsertFontFamily {
            family: egui::FontFamily::Name(ICONS_FAMILY.into()),
            priority: egui::epaint::text::FontPriority::Highest,
        }],
    ));

    ctx.data_mut(|d| {
        d.insert_temp(
            fa_state_id(),
            FontAwesomeState {
                loaded: true,
                glyphs,
            },
        )
    });

    true
}

pub fn fontawesome_loaded(ctx: &egui::Context) -> bool {
    ctx.data(|d| {
        d.get_temp::<FontAwesomeState>(fa_state_id())
            .map(|s| s.loaded)
            .unwrap_or(false)
    })
}

fn fontawesome_glyph(ctx: &egui::Context, icon: Icon) -> Option<char> {
    ctx.data(|d| d.get_temp::<FontAwesomeState>(fa_state_id()))
        .and_then(|s| s.glyphs.get(&icon).copied())
}

fn fontawesome_ttf_search_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ODON_FA_SOLID_TTF") {
        out.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("RUST_OZ_FA_SOLID_TTF") {
        out.push(PathBuf::from(p));
    }
    out.extend(resource_search_paths("assets/fonts/fa-solid-900.ttf"));
    out.extend(resource_search_paths("assets/fontawesome/webfonts/fa-solid-900.ttf"));
    dedupe_paths(&mut out);
    out
}

fn fontawesome_css_search_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ODON_FA_CSS") {
        out.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("RUST_OZ_FA_CSS") {
        out.push(PathBuf::from(p));
    }
    out.extend(resource_search_paths("assets/fontawesome/css/all.min.css"));
    out.extend(resource_search_paths("assets/fontawesome/css/all.css"));
    out.extend(resource_search_paths("assets/fonts/all.min.css"));
    dedupe_paths(&mut out);
    out
}

fn dedupe_paths(paths: &mut Vec<PathBuf>) {
    let mut seen = std::collections::HashSet::new();
    paths.retain(|p| seen.insert(p.clone()));
}

fn resource_search_paths(relative: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(p) = app_bundle_resource_path(relative) {
        out.push(p);
    }

    let rel = PathBuf::from(relative);
    out.push(rel.clone());

    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd.join(&rel));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    out.push(manifest_dir.join(&rel));

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            out.push(exe_dir.join(&rel));
            for ancestor in exe_dir.ancestors().take(4) {
                out.push(ancestor.join(&rel));
            }
        }
    }

    out
}

fn app_bundle_resource_path(relative: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let macos_dir = exe.parent()?;
    if macos_dir.file_name().and_then(|name| name.to_str()) != Some("MacOS") {
        return None;
    }
    let contents_dir = macos_dir.parent()?;
    if contents_dir.file_name().and_then(|name| name.to_str()) != Some("Contents") {
        return None;
    }
    Some(contents_dir.join("Resources").join(relative))
}

fn resolve_fontawesome_glyph(css: &str, icon: Icon) -> Option<char> {
    let candidates: &[&str] = match icon {
        Icon::Pan => &["fa-hand", "fa-hand-paper", "fa-hand-paper-o"],
        Icon::Move => &[
            "fa-up-down-left-right",
            "fa-arrows-up-down-left-right",
            "fa-arrows",
        ],
        Icon::Transform => &[
            "fa-vector-square",
            "fa-up-right-and-down-left-from-center",
            "fa-maximize",
        ],
        Icon::Polygon => &["fa-draw-polygon", "fa-vector-square"],
        Icon::RectSelect => &["fa-vector-square", "fa-square"],
        Icon::LassoSelect => &["fa-bezier-curve", "fa-signature", "fa-pen-fancy"],
        Icon::Image => &["fa-image"],
        Icon::Points => &["fa-braille", "fa-circle-dot", "fa-dot-circle"],
        Icon::Text => &["fa-font", "fa-i-cursor", "fa-text-height"],
    };

    for name in candidates {
        if let Some(cp) = find_fa_codepoint(css, name) {
            if let Some(ch) = char::from_u32(cp) {
                return Some(ch);
            }
        }
    }
    None
}

fn find_fa_codepoint(css: &str, class_name: &str) -> Option<u32> {
    let selector = format!(".{class_name}");
    for (start, _) in css.match_indices(&selector) {
        let after = start + selector.len();
        let Some(next) = css.as_bytes().get(after).copied() else {
            continue;
        };
        if is_css_ident_continue(next) {
            continue;
        }

        let rest = &css[after..];
        let Some(open) = rest.find('{') else {
            continue;
        };
        let Some(close) = rest[open + 1..].find('}') else {
            continue;
        };
        let body = &rest[open + 1..open + 1 + close];

        if let Some(cp) = find_codepoint_in_rule_body(body, "content:") {
            return Some(cp);
        }
        if let Some(cp) = find_codepoint_in_rule_body(body, "--fa:") {
            return Some(cp);
        }
    }

    None
}

fn is_css_ident_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

fn find_codepoint_in_rule_body(body: &str, prop: &str) -> Option<u32> {
    let start = body.find(prop)?;
    let mut i = start + prop.len();
    let bytes = body.as_bytes();

    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'\\' {
        return None;
    }
    i += 1;

    let hex_start = i;
    while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
        i += 1;
    }
    if i == hex_start {
        return None;
    }

    u32::from_str_radix(&body[hex_start..i], 16).ok()
}

pub fn icon_button(
    ui: &mut egui::Ui,
    icon: Icon,
    selected: bool,
    sense: egui::Sense,
) -> egui::Response {
    let size = egui::vec2(22.0, 22.0);
    let (rect, response) = ui.allocate_exact_size(size, sense);

    if ui.is_rect_visible(rect) {
        let mut visuals = ui.style().interact(&response).clone();
        if selected {
            visuals.bg_fill = ui.visuals().selection.bg_fill;
            visuals.fg_stroke = ui.visuals().selection.stroke;
            visuals.bg_stroke = ui.visuals().selection.stroke;
        }

        let rounding = egui::CornerRadius::same(4);
        ui.painter().rect_filled(rect, rounding, visuals.bg_fill);

        // Subtle border for contrast with dark panels.
        ui.painter()
            .rect_stroke(rect, rounding, visuals.bg_stroke, egui::StrokeKind::Inside);

        let stroke = egui::Stroke::new(1.6, visuals.fg_stroke.color);
        paint_icon_in_rect(ui.ctx(), ui.painter(), rect.shrink(4.0), icon, stroke);
    }

    response
}

pub fn paint_icon_in_rect(
    ctx: &egui::Context,
    p: &egui::Painter,
    rect: egui::Rect,
    icon: Icon,
    stroke: egui::Stroke,
) {
    if fontawesome_loaded(ctx) {
        if let Some(ch) = fontawesome_glyph(ctx, icon) {
            let family = egui::FontFamily::Name(ICONS_FAMILY.into());
            let size = rect.height().min(rect.width()) * 1.05;
            p.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                ch.to_string(),
                egui::FontId::new(size, family),
                stroke.color,
            );
            return;
        }
    }

    match icon {
        Icon::Pan => paint_hand(p, rect, stroke),
        Icon::Move => paint_move(p, rect, stroke),
        Icon::Transform => paint_transform(p, rect, stroke),
        Icon::Polygon => paint_polygon(p, rect, stroke),
        Icon::RectSelect => paint_rect_select(p, rect, stroke),
        Icon::LassoSelect => paint_lasso_select(p, rect, stroke),
        Icon::Image => paint_image(p, rect, stroke),
        Icon::Points => paint_points(p, rect, stroke),
        Icon::Text => paint_text(p, rect, stroke),
    }
}

fn paint_hand(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // Simple "hand" icon: palm + four fingers + thumb.
    let r = rect.shrink(1.0);
    let w = r.width().max(1.0);
    let h = r.height().max(1.0);

    let palm = egui::Rect::from_center_size(
        egui::pos2(r.center().x - w * 0.06, r.center().y + h * 0.10),
        egui::vec2(w * 0.55, h * 0.42),
    );
    p.rect_stroke(
        palm,
        egui::CornerRadius::same(((w.min(h) * 0.10).max(2.0)).round().clamp(0.0, 255.0) as u8),
        stroke,
        egui::StrokeKind::Inside,
    );

    // Fingers.
    let top = palm.top();
    let finger_h = h * 0.34;
    let fx0 = palm.left() + palm.width() * 0.10;
    let fx1 = palm.left() + palm.width() * 0.30;
    let fx2 = palm.left() + palm.width() * 0.52;
    let fx3 = palm.left() + palm.width() * 0.78;
    for (i, fx) in [fx0, fx1, fx2, fx3].into_iter().enumerate() {
        let t = i as f32 / 3.0;
        let len = finger_h * (0.90 + 0.10 * t);
        p.line_segment([egui::pos2(fx, top), egui::pos2(fx, top - len)], stroke);
    }

    // Thumb.
    let thumb_a = egui::pos2(palm.left(), palm.center().y);
    let thumb_b = egui::pos2(palm.left() - w * 0.18, palm.center().y - h * 0.10);
    p.line_segment([thumb_a, thumb_b], stroke);
}

fn paint_move(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // A small box with a diagonal arrow, to indicate moving a layer (as opposed to panning the view).
    let r = rect.shrink(1.0);
    p.rect_stroke(
        r,
        egui::CornerRadius::same(2),
        stroke,
        egui::StrokeKind::Inside,
    );

    let a = egui::pos2(r.left() + r.width() * 0.28, r.bottom() - r.height() * 0.28);
    let b = egui::pos2(r.right() - r.width() * 0.20, r.top() + r.height() * 0.20);
    p.line_segment([a, b], stroke);

    let ah = (r.width().min(r.height()) * 0.16).max(2.0);
    let fill = stroke.color;
    p.add(egui::Shape::convex_polygon(
        vec![b, egui::pos2(b.x - ah, b.y), egui::pos2(b.x, b.y + ah)],
        fill,
        egui::Stroke::NONE,
    ));
}

fn paint_polygon(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    let c = rect.center();
    let r = rect.width().min(rect.height()) * 0.46;
    let mut pts = Vec::with_capacity(6);
    for i in 0..5 {
        let a = (i as f32) / 5.0 * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        pts.push(egui::pos2(c.x + r * a.cos(), c.y + r * a.sin()));
    }
    pts.push(pts[0]);
    p.add(egui::Shape::line(pts, stroke));
}

fn paint_rect_select(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    let r = rect.shrink(1.5);
    p.rect_stroke(
        r,
        egui::CornerRadius::same(2),
        stroke,
        egui::StrokeKind::Inside,
    );

    let handle_fill = stroke.color;
    let handle_r = (r.width().min(r.height()) * 0.075).max(1.5);
    for corner in [
        r.left_top(),
        r.right_top(),
        r.right_bottom(),
        r.left_bottom(),
    ] {
        p.circle_filled(corner, handle_r, handle_fill);
    }
}

fn paint_lasso_select(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    let r = rect.shrink(1.0);
    let pts = vec![
        egui::pos2(r.left() + r.width() * 0.18, r.top() + r.height() * 0.42),
        egui::pos2(r.left() + r.width() * 0.28, r.top() + r.height() * 0.20),
        egui::pos2(r.left() + r.width() * 0.56, r.top() + r.height() * 0.16),
        egui::pos2(r.left() + r.width() * 0.80, r.top() + r.height() * 0.34),
        egui::pos2(r.left() + r.width() * 0.74, r.top() + r.height() * 0.64),
        egui::pos2(r.left() + r.width() * 0.48, r.top() + r.height() * 0.78),
        egui::pos2(r.left() + r.width() * 0.24, r.top() + r.height() * 0.66),
        egui::pos2(r.left() + r.width() * 0.18, r.top() + r.height() * 0.42),
    ];
    p.add(egui::Shape::line(pts, stroke));

    let knot = egui::pos2(r.left() + r.width() * 0.80, r.top() + r.height() * 0.34);
    p.circle_filled(
        knot,
        (r.width().min(r.height()) * 0.07).max(1.5),
        stroke.color,
    );
}

fn paint_transform(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // A bounding box with corner handles and a small rotate marker.
    let r = rect.shrink(1.0);
    let rounding = egui::CornerRadius::same(2);
    p.rect_stroke(r, rounding, stroke, egui::StrokeKind::Inside);

    let handle_r = (r.width().min(r.height()) * 0.10).max(1.8);
    let fill = stroke.color;
    for corner in [
        r.left_top(),
        r.right_top(),
        r.right_bottom(),
        r.left_bottom(),
    ] {
        p.circle_filled(corner, handle_r, fill);
    }

    // Rotate marker: small circle above the top edge, connected by a line.
    let top_mid = egui::pos2(r.center().x, r.top());
    let marker = top_mid + egui::vec2(0.0, -r.height() * 0.22);
    p.line_segment([top_mid, marker], stroke);
    p.circle_stroke(marker, handle_r * 0.9, stroke);
}

fn paint_image(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // A "photo" icon: frame + mountain + sun.
    let r = rect.shrink(1.0);
    p.rect_stroke(
        r,
        egui::CornerRadius::same(2),
        stroke,
        egui::StrokeKind::Inside,
    );

    let sun = egui::pos2(r.left() + r.width() * 0.70, r.top() + r.height() * 0.30);
    p.circle_stroke(sun, (r.width().min(r.height()) * 0.10).max(1.8), stroke);

    let a = egui::pos2(r.left() + r.width() * 0.18, r.bottom() - r.height() * 0.18);
    let b = egui::pos2(r.left() + r.width() * 0.44, r.top() + r.height() * 0.46);
    let c = egui::pos2(r.left() + r.width() * 0.62, r.bottom() - r.height() * 0.26);
    let d = egui::pos2(r.right() - r.width() * 0.14, r.top() + r.height() * 0.52);
    let e = egui::pos2(r.right() - r.width() * 0.14, r.bottom() - r.height() * 0.18);
    p.add(egui::Shape::line(vec![a, b, c, d, e], stroke));
}

fn paint_points(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // A cluster of dots.
    let r = rect.shrink(1.0);
    let dot_r = (r.width().min(r.height()) * 0.10).max(1.6);
    let fill = stroke.color;
    for (x, y) in [
        (0.30, 0.35),
        (0.55, 0.28),
        (0.72, 0.52),
        (0.40, 0.62),
        (0.58, 0.64),
    ] {
        p.circle_filled(
            egui::pos2(r.left() + r.width() * x, r.top() + r.height() * y),
            dot_r,
            fill,
        );
    }
}

fn paint_text(p: &egui::Painter, rect: egui::Rect, stroke: egui::Stroke) {
    // A simple "T" glyph.
    let r = rect.shrink(2.0);
    let w = r.width().max(1.0);
    let h = r.height().max(1.0);

    let top_y = r.top() + h * 0.18;
    let mid_x = r.center().x;
    let left_x = r.left() + w * 0.18;
    let right_x = r.right() - w * 0.18;
    let bottom_y = r.bottom() - h * 0.12;

    p.line_segment(
        [egui::pos2(left_x, top_y), egui::pos2(right_x, top_y)],
        stroke,
    );
    p.line_segment(
        [egui::pos2(mid_x, top_y), egui::pos2(mid_x, bottom_y)],
        stroke,
    );
}
