use eframe::egui;

#[derive(Debug, Clone)]
pub struct PointsStyle {
    pub radius_screen_px: f32,
    pub fill_positive: egui::Color32,
    pub fill_negative: egui::Color32,
    pub stroke_positive: egui::Stroke,
    pub stroke_negative: egui::Stroke,
}

impl Default for PointsStyle {
    fn default() -> Self {
        Self {
            radius_screen_px: 5.0,
            fill_positive: egui::Color32::from_rgba_unmultiplied(255, 255, 255, 230),
            fill_negative: egui::Color32::TRANSPARENT,
            stroke_positive: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
            stroke_negative: egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Point {
    pub world_lvl0: egui::Pos2,
    pub positive: bool,
}

#[derive(Debug, Clone)]
pub struct PointsLayer {
    pub name: String,
    pub visible: bool,
    pub style: PointsStyle,
    pub points: Vec<Point>,
}

impl PointsLayer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            visible: true,
            style: PointsStyle::default(),
            points: Vec::new(),
        }
    }

    fn effective_radius_points(base_radius_points: f32, zoom_screen_per_world_px: f32) -> f32 {
        // Make points smaller when zoomed out, larger when zoomed in.
        // The sqrt keeps it from growing/shrinking too aggressively.
        let zoom = zoom_screen_per_world_px.max(1e-6);
        (base_radius_points.max(0.0) * zoom.sqrt()).clamp(0.75, 40.0)
    }

    pub fn draw(
        &self,
        painter: &egui::Painter,
        viewport: egui::Rect,
        world_to_screen: impl Fn(egui::Pos2) -> egui::Pos2,
        world_visible: egui::Rect,
        zoom_screen_per_world_px: f32,
    ) {
        if !self.visible || self.points.is_empty() {
            return;
        }

        let radius_px =
            Self::effective_radius_points(self.style.radius_screen_px, zoom_screen_per_world_px);
        let world_margin = radius_px / zoom_screen_per_world_px.max(1e-6);
        let world_visible = world_visible.expand(world_margin);

        let mut shapes = Vec::new();
        shapes.reserve(self.points.len().min(50_000) * 2);

        for p in &self.points {
            if !world_visible.contains(p.world_lvl0) {
                continue;
            }
            let screen = world_to_screen(p.world_lvl0);
            if !viewport.expand(radius_px + 2.0).contains(screen) {
                continue;
            }

            let fill = if p.positive {
                self.style.fill_positive
            } else {
                self.style.fill_negative
            };
            shapes.push(egui::Shape::circle_filled(screen, radius_px, fill));
            let stroke = if p.positive {
                self.style.stroke_positive
            } else {
                self.style.stroke_negative
            };
            if stroke.width > 0.0 && stroke.color != egui::Color32::TRANSPARENT {
                shapes.push(egui::Shape::circle_stroke(screen, radius_px, stroke));
            }
        }

        painter.extend(shapes);
    }
}
