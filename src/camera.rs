use eframe::egui;

#[derive(Debug, Clone)]
pub struct Camera {
    pub center_world_lvl0: egui::Pos2,
    pub zoom_screen_per_lvl0_px: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center_world_lvl0: egui::pos2(0.0, 0.0),
            zoom_screen_per_lvl0_px: 1.0,
        }
    }
}

impl Camera {
    const MIN_ZOOM_SCREEN_PER_WORLD: f32 = 0.000_01;

    pub fn world_to_screen(&self, world_lvl0: egui::Pos2, viewport: egui::Rect) -> egui::Pos2 {
        let screen_center = viewport.center();
        let delta = world_lvl0 - self.center_world_lvl0;
        screen_center + delta * self.zoom_screen_per_lvl0_px
    }

    pub fn screen_to_world(&self, screen: egui::Pos2, viewport: egui::Rect) -> egui::Pos2 {
        let screen_center = viewport.center();
        let delta = screen - screen_center;
        self.center_world_lvl0 + delta / self.zoom_screen_per_lvl0_px
    }

    pub fn pan_by_screen_delta(&mut self, screen_delta: egui::Vec2) {
        self.center_world_lvl0 -= screen_delta / self.zoom_screen_per_lvl0_px;
    }

    pub fn zoom_about_screen_point(
        &mut self,
        viewport: egui::Rect,
        screen_point: egui::Pos2,
        zoom_factor: f32,
    ) {
        if !zoom_factor.is_finite() || zoom_factor <= 0.0 {
            return;
        }

        let world_before = self.screen_to_world(screen_point, viewport);
        self.zoom_screen_per_lvl0_px = (self.zoom_screen_per_lvl0_px * zoom_factor)
            .clamp(Self::MIN_ZOOM_SCREEN_PER_WORLD, 5000.0);
        let world_after = self.screen_to_world(screen_point, viewport);
        let correction = world_before - world_after;
        self.center_world_lvl0 += correction;
    }

    pub fn fit_to_world_rect(&mut self, viewport: egui::Rect, world_rect_lvl0: egui::Rect) {
        let world_w = world_rect_lvl0.width().max(1.0);
        let world_h = world_rect_lvl0.height().max(1.0);
        let scale_x = viewport.width() / world_w;
        let scale_y = viewport.height() / world_h;
        self.zoom_screen_per_lvl0_px =
            (scale_x.min(scale_y) * 0.95).clamp(Self::MIN_ZOOM_SCREEN_PER_WORLD, 5000.0);
        self.center_world_lvl0 = world_rect_lvl0.center();
    }
}
