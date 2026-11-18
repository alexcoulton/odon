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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_pos_close(actual: egui::Pos2, expected: egui::Pos2) {
        let tolerance = 1.0e-4;
        assert!(
            (actual.x - expected.x).abs() <= tolerance
                && (actual.y - expected.y).abs() <= tolerance,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn viewport() -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(100.0, 50.0), egui::vec2(800.0, 600.0))
    }

    #[test]
    fn world_and_screen_transforms_round_trip() {
        let camera = Camera {
            center_world_lvl0: egui::pos2(250.0, 400.0),
            zoom_screen_per_lvl0_px: 2.5,
        };
        let world = egui::pos2(310.0, 372.0);

        let screen = camera.world_to_screen(world, viewport());
        assert_pos_close(camera.screen_to_world(screen, viewport()), world);
        assert_pos_close(
            camera.world_to_screen(camera.center_world_lvl0, viewport()),
            viewport().center(),
        );
    }

    #[test]
    fn panning_uses_screen_delta_at_current_zoom() {
        let mut camera = Camera {
            center_world_lvl0: egui::pos2(100.0, 200.0),
            zoom_screen_per_lvl0_px: 4.0,
        };

        camera.pan_by_screen_delta(egui::vec2(40.0, -20.0));

        assert_pos_close(camera.center_world_lvl0, egui::pos2(90.0, 205.0));
    }

    #[test]
    fn zoom_keeps_world_position_under_pointer_fixed() {
        let mut camera = Camera {
            center_world_lvl0: egui::pos2(250.0, 400.0),
            zoom_screen_per_lvl0_px: 1.25,
        };
        let pointer = egui::pos2(725.0, 215.0);
        let before = camera.screen_to_world(pointer, viewport());

        camera.zoom_about_screen_point(viewport(), pointer, 3.0);

        assert_eq!(camera.zoom_screen_per_lvl0_px, 3.75);
        assert_pos_close(camera.screen_to_world(pointer, viewport()), before);
    }

    #[test]
    fn zoom_rejects_invalid_factors_and_clamps_extremes() {
        let initial = Camera {
            center_world_lvl0: egui::pos2(12.0, 34.0),
            zoom_screen_per_lvl0_px: 2.0,
        };

        for factor in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut camera = initial.clone();
            camera.zoom_about_screen_point(viewport(), viewport().center(), factor);
            assert_pos_close(camera.center_world_lvl0, initial.center_world_lvl0);
            assert_eq!(
                camera.zoom_screen_per_lvl0_px,
                initial.zoom_screen_per_lvl0_px
            );
        }

        let mut camera = initial.clone();
        camera.zoom_about_screen_point(viewport(), viewport().center(), 1.0e20);
        assert_eq!(camera.zoom_screen_per_lvl0_px, 5000.0);

        camera.zoom_about_screen_point(viewport(), viewport().center(), 1.0e-20);
        assert_eq!(
            camera.zoom_screen_per_lvl0_px,
            Camera::MIN_ZOOM_SCREEN_PER_WORLD
        );
    }

    #[test]
    fn fit_centers_world_rect_and_uses_limiting_viewport_dimension() {
        let mut camera = Camera::default();
        let world = egui::Rect::from_min_size(egui::pos2(50.0, 75.0), egui::vec2(400.0, 100.0));

        camera.fit_to_world_rect(viewport(), world);

        assert_pos_close(camera.center_world_lvl0, world.center());
        assert!((camera.zoom_screen_per_lvl0_px - 1.9).abs() <= 1.0e-6);

        let screen_min = camera.world_to_screen(world.left_top(), viewport());
        let screen_max = camera.world_to_screen(world.right_bottom(), viewport());
        assert!(screen_min.x >= viewport().left());
        assert!(screen_min.y >= viewport().top());
        assert!(screen_max.x <= viewport().right());
        assert!(screen_max.y <= viewport().bottom());
    }
}

/// Renderer-side ownership of the camera while a native pan/zoom gesture is in progress.
///
/// Camera gestures are applied locally first and then reported to the control actor. The actor's
/// projection of an earlier gesture step can arrive after the renderer has already moved on, so
/// while a hold is active the local camera stays authoritative and projected cameras are only
/// observed. At most one camera command is kept in flight; newer positions replace the unsent one.
/// Once the actor has caught up (or stopped answering) the hold settles and any differing camera
/// the actor published in the meantime (for example from a remote client) is handed back.
#[derive(Debug, Clone)]
pub struct CameraIntentHold {
    camera: Camera,
    unsent: bool,
    last_local_change: std::time::Instant,
    matched: bool,
    suppressed: Option<Camera>,
}

impl CameraIntentHold {
    /// Quiet period after the last local change before the actor may own the camera again.
    const SETTLE: std::time::Duration = std::time::Duration::from_millis(150);
    /// Upper bound on waiting for the actor to echo the local camera.
    const ECHO_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);
    /// How soon the renderer should look at an active hold again.
    pub const POLL: std::time::Duration = std::time::Duration::from_millis(16);

    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            unsent: true,
            last_local_change: std::time::Instant::now(),
            matched: false,
            suppressed: None,
        }
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn record_local(&mut self, camera: Camera) {
        self.camera = camera;
        self.unsent = true;
        self.matched = false;
        self.last_local_change = std::time::Instant::now();
    }

    /// Returns the camera to submit when nothing is in flight and the latest value is unsent.
    pub fn take_unsent(&mut self, in_flight: bool) -> Option<Camera> {
        if !self.unsent || in_flight {
            return None;
        }
        self.unsent = false;
        Some(self.camera.clone())
    }

    /// Marks the latest camera as unsent again after a failed submission.
    pub fn mark_unsent(&mut self) {
        self.unsent = true;
    }

    /// Records the actor's camera and returns the camera the renderer should keep showing.
    pub fn observe_projected(&mut self, projected: &Camera) -> Camera {
        if Self::same(projected, &self.camera) {
            self.matched = !self.unsent;
            self.suppressed = None;
        } else {
            self.matched = false;
            self.suppressed = Some(projected.clone());
        }
        self.camera.clone()
    }

    pub fn settled(&self, in_flight: bool) -> bool {
        if self.unsent || in_flight {
            return false;
        }
        let quiet = self.last_local_change.elapsed();
        quiet >= Self::SETTLE && (self.matched || quiet >= Self::ECHO_TIMEOUT)
    }

    /// Whether the renderer has to wake itself to finish this hold: either a newer camera is
    /// waiting for the in-flight command to be answered, or a hidden actor camera must be adopted
    /// once the hold settles. Otherwise the hold is released lazily on the next frame or
    /// projection, so an idle hold never keeps the renderer repainting.
    pub fn needs_wake(&self) -> bool {
        self.unsent || self.suppressed.is_some()
    }

    /// Ends the hold, returning an actor camera that was hidden while it was active.
    pub fn finish(self) -> Option<Camera> {
        self.suppressed
    }

    fn same(a: &Camera, b: &Camera) -> bool {
        let close = |x: f32, y: f32| (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1.0);
        close(a.center_world_lvl0.x, b.center_world_lvl0.x)
            && close(a.center_world_lvl0.y, b.center_world_lvl0.y)
            && close(a.zoom_screen_per_lvl0_px, b.zoom_screen_per_lvl0_px)
    }
}

#[cfg(test)]
mod camera_intent_hold_tests {
    use super::*;

    fn camera(x: f32) -> Camera {
        Camera {
            center_world_lvl0: egui::pos2(x, 0.0),
            zoom_screen_per_lvl0_px: 1.0,
        }
    }

    #[test]
    fn stale_projection_never_replaces_the_local_camera_and_one_command_is_in_flight() {
        let mut hold = CameraIntentHold::new(camera(1.0));
        assert_eq!(hold.take_unsent(false).unwrap().center_world_lvl0.x, 1.0);
        hold.record_local(camera(2.0));
        assert!(hold.take_unsent(true).is_none());
        hold.record_local(camera(3.0));
        // The projection of the first step arrives late.
        assert_eq!(hold.observe_projected(&camera(1.0)).center_world_lvl0.x, 3.0);
        assert!(!hold.settled(false));
        // Only the newest position is sent once the first command has been answered.
        assert_eq!(hold.take_unsent(false).unwrap().center_world_lvl0.x, 3.0);
        assert!(hold.take_unsent(false).is_none());
        assert_eq!(hold.observe_projected(&camera(3.0)).center_world_lvl0.x, 3.0);
        hold.last_local_change -= CameraIntentHold::SETTLE;
        assert!(!hold.settled(true));
        assert!(hold.settled(false));
        assert!(hold.finish().is_none());
    }

    #[test]
    fn differing_actor_camera_is_returned_when_the_hold_ends() {
        let mut hold = CameraIntentHold::new(camera(1.0));
        hold.take_unsent(false);
        hold.observe_projected(&camera(9.0));
        hold.last_local_change -= CameraIntentHold::SETTLE;
        assert!(!hold.settled(false));
        hold.last_local_change -= CameraIntentHold::ECHO_TIMEOUT;
        assert!(hold.settled(false));
        assert_eq!(hold.finish().unwrap().center_world_lvl0.x, 9.0);
    }
}
