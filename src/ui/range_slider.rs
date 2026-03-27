use eframe::egui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveHandle {
    Low,
    High,
}

impl Default for ActiveHandle {
    fn default() -> Self {
        Self::Low
    }
}

fn value_to_x(value: f32, min: f32, max: f32, rect: egui::Rect) -> f32 {
    let t = if max > min {
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    egui::lerp(rect.left()..=rect.right(), t)
}

fn x_to_value(x: f32, min: f32, max: f32, rect: egui::Rect) -> f32 {
    let t = if rect.width() > 0.0 {
        ((x - rect.left()) / rect.width()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    egui::lerp(min..=max, t)
}

/// A compact two-handle "bookend" slider for contrast limits.
///
/// - `lo` and `hi` are clamped to `[min,max]`
/// - `hi` is kept `>= lo + min_separation`
pub fn range_slider(
    ui: &mut egui::Ui,
    id: egui::Id,
    lo: &mut f32,
    hi: &mut f32,
    min: f32,
    max: f32,
    min_separation: f32,
) -> egui::Response {
    let desired_size = egui::vec2(170.0, 18.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click_and_drag());
    response = response.on_hover_text(format!(
        "min {:.0}\nmax {:.0}",
        (*lo).clamp(min, max),
        (*hi).clamp(min, max)
    ));

    if !ui.is_rect_visible(rect) {
        return response;
    }

    let min_sep = min_separation.max(0.0);
    let mut lo_v = (*lo).clamp(min, max);
    let mut hi_v = (*hi).clamp(min, max);
    if hi_v < lo_v + min_sep {
        hi_v = (lo_v + min_sep).min(max);
        lo_v = lo_v.min(hi_v - min_sep);
    }

    let visuals = ui.style().interact(&response);
    let bg = ui.visuals().extreme_bg_color;
    let track_bg = ui.visuals().widgets.inactive.bg_fill;
    let sel_fill = ui.visuals().selection.bg_fill;

    let painter = ui.painter();
    painter.rect_filled(rect, 4.0, bg);

    let track = rect.shrink2(egui::vec2(6.0, 6.0));
    let track = egui::Rect::from_min_max(
        egui::pos2(track.min.x, rect.center().y - 3.0),
        egui::pos2(track.max.x, rect.center().y + 3.0),
    );
    painter.rect_filled(track, 3.0, track_bg);

    let x_lo = value_to_x(lo_v, min, max, track);
    let x_hi = value_to_x(hi_v, min, max, track);
    let (x0, x1) = if x_lo <= x_hi {
        (x_lo, x_hi)
    } else {
        (x_hi, x_lo)
    };
    let sel = egui::Rect::from_min_max(egui::pos2(x0, track.min.y), egui::pos2(x1, track.max.y));
    painter.rect_filled(sel, 3.0, sel_fill);

    let handle_w = 6.0;
    let handle_h = rect.height() - 4.0;
    let handle = |x: f32| {
        egui::Rect::from_center_size(
            egui::pos2(x, rect.center().y),
            egui::vec2(handle_w, handle_h),
        )
    };
    let h_lo = handle(x_lo);
    let h_hi = handle(x_hi);
    painter.rect_filled(h_lo, 2.0, visuals.fg_stroke.color);
    painter.rect_filled(h_hi, 2.0, visuals.fg_stroke.color);

    let active_id = id.with("active_handle");
    if response.drag_started() {
        let pointer_x = ui.input(|i| i.pointer.interact_pos()).map(|p| p.x);
        let which = pointer_x.map(|x| {
            let d_lo = (x - x_lo).abs();
            let d_hi = (x - x_hi).abs();
            if d_lo <= d_hi {
                ActiveHandle::Low
            } else {
                ActiveHandle::High
            }
        });
        if let Some(which) = which {
            ui.data_mut(|d| d.insert_temp(active_id, which));
        }
    }

    if response.dragged() {
        let which = ui
            .data(|d| d.get_temp::<ActiveHandle>(active_id))
            .unwrap_or(ActiveHandle::High);
        if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
            let v = x_to_value(pos.x, min, max, track);
            match which {
                ActiveHandle::Low => {
                    lo_v = v.min(hi_v - min_sep).clamp(min, max);
                }
                ActiveHandle::High => {
                    hi_v = v.max(lo_v + min_sep).clamp(min, max);
                }
            }
        }
    }

    if response.drag_stopped() {
        ui.data_mut(|d| {
            d.remove_temp::<ActiveHandle>(active_id);
        });
    }

    if (lo_v - *lo).abs() > f32::EPSILON || (hi_v - *hi).abs() > f32::EPSILON {
        *lo = lo_v;
        *hi = hi_v;
        response.mark_changed();
    }

    response
}
