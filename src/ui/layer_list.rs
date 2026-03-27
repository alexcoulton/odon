use eframe::egui;

use crate::ui::icons::{Icon, paint_icon_in_rect};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerGroup {
    Overlays,
    Channels,
}

#[derive(Debug, Clone)]
pub struct LayerDragState<K: Copy + Eq> {
    pub group: LayerGroup,
    pub from: usize,
    pub dragged: K,
    pub hovering: Option<usize>,
    pub insert_after: bool,
}

pub struct LayerRowOptions {
    pub available: bool,
    pub selected: bool,
    pub icon: Icon,
    pub visible: Option<bool>,
    pub color_rgb: Option<[u8; 3]>,
}

#[derive(Debug, Clone)]
pub struct LayerRowResponse {
    pub selected_clicked: bool,
    pub changed: bool,
    pub visible_changed: Option<bool>,
    pub color_rgb_changed: Option<[u8; 3]>,
    pub row_response: egui::Response,
}

pub fn reorder_vec<T>(v: &mut Vec<T>, from: usize, to: usize) {
    if from >= v.len() || to > v.len() {
        return;
    }
    if from == to || from + 1 == to {
        return;
    }
    let item = v.remove(from);
    let insert_at = if to > from { to.saturating_sub(1) } else { to };
    v.insert(insert_at, item);
}

pub fn ui_layer_row<K: Copy + Eq + std::hash::Hash>(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    drag: &mut Option<LayerDragState<K>>,
    group: LayerGroup,
    index: usize,
    id: K,
    name: &str,
    opts: LayerRowOptions,
) -> LayerRowResponse {
    let is_dragged = drag
        .as_ref()
        .is_some_and(|d| d.group == group && d.dragged == id);

    let height = 28.0;
    let width = ui.available_width().max(1.0);
    let (rect, _reserve) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let group_tag: u8 = match group {
        LayerGroup::Overlays => 0,
        LayerGroup::Channels => 1,
    };
    let row_id = ui.id().with(("layer-row", group_tag, id, index));
    let row = ui.interact(rect, row_id, egui::Sense::click_and_drag());

    // Background.
    if ui.is_rect_visible(rect) {
        let rounding = egui::CornerRadius::same(4);
        if is_dragged {
            ui.painter()
                .rect_filled(rect, rounding, egui::Color32::from_gray(20));
            ui.painter().rect_stroke(
                rect,
                rounding,
                egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
                egui::StrokeKind::Inside,
            );
        } else {
            let mut visuals = ui.style().interact(&row).clone();
            if opts.selected {
                visuals.bg_fill = ui.visuals().selection.bg_fill;
                visuals.bg_stroke = ui.visuals().selection.stroke;
            } else if row.hovered() {
                visuals.bg_fill = ui.visuals().widgets.hovered.bg_fill;
                visuals.bg_stroke = ui.visuals().widgets.hovered.bg_stroke;
            } else {
                visuals.bg_fill = ui.visuals().widgets.inactive.bg_fill;
                visuals.bg_stroke = ui.visuals().widgets.inactive.bg_stroke;
            }
            ui.painter().rect_filled(rect, rounding, visuals.bg_fill);
            ui.painter()
                .rect_stroke(rect, rounding, visuals.bg_stroke, egui::StrokeKind::Inside);
        }
    }

    // Start drag from anywhere on the row.
    if row.drag_started() && drag.is_none() && !is_dragged {
        *drag = Some(LayerDragState {
            group,
            from: index,
            dragged: id,
            hovering: Some(index),
            insert_after: false,
        });
    }

    let mut changed = false;
    let mut selected_clicked = false;
    let mut visible_changed: Option<bool> = None;
    let mut color_rgb_changed: Option<[u8; 3]> = None;
    let mut checkbox_rect: Option<egui::Rect> = None;
    let mut menu_response = row.clone();

    if !is_dragged {
        let mut row_ui = ui.child_ui(
            rect.shrink2(egui::vec2(6.0, 0.0)),
            egui::Layout::left_to_right(egui::Align::Center),
            None,
        );
        row_ui.add_enabled_ui(opts.available, |ui| {
            // Layer type icon.
            let (icon_rect, _) =
                ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
            if ui.is_rect_visible(icon_rect) {
                let mut c = ui.visuals().widgets.inactive.fg_stroke.color;
                if opts.selected {
                    c = ui.visuals().selection.stroke.color;
                }
                if !opts.available {
                    c = egui::Color32::from_gray(120);
                }
                paint_icon_in_rect(
                    ui.ctx(),
                    ui.painter(),
                    icon_rect.shrink(1.0),
                    opts.icon,
                    egui::Stroke::new(1.4, c),
                );
            }

            if let Some(mut vis) = opts.visible {
                let vis_resp = ui.checkbox(&mut vis, "");
                checkbox_rect = Some(vis_resp.rect);
                if vis_resp.changed() {
                    changed = true;
                    visible_changed = Some(vis);
                }
            } else {
                let mut dummy = false;
                ui.add_enabled(false, egui::Checkbox::new(&mut dummy, ""));
            }

            // Prevent the text-selection cursor and avoid capturing right-clicks on the label.
            ui.add(egui::Label::new(name).selectable(false));

            if let Some(mut rgb) = opts.color_rgb {
                let color_resp = ui.color_edit_button_srgb(&mut rgb);
                // Include the color button in the "row response" used for context menus etc.
                // (But intentionally exclude the visibility checkbox.)
                menu_response = menu_response.union(color_resp.clone());
                if color_resp.changed() {
                    changed = true;
                    color_rgb_changed = Some(rgb);
                }
            }
        });
    }

    // Make the whole row selectable (not just the label text), but exclude the visibility checkbox.
    // We use pointer position instead of `row.hovered()` so clicks on child widgets still count.
    if !is_dragged
        && ctx.input(|i| i.pointer.button_clicked(egui::PointerButton::Primary))
        && ctx
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| rect.contains(p) && !checkbox_rect.is_some_and(|r| r.contains(p)))
    {
        selected_clicked = true;
    }

    if let Some(drag) = drag.as_mut() {
        if drag.group == group
            && ctx
                .input(|i| i.pointer.hover_pos())
                .is_some_and(|p| rect.contains(p))
        {
            drag.hovering = Some(index);
            if let Some(p) = ctx.input(|i| i.pointer.hover_pos()) {
                drag.insert_after = p.y >= rect.center().y;
            }
        }
    }

    // Drop indicator (insertion line).
    if let Some(drag) = drag.as_ref() {
        if drag.group == group && drag.hovering == Some(index) && ui.is_rect_visible(rect) {
            let y = if drag.insert_after {
                rect.bottom()
            } else {
                rect.top()
            };
            ui.painter().line_segment(
                [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 200, 255)),
            );
        }
    }

    LayerRowResponse {
        selected_clicked,
        changed,
        visible_changed,
        color_rgb_changed,
        row_response: menu_response,
    }
}

pub fn paint_drag_preview<K: Copy + Eq>(
    ctx: &egui::Context,
    drag: Option<&LayerDragState<K>>,
    display_name: impl Fn(K) -> String,
) {
    let Some(drag) = drag else {
        return;
    };
    let Some(pointer) = ctx.input(|i| i.pointer.hover_pos()) else {
        return;
    };
    ctx.set_cursor_icon(egui::CursorIcon::Grabbing);

    let name = display_name(drag.dragged);
    let font = egui::FontId::proportional(13.0);
    let galley = ctx.fonts_mut(|f| {
        f.layout_no_wrap(
            name.to_string(),
            font.clone(),
            egui::Color32::from_gray(235),
        )
    });

    let pad = egui::vec2(10.0, 6.0);
    let size = galley.size() + pad * 2.0;
    let mut rect = egui::Rect::from_min_size(pointer + egui::vec2(16.0, 16.0), size);

    // Keep it roughly on-screen.
    let screen = ctx.content_rect();
    if rect.right() > screen.right() {
        rect = rect.translate(egui::vec2(screen.right() - rect.right(), 0.0));
    }
    if rect.bottom() > screen.bottom() {
        rect = rect.translate(egui::vec2(0.0, screen.bottom() - rect.bottom()));
    }

    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::new("layer-drag-preview"),
    ));

    let rounding = egui::CornerRadius::same(6);
    let shadow = rect.translate(egui::vec2(0.0, 6.0)).expand(2.0);
    painter.rect_filled(
        shadow,
        rounding,
        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 140),
    );
    painter.rect_filled(rect, rounding, egui::Color32::from_rgb(34, 34, 40));
    painter.rect_stroke(
        rect,
        rounding,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(90, 120, 190)),
        egui::StrokeKind::Inside,
    );
    painter.galley(rect.min + pad, galley, egui::Color32::from_gray(235));
}

pub fn finish_drag_if_released<K: Copy + Eq>(
    ctx: &egui::Context,
    drag: &mut Option<LayerDragState<K>>,
    mut on_drop: impl FnMut(LayerGroup, usize, usize),
) {
    if drag.is_none() {
        return;
    }
    if !ctx.input(|i| i.pointer.any_released()) {
        return;
    }

    let Some(drag_state) = drag.take() else {
        return;
    };
    let Some(hover) = drag_state.hovering else {
        return;
    };
    let to = if drag_state.insert_after {
        hover.saturating_add(1)
    } else {
        hover
    };
    on_drop(drag_state.group, drag_state.from, to);
}
