//! Category legend shared by the single-image and mosaic object panels, plus the project-wide
//! label-to-colour map it edits.

use std::collections::BTreeMap;

use eframe::egui;

use super::ObjectColorLegendEntry;

const INLINE_HEIGHT: f32 = 500.0;
const LABEL_MIN_WIDTH: f32 = 70.0;
const CHECKBOX_WIDTH: f32 = 20.0;
const SWATCH_WIDTH: f32 = 30.0;
const WINDOW_MIN_WIDTH: f32 = 320.0;
/// Project updates arrive asynchronously, so a committed edit is held until the project reports
/// it. The limit stops a rejected command from freezing the local colours.
const COMMIT_WAIT_SYNCS: u32 = 120;

const DISTINCT_PALETTE: [[u8; 3]; 20] = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [188, 189, 34],
    [23, 190, 207],
    [174, 199, 232],
    [255, 187, 120],
    [152, 223, 138],
    [255, 152, 150],
    [197, 176, 213],
    [196, 156, 148],
    [247, 182, 210],
    [219, 219, 141],
    [158, 218, 229],
    [127, 127, 127],
    [199, 199, 199],
];
const COLORBLIND_PALETTE: [[u8; 3]; 8] = [
    [0, 114, 178],
    [230, 159, 0],
    [0, 158, 115],
    [204, 121, 167],
    [86, 180, 233],
    [213, 94, 0],
    [240, 228, 66],
    [150, 150, 150],
];
const PALETTES: [(&str, &[[u8; 3]]); 2] = [
    ("Distinct", &DISTINCT_PALETTE),
    ("Colour-blind safe", &COLORBLIND_PALETTE),
];

/// Local copy of the project's category colours. Edits show immediately and are committed to the
/// project once the pointer is released, so dragging in a colour picker is not one project command
/// per frame.
#[derive(Debug, Clone, Default)]
pub(crate) struct CategoryColors {
    colors: BTreeMap<String, [u8; 3]>,
    uncommitted: bool,
    awaiting: Option<(BTreeMap<String, [u8; 3]>, u32)>,
}

impl CategoryColors {
    pub(crate) fn get(&self, label: &str) -> Option<[u8; 3]> {
        self.colors.get(label).copied()
    }

    pub(crate) fn colors(&self) -> &BTreeMap<String, [u8; 3]> {
        &self.colors
    }

    pub(crate) fn sync(&mut self, project: &BTreeMap<String, [u8; 3]>) {
        if self.uncommitted {
            return;
        }
        if let Some((expected, remaining)) = self.awaiting.as_mut() {
            if expected != project && *remaining > 0 {
                *remaining -= 1;
                return;
            }
            self.awaiting = None;
        }
        if &self.colors != project {
            self.colors.clone_from(project);
        }
    }

    pub(crate) fn set(&mut self, label: &str, rgb: [u8; 3]) {
        self.colors.insert(label.to_string(), rgb);
        self.uncommitted = true;
    }

    pub(crate) fn remove(&mut self, label: &str) {
        self.colors.remove(label);
        self.uncommitted = true;
    }

    pub(crate) fn take_commit(&mut self, pointer_down: bool) -> Option<BTreeMap<String, [u8; 3]>> {
        if !self.uncommitted || pointer_down {
            return None;
        }
        self.uncommitted = false;
        self.awaiting = Some((self.colors.clone(), COMMIT_WAIT_SYNCS));
        Some(self.colors.clone())
    }
}

pub(crate) fn category_colors_params(colors: &BTreeMap<String, [u8; 3]>) -> serde_json::Value {
    serde_json::json!({ "category_colors": colors })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LegendRow {
    pub label: String,
    pub count: usize,
    pub color_rgb: [u8; 3],
    pub visible: bool,
    pub custom_color: bool,
}

/// Builds display rows. A colour set for this view wins over the project-wide colour, which wins
/// over the hashed default in `entry`.
pub(crate) fn legend_rows(
    entries: Vec<ObjectColorLegendEntry>,
    colors: &CategoryColors,
    view_style: impl Fn(&str) -> (bool, Option<[u8; 3]>),
) -> Vec<LegendRow> {
    entries
        .into_iter()
        .map(|entry| {
            let (visible, view_color) = view_style(&entry.value_label);
            let project_color = colors.get(&entry.value_label);
            LegendRow {
                count: entry.count,
                color_rgb: view_color.or(project_color).unwrap_or(entry.color_rgb),
                visible,
                custom_color: view_color.is_some() || project_color.is_some(),
                label: entry.value_label,
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LegendAction {
    SetVisible { label: String, visible: bool },
    SetColor { label: String, rgb: [u8; 3] },
    ResetColor { label: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LegendSort {
    #[default]
    Count,
    Name,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct LegendUiState {
    popped_out: bool,
    sort: LegendSort,
}

fn sorted_rows(rows: &[LegendRow], sort: LegendSort) -> Vec<&LegendRow> {
    let mut sorted = rows.iter().collect::<Vec<_>>();
    match sort {
        LegendSort::Count => {
            sorted.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));
        }
        LegendSort::Name => {
            sorted.sort_by_key(|row| row.label.to_lowercase());
        }
    }
    sorted
}

fn visibility_actions(
    rows: &[LegendRow],
    visible: impl Fn(&LegendRow) -> bool,
) -> Vec<LegendAction> {
    rows.iter()
        .filter(|row| row.visible != visible(row))
        .map(|row| LegendAction::SetVisible {
            label: row.label.clone(),
            visible: visible(row),
        })
        .collect()
}

/// Assigned in name order, so the same categories get the same colours whatever the current sort.
fn palette_actions(rows: &[LegendRow], palette: &[[u8; 3]]) -> Vec<LegendAction> {
    sorted_rows(rows, LegendSort::Name)
        .into_iter()
        .zip(palette.iter().cycle())
        .map(|(row, rgb)| LegendAction::SetColor {
            label: row.label.clone(),
            rgb: *rgb,
        })
        .collect()
}

fn ui_toolbar(
    ui: &mut egui::Ui,
    id_salt: &str,
    rows: &[LegendRow],
    state: &mut LegendUiState,
    actions: &mut Vec<LegendAction>,
) {
    ui.horizontal_wrapped(|ui| {
        if ui.button("All").clicked() {
            actions.extend(visibility_actions(rows, |_| true));
        }
        if ui.button("None").clicked() {
            actions.extend(visibility_actions(rows, |_| false));
        }
        if ui.button("Invert").clicked() {
            actions.extend(visibility_actions(rows, |row| !row.visible));
        }
        egui::ComboBox::from_id_salt((id_salt, "sort"))
            .selected_text(match state.sort {
                LegendSort::Count => "Sort: count",
                LegendSort::Name => "Sort: name",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.sort, LegendSort::Count, "Count");
                ui.selectable_value(&mut state.sort, LegendSort::Name, "Name");
            });
        ui.menu_button("Colours", |ui| {
            for (name, palette) in PALETTES {
                if ui.button(name).clicked() {
                    actions.extend(palette_actions(rows, palette));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Reset all").clicked() {
                actions.extend(rows.iter().filter(|row| row.custom_color).map(|row| {
                    LegendAction::ResetColor {
                        label: row.label.clone(),
                    }
                }));
                ui.close();
            }
        })
        .response
        .on_hover_text("Colours apply to matching labels across the whole project.");
        let toggle = if state.popped_out { "Dock" } else { "Pop out" };
        if ui.button(toggle).clicked() {
            state.popped_out = !state.popped_out;
        }
    });
}

/// Column widths for one legend row. Everything is sized to fit `available`, dropping the least
/// important columns when narrow, so a row can never run past the edge of its container.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RowColumns {
    label: f32,
    count: Option<f32>,
    percent: Option<f32>,
    only: f32,
    reset: f32,
}

fn row_columns(
    available: f32,
    item_spacing: f32,
    count: f32,
    percent: f32,
    only: f32,
    reset: f32,
) -> RowColumns {
    let required = CHECKBOX_WIDTH + SWATCH_WIDTH + only + reset + item_spacing * 4.0;
    let with_count = required + count + item_spacing;
    let with_percent = with_count + percent + item_spacing;
    let (used, count, percent) = if available - with_percent >= LABEL_MIN_WIDTH {
        (with_percent, Some(count), Some(percent))
    } else if available - with_count >= LABEL_MIN_WIDTH {
        (with_count, Some(count), None)
    } else {
        (required, None, None)
    };
    RowColumns {
        label: (available - used).max(LABEL_MIN_WIDTH),
        count,
        percent,
        only,
        reset,
    }
}

fn text_width(ui: &egui::Ui, text: &str, style: egui::TextStyle) -> f32 {
    egui::WidgetText::from(text)
        .into_galley(ui, None, f32::INFINITY, style)
        .size()
        .x
}

fn cell<R>(
    ui: &mut egui::Ui,
    width: f32,
    align: egui::Align,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let layout = if align == egui::Align::Max {
        egui::Layout::right_to_left(egui::Align::Center)
    } else {
        egui::Layout::left_to_right(egui::Align::Center)
    };
    ui.allocate_ui_with_layout(
        egui::vec2(width, ui.spacing().interact_size.y),
        layout,
        |ui| {
            ui.set_width(width);
            add(ui)
        },
    )
    .inner
}

fn ui_rows(
    ui: &mut egui::Ui,
    rows: &[LegendRow],
    sort: LegendSort,
    actions: &mut Vec<LegendAction>,
) {
    let total = rows.iter().map(|row| row.count).sum::<usize>().max(1);
    let percent_text = |row: &LegendRow| format!("{:.1}%", row.count as f64 * 100.0 / total as f64);
    let max_text_width = |text: &dyn Fn(&LegendRow) -> String| {
        rows.iter()
            .map(|row| text_width(ui, &text(row), egui::TextStyle::Body))
            .fold(0.0, f32::max)
    };
    let button_padding = ui.spacing().button_padding.x * 2.0;
    let columns = row_columns(
        ui.available_width(),
        ui.spacing().item_spacing.x,
        max_text_width(&|row| row.count.to_string()),
        max_text_width(&percent_text),
        text_width(ui, "Only", egui::TextStyle::Button) + button_padding,
        text_width(ui, "Reset", egui::TextStyle::Button) + button_padding,
    );
    let button_height = ui.spacing().interact_size.y;

    for row in sorted_rows(rows, sort) {
        ui.horizontal(|ui| {
            cell(ui, CHECKBOX_WIDTH, egui::Align::Min, |ui| {
                let mut visible = row.visible;
                if ui.add(egui::Checkbox::without_text(&mut visible)).changed() {
                    actions.push(LegendAction::SetVisible {
                        label: row.label.clone(),
                        visible,
                    });
                }
            });
            cell(ui, SWATCH_WIDTH, egui::Align::Min, |ui| {
                let [r, g, b] = row.color_rgb;
                let mut color = egui::Color32::from_rgb(r, g, b);
                ui.spacing_mut().interact_size = egui::vec2(SWATCH_WIDTH, 18.0);
                if ui.color_edit_button_srgba(&mut color).changed() {
                    actions.push(LegendAction::SetColor {
                        label: row.label.clone(),
                        rgb: [color.r(), color.g(), color.b()],
                    });
                }
            });
            cell(ui, columns.label, egui::Align::Min, |ui| {
                ui.add(egui::Label::new(row.label.as_str()).truncate())
                    .on_hover_text(row.label.as_str());
            });
            if let Some(width) = columns.count {
                cell(ui, width, egui::Align::Max, |ui| {
                    ui.label(row.count.to_string());
                });
            }
            if let Some(width) = columns.percent {
                cell(ui, width, egui::Align::Max, |ui| {
                    ui.weak(percent_text(row));
                });
            }
            if ui
                .add_sized(
                    [columns.only, button_height],
                    egui::Button::new("Only").small(),
                )
                .on_hover_text("Show only this category")
                .clicked()
            {
                actions.extend(visibility_actions(rows, |other| other.label == row.label));
            }
            if ui
                .add_enabled_ui(row.custom_color, |ui| {
                    ui.add_sized(
                        [columns.reset, button_height],
                        egui::Button::new("Reset").small(),
                    )
                })
                .inner
                .on_hover_text("Return to the default colour")
                .clicked()
            {
                actions.push(LegendAction::ResetColor {
                    label: row.label.clone(),
                });
            }
        });
    }
}

fn ui_contents(
    ui: &mut egui::Ui,
    id_salt: &str,
    rows: &[LegendRow],
    state: &mut LegendUiState,
    actions: &mut Vec<LegendAction>,
    inline: bool,
) {
    ui_toolbar(ui, id_salt, rows, state, actions);
    let scroll = egui::ScrollArea::vertical().id_salt((id_salt, "scroll"));
    // A nested scroll area otherwise collapses to egui's small default height.
    let scroll = if inline {
        scroll
            .min_scrolled_height(INLINE_HEIGHT)
            .max_height(INLINE_HEIGHT)
            .auto_shrink([false, true])
    } else {
        scroll.auto_shrink([false, false])
    };
    scroll.show(ui, |ui| ui_rows(ui, rows, state.sort, actions));
}

/// Draws a collapsible legend, or a floating window when popped out, and returns requested edits.
pub(crate) fn ui_legend(
    ui: &mut egui::Ui,
    id_salt: &str,
    property: &str,
    rows: &[LegendRow],
    state: &mut LegendUiState,
) -> Vec<LegendAction> {
    let mut actions = Vec::new();
    let title = format!("Legend: {property}");
    egui::CollapsingHeader::new(title.as_str())
        .id_salt((id_salt, "section"))
        .default_open(true)
        .show(ui, |ui| {
            if state.popped_out {
                ui.horizontal(|ui| {
                    ui.weak("Open in a separate window.");
                    if ui.button("Dock").clicked() {
                        state.popped_out = false;
                    }
                });
            } else {
                ui_contents(ui, id_salt, rows, state, &mut actions, true);
            }
        });
    if state.popped_out {
        let mut open = true;
        egui::Window::new(title.as_str())
            .id(egui::Id::new((id_salt, "window")))
            .open(&mut open)
            .default_size([500.0, 560.0])
            .min_width(WINDOW_MIN_WIDTH)
            .resizable(true)
            .show(ui.ctx(), |ui| {
                ui_contents(ui, id_salt, rows, state, &mut actions, false);
            });
        if !open {
            state.popped_out = false;
        }
    }
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: &str, count: usize, visible: bool) -> LegendRow {
        LegendRow {
            label: label.to_string(),
            count,
            color_rgb: [1, 2, 3],
            visible,
            custom_color: false,
        }
    }

    fn entry(label: &str) -> ObjectColorLegendEntry {
        ObjectColorLegendEntry {
            value_label: label.to_string(),
            count: 1,
            color_rgb: [9, 9, 9],
        }
    }

    #[test]
    fn row_columns_fit_the_available_width_and_drop_columns_when_narrow() {
        let width = |c: RowColumns, spacing: f32| {
            let optional = [c.count, c.percent].into_iter().flatten();
            let optional_count = optional.clone().count() as f32;
            CHECKBOX_WIDTH
                + SWATCH_WIDTH
                + c.label
                + c.only
                + c.reset
                + optional.sum::<f32>()
                + spacing * (4.0 + optional_count)
        };
        let columns = |available| row_columns(available, 6.0, 50.0, 45.0, 44.0, 50.0);

        for available in [280.0, 320.0, 360.0, 400.0, 540.0, 900.0] {
            let fitted = columns(available);
            assert!(
                width(fitted, 6.0) <= available + 0.01,
                "{available}: {fitted:?}"
            );
            assert!(fitted.label >= LABEL_MIN_WIDTH);
        }
        assert!(columns(540.0).percent.is_some());
        assert!(columns(320.0).percent.is_none() && columns(320.0).count.is_some());
        assert!(columns(280.0).count.is_none());
    }

    #[test]
    fn colour_precedence_is_view_then_project_then_default() {
        let mut colors = CategoryColors::default();
        colors.set("project", [1, 1, 1]);
        colors.set("both", [1, 1, 1]);
        let rows = legend_rows(
            vec![entry("default"), entry("project"), entry("both")],
            &colors,
            |label| (true, (label == "both").then_some([2, 2, 2])),
        );
        assert_eq!(rows[0].color_rgb, [9, 9, 9]);
        assert!(!rows[0].custom_color);
        assert_eq!(rows[1].color_rgb, [1, 1, 1]);
        assert!(rows[1].custom_color);
        assert_eq!(rows[2].color_rgb, [2, 2, 2]);
    }

    #[test]
    fn sorting_orders_by_count_then_name() {
        let rows = [row("b", 5, true), row("A", 5, true), row("c", 9, true)];
        let labels = |sort| {
            sorted_rows(&rows, sort)
                .iter()
                .map(|row| row.label.as_str())
                .collect::<Vec<_>>()
        };
        assert_eq!(labels(LegendSort::Count), ["c", "A", "b"]);
        assert_eq!(labels(LegendSort::Name), ["A", "b", "c"]);
    }

    #[test]
    fn visibility_actions_only_change_rows_that_differ() {
        let rows = [row("a", 1, true), row("b", 1, false), row("c", 1, true)];
        let only_b = visibility_actions(&rows, |row| row.label == "b");
        assert_eq!(
            only_b,
            [
                LegendAction::SetVisible {
                    label: "a".to_string(),
                    visible: false,
                },
                LegendAction::SetVisible {
                    label: "b".to_string(),
                    visible: true,
                },
                LegendAction::SetVisible {
                    label: "c".to_string(),
                    visible: false,
                },
            ]
        );
        assert_eq!(visibility_actions(&rows, |row| row.visible), []);
    }

    #[test]
    fn palette_assignment_is_independent_of_input_order_and_wraps() {
        let palette = [[1, 0, 0], [2, 0, 0]];
        let rows = [row("c", 9, true), row("a", 1, true), row("b", 5, true)];
        assert_eq!(
            palette_actions(&rows, &palette),
            [
                LegendAction::SetColor {
                    label: "a".to_string(),
                    rgb: [1, 0, 0],
                },
                LegendAction::SetColor {
                    label: "b".to_string(),
                    rgb: [2, 0, 0],
                },
                LegendAction::SetColor {
                    label: "c".to_string(),
                    rgb: [1, 0, 0],
                },
            ]
        );
    }

    #[test]
    fn edits_commit_on_pointer_release_and_survive_until_the_project_catches_up() {
        let mut colors = CategoryColors::default();
        let before = BTreeMap::new();
        colors.set("Tumour", [1, 2, 3]);

        assert_eq!(colors.take_commit(true), None);
        colors.sync(&before);
        assert_eq!(
            colors.get("Tumour"),
            Some([1, 2, 3]),
            "drag is not reverted"
        );

        let committed = colors.take_commit(false).expect("released edit commits");
        assert_eq!(colors.take_commit(false), None, "commits once");
        colors.sync(&before);
        assert_eq!(
            colors.get("Tumour"),
            Some([1, 2, 3]),
            "stale project ignored"
        );

        colors.sync(&committed);
        let mut renamed = BTreeMap::new();
        renamed.insert("Stroma".to_string(), [4, 5, 6]);
        colors.sync(&renamed);
        assert_eq!(
            colors.get("Stroma"),
            Some([4, 5, 6]),
            "follows project again"
        );
        assert_eq!(colors.get("Tumour"), None);
    }

    #[test]
    fn rejected_commit_eventually_returns_to_the_project_colours() {
        let mut colors = CategoryColors::default();
        let project = BTreeMap::new();
        colors.set("Tumour", [1, 2, 3]);
        colors.take_commit(false);
        for _ in 0..=COMMIT_WAIT_SYNCS {
            colors.sync(&project);
        }
        assert_eq!(colors.get("Tumour"), None);
    }
}
