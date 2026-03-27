use std::collections::{BTreeMap, BTreeSet};

use eframe::egui;

use crate::data::project_config::ProjectRoi;

const DATASET_COLUMN: &str = "dataset";

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RoiBrowseClause {
    pub column: String,
    pub value: String,
}

#[derive(Debug, Clone, Default)]
pub struct RoiBrowseState {
    pub clauses: Vec<RoiBrowseClause>,
}

impl RoiBrowseState {
    pub fn clear(&mut self) {
        self.clauses.clear();
    }

    pub fn has_filters(&self) -> bool {
        !self.clauses.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct RoiBrowseUiResult {
    pub changed: bool,
    pub filtered_indices: Vec<usize>,
    pub total_count: usize,
}

pub fn filtered_roi_indices(
    rois: &[ProjectRoi],
    dataset_filter: Option<&str>,
    clauses: &[RoiBrowseClause],
) -> Vec<usize> {
    rois.iter()
        .enumerate()
        .filter_map(|(idx, roi)| {
            (roi_matches_dataset(roi, dataset_filter) && roi_matches_clauses(roi, clauses))
                .then_some(idx)
        })
        .collect()
}

pub fn ui(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    rois: &[ProjectRoi],
    dataset_filter: Option<&str>,
    state: &mut RoiBrowseState,
) -> RoiBrowseUiResult {
    let mut changed = sanitize_state(rois, dataset_filter, state);
    let total_count = filtered_roi_indices(rois, dataset_filter, &[]).len();

    ui.push_id(id_salt, |ui| {
        let all_columns = available_columns(rois, dataset_filter);
        let addable_columns = unused_columns(&all_columns, &state.clauses, None);

        ui.horizontal(|ui| {
            let add_clicked = ui
                .add_enabled(
                    !addable_columns.is_empty(),
                    egui::Button::new("+ Add filter"),
                )
                .clicked();
            let clear_clicked = ui
                .add_enabled(state.has_filters(), egui::Button::new("Clear filters"))
                .clicked();
            if add_clicked {
                if let Some(column) = addable_columns.first().cloned() {
                    let prefix = state.clauses.clone();
                    let value = available_values(rois, dataset_filter, &prefix, &column)
                        .into_iter()
                        .next()
                        .map(|(value, _count)| value)
                        .unwrap_or_default();
                    state.clauses.push(RoiBrowseClause { column, value });
                    changed = true;
                }
            }
            if clear_clicked {
                state.clear();
                changed = true;
            }
        });

        let mut remove_idx = None;
        for idx in 0..state.clauses.len() {
            let prefix = state.clauses[..idx].to_vec();
            let columns_for_row = unused_columns(&all_columns, &state.clauses, Some(idx));
            let clause = &mut state.clauses[idx];

            if !columns_for_row
                .iter()
                .any(|column| column == &clause.column)
            {
                if let Some(first) = columns_for_row.first() {
                    clause.column = first.clone();
                    changed = true;
                }
            }

            let values = available_values(rois, dataset_filter, &prefix, &clause.column);
            if !values.iter().any(|(value, _count)| value == &clause.value) {
                clause.value = values
                    .first()
                    .map(|(value, _count)| value.clone())
                    .unwrap_or_default();
                changed = true;
            }

            ui.horizontal(|ui| {
                ui.label(format!("{}. ", idx + 1));
                egui::ComboBox::from_id_salt(("column", idx))
                    .selected_text(clause.column.as_str())
                    .width(140.0)
                    .show_ui(ui, |ui| {
                        for column in &columns_for_row {
                            if ui
                                .selectable_value(&mut clause.column, column.clone(), column)
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });

                let values = available_values(rois, dataset_filter, &prefix, &clause.column);
                egui::ComboBox::from_id_salt(("value", idx))
                    .selected_text(display_value(&clause.value))
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for (value, count) in &values {
                            let label = format!("{} ({count})", display_value(value));
                            if ui
                                .selectable_value(&mut clause.value, value.clone(), label)
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });

                if ui.small_button("Remove").clicked() {
                    remove_idx = Some(idx);
                }
            });
        }

        if let Some(idx) = remove_idx {
            state.clauses.remove(idx);
            changed = true;
        }

        if state.has_filters() {
            let summary = state
                .clauses
                .iter()
                .map(|clause| format!("{}={}", clause.column, display_value(&clause.value)))
                .collect::<Vec<_>>()
                .join(" > ");
            ui.label(format!("Hierarchy: {summary}"));
        }
    });

    changed |= sanitize_state(rois, dataset_filter, state);
    let filtered_indices = filtered_roi_indices(rois, dataset_filter, &state.clauses);

    RoiBrowseUiResult {
        changed,
        filtered_indices,
        total_count,
    }
}

fn sanitize_state(
    rois: &[ProjectRoi],
    dataset_filter: Option<&str>,
    state: &mut RoiBrowseState,
) -> bool {
    let all_columns = available_columns(rois, dataset_filter);
    let mut changed = false;
    let mut used = BTreeSet::new();
    let mut keep = Vec::new();

    for clause in state.clauses.drain(..) {
        if clause.column.trim().is_empty()
            || !all_columns.iter().any(|column| column == &clause.column)
            || !used.insert(clause.column.clone())
        {
            changed = true;
            continue;
        }
        keep.push(clause);
    }

    for idx in 0..keep.len() {
        let prefix = keep[..idx].to_vec();
        let values = available_values(rois, dataset_filter, &prefix, &keep[idx].column);
        if !values
            .iter()
            .any(|(value, _count)| value == &keep[idx].value)
        {
            keep[idx].value = values
                .first()
                .map(|(value, _count)| value.clone())
                .unwrap_or_default();
            changed = true;
        }
    }

    state.clauses = keep;
    changed
}

fn available_columns(rois: &[ProjectRoi], dataset_filter: Option<&str>) -> Vec<String> {
    let mut columns = BTreeSet::new();
    let mut saw_any = false;

    for roi in rois {
        if !roi_matches_dataset(roi, dataset_filter) {
            continue;
        }
        saw_any = true;
        columns.insert(DATASET_COLUMN.to_string());
        for key in roi.meta.keys() {
            let key = key.trim();
            if !key.is_empty() {
                columns.insert(key.to_string());
            }
        }
    }

    if !saw_any {
        return Vec::new();
    }

    columns.into_iter().collect()
}

fn unused_columns(
    all_columns: &[String],
    clauses: &[RoiBrowseClause],
    current_idx: Option<usize>,
) -> Vec<String> {
    let used_elsewhere = clauses
        .iter()
        .enumerate()
        .filter(|(idx, _clause)| Some(*idx) != current_idx)
        .map(|(_idx, clause)| clause.column.as_str())
        .collect::<BTreeSet<_>>();

    all_columns
        .iter()
        .filter(|column| !used_elsewhere.contains(column.as_str()))
        .cloned()
        .collect()
}

fn available_values(
    rois: &[ProjectRoi],
    dataset_filter: Option<&str>,
    clauses: &[RoiBrowseClause],
    column: &str,
) -> Vec<(String, usize)> {
    let mut counts = BTreeMap::<String, usize>::new();

    for roi in rois {
        if !roi_matches_dataset(roi, dataset_filter) || !roi_matches_clauses(roi, clauses) {
            continue;
        }
        *counts.entry(roi_value_for_column(roi, column)).or_default() += 1;
    }

    counts.into_iter().collect()
}

fn roi_matches_clauses(roi: &ProjectRoi, clauses: &[RoiBrowseClause]) -> bool {
    clauses
        .iter()
        .all(|clause| roi_value_for_column(roi, &clause.column) == clause.value)
}

fn roi_matches_dataset(roi: &ProjectRoi, dataset_filter: Option<&str>) -> bool {
    dataset_filter.is_none_or(|dataset| roi_value_for_column(roi, DATASET_COLUMN) == dataset)
}

fn roi_value_for_column(roi: &ProjectRoi, column: &str) -> String {
    if column == DATASET_COLUMN {
        return roi.dataset.clone().unwrap_or_default();
    }
    roi.meta.get(column).cloned().unwrap_or_default()
}

fn display_value(value: &str) -> &str {
    if value.trim().is_empty() {
        "(blank)"
    } else {
        value
    }
}
