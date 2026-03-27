use std::collections::HashSet;
use std::hash::Hash;
use std::time::{Duration, Instant};

use eframe::egui;

#[derive(Debug, Clone)]
pub struct SystemMemorySnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRiskLevel {
    Warning,
    Danger,
}

#[derive(Debug, Clone)]
pub struct MemoryRisk {
    pub level: MemoryRiskLevel,
    pub requested_bytes: u64,
    pub pinned_bytes: u64,
    pub projected_bytes: u64,
    pub available_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct PendingMemoryAction<T> {
    pub summary: String,
    pub payload: T,
    pub risk: MemoryRisk,
}

#[derive(Debug, Clone)]
pub struct MemoryChannelRow {
    pub id: usize,
    pub label: String,
    pub visible: bool,
}

pub fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

pub fn read_system_memory_snapshot() -> Option<SystemMemorySnapshot> {
    use sysinfo::System;

    let mut system = System::new();
    system.refresh_memory();
    let total_bytes = system.total_memory();
    let available_bytes = system.available_memory();
    (total_bytes > 0).then_some(SystemMemorySnapshot {
        total_bytes,
        available_bytes,
    })
}

pub fn refresh_system_memory_if_needed(
    system_memory: &mut Option<SystemMemorySnapshot>,
    last_refresh: &mut Option<Instant>,
    refresh_every: Duration,
) {
    let should_refresh = last_refresh.is_none_or(|t| t.elapsed() >= refresh_every);
    if !should_refresh {
        return;
    }
    *system_memory = read_system_memory_snapshot();
    *last_refresh = Some(Instant::now());
}

pub fn memory_risk(
    system_memory: Option<&SystemMemorySnapshot>,
    pinned_bytes: u64,
    requested_bytes: u64,
) -> Option<MemoryRisk> {
    let snapshot = system_memory?;
    let projected_bytes = pinned_bytes.saturating_add(requested_bytes);
    let available_bytes = snapshot.available_bytes.max(1);
    let level = if projected_bytes > snapshot.available_bytes {
        Some(MemoryRiskLevel::Danger)
    } else if projected_bytes.saturating_mul(100) >= available_bytes.saturating_mul(75) {
        Some(MemoryRiskLevel::Warning)
    } else {
        None
    }?;
    Some(MemoryRisk {
        level,
        requested_bytes,
        pinned_bytes,
        projected_bytes,
        available_bytes: snapshot.available_bytes,
        total_bytes: snapshot.total_bytes,
    })
}

pub fn ui_memory_overview(
    ui: &mut egui::Ui,
    description: &str,
    total: Option<(&str, u64)>,
    system_memory: Option<&SystemMemorySnapshot>,
) {
    ui.heading("Memory");
    ui.label(description);
    if let Some((total_label, total_bytes)) = total {
        ui.label(format!("{total_label}: {}", format_bytes(total_bytes)));
    }
    match system_memory {
        Some(memory) => {
            ui.label(format!(
                "System RAM: {} total, {} available",
                format_bytes(memory.total_bytes),
                format_bytes(memory.available_bytes)
            ));
            ui.label("Loads above 75% of available RAM require confirmation.");
        }
        None => {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "System RAM unavailable. Load estimates still work, but no risk warning is shown.",
            );
        }
    }
}

pub fn ui_memory_channel_selector(
    ui: &mut egui::Ui,
    id_salt: impl Hash,
    rows: &[MemoryChannelRow],
    selected: &mut HashSet<usize>,
) {
    ui.horizontal(|ui| {
        ui.label(format!(
            "Selected channels: {}/{}",
            selected.len(),
            rows.len()
        ));
        if ui.button("All").clicked() {
            selected.clear();
            selected.extend(rows.iter().map(|row| row.id));
        }
        if ui.button("None").clicked() {
            selected.clear();
        }
        if ui.button("Visible").clicked() {
            selected.clear();
            selected.extend(rows.iter().filter(|row| row.visible).map(|row| row.id));
        }
    });

    if rows.is_empty() {
        ui.label("No channels available.");
        return;
    }

    egui::ScrollArea::vertical()
        .id_salt(id_salt)
        .max_height(160.0)
        .show(ui, |ui| {
            for row in rows {
                let mut is_selected = selected.contains(&row.id);
                if ui.checkbox(&mut is_selected, &row.label).changed() {
                    if is_selected {
                        selected.insert(row.id);
                    } else {
                        selected.remove(&row.id);
                    }
                }
            }
        });
}

pub fn ui_pending_memory_action_dialog<T: Clone>(
    ctx: &egui::Context,
    pending: &mut Option<PendingMemoryAction<T>>,
) -> Option<(String, T)> {
    let Some(pending_value) = pending.clone() else {
        return None;
    };

    let mut open = true;
    let mut proceed = false;
    let mut cancel = false;
    egui::Window::new("High memory load")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
        .open(&mut open)
        .show(ctx, |ui| {
            let (label, color) = match pending_value.risk.level {
                MemoryRiskLevel::Warning => ("Warning", ui.visuals().warn_fg_color),
                MemoryRiskLevel::Danger => ("Danger", ui.visuals().error_fg_color),
            };
            ui.colored_label(color, label);
            ui.label(pending_value.summary.clone());
            ui.add_space(8.0);
            ui.label(format!(
                "Requested load: {}",
                format_bytes(pending_value.risk.requested_bytes)
            ));
            ui.label(format!(
                "Already pinned: {}",
                format_bytes(pending_value.risk.pinned_bytes)
            ));
            ui.label(format!(
                "Projected pinned total: {}",
                format_bytes(pending_value.risk.projected_bytes)
            ));
            ui.label(format!(
                "Available system RAM: {}",
                format_bytes(pending_value.risk.available_bytes)
            ));
            ui.label(format!(
                "Total system RAM: {}",
                format_bytes(pending_value.risk.total_bytes)
            ));
            if pending_value.risk.level == MemoryRiskLevel::Danger {
                ui.colored_label(
                    ui.visuals().error_fg_color,
                    "Projected pinned memory exceeds currently available RAM.",
                );
            } else {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Projected pinned memory uses at least 75% of currently available RAM.",
                );
            }

            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                cancel = true;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                proceed = true;
            }

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
                if ui.button("Load anyway").clicked() {
                    proceed = true;
                }
            });
        });

    if !open || cancel {
        *pending = None;
        None
    } else if proceed {
        *pending = None;
        Some((pending_value.summary, pending_value.payload))
    } else {
        None
    }
}
