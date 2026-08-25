//! Read-only snapshots reported by the renderer to the control actor.
//!
//! These adapters deliberately observe native UI state. Command semantics and mutation remain
//! actor-owned; this module only detects changes initiated through Odon's native interface.

use std::collections::HashSet;

use crate::data::project_config::ProjectRoi;

use super::{Mode, RootApp};

impl RootApp {
    pub(super) fn actor_renderer_observation(&mut self) -> serde_json::Value {
        let workspace = match &mut self.mode {
            Mode::Single(app) => app.control_viewport_workspace_snapshot(),
            _ => serde_json::Value::Null,
        };
        serde_json::json!({
            "view": self.renderer_view_snapshot(),
            "camera": self.renderer_camera_snapshot(),
            "channels": self.renderer_channels_snapshot(),
            "smooth": self.renderer_smooth_pixels_snapshot(),
            "loading": self.renderer_loading_snapshot(),
            "workspace": workspace,
            "selection": match &self.mode {
                Mode::Single(app) => app.control_object_selection_signature(),
                _ => serde_json::Value::Null,
            },
            "shell": self.control_shell_projection,
        })
    }

    fn renderer_view_snapshot(&self) -> serde_json::Value {
        let (mode, view) = match &self.mode {
            Mode::Project { .. } => ("project", serde_json::Value::Null),
            Mode::Single(app) => ("single", app.control_view_snapshot()),
            Mode::Mosaic { mosaic, .. } => ("mosaic", mosaic.control_view_snapshot()),
            Mode::Transition => ("transition", serde_json::Value::Null),
        };
        serde_json::json!({
            "mode": mode,
            "view": view,
            "project": self.renderer_project_snapshot(),
        })
    }

    fn renderer_project_snapshot(&self) -> serde_json::Value {
        let Some(project_space) = self.current_project_space() else {
            return serde_json::json!({"project": null, "rois": []});
        };
        let selected = project_space
            .selected_rois()
            .into_iter()
            .filter_map(|roi| roi.source_key())
            .collect::<HashSet<_>>();
        let focused = project_space.focused_roi().and_then(ProjectRoi::source_key);
        let rois = project_space
            .rois()
            .iter()
            .map(|roi| {
                let source_key = roi.source_key();
                serde_json::json!({
                    "id": roi.id,
                    "display_name": roi.display_name,
                    "dataset": roi.dataset,
                    "source_key": source_key,
                    "source": roi.source_display(),
                    "segmentation_path": roi.segpath.as_ref().map(|p| p.to_string_lossy().to_string()),
                    "selected": source_key.as_ref().is_some_and(|key| selected.contains(key)),
                    "focused": source_key == focused,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "project_path": project_space
                .saved_project_path()
                .map(|path| path.to_string_lossy().to_string()),
            "roi_count": rois.len(),
            "rois": rois,
        })
    }

    fn renderer_camera_snapshot(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "camera": app.control_camera_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "camera": mosaic.control_camera_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn renderer_channels_snapshot(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "channels": app.control_channel_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "channels": mosaic.control_channel_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "channels": [],
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "channels": [],
            }),
        }
    }

    fn renderer_smooth_pixels_snapshot(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "smooth_pixels": app.control_smooth_pixels_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "smooth_pixels": mosaic.control_smooth_pixels_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "error": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "error": "Odon is currently transitioning between views.",
            }),
        }
    }

    fn renderer_loading_snapshot(&self) -> serde_json::Value {
        match &self.mode {
            Mode::Single(app) => serde_json::json!({
                "mode": "single",
                "loading": app.control_loading_state_snapshot(),
            }),
            Mode::Mosaic { mosaic, .. } => serde_json::json!({
                "mode": "mosaic",
                "loading": mosaic.control_loading_state_snapshot(),
            }),
            Mode::Project { .. } => serde_json::json!({
                "mode": "project",
                "busy": false,
                "note": "No dataset viewer is currently open.",
            }),
            Mode::Transition => serde_json::json!({
                "mode": "transition",
                "busy": true,
                "reasons": ["transition"],
            }),
        }
    }
}
