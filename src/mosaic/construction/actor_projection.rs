//! Section-level application of actor projections to the mosaic renderer.

use super::*;

impl MosaicViewerApp {
    /// Applies an actor projection, skipping the state section when it is the one this renderer
    /// already applied. In that case only the camera, which is projected separately, can differ.
    pub fn apply_control_actor_projection(
        &mut self,
        state: &serde_json::Value,
        memory_state: &serde_json::Value,
        object_resources: &[(usize, Arc<odon::model::ControlObjectResource>)],
        pinned_levels: &[(usize, Arc<odon::model::ControlPinnedLevelResource>)],
        fingerprint: u64,
        camera: Option<odon::model::ProjectedCamera>,
    ) -> Result<(), String> {
        if self.applied_state_fingerprint == Some(fingerprint) {
            if let Some(camera) = camera {
                self.apply_projected_camera(Camera {
                    center_world_lvl0: egui::pos2(
                        camera.center_world_lvl0[0],
                        camera.center_world_lvl0[1],
                    ),
                    zoom_screen_per_lvl0_px: camera.zoom_screen_per_lvl0_px,
                });
            }
            // The echo of a native contrast commit can be a projection that changed nothing.
            self.reconcile_channel_contrast_projection();
            return Ok(());
        }
        self.applied_state_fingerprint = None;
        self.apply_control_actor_state(state, memory_state, object_resources, pinned_levels)?;
        self.applied_state_fingerprint = Some(fingerprint);
        Ok(())
    }

    /// A pan/zoom gesture owns the camera until the actor has caught up; a projection of an
    /// earlier gesture step must not rewind it.
    pub(super) fn apply_projected_camera(&mut self, projected: Camera) {
        self.release_settled_native_camera_hold();
        self.camera = match self.native_camera_hold.as_mut() {
            Some(hold) => hold.observe_projected(&projected),
            None => projected,
        };
    }

    /// Fingerprint of everything in the projected state that tile requests depend on. Camera,
    /// selection, contrast and colour do not: the camera is part of the per-frame request
    /// signature and the others are applied when tiles are drawn.
    pub(super) fn projected_tile_state_fingerprint(
        state: &serde_json::Value,
        pinned_levels: &[(usize, Arc<odon::model::ControlPinnedLevelResource>)],
    ) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for item in state["items"].as_array().into_iter().flatten() {
            odon::control::actor::json_object_fingerprint(
                item,
                &["selected", "focused"],
                &mut hasher,
            );
        }
        for channel in state["channels"].as_array().into_iter().flatten() {
            odon::control::actor::json_fingerprint(&channel["index"], &mut hasher);
            odon::control::actor::json_fingerprint(&channel["visible"], &mut hasher);
        }
        for key in ["generation", "channel_order", "layer_groups", "smooth_pixels"] {
            odon::control::actor::json_fingerprint(&state[key], &mut hasher);
        }
        for layer in state["native_layers"].as_array().into_iter().flatten() {
            odon::control::actor::json_fingerprint(&layer["layer_id"], &mut hasher);
            odon::control::actor::json_fingerprint(&layer["visible"], &mut hasher);
        }
        for (item_id, resource) in pinned_levels {
            item_id.hash(&mut hasher);
            (Arc::as_ptr(resource) as usize).hash(&mut hasher);
        }
        hasher.finish()
    }
}
