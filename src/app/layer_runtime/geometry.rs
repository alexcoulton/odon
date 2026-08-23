use super::*;

impl OmeZarrViewerApp {
    pub(in crate::app) fn any_visible_channel_offset(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if off.x.abs() > 1e-6 || off.y.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    pub(in crate::app) fn any_visible_channel_affine(&self) -> bool {
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);
            if (scale.x - 1.0).abs() > 1e-6 || (scale.y - 1.0).abs() > 1e-6 || rot.abs() > 1e-6 {
                return true;
            }
        }
        false
    }

    pub(in crate::app) fn union_visible_world_for_visible_channels(
        &self,
        visible_world: egui::Rect,
    ) -> egui::Rect {
        let mut min_off_x = 0.0f32;
        let mut max_off_x = 0.0f32;
        let mut min_off_y = 0.0f32;
        let mut max_off_y = 0.0f32;
        let mut any = false;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            if !any {
                min_off_x = off.x;
                max_off_x = off.x;
                min_off_y = off.y;
                max_off_y = off.y;
                any = true;
            } else {
                min_off_x = min_off_x.min(off.x);
                max_off_x = max_off_x.max(off.x);
                min_off_y = min_off_y.min(off.y);
                max_off_y = max_off_y.max(off.y);
            }
        }
        if !any {
            return visible_world;
        }

        // For a channel with offset `off`, the region of *data* that must be fetched is
        // `visible_world - off`. Union all of those to avoid missing tiles.
        egui::Rect::from_min_max(
            egui::pos2(
                visible_world.min.x - max_off_x,
                visible_world.min.y - max_off_y,
            ),
            egui::pos2(
                visible_world.max.x - min_off_x,
                visible_world.max.y - min_off_y,
            ),
        )
    }

    pub(in crate::app) fn union_visible_world_for_visible_channels_xform(
        &self,
        visible_world: egui::Rect,
    ) -> egui::Rect {
        let img_world = self.image_local_rect_lvl0();
        let pivot = img_world.center();

        let corners = [
            visible_world.left_top(),
            egui::pos2(visible_world.right(), visible_world.top()),
            visible_world.right_bottom(),
            egui::pos2(visible_world.left(), visible_world.bottom()),
        ];

        let mut acc: Option<egui::Rect> = None;
        for (i, ch) in self.channels.iter().enumerate() {
            if !ch.visible {
                continue;
            }
            let off = self
                .channel_offsets_world
                .get(i)
                .copied()
                .unwrap_or_default();
            let scale = self
                .channel_scales
                .get(i)
                .copied()
                .unwrap_or(egui::Vec2::splat(1.0));
            let rot = self.channel_rotations_rad.get(i).copied().unwrap_or(0.0);

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for &c in &corners {
                let p = inv_xform_world_point(c, pivot, off, scale, rot);
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_y = min_y.min(p.y);
                max_y = max_y.max(p.y);
            }
            let r = egui::Rect::from_min_max(egui::pos2(min_x, min_y), egui::pos2(max_x, max_y));
            acc = Some(match acc {
                None => r,
                Some(prev) => prev.union(r),
            });
        }

        acc.unwrap_or(visible_world)
    }
}
