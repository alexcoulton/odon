//! Object-local, world, and screen transform helpers shared by render paths.

use super::*;

impl ObjectsLayer {
    pub(in crate::objects) fn display_scale(&self) -> egui::Vec2 {
        egui::vec2(
            self.display_transform.scale[0].max(1e-6),
            self.display_transform.scale[1].max(1e-6),
        )
    }

    pub(in crate::objects) fn display_offset(
        &self,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Vec2 {
        egui::vec2(
            local_to_world_offset.x + self.display_transform.translation[0],
            local_to_world_offset.y + self.display_transform.translation[1],
        )
    }

    pub(in crate::objects) fn local_to_world_point(
        &self,
        local: egui::Pos2,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Pos2 {
        let scale = self.display_scale();
        let offset = self.display_offset(local_to_world_offset);
        egui::pos2(local.x * scale.x + offset.x, local.y * scale.y + offset.y)
    }

    pub(in crate::objects) fn world_to_local_point(
        &self,
        world: egui::Pos2,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Pos2 {
        let scale = self.display_scale();
        let offset = self.display_offset(local_to_world_offset);
        egui::pos2(
            (world.x - offset.x) / scale.x,
            (world.y - offset.y) / scale.y,
        )
    }

    pub(in crate::objects) fn world_to_local_rect(
        &self,
        world: egui::Rect,
        local_to_world_offset: egui::Vec2,
    ) -> egui::Rect {
        let min = self.world_to_local_point(world.min, local_to_world_offset);
        let max = self.world_to_local_point(world.max, local_to_world_offset);
        egui::Rect::from_min_max(
            egui::pos2(min.x.min(max.x), min.y.min(max.y)),
            egui::pos2(min.x.max(max.x), min.y.max(max.y)),
        )
    }
}
