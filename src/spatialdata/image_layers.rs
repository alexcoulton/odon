use std::path::{Path, PathBuf};
use std::sync::Arc;

use eframe::egui;

use crate::camera::Camera;
use crate::data::ome::{ChannelInfo, OmeZarrDataset};
use crate::spatialdata::SpatialDataElement;
use crate::render::tiles::{RenderChannel, recommended_tile_loader_threads};
use crate::render::tiles_gl::{ChannelDraw, TileDraw, TilesGl};
use crate::render::tiles_raw::{
    RawTileKey, RawTileLoaderHandle, RawTileWorkerResponse, spawn_raw_tile_loader,
};
use crate::imaging::tiling::{TileCoord, choose_level_auto, tiles_needed_lvl0_rect};

#[derive(Debug, Default)]
pub struct SpatialImageLayers {
    pub images: Vec<SpatialImageLayer>,
    next_id: u64,
}

impl SpatialImageLayers {
    pub fn clear(&mut self) {
        self.images.clear();
        self.next_id = 1;
    }

    pub fn load_image(
        &mut self,
        root: &Path,
        element: &SpatialDataElement,
        gpu_available: bool,
        smooth_pixels: bool,
    ) -> anyhow::Result<u64> {
        let id = self.next_id.max(1);
        self.next_id = id.wrapping_add(1).max(1);
        let layer = SpatialImageLayer::open(
            id,
            format!("Image: {}", element.name),
            root.join(&element.rel_group),
            gpu_available,
            smooth_pixels,
        )?;
        self.images.push(layer);
        Ok(id)
    }

    pub fn tick(&mut self) {
        for layer in &mut self.images {
            layer.tick();
        }
    }

    pub fn set_smooth_pixels(&mut self, smooth_pixels: bool) {
        for layer in &mut self.images {
            layer.set_smooth_pixels(smooth_pixels);
        }
    }

    pub fn is_busy(&self) -> bool {
        self.images.iter().any(|layer| layer.is_busy())
    }
}

#[derive(Debug)]
pub struct SpatialImageLayer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub offset_world: egui::Vec2,

    pub dataset: OmeZarrDataset,
    pub channels: Vec<ChannelInfo>,
    pub path: PathBuf,
    raw_loader: Option<RawTileLoaderHandle>,
    tiles_gl: Option<TilesGl>,
    status: String,
}

impl SpatialImageLayer {
    pub fn open(
        id: u64,
        name: String,
        path: PathBuf,
        gpu_available: bool,
        smooth_pixels: bool,
    ) -> anyhow::Result<Self> {
        let (dataset, store) = OmeZarrDataset::open_local(&path)?;
        let mut status = String::new();
        let (raw_loader, tiles_gl) = if gpu_available {
            let level_paths = dataset
                .levels
                .iter()
                .map(|l| l.path.clone())
                .collect::<Vec<_>>();
            let level_shapes = dataset
                .levels
                .iter()
                .map(|l| l.shape.clone())
                .collect::<Vec<_>>();
            let level_chunks = dataset
                .levels
                .iter()
                .map(|l| l.chunks.clone())
                .collect::<Vec<_>>();
            let level_dtypes = dataset
                .levels
                .iter()
                .map(|l| l.dtype.clone())
                .collect::<Vec<_>>();
            let dims_cyx = (dataset.dims.c, dataset.dims.y, dataset.dims.x);
            let raw_loader = spawn_raw_tile_loader(
                store,
                level_paths,
                level_shapes,
                level_chunks,
                level_dtypes,
                dims_cyx,
                recommended_tile_loader_threads(),
            )?;
            let tiles_gl = TilesGl::new(4096);
            tiles_gl.set_smooth_pixels(smooth_pixels);
            (Some(raw_loader), Some(tiles_gl))
        } else {
            status = "Secondary SpatialData image layers require the GPU renderer.".to_string();
            (None, None)
        };

        Ok(Self {
            id,
            name,
            visible: true,
            opacity: 1.0,
            offset_world: egui::Vec2::ZERO,
            channels: dataset.channels.clone(),
            dataset,
            path,
            raw_loader,
            tiles_gl,
            status,
        })
    }

    pub fn set_smooth_pixels(&mut self, smooth_pixels: bool) {
        if let Some(tiles_gl) = self.tiles_gl.as_ref() {
            tiles_gl.set_smooth_pixels(smooth_pixels);
        }
    }

    pub fn tick(&mut self) {
        let (Some(loader), Some(tiles_gl)) = (self.raw_loader.as_ref(), self.tiles_gl.as_ref())
        else {
            return;
        };
        while let Ok(msg) = loader.rx.try_recv() {
            match msg {
                RawTileWorkerResponse::Tile(msg) => tiles_gl.insert_pending(msg),
                RawTileWorkerResponse::Failed { key, error } => {
                    tiles_gl.cancel_in_flight(&key);
                    crate::log_warn!(
                        "spatial image raw tile load failed for {:?}: {}",
                        key,
                        error
                    );
                }
            }
        }
    }

    pub fn is_busy(&self) -> bool {
        self.tiles_gl.as_ref().is_some_and(|gl| gl.is_busy())
    }

    pub fn ui_properties(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.checkbox(&mut self.visible, "Visible");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.opacity, 0.0..=1.0)
                    .text("Opacity")
                    .clamping(egui::SliderClamping::Always),
            )
            .changed();
        ui.label(format!("Channels: {}", self.channels.len()));
        ui.label(self.path.to_string_lossy());
        if !self.status.is_empty() {
            ui.label(self.status.clone());
        }
        ui.separator();
        ui.label("Channels");
        for ch in &mut self.channels {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut ch.visible, "").changed();
                ui.label(ch.name.clone());
                let mut color =
                    egui::Color32::from_rgb(ch.color_rgb[0], ch.color_rgb[1], ch.color_rgb[2]);
                if ui.color_edit_button_srgba(&mut color).changed() {
                    ch.color_rgb = [color.r(), color.g(), color.b()];
                    changed = true;
                }
            });
        }
        changed
    }

    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        camera: &Camera,
        viewport: egui::Rect,
        visible_world: egui::Rect,
    ) {
        if !self.visible {
            return;
        }
        let (Some(raw_loader), Some(tiles_gl)) = (self.raw_loader.as_ref(), self.tiles_gl.clone())
        else {
            return;
        };
        let render_channels = self.render_channels_for_request();
        if render_channels.is_empty() {
            return;
        }

        let visible_world = visible_world.translate(-self.offset_world);
        let target_level =
            choose_level_auto(&self.dataset.levels, camera.zoom_screen_per_lvl0_px, 1.0);
        let level_info = self.dataset.levels[target_level].clone();
        let coords: Vec<TileCoord> =
            tiles_needed_lvl0_rect(visible_world, &level_info, &self.dataset.dims, 1);
        let mut needed: Vec<RawTileKey> = coords
            .into_iter()
            .map(|c| RawTileKey {
                level: target_level,
                tile_y: c.tile_y,
                tile_x: c.tile_x,
                channel: 0,
            })
            .collect();
        needed.sort_unstable_by_key(|k| (k.tile_y, k.tile_x));

        let mut keep = std::collections::HashSet::new();
        let mut requested_this_frame = 0usize;
        let max_requests_per_frame = 128usize;
        for key in &needed {
            for ch in &render_channels {
                let raw_key = RawTileKey {
                    level: key.level,
                    tile_y: key.tile_y,
                    tile_x: key.tile_x,
                    channel: ch.index,
                };
                keep.insert(raw_key);
                if requested_this_frame >= max_requests_per_frame {
                    continue;
                }
                if tiles_gl.mark_in_flight(raw_key) {
                    let _ = raw_loader
                        .tx
                        .send(crate::render::tiles_raw::RawTileRequest { key: raw_key });
                    requested_this_frame += 1;
                }
            }
        }
        tiles_gl.prune_in_flight(&keep);

        let draws = needed
            .into_iter()
            .map(|key| TileDraw {
                level: key.level,
                tile_y: key.tile_y,
                tile_x: key.tile_x,
                screen_rect: self.tile_screen_rect(camera, viewport, &level_info, &key),
            })
            .filter(|draw| draw.screen_rect.intersects(viewport))
            .collect::<Vec<_>>();
        if draws.is_empty() {
            return;
        }

        let channels = render_channels
            .into_iter()
            .map(ChannelDraw::from)
            .collect::<Vec<_>>();
        let opacity = self.opacity;
        let cb = egui_glow::CallbackFn::new(move |info, painter| {
            tiles_gl.paint_overlay(info, painter, &draws, &channels, opacity);
        });
        ui.painter().add(egui::PaintCallback {
            rect: viewport,
            callback: Arc::new(cb),
        });
    }

    fn render_channels_for_request(&self) -> Vec<RenderChannel> {
        let mut out = Vec::new();
        for ch in &self.channels {
            if !ch.visible {
                continue;
            }
            out.push(RenderChannel {
                index: ch.index as u64,
                color_rgb: [
                    (ch.color_rgb[0] as f32 / 255.0) * self.opacity,
                    (ch.color_rgb[1] as f32 / 255.0) * self.opacity,
                    (ch.color_rgb[2] as f32 / 255.0) * self.opacity,
                ],
                window: ch.window.unwrap_or((0.0, self.dataset.abs_max.max(1.0))),
            });
        }
        out
    }

    fn tile_screen_rect(
        &self,
        camera: &Camera,
        viewport: egui::Rect,
        level_info: &crate::data::ome::LevelInfo,
        key: &RawTileKey,
    ) -> egui::Rect {
        let y_dim = self.dataset.dims.y;
        let x_dim = self.dataset.dims.x;
        let chunk_y = level_info.chunks[y_dim] as f32;
        let chunk_x = level_info.chunks[x_dim] as f32;
        let y0 = key.tile_y as f32 * chunk_y;
        let x0 = key.tile_x as f32 * chunk_x;
        let y1 = (y0 + chunk_y).min(level_info.shape[y_dim] as f32);
        let x1 = (x0 + chunk_x).min(level_info.shape[x_dim] as f32);
        let downsample = level_info.downsample;
        let world_min = egui::pos2(x0 * downsample, y0 * downsample) + self.offset_world;
        let world_max = egui::pos2(x1 * downsample, y1 * downsample) + self.offset_world;
        let screen_min = camera.world_to_screen(world_min, viewport);
        let screen_max = camera.world_to_screen(world_max, viewport);
        egui::Rect::from_min_max(screen_min, screen_max)
    }
}
