use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::{Context, anyhow};
use crossbeam_channel::{Receiver, Sender};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::{IfdPointer, Tag};

use crate::data::ome::{ChannelInfo, Dims, LevelInfo};
use crate::render::tiles::{
    CpuDecodedTileCache, DecodedTile, DecodedTileKey, RenderChannel, TileKey, TileLoaderHandle,
    TileRequest, TileResponse, TileWorkerResponse,
};
use crate::render::tiles_raw::{
    RawTileKey, RawTileLoaderHandle, RawTileRequest, RawTileResponse, RawTileWorkerResponse,
};
use crate::{log_debug, log_warn};

mod inspection;
mod loaders;
mod metadata;
#[cfg(test)]
mod tests;

use inspection::*;
#[cfg(test)]
use loaders::decode_tiff_channel_chunk;
pub use loaders::{spawn_tiff_raw_tile_loader, spawn_tiff_tile_loader};
use metadata::*;

// This file adapts TIFF and OME-TIFF pyramids into the viewer's shared tile/level
// model. The tricky part is that different producers encode channels and pyramid
// levels differently, so the loader has to normalize several TIFF layouts behind
// one consistent runtime API.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffChannelLayout {
    Single,
    Chunky,
    Planar,
    SeparateIfds,
}

#[derive(Debug, Clone)]
pub struct TiffLevel {
    pub ifd_pointers: Vec<IfdPointer>,
    pub width: u32,
    pub height: u32,
    pub chunk_type: ChunkType,
    pub chunk_w: u32,
    pub chunk_h: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub chunks_per_plane: u32,
    pub channels: usize,
    pub channel_layout: TiffChannelLayout,
}

#[derive(Debug, Clone)]
pub struct TiffPyramid {
    pub path: PathBuf,
    pub levels: Vec<TiffLevel>,
    pub pixel_dtype: String,
    pub channel_count: usize,
    pub abs_max: f32,
    pub ome: Option<OmeTiffMetadata>,
    pub size_z: usize,
    pub size_t: usize,
    pub plane_selection: TiffPlaneSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiffPlaneSelection {
    pub z: usize,
    pub t: usize,
}

#[derive(Debug, Clone)]
struct IfdInfo {
    main_ifd_index: usize,
    ifd_pointer: IfdPointer,
    subifd_pointers: Vec<IfdPointer>,
    width: u32,
    height: u32,
    chunk_type: ChunkType,
    chunk_w: u32,
    chunk_h: u32,
    tiles_x: u32,
    tiles_y: u32,
    chunks_per_plane: u32,
    channels: usize,
    channel_layout: TiffChannelLayout,
    pixel_dtype: String,
    abs_max: f32,
}

#[derive(Debug, Clone)]
pub struct OmeTiffMetadata {
    pub dimension_order: Option<String>,
    pub size_z: Option<usize>,
    pub size_t: Option<usize>,
    pub size_c: Option<usize>,
    pub physical_size_x: Option<f32>,
    pub physical_size_x_unit: Option<String>,
    pub physical_size_y: Option<f32>,
    pub physical_size_y_unit: Option<String>,
    pub channels: Vec<OmeTiffChannel>,
    pub tiff_data: Vec<OmeTiffData>,
}

#[derive(Debug, Clone)]
pub struct OmeTiffChannel {
    pub name: Option<String>,
    pub color_rgb: Option<[u8; 3]>,
}

#[derive(Debug, Clone)]
pub struct OmeTiffData {
    pub ifd: Option<usize>,
    pub first_c: Option<usize>,
    pub first_z: Option<usize>,
    pub first_t: Option<usize>,
    pub plane_count: Option<usize>,
}

impl TiffPyramid {
    pub fn open_with_selection(
        path: &Path,
        plane_selection: TiffPlaneSelection,
    ) -> anyhow::Result<Self> {
        // Opening a TIFF means more than reading the first image: we inspect the
        // main IFDs, optional OME plane metadata, and channel layout so the rest of
        // the app can treat the result like any other multi-level image source.
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut dec = open_decoder(&path)?;
        let ome = read_ome_tiff_metadata(&mut dec).context("read OME-TIFF metadata")?;
        let size_z = ome.as_ref().and_then(|m| m.size_z).unwrap_or(1).max(1);
        let size_t = ome.as_ref().and_then(|m| m.size_t).unwrap_or(1).max(1);
        if plane_selection.z >= size_z || plane_selection.t >= size_t {
            return Err(anyhow!(
                "requested TIFF plane Z={}, T={} is outside the available range Z=0..{}, T=0..{}",
                plane_selection.z,
                plane_selection.t,
                size_z.saturating_sub(1),
                size_t.saturating_sub(1),
            ));
        }

        let mut ifds: Vec<IfdInfo> = Vec::new();
        let mut main_ifd_index = 0usize;
        loop {
            ifds.push(current_ifd_info(&mut dec, main_ifd_index)?);

            if !dec.more_images() {
                break;
            }
            dec.next_image().context("advance to next TIFF image")?;
            main_ifd_index += 1;
        }

        if ifds.is_empty() {
            return Err(anyhow!("tiff has no image directories"));
        }

        let pixel_dtype = ifds[0].pixel_dtype.clone();
        let abs_max = ifds[0].abs_max;
        let levels = build_levels_from_main_ifds(&path, &ifds, ome.as_ref(), plane_selection)?;

        let channel_count = levels.first().map(|lvl| lvl.channels).unwrap_or(1);
        if levels.iter().any(|lvl| lvl.channels != channel_count) {
            return Err(anyhow!(
                "tiff channel count varies across pyramid levels; not supported yet"
            ));
        }

        log_debug!("tiff pyramid: path={:?} levels={}", path, levels.len());
        for (i, l) in levels.iter().enumerate() {
            log_debug!(
                "  lvl {i}: ifds={:?} size={}x{} chunk={:?} {}x{} tiles={}x{} channels={} layout={:?}",
                l.ifd_pointers,
                l.width,
                l.height,
                l.chunk_type,
                l.chunk_w,
                l.chunk_h,
                l.tiles_x,
                l.tiles_y,
                l.channels,
                l.channel_layout
            );
        }

        Ok(Self {
            path,
            levels,
            pixel_dtype,
            channel_count,
            abs_max,
            ome,
            size_z,
            size_t,
            plane_selection,
        })
    }

    pub fn to_levels_info(&self) -> Vec<LevelInfo> {
        if self.levels.is_empty() {
            return Vec::new();
        }
        let base_w = self.levels[0].width.max(1) as f32;
        let base_h = self.levels[0].height.max(1) as f32;
        self.levels
            .iter()
            .enumerate()
            .map(|(i, lvl)| {
                let sx = base_w / lvl.width.max(1) as f32;
                let sy = base_h / lvl.height.max(1) as f32;
                let downsample = (sx + sy) * 0.5;
                let (shape, chunks) = if self.channel_count > 1 {
                    (
                        vec![
                            self.channel_count as u64,
                            lvl.height as u64,
                            lvl.width as u64,
                        ],
                        vec![1, lvl.chunk_h as u64, lvl.chunk_w as u64],
                    )
                } else {
                    (
                        vec![lvl.height as u64, lvl.width as u64],
                        vec![lvl.chunk_h as u64, lvl.chunk_w as u64],
                    )
                };
                LevelInfo {
                    index: i,
                    path: format!("tiff/ifd/{i}"),
                    shape,
                    chunks,
                    downsample,
                    dtype: self.pixel_dtype.clone(),
                    scale: if self.channel_count > 1 {
                        vec![1.0, downsample, downsample]
                    } else {
                        vec![downsample, downsample]
                    },
                    translation: vec![0.0; if self.channel_count > 1 { 3 } else { 2 }],
                }
            })
            .collect()
    }

    pub fn dims(&self) -> Dims {
        if self.channel_count > 1 {
            Dims {
                c: Some(0),
                z: None,
                y: 1,
                x: 2,
                ndim: 3,
            }
        } else {
            Self::default_dims()
        }
    }

    pub fn default_dims() -> Dims {
        Dims {
            c: None,
            z: None,
            y: 0,
            x: 1,
            ndim: 2,
        }
    }

    pub fn default_channels_named(&self, single_name: &str) -> Vec<ChannelInfo> {
        let palette = [
            [255, 255, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 128, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ];
        if self.channel_count <= 1 {
            return vec![ChannelInfo {
                index: 0,
                name: single_name.to_string(),
                color_rgb: palette[0],
                window: Some((0.0, self.abs_max)),
                visible: true,
                note: String::new(),
            }];
        }

        let rgb_names = ["red", "green", "blue"];
        (0..self.channel_count)
            .map(|i| ChannelInfo {
                index: i,
                name: self
                    .ome
                    .as_ref()
                    .and_then(|ome| ome.channels.get(i))
                    .and_then(|ch| ch.name.clone())
                    .unwrap_or_else(|| {
                        if self.channel_count == 3 && i < 3 {
                            rgb_names[i].to_string()
                        } else {
                            format!("channel {}", i + 1)
                        }
                    }),
                color_rgb: self
                    .ome
                    .as_ref()
                    .and_then(|ome| ome.channels.get(i))
                    .and_then(|ch| ch.color_rgb)
                    .unwrap_or_else(|| palette[(i + 1).min(palette.len() - 1)]),
                window: Some((0.0, self.abs_max)),
                visible: i < 3,
                note: String::new(),
            })
            .collect()
    }

    pub fn validate_supported_ome_layout(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub fn has_plane_selection(&self) -> bool {
        self.size_z > 1 || self.size_t > 1
    }

    pub fn physical_pixel_size_xy(&self) -> Option<([f32; 2], [Option<String>; 2])> {
        let ome = self.ome.as_ref()?;
        let x = ome.physical_size_x?;
        let y = ome.physical_size_y.unwrap_or(x);
        let x_unit = ome
            .physical_size_x_unit
            .clone()
            .or_else(|| Some("µm".to_string()));
        let y_unit = ome
            .physical_size_y_unit
            .clone()
            .or_else(|| ome.physical_size_x_unit.clone())
            .or_else(|| Some("µm".to_string()));
        Some(([y, x], [y_unit, x_unit]))
    }
}
