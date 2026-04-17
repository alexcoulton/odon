use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow};
use crossbeam_channel::{Receiver, Sender};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::{IfdPointer, Tag};

use crate::data::ome::{ChannelInfo, Dims, LevelInfo};
use crate::imaging::channel_max::{ChannelMaxLoaderHandle, ChannelMaxRequest, ChannelMaxResponse};
use crate::imaging::histogram::{
    HistogramLoaderHandle, HistogramRequest, HistogramResponse, HistogramStats,
};
use crate::render::tiles::{
    RenderChannel, TileKey, TileLoaderHandle, TileRequest, TileResponse, TileWorkerResponse,
};
use crate::render::tiles_raw::{
    RawTileLoaderHandle, RawTileRequest, RawTileResponse, RawTileWorkerResponse,
};
use crate::{log_debug, log_warn};

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

fn color_type_meta(color_type: tiff::ColorType) -> anyhow::Result<(usize, String, f32)> {
    match color_type {
        tiff::ColorType::Gray(8) => Ok((1, "|u1".to_string(), 255.0)),
        tiff::ColorType::Gray(16) => Ok((1, "<u2".to_string(), 65535.0)),
        tiff::ColorType::RGB(8) => Ok((3, "|u1".to_string(), 255.0)),
        tiff::ColorType::RGB(16) => Ok((3, "<u2".to_string(), 65535.0)),
        other => Err(anyhow!("unsupported TIFF color type: {other:?}")),
    }
}

fn open_decoder(path: &Path) -> anyhow::Result<Decoder<BufReader<File>>> {
    let f = File::open(path).with_context(|| format!("open tiff: {path:?}"))?;
    Decoder::new(BufReader::new(f)).context("tiff decoder")
}

fn current_ifd_info(
    dec: &mut Decoder<BufReader<File>>,
    main_ifd_index: usize,
) -> anyhow::Result<IfdInfo> {
    let ifd_pointer = dec
        .ifd_pointer()
        .context("missing current TIFF IFD pointer")?;
    let subifd_pointers = dec
        .find_tag(Tag::SubIfd)
        .ok()
        .flatten()
        .and_then(|value| value.into_ifd_vec().ok())
        .unwrap_or_default();
    let (w, h) = dec.dimensions().context("tiff dimensions")?;
    let chunk_type = dec.get_chunk_type();
    let (chunk_w, chunk_h) = dec.chunk_dimensions();
    if chunk_w == 0 || chunk_h == 0 {
        return Err(anyhow!("tiff chunk dimensions are 0"));
    }
    let tiles_x = (w + chunk_w - 1) / chunk_w;
    let tiles_y = (h + chunk_h - 1) / chunk_h;
    let color_type = dec.colortype().context("tiff color type")?;
    let (channels, pixel_dtype, abs_max) = color_type_meta(color_type)?;
    let planar = dec
        .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
        .ok()
        .flatten()
        .unwrap_or(1)
        == 2;
    let chunks_per_plane = match chunk_type {
        ChunkType::Tile => tiles_x.saturating_mul(tiles_y),
        ChunkType::Strip => tiles_y,
    };
    let channel_layout = if channels <= 1 {
        TiffChannelLayout::Single
    } else if planar {
        TiffChannelLayout::Planar
    } else {
        TiffChannelLayout::Chunky
    };

    Ok(IfdInfo {
        main_ifd_index,
        ifd_pointer,
        subifd_pointers,
        width: w,
        height: h,
        chunk_type,
        chunk_w,
        chunk_h,
        tiles_x,
        tiles_y,
        chunks_per_plane,
        channels,
        channel_layout,
        pixel_dtype,
        abs_max,
    })
}

fn inspect_ifd_pointer(path: &Path, ifd_pointer: IfdPointer) -> anyhow::Result<IfdInfo> {
    let mut dec = open_decoder(path)?;
    dec.seek_to_ifd_pointer(ifd_pointer)
        .with_context(|| format!("seek to TIFF IFD pointer {}", ifd_pointer.0))?;
    current_ifd_info(&mut dec, usize::MAX)
}

fn same_channel_model(a: &IfdInfo, b: &IfdInfo) -> bool {
    a.channels == b.channels
        && a.channel_layout == b.channel_layout
        && a.pixel_dtype == b.pixel_dtype
        && a.abs_max == b.abs_max
}

fn same_geometry(a: &IfdInfo, b: &IfdInfo) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.chunk_type == b.chunk_type
        && a.chunk_w == b.chunk_w
        && a.chunk_h == b.chunk_h
        && a.pixel_dtype == b.pixel_dtype
        && a.channels == 1
        && b.channels == 1
}

fn build_level(group: &[IfdInfo]) -> anyhow::Result<TiffLevel> {
    let first = group
        .first()
        .ok_or_else(|| anyhow!("cannot build TIFF level from empty group"))?;
    if group.len() > 1 {
        if group.iter().any(|ifd| !same_geometry(first, ifd)) {
            return Err(anyhow!(
                "TIFF channel IFDs within the same pyramid level do not share geometry"
            ));
        }
        if group.iter().any(|ifd| ifd.channels != 1) {
            return Err(anyhow!(
                "TIFF levels with multiple source IFDs must be single-channel per IFD"
            ));
        }
    }
    Ok(TiffLevel {
        ifd_pointers: group.iter().map(|ifd| ifd.ifd_pointer).collect(),
        width: first.width,
        height: first.height,
        chunk_type: first.chunk_type,
        chunk_w: first.chunk_w,
        chunk_h: first.chunk_h,
        tiles_x: first.tiles_x,
        tiles_y: first.tiles_y,
        chunks_per_plane: first.chunks_per_plane,
        channels: if group.len() > 1 {
            group.len()
        } else {
            first.channels
        },
        channel_layout: if group.len() > 1 {
            TiffChannelLayout::SeparateIfds
        } else {
            first.channel_layout
        },
    })
}

fn separate_ifd_group_ranges(ifds: &[IfdInfo]) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut i = 0usize;
    while i < ifds.len() {
        let first = &ifds[i];
        let mut j = i + 1;
        while j < ifds.len() && same_geometry(first, &ifds[j]) {
            j += 1;
        }
        groups.push((i, j));
        i = j;
    }
    groups
}

fn ome_plane_axis_order(ome: &OmeTiffMetadata) -> anyhow::Result<[char; 3]> {
    let dim_order = ome
        .dimension_order
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("XYZCT");
    let mut axes = Vec::new();
    for ch in dim_order.chars() {
        if ch != 'X' && ch != 'Y' {
            axes.push(ch);
        }
    }
    if axes.len() != 3 || !axes.contains(&'Z') || !axes.contains(&'C') || !axes.contains(&'T') {
        return Err(anyhow!(
            "unsupported OME dimension order for TIFF planes: {dim_order}"
        ));
    }
    Ok([axes[0], axes[1], axes[2]])
}

fn ome_plane_axis_size(axis: char, size_z: usize, size_c: usize, size_t: usize) -> usize {
    match axis {
        'Z' => size_z,
        'C' => size_c,
        'T' => size_t,
        _ => 0,
    }
}

fn ome_plane_axis_coord(axis: char, z: usize, c: usize, t: usize) -> usize {
    match axis {
        'Z' => z,
        'C' => c,
        'T' => t,
        _ => 0,
    }
}

fn ome_linear_plane_index(
    axis_order: [char; 3],
    size_z: usize,
    size_c: usize,
    size_t: usize,
    z: usize,
    c: usize,
    t: usize,
) -> anyhow::Result<usize> {
    if z >= size_z || c >= size_c || t >= size_t {
        return Err(anyhow!("OME plane coordinates out of range"));
    }

    let a0 = axis_order[0];
    let a1 = axis_order[1];
    let a2 = axis_order[2];
    let s0 = ome_plane_axis_size(a0, size_z, size_c, size_t);
    let s1 = ome_plane_axis_size(a1, size_z, size_c, size_t);
    let c0 = ome_plane_axis_coord(a0, z, c, t);
    let c1 = ome_plane_axis_coord(a1, z, c, t);
    let c2 = ome_plane_axis_coord(a2, z, c, t);
    Ok(c0 + s0.saturating_mul(c1 + s1.saturating_mul(c2)))
}

fn ome_plane_coords_from_linear(
    axis_order: [char; 3],
    size_z: usize,
    size_c: usize,
    size_t: usize,
    linear: usize,
) -> anyhow::Result<(usize, usize, usize)> {
    let total = size_z.saturating_mul(size_c).saturating_mul(size_t);
    if linear >= total {
        return Err(anyhow!(
            "OME plane index {linear} out of range for {total} planes"
        ));
    }

    let a0 = axis_order[0];
    let a1 = axis_order[1];
    let a2 = axis_order[2];
    let s0 = ome_plane_axis_size(a0, size_z, size_c, size_t);
    let s1 = ome_plane_axis_size(a1, size_z, size_c, size_t);
    let mut remainder = linear;
    let v0 = remainder % s0;
    remainder /= s0;
    let v1 = remainder % s1;
    remainder /= s1;
    let v2 = remainder;

    let mut z = 0usize;
    let mut c = 0usize;
    let mut t = 0usize;
    for (axis, value) in [(a0, v0), (a1, v1), (a2, v2)] {
        match axis {
            'Z' => z = value,
            'C' => c = value,
            'T' => t = value,
            _ => {}
        }
    }
    Ok((z, c, t))
}

fn ome_channel_ifd_order(
    ome: &OmeTiffMetadata,
    channel_count: usize,
    target_z: usize,
    target_t: usize,
) -> anyhow::Result<Option<Vec<usize>>> {
    let expected_channels = ome.size_c.unwrap_or(channel_count);
    if expected_channels != channel_count {
        return Err(anyhow!(
            "OME SizeC ({expected_channels}) does not match TIFF channel count ({channel_count})"
        ));
    }
    let size_z = ome.size_z.unwrap_or(1);
    let size_t = ome.size_t.unwrap_or(1);
    if target_z >= size_z || target_t >= size_t {
        return Err(anyhow!(
            "requested OME plane Z={target_z}, T={target_t} is outside the supported range"
        ));
    }
    let axis_order = ome_plane_axis_order(ome)?;
    let total_planes = size_z
        .checked_mul(channel_count)
        .and_then(|v| v.checked_mul(size_t))
        .ok_or_else(|| anyhow!("OME plane count overflow"))?;

    let tiff_data_entries: Vec<OmeTiffData> = if ome.tiff_data.is_empty() {
        vec![OmeTiffData {
            ifd: Some(0),
            first_c: Some(0),
            first_z: Some(0),
            first_t: Some(0),
            plane_count: Some(total_planes),
        }]
    } else {
        ome.tiff_data.clone()
    };

    let mut channel_ifds: Vec<Option<usize>> = vec![None; channel_count];
    let mut next_ifd = 0usize;
    let mut next_c = 0usize;
    let mut next_z = 0usize;
    let mut next_t = 0usize;
    for td in &tiff_data_entries {
        let ifd_start = td.ifd.unwrap_or(next_ifd);
        let first_z = td.first_z.unwrap_or(next_z);
        let first_c = td.first_c.unwrap_or(next_c);
        let first_t = td.first_t.unwrap_or(next_t);
        let start_linear = ome_linear_plane_index(
            axis_order,
            size_z,
            channel_count,
            size_t,
            first_z,
            first_c,
            first_t,
        )?;
        let plane_count = td
            .plane_count
            .unwrap_or(total_planes.saturating_sub(start_linear));
        if plane_count == 0 {
            continue;
        }
        if start_linear + plane_count > total_planes {
            return Err(anyhow!(
                "OME-TIFF TiffData plane range exceeds available planes"
            ));
        }

        for offset in 0..plane_count {
            let (z, c, t) = ome_plane_coords_from_linear(
                axis_order,
                size_z,
                channel_count,
                size_t,
                start_linear + offset,
            )?;
            if z == target_z && t == target_t {
                let ifd_index = ifd_start + offset;
                if channel_ifds[c].replace(ifd_index).is_some() {
                    return Err(anyhow!(
                        "OME-TIFF TiffData maps multiple IFDs to channel {c} at Z={target_z}, T={target_t}"
                    ));
                }
            }
        }

        next_ifd = ifd_start + plane_count;
        let (z, c, t) = ome_plane_coords_from_linear(
            axis_order,
            size_z,
            channel_count,
            size_t,
            start_linear + plane_count - 1,
        )?;
        if start_linear + plane_count < total_planes {
            let (next_plane_z, next_plane_c, next_plane_t) = ome_plane_coords_from_linear(
                axis_order,
                size_z,
                channel_count,
                size_t,
                start_linear + plane_count,
            )?;
            next_z = next_plane_z;
            next_c = next_plane_c;
            next_t = next_plane_t;
        } else {
            next_z = z;
            next_c = c;
            next_t = t;
        }
    }

    if channel_ifds.iter().any(|mapped| mapped.is_none()) {
        return Err(anyhow!(
            "OME-TIFF mapping did not assign every channel at Z={target_z}, T={target_t}"
        ));
    }

    Ok(Some(
        channel_ifds
            .into_iter()
            .map(|mapped| mapped.expect("checked above"))
            .collect(),
    ))
}

fn reorder_ifd_group_by_tiff_data(
    group: &[IfdInfo],
    ome: Option<&OmeTiffMetadata>,
) -> anyhow::Result<Vec<IfdInfo>> {
    let Some(ome) = ome else {
        return Ok(group.to_vec());
    };
    let Some(channel_ifds) = ome_channel_ifd_order(ome, group.len(), 0, 0)? else {
        return Ok(group.to_vec());
    };
    if group.iter().any(|ifd| ifd.main_ifd_index == usize::MAX) {
        return Ok(group.to_vec());
    }

    let group_indices: Vec<_> = group.iter().map(|ifd| ifd.main_ifd_index).collect();
    if channel_ifds
        .iter()
        .any(|ifd_index| !group_indices.contains(ifd_index))
    {
        return Err(anyhow!(
            "OME-TIFF TiffData mapping references IFDs outside the base channel group"
        ));
    }

    if channel_ifds
        .iter()
        .zip(group_indices.iter())
        .all(|(expected, existing)| expected == existing)
    {
        return Ok(group.to_vec());
    }

    channel_ifds
        .into_iter()
        .map(|main_ifd_index| {
            group
                .iter()
                .find(|ifd| ifd.main_ifd_index == main_ifd_index)
                .cloned()
                .ok_or_else(|| {
                    anyhow!("missing TIFF IFD {main_ifd_index} referenced by OME TiffData")
                })
        })
        .collect()
}

fn select_base_ifd_group(
    ifds: &[IfdInfo],
    groups: &[(usize, usize)],
    ome: Option<&OmeTiffMetadata>,
    plane_selection: TiffPlaneSelection,
) -> anyhow::Result<(usize, Vec<IfdInfo>)> {
    let Some(ome) = ome else {
        let (start, end) = groups[0];
        return Ok((0, ifds[start..end].to_vec()));
    };

    let size_z = ome.size_z.unwrap_or(1);
    let size_t = ome.size_t.unwrap_or(1);
    if size_z == 1 && size_t == 1 {
        let (start, end) = groups[0];
        return Ok((
            0,
            reorder_ifd_group_by_tiff_data(&ifds[start..end], Some(ome))?,
        ));
    }

    let channel_ifds = ome_channel_ifd_order(
        ome,
        ome.size_c.unwrap_or(1),
        plane_selection.z,
        plane_selection.t,
    )?
    .ok_or_else(|| anyhow!("OME plane selection requires channel-to-IFD mapping"))?;

    let base_group_index = groups
        .iter()
        .enumerate()
        .find_map(|(group_index, (start, end))| {
            let group_indices: Vec<_> = ifds[*start..*end]
                .iter()
                .map(|ifd| ifd.main_ifd_index)
                .collect();
            let matches = channel_ifds.len() == group_indices.len()
                && channel_ifds
                    .iter()
                    .all(|ifd_index| group_indices.contains(ifd_index));
            matches.then_some(group_index)
        })
        .ok_or_else(|| {
            anyhow!(
                "could not find OME plane Z={}, T={} in TIFF IFD groups",
                plane_selection.z,
                plane_selection.t
            )
        })?;

    let (start, end) = groups[base_group_index];
    Ok((
        base_group_index,
        reorder_ifd_group_by_tiff_data(&ifds[start..end], Some(ome))?,
    ))
}

fn ome_multichannel_plane_order(ome: &OmeTiffMetadata) -> anyhow::Result<[char; 2]> {
    let full = ome_plane_axis_order(ome)?;
    let axes = full
        .into_iter()
        .filter(|axis| *axis != 'C')
        .collect::<Vec<_>>();
    if axes.len() != 2 || !axes.contains(&'Z') || !axes.contains(&'T') {
        return Err(anyhow!(
            "unsupported OME multichannel plane order: {:?}",
            ome.dimension_order
        ));
    }
    Ok([axes[0], axes[1]])
}

fn ome_multichannel_plane_index(
    ome: &OmeTiffMetadata,
    target_z: usize,
    target_t: usize,
) -> anyhow::Result<usize> {
    let size_z = ome.size_z.unwrap_or(1).max(1);
    let size_t = ome.size_t.unwrap_or(1).max(1);
    if target_z >= size_z || target_t >= size_t {
        return Err(anyhow!("requested OME plane coordinates out of range"));
    }
    let [a0, a1] = ome_multichannel_plane_order(ome)?;
    let s0 = match a0 {
        'Z' => size_z,
        'T' => size_t,
        _ => 0,
    };
    let c0 = match a0 {
        'Z' => target_z,
        'T' => target_t,
        _ => 0,
    };
    let c1 = match a1 {
        'Z' => target_z,
        'T' => target_t,
        _ => 0,
    };
    Ok(c0 + s0.saturating_mul(c1))
}

fn select_multichannel_base_ifd<'a>(
    ifds: &'a [IfdInfo],
    ome: Option<&OmeTiffMetadata>,
    plane_selection: TiffPlaneSelection,
) -> anyhow::Result<&'a IfdInfo> {
    let Some(ome) = ome else {
        return ifds
            .first()
            .ok_or_else(|| anyhow!("missing base TIFF level"));
    };
    let size_z = ome.size_z.unwrap_or(1).max(1);
    let size_t = ome.size_t.unwrap_or(1).max(1);
    if size_z == 1 && size_t == 1 {
        return ifds
            .first()
            .ok_or_else(|| anyhow!("missing base TIFF level"));
    }

    let plane_index = ome_multichannel_plane_index(ome, plane_selection.z, plane_selection.t)?;
    ifds.get(plane_index).ok_or_else(|| {
        anyhow!(
            "OME multichannel plane Z={}, T={} resolved to IFD {}, but TIFF has only {} main IFDs",
            plane_selection.z,
            plane_selection.t,
            plane_index,
            ifds.len(),
        )
    })
}

fn build_multichannel_levels(
    path: &Path,
    ifds: &[IfdInfo],
    ome: Option<&OmeTiffMetadata>,
    plane_selection: TiffPlaneSelection,
) -> anyhow::Result<Vec<TiffLevel>> {
    // Chunky/planar TIFFs keep all channels in one IFD (or one SubIFD chain), so
    // we first choose the requested Z/T plane and then derive the pyramid from that
    // one channel model. Mixed layouts are rejected to keep decoding predictable.
    let base = select_multichannel_base_ifd(ifds, ome, plane_selection)?;
    let mut levels = vec![build_level(&ifds[..1])?];
    levels[0].ifd_pointers = vec![base.ifd_pointer];

    if !base.subifd_pointers.is_empty() {
        if ifds.len() > 1 {
            log_warn!(
                "TIFF has {} main IFDs plus SubIFDs; using the selected plane as the pyramid root",
                ifds.len()
            );
        }
        for &subifd in &base.subifd_pointers {
            let info = inspect_ifd_pointer(path, subifd)?;
            if !same_channel_model(base, &info) {
                return Err(anyhow!(
                    "SubIFD channel model does not match the base TIFF level"
                ));
            }
            if !info.subifd_pointers.is_empty() {
                log_warn!("nested TIFF SubIFDs are ignored");
            }
            let group = [info];
            levels.push(build_level(&group)?);
        }
        return Ok(levels);
    }

    if ifds.iter().any(|ifd| !same_channel_model(base, ifd)) {
        return Err(anyhow!("mixed TIFF channel layouts are not supported yet"));
    }

    if ome
        .map(|m| m.size_z.unwrap_or(1) > 1 || m.size_t.unwrap_or(1) > 1)
        .unwrap_or(false)
    {
        return Ok(levels);
    }

    ifds.iter()
        .map(|ifd| {
            let group = [ifd.clone()];
            build_level(&group)
        })
        .collect()
}

fn build_separate_ifd_levels(
    path: &Path,
    ifds: &[IfdInfo],
    ome: Option<&OmeTiffMetadata>,
    plane_selection: TiffPlaneSelection,
) -> anyhow::Result<Vec<TiffLevel>> {
    // Some TIFFs store one channel per IFD. In that case we group compatible IFDs
    // into per-level channel sets, then decide whether extra groups represent
    // pyramid levels or alternative Z/T planes based on the available metadata.
    let groups = separate_ifd_group_ranges(ifds);
    if groups.is_empty() {
        return Ok(Vec::new());
    }

    let (base_group_index, base_group) =
        select_base_ifd_group(ifds, &groups, ome, plane_selection)?;
    let mut levels = vec![build_level(&base_group)?];
    let size_z = ome.and_then(|m| m.size_z).unwrap_or(1);
    let size_t = ome.and_then(|m| m.size_t).unwrap_or(1);

    if base_group.iter().any(|ifd| !ifd.subifd_pointers.is_empty()) {
        if groups.len() > 1 {
            log_warn!(
                "TIFF has {} main-IFD groups plus SubIFDs; using the selected Z=0, T=0 group as the pyramid root",
                groups.len()
            );
        }
        let subifd_count = base_group[0].subifd_pointers.len();
        if base_group
            .iter()
            .any(|ifd| ifd.subifd_pointers.len() != subifd_count)
        {
            return Err(anyhow!(
                "separate-IFD TIFF channels expose different numbers of SubIFD levels"
            ));
        }

        for level_index in 0..subifd_count {
            let mut sublevel = Vec::with_capacity(base_group.len());
            for ifd in &base_group {
                let info = inspect_ifd_pointer(path, ifd.subifd_pointers[level_index])?;
                if !info.subifd_pointers.is_empty() {
                    log_warn!("nested TIFF SubIFDs are ignored");
                }
                sublevel.push(info);
            }
            levels.push(build_level(&sublevel)?);
        }
        return Ok(levels);
    }

    if size_z > 1 || size_t > 1 {
        if groups.len() > 1 {
            log_warn!(
                "TIFF has additional main-IFD groups that are treated as other Z/T planes and ignored without plane-selection UI"
            );
        }
        return Ok(levels);
    }

    for (group_index, (start, end)) in groups.into_iter().enumerate() {
        if group_index == base_group_index {
            continue;
        }
        levels.push(build_level(&ifds[start..end])?);
    }

    Ok(levels)
}

fn build_levels_from_main_ifds(
    path: &Path,
    ifds: &[IfdInfo],
    ome: Option<&OmeTiffMetadata>,
    plane_selection: TiffPlaneSelection,
) -> anyhow::Result<Vec<TiffLevel>> {
    // The runtime only needs one normalized pyramid description, but the way we
    // construct it depends on whether channels live inside IFDs or across them.
    if ifds.iter().any(|ifd| ifd.channels > 1) {
        build_multichannel_levels(path, ifds, ome, plane_selection)
    } else {
        build_separate_ifd_levels(path, ifds, ome, plane_selection)
    }
}

fn base_chunk_index(lvl: &TiffLevel, tile_y: u64, tile_x: u64) -> u32 {
    match lvl.chunk_type {
        ChunkType::Tile => {
            let ix = tile_x.min(lvl.tiles_x.saturating_sub(1) as u64) as u32;
            let iy = tile_y.min(lvl.tiles_y.saturating_sub(1) as u64) as u32;
            iy.saturating_mul(lvl.tiles_x).saturating_add(ix)
        }
        ChunkType::Strip => tile_y.min(lvl.tiles_y.saturating_sub(1) as u64) as u32,
    }
}

fn decode_result_u16(decoded: DecodingResult) -> Option<Vec<u16>> {
    match decoded {
        DecodingResult::U16(v) => Some(v),
        DecodingResult::U8(v) => Some(v.into_iter().map(|b| b as u16).collect()),
        _ => None,
    }
}

fn compute_hist_u16(values: &[u16], bins: usize, abs_max: f32) -> Vec<u32> {
    let bins = bins.max(8);
    let mut out = vec![0u32; bins];
    if values.is_empty() {
        return out;
    }
    let inv = (bins as f32 - 1.0) / abs_max.max(1.0);
    for &v in values {
        let vf = (v as f32).clamp(0.0, abs_max);
        let idx = (vf * inv).floor() as usize;
        let idx = idx.min(bins - 1);
        out[idx] = out[idx].saturating_add(1);
    }
    out
}

fn compute_stats_u16(values: &[u16]) -> Option<HistogramStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    Some(HistogramStats {
        min: sorted[0] as f32,
        q1: sorted[(n * 25) / 100] as f32,
        median: sorted[(n * 50) / 100] as f32,
        q3: sorted[(n * 75) / 100] as f32,
        max: sorted[n - 1] as f32,
        n,
    })
}

fn read_ome_tiff_metadata(
    dec: &mut Decoder<BufReader<File>>,
) -> anyhow::Result<Option<OmeTiffMetadata>> {
    let Some(raw_desc) = dec.find_tag(Tag::ImageDescription).ok().flatten() else {
        return Ok(None);
    };
    let Ok(xml) = raw_desc.into_string() else {
        return Ok(None);
    };
    let xml_trimmed = xml.trim_start();
    if !xml_trimmed.starts_with('<') || !xml_trimmed.contains("OME") {
        return Ok(None);
    }
    parse_ome_xml(&xml).map(Some)
}

fn parse_ome_xml(xml: &str) -> anyhow::Result<OmeTiffMetadata> {
    // We intentionally consume only the first Pixels block. That matches the files
    // we support today and gives us enough information to name channels, choose a
    // Z/T plane, and understand how TIFF IFDs map onto logical image planes.
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut metadata = OmeTiffMetadata {
        dimension_order: None,
        size_z: None,
        size_t: None,
        size_c: None,
        physical_size_x: None,
        physical_size_x_unit: None,
        physical_size_y: None,
        physical_size_y_unit: None,
        channels: Vec::new(),
        tiff_data: Vec::new(),
    };

    let mut in_first_pixels = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let is_pixels = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"Pixels"
                };
                let is_channel = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"Channel"
                };
                let is_tiff_data = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"TiffData"
                };
                if is_pixels && !in_first_pixels {
                    in_first_pixels = true;
                    apply_pixels_attrs(&mut metadata, e, &reader)?;
                } else if is_channel && in_first_pixels {
                    metadata.channels.push(parse_channel(e, &reader)?);
                } else if is_tiff_data && in_first_pixels {
                    metadata.tiff_data.push(parse_tiff_data(e, &reader)?);
                }
            }
            Ok(Event::Empty(ref e)) => {
                let is_pixels = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"Pixels"
                };
                let is_channel = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"Channel"
                };
                let is_tiff_data = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"TiffData"
                };
                if is_pixels && !in_first_pixels {
                    apply_pixels_attrs(&mut metadata, e, &reader)?;
                    break;
                } else if is_channel && in_first_pixels {
                    metadata.channels.push(parse_channel(e, &reader)?);
                } else if is_tiff_data && in_first_pixels {
                    metadata.tiff_data.push(parse_tiff_data(e, &reader)?);
                }
            }
            Ok(Event::End(ref e)) => {
                let is_pixels = {
                    let name = e.name();
                    local_name(name.as_ref()) == b"Pixels"
                };
                if in_first_pixels && is_pixels {
                    break;
                }
            }
            Ok(Event::Eof) => break,
            Err(err) => return Err(anyhow!("OME-XML parse error: {err}")),
            _ => {}
        }
    }

    Ok(metadata)
}

fn parse_channel(e: &BytesStart<'_>, reader: &Reader<&[u8]>) -> anyhow::Result<OmeTiffChannel> {
    let mut name = None;
    let mut color_rgb = None;
    for attr in e.attributes() {
        let attr = attr.context("OME-XML channel attribute")?;
        let key = local_name(attr.key.as_ref());
        let value = attr
            .decode_and_unescape_value(reader.decoder())
            .context("decode OME-XML channel attribute")?
            .to_string();
        match key {
            b"Name" if !value.trim().is_empty() => name = Some(value),
            b"Color" => color_rgb = parse_ome_color(&value),
            _ => {}
        }
    }
    Ok(OmeTiffChannel { name, color_rgb })
}

fn parse_tiff_data(e: &BytesStart<'_>, reader: &Reader<&[u8]>) -> anyhow::Result<OmeTiffData> {
    let mut tiff_data = OmeTiffData {
        ifd: None,
        first_c: None,
        first_z: None,
        first_t: None,
        plane_count: None,
    };
    for attr in e.attributes() {
        let attr = attr.context("OME-XML TiffData attribute")?;
        let key = local_name(attr.key.as_ref());
        let value = attr
            .decode_and_unescape_value(reader.decoder())
            .context("decode OME-XML TiffData attribute")?
            .to_string();
        match key {
            b"IFD" => tiff_data.ifd = value.parse::<usize>().ok(),
            b"FirstC" => tiff_data.first_c = value.parse::<usize>().ok(),
            b"FirstZ" => tiff_data.first_z = value.parse::<usize>().ok(),
            b"FirstT" => tiff_data.first_t = value.parse::<usize>().ok(),
            b"PlaneCount" => tiff_data.plane_count = value.parse::<usize>().ok(),
            _ => {}
        }
    }
    Ok(tiff_data)
}

fn apply_pixels_attrs(
    metadata: &mut OmeTiffMetadata,
    e: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> anyhow::Result<()> {
    for attr in e.attributes() {
        let attr = attr.context("OME-XML pixels attribute")?;
        let key = local_name(attr.key.as_ref());
        let value = attr
            .decode_and_unescape_value(reader.decoder())
            .context("decode OME-XML pixels attribute")?
            .to_string();
        match key {
            b"DimensionOrder" => metadata.dimension_order = Some(value),
            b"SizeZ" => metadata.size_z = value.parse::<usize>().ok(),
            b"SizeT" => metadata.size_t = value.parse::<usize>().ok(),
            b"SizeC" => metadata.size_c = value.parse::<usize>().ok(),
            b"PhysicalSizeX" => metadata.physical_size_x = value.parse::<f32>().ok(),
            b"PhysicalSizeXUnit" if !value.trim().is_empty() => {
                metadata.physical_size_x_unit = Some(value)
            }
            b"PhysicalSizeY" => metadata.physical_size_y = value.parse::<f32>().ok(),
            b"PhysicalSizeYUnit" if !value.trim().is_empty() => {
                metadata.physical_size_y_unit = Some(value)
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_ome_color(s: &str) -> Option<[u8; 3]> {
    let raw = s.trim().parse::<i32>().ok()? as u32;
    Some([
        ((raw >> 16) & 0xff) as u8,
        ((raw >> 8) & 0xff) as u8,
        (raw & 0xff) as u8,
    ])
}

fn local_name(name: &[u8]) -> &[u8] {
    name.rsplit(|b| *b == b':').next().unwrap_or(name)
}

fn decode_tiff_channel_chunk(
    dec: &mut Decoder<BufReader<File>>,
    lvl: &TiffLevel,
    tile_y: u64,
    tile_x: u64,
    channel: usize,
) -> anyhow::Result<(usize, usize, Vec<u16>)> {
    // Channel lookup depends on the TIFF layout: chunky tiles interleave channels
    // inside one chunk, planar tiles append planes within one IFD, and separate-IFD
    // layouts route each channel through a different IFD pointer.
    if channel >= lvl.channels {
        anyhow::bail!(
            "requested TIFF channel {channel} out of range for {}",
            lvl.channels
        );
    }

    let base_index = base_chunk_index(lvl, tile_y, tile_x);
    let (ifd_pointer, chunk_index) = match lvl.channel_layout {
        TiffChannelLayout::Single | TiffChannelLayout::Chunky => (lvl.ifd_pointers[0], base_index),
        TiffChannelLayout::Planar => (
            lvl.ifd_pointers[0],
            base_index.saturating_add((channel as u32).saturating_mul(lvl.chunks_per_plane)),
        ),
        TiffChannelLayout::SeparateIfds => (lvl.ifd_pointers[channel], base_index),
    };

    dec.seek_to_ifd_pointer(ifd_pointer)
        .with_context(|| format!("seek to TIFF IFD pointer {}", ifd_pointer.0))?;
    let (w, h) = dec.chunk_data_dimensions(chunk_index);
    let decoded = dec.read_chunk(chunk_index)?;
    let data = decode_result_u16(decoded).ok_or_else(|| anyhow!("unsupported TIFF chunk dtype"))?;
    let width = w as usize;
    let height = h as usize;
    let plane_len = width.saturating_mul(height);

    let data_u16 = match lvl.channel_layout {
        TiffChannelLayout::Single | TiffChannelLayout::Planar | TiffChannelLayout::SeparateIfds => {
            if data.len() != plane_len {
                anyhow::bail!(
                    "unexpected TIFF chunk length: got {}, expected {}",
                    data.len(),
                    plane_len
                );
            }
            data
        }
        TiffChannelLayout::Chunky => {
            let expected = plane_len.saturating_mul(lvl.channels);
            if data.len() != expected {
                anyhow::bail!(
                    "unexpected chunky TIFF chunk length: got {}, expected {}",
                    data.len(),
                    expected
                );
            }
            let mut out = Vec::with_capacity(plane_len);
            for px in 0..plane_len {
                out.push(data[px * lvl.channels + channel]);
            }
            out
        }
    };

    Ok((width, height, data_u16))
}

pub fn spawn_tiff_raw_tile_loader(
    pyramid: Arc<TiffPyramid>,
    dims_yx: (usize, usize),
) -> anyhow::Result<RawTileLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<RawTileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<RawTileWorkerResponse>();

    std::thread::Builder::new()
        .name("tiff-raw-tile-loader".to_string())
        .spawn(move || {
            if let Err(err) = tiff_raw_tile_loader_thread(pyramid, dims_yx, rx_req, tx_rsp) {
                eprintln!("tiff raw tile loader exited: {err:?}");
            }
        })
        .context("spawn tiff raw tile loader")?;

    Ok(RawTileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn tiff_raw_tile_loader_thread(
    pyramid: Arc<TiffPyramid>,
    _dims_yx: (usize, usize),
    rx_req: Receiver<RawTileRequest>,
    tx_rsp: Sender<RawTileWorkerResponse>,
) -> anyhow::Result<()> {
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;

    let mut err_count: u64 = 0;
    let mut ok_count: u64 = 0;
    let mut saw_req = false;
    for req in rx_req.iter() {
        if crate::debug_log::debug_io_enabled() && !saw_req {
            saw_req = true;
            log_debug!(
                "tiff loader (raw): first request level={} tile=({}, {}) ch={}",
                req.key.level,
                req.key.tile_y,
                req.key.tile_x,
                req.key.channel
            );
        }
        let level = req.key.level;
        let Some(lvl) = pyramid.levels.get(level) else {
            continue;
        };

        let decoded = decode_tiff_channel_chunk(
            &mut dec,
            lvl,
            req.key.tile_y,
            req.key.tile_x,
            req.key.channel as usize,
        );
        let (width, height, data_u16) = match decoded {
            Ok(v) => v,
            Err(err) => {
                err_count += 1;
                if crate::debug_log::debug_io_enabled() && (err_count <= 20 || err_count % 200 == 0)
                {
                    log_warn!(
                        "tiff read_chunk failed (raw): lvl={} ifds={:?} key=({},{},ch={}) err={err:?}",
                        level,
                        lvl.ifd_pointers,
                        req.key.tile_y,
                        req.key.tile_x,
                        req.key.channel
                    );
                }
                continue;
            }
        };

        ok_count += 1;
        if crate::debug_log::debug_io_enabled() && ok_count == 1 {
            log_debug!(
                "tiff first tile ok (raw): lvl={} ifds={:?} {}x{}",
                level,
                lvl.ifd_pointers,
                width,
                height
            );
        }
        let _ = tx_rsp.send(RawTileWorkerResponse::Tile(RawTileResponse {
            key: req.key,
            width,
            height,
            data_u16,
        }));
    }

    Ok(())
}

pub fn spawn_tiff_tile_loader(
    pyramid: Arc<TiffPyramid>,
    dims_yx: (usize, usize),
) -> anyhow::Result<TileLoaderHandle> {
    // TIFF decoding is isolated in a dedicated worker so frame rendering never
    // blocks on chunk IO or format conversion.
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<TileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<TileWorkerResponse>();

    std::thread::Builder::new()
        .name("tiff-tile-loader".to_string())
        .spawn(move || {
            if let Err(err) = tiff_tile_loader_thread(pyramid, dims_yx, rx_req, tx_rsp) {
                eprintln!("tiff tile loader exited: {err:?}");
            }
        })
        .context("spawn tiff tile loader")?;

    Ok(TileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn tiff_tile_loader_thread(
    pyramid: Arc<TiffPyramid>,
    _dims_yx: (usize, usize),
    rx_req: Receiver<TileRequest>,
    tx_rsp: Sender<TileWorkerResponse>,
) -> anyhow::Result<()> {
    // Each request may need several channel chunks, which are decoded separately
    // and then composited into one RGBA tile. Failures are reported per tile so a
    // bad chunk does not tear down the entire worker thread.
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;

    let mut err_count: u64 = 0;
    let mut ok_count: u64 = 0;
    let mut saw_req = false;
    for req in rx_req.iter() {
        if crate::debug_log::debug_io_enabled() && !saw_req {
            saw_req = true;
            log_debug!(
                "tiff loader: first request level={} tile=({}, {})",
                req.key.level,
                req.key.tile_y,
                req.key.tile_x
            );
        }
        let TileKey {
            level,
            tile_y,
            tile_x,
            ..
        } = req.key;
        let Some(lvl) = pyramid.levels.get(level) else {
            continue;
        };

        let channels = if req.channels.is_empty() {
            vec![RenderChannel {
                index: 0,
                color_rgb: [1.0, 1.0, 1.0],
                window: (0.0, pyramid.abs_max),
            }]
        } else {
            req.channels.clone()
        };

        let mut width = 0usize;
        let mut height = 0usize;
        let mut acc: Vec<f32> = Vec::new();
        let mut failed = false;

        for ch in &channels {
            let decoded =
                decode_tiff_channel_chunk(&mut dec, lvl, tile_y, tile_x, ch.index as usize);
            let (w, h, data_u16) = match decoded {
                Ok(v) => v,
                Err(err) => {
                    err_count += 1;
                    failed = true;
                    if crate::debug_log::debug_io_enabled()
                        && (err_count <= 20 || err_count % 200 == 0)
                    {
                        log_warn!(
                            "tiff read_chunk failed: lvl={} ifds={:?} key=({},{},ch={}) err={err:?}",
                            level,
                            lvl.ifd_pointers,
                            tile_y,
                            tile_x,
                            ch.index
                        );
                    }
                    break;
                }
            };
            if acc.is_empty() {
                width = w;
                height = h;
                acc.resize(width.saturating_mul(height).saturating_mul(3), 0.0);
            } else if width != w || height != h {
                failed = true;
                break;
            }

            let (w0, w1) = ch.window;
            let denom = (w1 - w0).max(1.0);
            for (i, &val) in data_u16.iter().enumerate() {
                let t = ((val as f32 - w0) / denom).clamp(0.0, 1.0);
                acc[i * 3 + 0] += t * ch.color_rgb[0];
                acc[i * 3 + 1] += t * ch.color_rgb[1];
                acc[i * 3 + 2] += t * ch.color_rgb[2];
            }
        }
        if failed || width == 0 || height == 0 {
            continue;
        }

        ok_count += 1;
        if crate::debug_log::debug_io_enabled() && ok_count == 1 {
            log_debug!(
                "tiff first tile ok: lvl={} ifds={:?} {}x{}",
                level,
                lvl.ifd_pointers,
                width,
                height
            );
        }

        let mut rgba = vec![0u8; width * height * 4];
        for i in 0..(width * height) {
            rgba[i * 4 + 0] = (acc[i * 3 + 0].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 1] = (acc[i * 3 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 2] = (acc[i * 3 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba[i * 4 + 3] = 255;
        }

        let _ = tx_rsp.send(TileWorkerResponse::Tile(TileResponse {
            key: req.key,
            width,
            height,
            rgba,
        }));
    }

    Ok(())
}

pub fn spawn_tiff_histogram_loader(
    pyramid: Arc<TiffPyramid>,
) -> anyhow::Result<HistogramLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<HistogramRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<HistogramResponse>();

    std::thread::Builder::new()
        .name("tiff-hist-loader".to_string())
        .spawn(move || {
            if let Err(err) = tiff_histogram_loader_thread(pyramid, rx_req, tx_rsp) {
                eprintln!("tiff histogram loader exited: {err:?}");
            }
        })
        .context("spawn tiff histogram loader")?;

    Ok(HistogramLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn tiff_histogram_loader_thread(
    pyramid: Arc<TiffPyramid>,
    rx_req: Receiver<HistogramRequest>,
    tx_rsp: Sender<HistogramResponse>,
) -> anyhow::Result<()> {
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;

    for req in rx_req.iter() {
        let Some(lvl) = pyramid.levels.get(req.level) else {
            continue;
        };
        let bins = req.bins.clamp(8, 4096);
        let abs_max = if req.abs_max.is_finite() && req.abs_max > 0.0 {
            req.abs_max
        } else {
            pyramid.abs_max.max(1.0)
        };

        let y0 = req.y0.min(lvl.height as u64);
        let y1 = req.y1.min(lvl.height as u64).max(y0);
        let x0 = req.x0.min(lvl.width as u64);
        let x1 = req.x1.min(lvl.width as u64).max(x0);
        if y1 <= y0 || x1 <= x0 {
            let _ = tx_rsp.send(HistogramResponse {
                request_id: req.request_id,
                bins: vec![0u32; bins],
                stats: None,
            });
            continue;
        }

        let tile_y0 = (y0 / lvl.chunk_h.max(1) as u64) as u32;
        let tile_y1 = ((y1 - 1) / lvl.chunk_h.max(1) as u64) as u32;
        let tile_x0 = (x0 / lvl.chunk_w.max(1) as u64) as u32;
        let tile_x1 = ((x1 - 1) / lvl.chunk_w.max(1) as u64) as u32;

        let mut values = Vec::<u16>::new();
        for tile_y in tile_y0..=tile_y1 {
            for tile_x in tile_x0..=tile_x1 {
                let Ok((width, height, data_u16)) = decode_tiff_channel_chunk(
                    &mut dec,
                    lvl,
                    tile_y as u64,
                    tile_x as u64,
                    req.channel as usize,
                ) else {
                    continue;
                };
                let tile_origin_y = tile_y as u64 * lvl.chunk_h as u64;
                let tile_origin_x = tile_x as u64 * lvl.chunk_w as u64;
                let local_y0 = y0.saturating_sub(tile_origin_y).min(height as u64) as usize;
                let local_y1 = y1
                    .saturating_sub(tile_origin_y)
                    .min(height as u64)
                    .max(local_y0 as u64) as usize;
                let local_x0 = x0.saturating_sub(tile_origin_x).min(width as u64) as usize;
                let local_x1 = x1
                    .saturating_sub(tile_origin_x)
                    .min(width as u64)
                    .max(local_x0 as u64) as usize;
                for row in local_y0..local_y1 {
                    let start = row.saturating_mul(width).saturating_add(local_x0);
                    let end = row.saturating_mul(width).saturating_add(local_x1);
                    values.extend_from_slice(&data_u16[start..end]);
                }
            }
        }

        let _ = tx_rsp.send(HistogramResponse {
            request_id: req.request_id,
            bins: compute_hist_u16(&values, bins, abs_max),
            stats: compute_stats_u16(&values),
        });
    }

    Ok(())
}

pub fn spawn_tiff_channel_max_loader(
    pyramid: Arc<TiffPyramid>,
) -> anyhow::Result<ChannelMaxLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<ChannelMaxRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<ChannelMaxResponse>();

    std::thread::Builder::new()
        .name("tiff-chan-max-loader".to_string())
        .spawn(move || {
            if let Err(err) = tiff_channel_max_loader_thread(pyramid, rx_req, tx_rsp) {
                eprintln!("tiff channel max loader exited: {err:?}");
            }
        })
        .context("spawn tiff channel max loader")?;

    Ok(ChannelMaxLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
    })
}

fn tiff_channel_max_loader_thread(
    pyramid: Arc<TiffPyramid>,
    rx_req: Receiver<ChannelMaxRequest>,
    tx_rsp: Sender<ChannelMaxResponse>,
) -> anyhow::Result<()> {
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;

    for req in rx_req.iter() {
        let Some(lvl) = pyramid.levels.get(req.level) else {
            continue;
        };

        let mut hist = vec![0u64; 65536];
        let mut n: u64 = 0;
        let mut max_v: u16 = 0;
        for tile_y in 0..lvl.tiles_y {
            for tile_x in 0..lvl.tiles_x {
                let Ok((_width, _height, data_u16)) = decode_tiff_channel_chunk(
                    &mut dec,
                    lvl,
                    tile_y as u64,
                    tile_x as u64,
                    req.channel as usize,
                ) else {
                    continue;
                };
                for v in data_u16 {
                    hist[v as usize] = hist[v as usize].saturating_add(1);
                    n = n.saturating_add(1);
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
        }

        let p97 = if n == 0 {
            0
        } else {
            let target = (n.saturating_mul(97).saturating_add(99)) / 100;
            let mut acc: u64 = 0;
            let mut out: u16 = 0;
            for (i, c) in hist.iter().enumerate() {
                acc = acc.saturating_add(*c);
                if acc >= target {
                    out = i as u16;
                    break;
                }
            }
            out
        };

        let _ = tx_rsp.send(ChannelMaxResponse {
            request_id: req.request_id,
            channel: req.channel,
            p97,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        TiffPlaneSelection, TiffPyramid, ome_channel_ifd_order, ome_multichannel_plane_index,
        parse_ome_xml,
    };

    #[test]
    fn parses_ome_xml_channel_names_and_sizes() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" SizeX="10" SizeY="20" SizeZ="1" SizeC="3" SizeT="1" PhysicalSizeX="0.65" PhysicalSizeXUnit="µm" PhysicalSizeY="0.70" PhysicalSizeYUnit="µm">
      <Channel ID="Channel:0:0" Name="CD3" Color="16711680"/>
      <Channel ID="Channel:0:1" Name="PanCK" Color="65280"/>
      <Channel ID="Channel:0:2" Name="DAPI" Color="255"/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        assert_eq!(meta.size_c, Some(3));
        assert_eq!(meta.size_z, Some(1));
        assert_eq!(meta.size_t, Some(1));
        assert_eq!(meta.physical_size_x, Some(0.65));
        assert_eq!(meta.physical_size_y, Some(0.70));
        assert_eq!(meta.channels.len(), 3);
        assert_eq!(meta.channels[0].name.as_deref(), Some("CD3"));
        assert_eq!(meta.channels[1].color_rgb, Some([0, 255, 0]));
        assert_eq!(meta.channels[2].color_rgb, Some([0, 0, 255]));
    }

    #[test]
    fn parses_ome_xml_tiff_data_mapping() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" SizeX="10" SizeY="20" SizeZ="1" SizeC="3" SizeT="1">
      <Channel ID="Channel:0:0" Name="A"/>
      <Channel ID="Channel:0:1" Name="B"/>
      <Channel ID="Channel:0:2" Name="C"/>
      <TiffData IFD="0" FirstC="1" PlaneCount="1"/>
      <TiffData IFD="1" FirstC="0" PlaneCount="1"/>
      <TiffData IFD="2" FirstC="2" PlaneCount="1"/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        assert_eq!(meta.tiff_data.len(), 3);
        assert_eq!(meta.tiff_data[0].ifd, Some(0));
        assert_eq!(meta.tiff_data[0].first_c, Some(1));
        assert_eq!(meta.tiff_data[1].ifd, Some(1));
        assert_eq!(meta.tiff_data[1].first_c, Some(0));
        assert_eq!(meta.tiff_data[2].ifd, Some(2));
        assert_eq!(meta.tiff_data[2].first_c, Some(2));
    }

    #[test]
    fn derives_channel_order_from_tiff_data() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" SizeX="10" SizeY="20" SizeZ="1" SizeC="3" SizeT="1">
      <TiffData IFD="0" FirstC="1" PlaneCount="1"/>
      <TiffData IFD="1" FirstC="0" PlaneCount="1"/>
      <TiffData IFD="2" FirstC="2" PlaneCount="1"/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        let order = ome_channel_ifd_order(&meta, 3, 0, 0)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn derives_default_channel_order_from_bare_tiff_data() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" SizeX="10" SizeY="20" SizeZ="1" SizeC="3" SizeT="1">
      <TiffData/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        let order = ome_channel_ifd_order(&meta, 3, 0, 0)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn derives_channel_order_for_z_plane_from_default_ome_mapping() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYZCT" SizeX="10" SizeY="20" SizeZ="2" SizeC="3" SizeT="1">
      <TiffData/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        let order_z0 = ome_channel_ifd_order(&meta, 3, 0, 0)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        let order_z1 = ome_channel_ifd_order(&meta, 3, 1, 0)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        assert_eq!(order_z0, vec![0, 2, 4]);
        assert_eq!(order_z1, vec![1, 3, 5]);
    }

    #[test]
    fn derives_channel_order_for_timepoint_from_default_ome_mapping() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYCZT" SizeX="10" SizeY="20" SizeZ="2" SizeC="3" SizeT="2">
      <TiffData/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        let order_t0_z0 = ome_channel_ifd_order(&meta, 3, 0, 0)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        let order_t1_z0 = ome_channel_ifd_order(&meta, 3, 0, 1)
            .expect("derive TIFF channel order")
            .expect("tiff data mapping present");
        assert_eq!(order_t0_z0, vec![0, 1, 2]);
        assert_eq!(order_t1_z0, vec![6, 7, 8]);
    }

    #[test]
    fn derives_multichannel_plane_index_from_dimension_order() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
    <Pixels DimensionOrder="XYTCZ" SizeX="10" SizeY="20" SizeZ="3" SizeC="4" SizeT="2">
      <TiffData/>
    </Pixels>
  </Image>
</OME>"#;

        let meta = parse_ome_xml(xml).expect("parse OME XML");
        assert_eq!(
            ome_multichannel_plane_index(&meta, 0, 0).expect("plane index"),
            0
        );
        assert_eq!(
            ome_multichannel_plane_index(&meta, 2, 0).expect("plane index"),
            4
        );
        assert_eq!(
            ome_multichannel_plane_index(&meta, 0, 1).expect("plane index"),
            1
        );
    }

    #[test]
    fn opens_imagej_hyperstack_fixture_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("1.tif");
        if !path.exists() {
            return;
        }

        let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
            .expect("open TIFF fixture");
        assert_eq!(pyramid.channel_count, 64);
        assert_eq!(pyramid.levels.len(), 1);
        assert_eq!(pyramid.levels[0].channels, 64);
    }

    #[test]
    fn opens_pyramidal_ome_fixture_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("1_pyramid_crop.ome.tif");
        if !path.exists() {
            return;
        }

        let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
            .expect("open pyramidal OME-TIFF fixture");
        assert_eq!(pyramid.channel_count, 64);
        assert_eq!(pyramid.levels.len(), 4);
        assert_eq!(pyramid.levels[0].width, 512);
        assert_eq!(pyramid.levels[1].width, 256);
        assert_eq!(pyramid.levels[2].width, 128);
        assert_eq!(pyramid.levels[3].width, 64);
    }
}
