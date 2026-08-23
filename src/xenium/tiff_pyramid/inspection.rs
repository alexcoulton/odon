//! TIFF IFD inspection, channel ordering, and pyramid-level construction.

use super::*;

pub(super) fn color_type_meta(color_type: tiff::ColorType) -> anyhow::Result<(usize, String, f32)> {
    match color_type {
        tiff::ColorType::Gray(8) => Ok((1, "|u1".to_string(), 255.0)),
        tiff::ColorType::Gray(16) => Ok((1, "<u2".to_string(), 65535.0)),
        tiff::ColorType::RGB(8) => Ok((3, "|u1".to_string(), 255.0)),
        tiff::ColorType::RGB(16) => Ok((3, "<u2".to_string(), 65535.0)),
        other => Err(anyhow!("unsupported TIFF color type: {other:?}")),
    }
}

pub(super) fn open_decoder(path: &Path) -> anyhow::Result<Decoder<BufReader<File>>> {
    let f = File::open(path).with_context(|| format!("open tiff: {path:?}"))?;
    Decoder::new(BufReader::new(f)).context("tiff decoder")
}

pub(super) fn current_ifd_info(
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

pub(super) fn inspect_ifd_pointer(path: &Path, ifd_pointer: IfdPointer) -> anyhow::Result<IfdInfo> {
    let mut dec = open_decoder(path)?;
    dec.seek_to_ifd_pointer(ifd_pointer)
        .with_context(|| format!("seek to TIFF IFD pointer {}", ifd_pointer.0))?;
    current_ifd_info(&mut dec, usize::MAX)
}

pub(super) fn same_channel_model(a: &IfdInfo, b: &IfdInfo) -> bool {
    a.channels == b.channels
        && a.channel_layout == b.channel_layout
        && a.pixel_dtype == b.pixel_dtype
        && a.abs_max == b.abs_max
}

pub(super) fn same_geometry(a: &IfdInfo, b: &IfdInfo) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.chunk_type == b.chunk_type
        && a.chunk_w == b.chunk_w
        && a.chunk_h == b.chunk_h
        && a.pixel_dtype == b.pixel_dtype
        && a.channels == 1
        && b.channels == 1
}

pub(super) fn build_level(group: &[IfdInfo]) -> anyhow::Result<TiffLevel> {
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

pub(super) fn separate_ifd_group_ranges(ifds: &[IfdInfo]) -> Vec<(usize, usize)> {
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

pub(super) fn ome_plane_axis_order(ome: &OmeTiffMetadata) -> anyhow::Result<[char; 3]> {
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

pub(super) fn ome_plane_axis_size(
    axis: char,
    size_z: usize,
    size_c: usize,
    size_t: usize,
) -> usize {
    match axis {
        'Z' => size_z,
        'C' => size_c,
        'T' => size_t,
        _ => 0,
    }
}

pub(super) fn ome_plane_axis_coord(axis: char, z: usize, c: usize, t: usize) -> usize {
    match axis {
        'Z' => z,
        'C' => c,
        'T' => t,
        _ => 0,
    }
}

pub(super) fn ome_linear_plane_index(
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

pub(super) fn ome_plane_coords_from_linear(
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

pub(super) fn ome_channel_ifd_order(
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

pub(super) fn reorder_ifd_group_by_tiff_data(
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

pub(super) fn select_base_ifd_group(
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

    // Equal-sized separate IFDs are ambiguous from geometry alone: they may be channels,
    // Z/T planes, or both. OME metadata already gives the exact IFD for every channel in
    // the requested plane, so use that mapping directly instead of relying on geometry
    // group boundaries.
    let selected = channel_ifds
        .iter()
        .map(|ifd_index| {
            ifds.iter()
                .find(|ifd| ifd.main_ifd_index == *ifd_index)
                .cloned()
                .ok_or_else(|| anyhow!("OME-TIFF mapping references missing IFD {ifd_index}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if let Some(first) = selected.first()
        && selected.iter().any(|ifd| !same_geometry(first, ifd))
    {
        return Err(anyhow!(
            "OME-TIFF channels for Z={}, T={} do not share geometry",
            plane_selection.z,
            plane_selection.t
        ));
    }
    Ok((0, selected))
}

pub(super) fn ome_multichannel_plane_order(ome: &OmeTiffMetadata) -> anyhow::Result<[char; 2]> {
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

pub(super) fn ome_multichannel_plane_index(
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

pub(super) fn select_multichannel_base_ifd<'a>(
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

pub(super) fn build_multichannel_levels(
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

pub(super) fn build_separate_ifd_levels(
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

pub(super) fn build_levels_from_main_ifds(
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
