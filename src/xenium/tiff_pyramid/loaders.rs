//! TIFF chunk decoding and asynchronous tile/statistics loaders.

use super::*;

pub(super) fn base_chunk_index(lvl: &TiffLevel, tile_y: u64, tile_x: u64) -> u32 {
    match lvl.chunk_type {
        ChunkType::Tile => {
            let ix = tile_x.min(lvl.tiles_x.saturating_sub(1) as u64) as u32;
            let iy = tile_y.min(lvl.tiles_y.saturating_sub(1) as u64) as u32;
            iy.saturating_mul(lvl.tiles_x).saturating_add(ix)
        }
        ChunkType::Strip => tile_y.min(lvl.tiles_y.saturating_sub(1) as u64) as u32,
    }
}

pub(super) fn decode_result_u16(decoded: DecodingResult) -> Option<Vec<u16>> {
    match decoded {
        DecodingResult::U16(v) => Some(v),
        DecodingResult::U8(v) => Some(v.into_iter().map(|b| b as u16).collect()),
        _ => None,
    }
}

pub(super) fn decode_tiff_channel_chunk(
    dec: &mut Decoder<BufReader<File>>,
    current_ifd: &mut Option<IfdPointer>,
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

    if *current_ifd != Some(ifd_pointer) {
        dec.seek_to_ifd_pointer(ifd_pointer)
            .with_context(|| format!("seek to TIFF IFD pointer {}", ifd_pointer.0))?;
        *current_ifd = Some(ifd_pointer);
    }
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

impl TiffPyramid {
    pub(crate) fn read_channel_region_u16(
        &self,
        level: usize,
        channel: usize,
        y0: u64,
        y1: u64,
        x0: u64,
        x1: u64,
    ) -> anyhow::Result<(Vec<u16>, usize, usize)> {
        let lvl = self
            .levels
            .get(level)
            .ok_or_else(|| anyhow::anyhow!("TIFF level {level} is out of range"))?;
        let y0 = y0.min(lvl.height as u64);
        let y1 = y1.min(lvl.height as u64);
        let x0 = x0.min(lvl.width as u64);
        let x1 = x1.min(lvl.width as u64);
        anyhow::ensure!(y1 > y0 && x1 > x0, "TIFF intensity region is empty");
        let tile_y0 = (y0 / lvl.chunk_h.max(1) as u64) as u32;
        let tile_y1 = ((y1 - 1) / lvl.chunk_h.max(1) as u64) as u32;
        let tile_x0 = (x0 / lvl.chunk_w.max(1) as u64) as u32;
        let tile_x1 = ((x1 - 1) / lvl.chunk_w.max(1) as u64) as u32;
        let file = File::open(&self.path)?;
        let mut decoder = Decoder::new(BufReader::new(file))?;
        let mut current_ifd = None;
        let mut values =
            Vec::with_capacity(usize::try_from((y1 - y0).saturating_mul(x1 - x0)).unwrap_or(0));
        for tile_y in tile_y0..=tile_y1 {
            for tile_x in tile_x0..=tile_x1 {
                let (width, height, data) = decode_tiff_channel_chunk(
                    &mut decoder,
                    &mut current_ifd,
                    lvl,
                    tile_y as u64,
                    tile_x as u64,
                    channel,
                )?;
                let origin_y = tile_y as u64 * lvl.chunk_h as u64;
                let origin_x = tile_x as u64 * lvl.chunk_w as u64;
                let local_y0 = y0.saturating_sub(origin_y).min(height as u64) as usize;
                let local_y1 = y1.saturating_sub(origin_y).min(height as u64) as usize;
                let local_x0 = x0.saturating_sub(origin_x).min(width as u64) as usize;
                let local_x1 = x1.saturating_sub(origin_x).min(width as u64) as usize;
                for row in local_y0..local_y1 {
                    let start = row.saturating_mul(width).saturating_add(local_x0);
                    let end = row.saturating_mul(width).saturating_add(local_x1);
                    values.extend_from_slice(&data[start..end]);
                }
            }
        }
        Ok((values, (x1 - x0) as usize, (y1 - y0) as usize))
    }
}

pub fn spawn_tiff_raw_tile_loader(
    pyramid: Arc<TiffPyramid>,
    dims_yx: (usize, usize),
    worker_threads: usize,
) -> anyhow::Result<RawTileLoaderHandle> {
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<RawTileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<RawTileWorkerResponse>();
    let threads = worker_threads.max(1);
    let active_keys = Arc::new(Mutex::new(HashSet::new()));

    for worker_idx in 0..threads {
        let pyramid = Arc::clone(&pyramid);
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let active_keys = Arc::clone(&active_keys);
        std::thread::Builder::new()
            .name(format!("tiff-raw-tile-loader-{worker_idx}"))
            .spawn(move || {
                if let Err(err) =
                    tiff_raw_tile_loader_thread(pyramid, dims_yx, rx_req, tx_rsp, active_keys)
                {
                    eprintln!("tiff raw tile loader worker {worker_idx} exited: {err:?}");
                }
            })
            .context("spawn tiff raw tile loader")?;
    }

    Ok(RawTileLoaderHandle {
        tx: tx_req,
        rx: rx_rsp,
        active_keys,
    })
}

pub(super) fn tiff_raw_tile_loader_thread(
    pyramid: Arc<TiffPyramid>,
    _dims_yx: (usize, usize),
    rx_req: Receiver<RawTileRequest>,
    tx_rsp: Sender<RawTileWorkerResponse>,
    active_keys: Arc<Mutex<HashSet<RawTileKey>>>,
) -> anyhow::Result<()> {
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;
    let mut current_ifd: Option<IfdPointer> = None;

    let mut err_count: u64 = 0;
    let mut ok_count: u64 = 0;
    let mut saw_req = false;
    for req in rx_req.iter() {
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
        }
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
            &mut current_ifd,
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
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
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
    worker_threads: usize,
) -> anyhow::Result<TileLoaderHandle> {
    // TIFF decoding is isolated in workers so frame rendering never blocks on
    // chunk IO or format conversion. Each worker owns a file handle and decoder
    // because TIFF decoders are stateful around the current IFD.
    let (tx_req, rx_req) = crossbeam_channel::unbounded::<TileRequest>();
    let (tx_rsp, rx_rsp) = crossbeam_channel::unbounded::<TileWorkerResponse>();
    let threads = worker_threads.max(1);
    let active_render_ids = Arc::new(Mutex::new(HashSet::new()));
    let active_keys = Arc::new(Mutex::new(HashSet::new()));
    let decoded_tiles = CpuDecodedTileCache::new(8192);

    for worker_idx in 0..threads {
        let pyramid = Arc::clone(&pyramid);
        let rx_req = rx_req.clone();
        let tx_rsp = tx_rsp.clone();
        let active_render_ids = Arc::clone(&active_render_ids);
        let active_keys = Arc::clone(&active_keys);
        let decoded_tiles = Arc::clone(&decoded_tiles);
        std::thread::Builder::new()
            .name(format!("tiff-tile-loader-{worker_idx}"))
            .spawn(move || {
                if let Err(err) = tiff_tile_loader_thread(
                    pyramid,
                    dims_yx,
                    rx_req,
                    tx_rsp,
                    active_render_ids,
                    active_keys,
                    decoded_tiles,
                ) {
                    eprintln!("tiff tile loader worker {worker_idx} exited: {err:?}");
                }
            })
            .context("spawn tiff tile loader")?;
    }

    Ok(TileLoaderHandle::with_decoded_tiles(
        tx_req,
        rx_rsp,
        active_render_ids,
        active_keys,
        decoded_tiles,
    ))
}

pub(super) fn tiff_tile_loader_thread(
    pyramid: Arc<TiffPyramid>,
    _dims_yx: (usize, usize),
    rx_req: Receiver<TileRequest>,
    tx_rsp: Sender<TileWorkerResponse>,
    active_render_ids: Arc<Mutex<HashSet<u64>>>,
    active_keys: Arc<Mutex<HashSet<TileKey>>>,
    decoded_tiles: Arc<CpuDecodedTileCache>,
) -> anyhow::Result<()> {
    // Each request may need several channel chunks, which are decoded separately
    // and then composited into one RGBA tile. Failures are reported per tile so a
    // bad chunk does not tear down the entire worker thread.
    let f = File::open(&pyramid.path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;
    let mut current_ifd: Option<IfdPointer> = None;

    let mut err_count: u64 = 0;
    let mut ok_count: u64 = 0;
    let mut saw_req = false;
    for req in rx_req.iter() {
        if let Ok(active) = active_render_ids.lock()
            && !active.is_empty()
            && !active.contains(&req.key.render_id)
        {
            continue;
        }
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
        }
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
            let decoded_key = DecodedTileKey {
                view: req.key.view,
                level,
                tile_y,
                tile_x,
                channel: ch.index,
            };
            let decoded = decoded_tiles.get_or_decode(decoded_key, || {
                decode_tiff_channel_chunk(
                    &mut dec,
                    &mut current_ifd,
                    lvl,
                    tile_y,
                    tile_x,
                    ch.index as usize,
                )
                .map(|(width, height, values)| DecodedTile {
                    width,
                    height,
                    values: Arc::new(values),
                })
                .map_err(|error| error.to_string())
            });
            let decoded = match decoded {
                Ok(value) => value,
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
                width = decoded.width;
                height = decoded.height;
                acc.resize(width.saturating_mul(height).saturating_mul(3), 0.0);
            } else if width != decoded.width || height != decoded.height {
                failed = true;
                break;
            }

            let (w0, w1) = ch.window;
            let denom = (w1 - w0).max(1.0);
            for (i, &val) in decoded.values.iter().enumerate() {
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

        if let Ok(active) = active_render_ids.lock()
            && !active.is_empty()
            && !active.contains(&req.key.render_id)
        {
            continue;
        }
        if let Ok(active) = active_keys.lock()
            && !active.is_empty()
            && !active.contains(&req.key)
        {
            continue;
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
