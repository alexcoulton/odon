//! OME-XML metadata extraction and parsing.

use super::*;

pub(super) fn read_ome_tiff_metadata(
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

pub(super) fn parse_ome_xml(xml: &str) -> anyhow::Result<OmeTiffMetadata> {
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

pub(super) fn parse_channel(
    e: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> anyhow::Result<OmeTiffChannel> {
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

pub(super) fn parse_tiff_data(
    e: &BytesStart<'_>,
    reader: &Reader<&[u8]>,
) -> anyhow::Result<OmeTiffData> {
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

pub(super) fn apply_pixels_attrs(
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

pub(super) fn parse_ome_color(s: &str) -> Option<[u8; 3]> {
    let raw = s.trim().parse::<i32>().ok()? as u32;
    Some([
        ((raw >> 16) & 0xff) as u8,
        ((raw >> 8) & 0xff) as u8,
        (raw & 0xff) as u8,
    ])
}

pub(super) fn local_name(name: &[u8]) -> &[u8] {
    name.rsplit(|b| *b == b':').next().unwrap_or(name)
}
