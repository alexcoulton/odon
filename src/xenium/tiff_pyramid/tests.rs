//! TIFF and OME-TIFF format, loader, and plane-order regression tests.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tiff::encoder::{TiffEncoder, colortype};
use tiff::tags::Tag;

use super::{
    TiffChannelLayout, TiffPlaneSelection, TiffPyramid, decode_tiff_channel_chunk,
    ome_channel_ifd_order, ome_multichannel_plane_index, open_decoder, parse_ome_xml,
    spawn_tiff_tile_loader,
};
use crate::imaging::view_plane::{ViewPlaneMode, ViewPlaneSelection};
use crate::render::tiles::{RenderChannel, TileKey, TileRequest, TileWorkerResponse};

static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

struct TestTiffDir {
    path: PathBuf,
}

impl TestTiffDir {
    fn new() -> Self {
        let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("odon-tiff-tests-{}-{sequence}", std::process::id()));
        fs::create_dir_all(&path).expect("create TIFF test directory");
        Self { path }
    }

    fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestTiffDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn decode_channel(path: &Path, pyramid: &TiffPyramid, channel: usize) -> Vec<u16> {
    let mut decoder = open_decoder(path).expect("open generated TIFF decoder");
    let mut current_ifd = None;
    let (width, height, pixels) = decode_tiff_channel_chunk(
        &mut decoder,
        &mut current_ifd,
        &pyramid.levels[0],
        0,
        0,
        channel,
    )
    .expect("decode generated TIFF channel");
    assert_eq!((width, height), (8, 6));
    pixels
}

#[test]
fn opens_generated_grayscale_tiff_and_decodes_pixels() {
    let dir = TestTiffDir::new();
    let path = dir.file("grayscale.tif");
    let pixels = (0..48).map(|value| value as u8).collect::<Vec<_>>();

    let file = File::create(&path).expect("create grayscale TIFF");
    let mut encoder = TiffEncoder::new(file).expect("create TIFF encoder");
    encoder
        .new_image::<colortype::Gray8>(8, 6)
        .expect("create grayscale image")
        .write_data(&pixels)
        .expect("write grayscale image");

    let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open generated grayscale TIFF");

    assert_eq!(pyramid.channel_count, 1);
    assert_eq!(pyramid.levels.len(), 1);
    assert_eq!(pyramid.levels[0].channel_layout, TiffChannelLayout::Single);
    assert_eq!(pyramid.pixel_dtype, "|u1");
    assert_eq!(pyramid.to_levels_info()[0].shape, vec![6, 8]);
    assert_eq!(
        decode_channel(&path, &pyramid, 0),
        pixels.iter().map(|&value| value as u16).collect::<Vec<_>>()
    );
}

#[test]
fn composited_tiff_loader_shares_decode_across_viewport_presentations() {
    let dir = TestTiffDir::new();
    let path = dir.file("shared-decode.tif");
    let pixels = (0..48).map(|value| value as u8).collect::<Vec<_>>();
    let file = File::create(&path).expect("create grayscale TIFF");
    TiffEncoder::new(file)
        .expect("create TIFF encoder")
        .new_image::<colortype::Gray8>(8, 6)
        .expect("create grayscale image")
        .write_data(&pixels)
        .expect("write grayscale image");
    let pyramid = Arc::new(
        TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
            .expect("open generated grayscale TIFF"),
    );
    let loader = spawn_tiff_tile_loader(pyramid, (6, 8), 1).expect("spawn TIFF loader");
    let left = TileKey {
        render_id: 51,
        view: ViewPlaneSelection {
            mode: ViewPlaneMode::Xy,
            slice_level0: 0,
        },
        level: 0,
        tile_y: 0,
        tile_x: 0,
    };
    let right = TileKey {
        render_id: 52,
        ..left
    };
    loader.set_active_render_ids(HashSet::from([left.render_id, right.render_id]));
    loader.set_active_keys(HashSet::from([left, right]));
    for (key, color_rgb) in [(left, [1.0, 0.0, 0.0]), (right, [0.0, 1.0, 0.0])] {
        loader
            .tx
            .send(TileRequest {
                key,
                channels: vec![RenderChannel {
                    index: 0,
                    color_rgb,
                    window: (0.0, 255.0),
                }],
            })
            .expect("queue viewport tile");
    }

    let mut responses = HashMap::new();
    for _ in 0..2 {
        let response = loader
            .rx
            .recv_timeout(Duration::from_secs(5))
            .expect("tile completion");
        let TileWorkerResponse::Tile(tile) = response else {
            panic!("expected composited tile");
        };
        responses.insert(tile.key.render_id, tile.rgba);
    }
    assert_ne!(responses[&left.render_id], responses[&right.render_id]);
    let stats = loader.stats();
    assert_eq!(stats.decode_requests, 2);
    assert_eq!(stats.source_reads, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.decoded_cache_entries, 1);
}

#[test]
fn opens_generated_two_channel_ome_tiff_and_decodes_ifds() {
    let dir = TestTiffDir::new();
    let path = dir.file("two-channel.ome.tif");
    let ome_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
<Pixels DimensionOrder="XYZCT" Type="uint8" SizeX="8" SizeY="6" SizeZ="1" SizeC="2" SizeT="1" PhysicalSizeX="0.5" PhysicalSizeXUnit="um" PhysicalSizeY="0.75" PhysicalSizeYUnit="um">
  <Channel ID="Channel:0:0" Name="DAPI" Color="255"/>
  <Channel ID="Channel:0:1" Name="CD3" Color="16711680"/>
  <TiffData/>
</Pixels>
  </Image>
</OME>"#;
    let dapi = vec![7u8; 48];
    let cd3 = vec![19u8; 48];

    let file = File::create(&path).expect("create OME-TIFF");
    let mut encoder = TiffEncoder::new(file).expect("create TIFF encoder");
    {
        let mut image = encoder
            .new_image::<colortype::Gray8>(8, 6)
            .expect("create first OME-TIFF channel");
        image
            .encoder()
            .write_tag(Tag::ImageDescription, ome_xml)
            .expect("write OME metadata");
        image.write_data(&dapi).expect("write DAPI pixels");
    }
    encoder
        .new_image::<colortype::Gray8>(8, 6)
        .expect("create second OME-TIFF channel")
        .write_data(&cd3)
        .expect("write CD3 pixels");

    let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open generated OME-TIFF");

    assert_eq!(pyramid.channel_count, 2);
    assert_eq!(pyramid.levels.len(), 1);
    assert_eq!(
        pyramid.levels[0].channel_layout,
        TiffChannelLayout::SeparateIfds
    );
    assert_eq!(pyramid.to_levels_info()[0].shape, vec![2, 6, 8]);
    assert_eq!(
        pyramid.physical_pixel_size_xy(),
        Some((
            [0.75, 0.5],
            [Some("um".to_string()), Some("um".to_string())]
        ))
    );
    let channels = pyramid.default_channels_named("unused");
    assert_eq!(channels[0].name, "DAPI");
    assert_eq!(channels[0].color_rgb, [0, 0, 255]);
    assert_eq!(channels[1].name, "CD3");
    assert_eq!(channels[1].color_rgb, [255, 0, 0]);
    assert_eq!(decode_channel(&path, &pyramid, 0), vec![7u16; 48]);
    assert_eq!(decode_channel(&path, &pyramid, 1), vec![19u16; 48]);
}

#[test]
fn generated_ome_tiff_selects_distinct_z_planes() {
    let dir = TestTiffDir::new();
    let path = dir.file("two-channel-two-z.ome.tif");
    let ome_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0">
<Pixels DimensionOrder="XYZCT" Type="uint8" SizeX="8" SizeY="6" SizeZ="2" SizeC="2" SizeT="1">
  <Channel ID="Channel:0:0" Name="DAPI"/>
  <Channel ID="Channel:0:1" Name="CD3"/>
  <TiffData/>
</Pixels>
  </Image>
</OME>"#;
    let file = File::create(&path).expect("create OME-TIFF");
    let mut encoder = TiffEncoder::new(file).expect("create TIFF encoder");
    for (ifd, value) in [10u8, 11, 20, 21].into_iter().enumerate() {
        let mut image = encoder
            .new_image::<colortype::Gray8>(8, 6)
            .expect("create plane");
        if ifd == 0 {
            image
                .encoder()
                .write_tag(Tag::ImageDescription, ome_xml)
                .expect("write OME metadata");
        }
        image
            .write_data(&vec![value; 48])
            .expect("write plane pixels");
    }

    let z0 = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open z0");
    let z1 = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 1, t: 0 })
        .expect("open z1");
    assert_eq!((z0.size_z, z0.size_t), (2, 1));
    assert_eq!(decode_channel(&path, &z0, 0), vec![10; 48]);
    assert_eq!(decode_channel(&path, &z0, 1), vec![20; 48]);
    assert_eq!(decode_channel(&path, &z1, 0), vec![11; 48]);
    assert_eq!(decode_channel(&path, &z1, 1), vec![21; 48]);
    assert!(
        TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 2, t: 0 }).is_err(),
        "out-of-range z must fail"
    );
}

#[test]
fn opens_generated_chunky_rgb_tiff_and_extracts_each_channel() {
    let dir = TestTiffDir::new();
    let path = dir.file("rgb.tif");
    let mut pixels = Vec::with_capacity(8 * 6 * 3);
    let mut red = Vec::with_capacity(8 * 6);
    let mut green = Vec::with_capacity(8 * 6);
    let mut blue = Vec::with_capacity(8 * 6);
    for y in 0..6u8 {
        for x in 0..8u8 {
            pixels.extend_from_slice(&[x, y, x + y]);
            red.push(x as u16);
            green.push(y as u16);
            blue.push((x + y) as u16);
        }
    }

    let file = File::create(&path).expect("create RGB TIFF");
    let mut encoder = TiffEncoder::new(file).expect("create TIFF encoder");
    encoder
        .new_image::<colortype::RGB8>(8, 6)
        .expect("create RGB image")
        .write_data(&pixels)
        .expect("write RGB image");

    let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open generated RGB TIFF");

    assert_eq!(pyramid.channel_count, 3);
    assert_eq!(pyramid.levels[0].channel_layout, TiffChannelLayout::Chunky);
    assert_eq!(pyramid.to_levels_info()[0].shape, vec![3, 6, 8]);
    assert_eq!(decode_channel(&path, &pyramid, 0), red);
    assert_eq!(decode_channel(&path, &pyramid, 1), green);
    assert_eq!(decode_channel(&path, &pyramid, 2), blue);
}

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
#[ignore = "requires 1.tif extended-test fixture"]
fn opens_imagej_hyperstack_extended_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("1.tif");
    assert!(path.exists(), "missing extended-test fixture: {path:?}");

    let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open TIFF fixture");
    assert_eq!(pyramid.channel_count, 64);
    assert_eq!(pyramid.levels.len(), 1);
    assert_eq!(pyramid.levels[0].channels, 64);
}

#[test]
#[ignore = "requires 1_pyramid_crop.ome.tif extended-test fixture"]
fn opens_pyramidal_ome_extended_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("1_pyramid_crop.ome.tif");
    assert!(path.exists(), "missing extended-test fixture: {path:?}");

    let pyramid = TiffPyramid::open_with_selection(&path, TiffPlaneSelection { z: 0, t: 0 })
        .expect("open pyramidal OME-TIFF fixture");
    assert_eq!(pyramid.channel_count, 64);
    assert_eq!(pyramid.levels.len(), 4);
    assert_eq!(pyramid.levels[0].width, 512);
    assert_eq!(pyramid.levels[1].width, 256);
    assert_eq!(pyramid.levels[2].width, 128);
    assert_eq!(pyramid.levels[3].width, 64);
}
