use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use anyhow::{Context, anyhow};
use tiff::decoder::Decoder;
use tiff::tags::Tag;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("usage: cargo run --bin inspect_tiff -- <path>"))?;

    let file = File::open(&path).with_context(|| format!("open {:?}", path))?;
    let mut dec = Decoder::new(BufReader::new(file)).context("create TIFF decoder")?;

    println!("path: {}", path.display());
    let mut index = 0usize;
    loop {
        let ifd = dec.ifd_pointer().context("missing current IFD pointer")?;
        let (w, h) = dec.dimensions().context("dimensions")?;
        let color = dec.colortype().context("color type")?;
        let chunk_type = dec.get_chunk_type();
        let (chunk_w, chunk_h) = dec.chunk_dimensions();
        let planar = dec
            .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
            .ok()
            .flatten()
            .unwrap_or(1);
        let subifds = dec
            .find_tag(Tag::SubIfd)
            .ok()
            .flatten()
            .and_then(|value| value.into_ifd_vec().ok())
            .unwrap_or_default();
        let image_description = dec
            .find_tag(Tag::ImageDescription)
            .ok()
            .flatten()
            .and_then(|value| value.into_string().ok());

        println!(
            "ifd[{index}] ptr={} size={}x{} color={color:?} chunk={chunk_type:?} {}x{} planar={} subifds={}",
            ifd.0,
            w,
            h,
            chunk_w,
            chunk_h,
            planar,
            subifds.len()
        );
        if index == 0 {
            if let Some(desc) = image_description {
                let snippet = desc
                    .chars()
                    .take(400)
                    .collect::<String>()
                    .replace('\n', " ");
                println!("image_description: {snippet}");
            }
        }

        if !dec.more_images() {
            break;
        }
        dec.next_image().context("advance image")?;
        index += 1;
    }

    Ok(())
}
