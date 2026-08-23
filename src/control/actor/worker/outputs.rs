//! Screenshot encoding and label-resource loading.

use super::*;

pub(in crate::control::actor) fn write_screenshot_on_worker(
    request: &OdonControlRequest,
    spec: &ScreenshotWriteSpec,
    pixels: PresentationPixels,
) -> anyhow::Result<u64> {
    let cancelled = || {
        request
            .task_id
            .as_deref()
            .and_then(|task_id| request.task_registry.get(task_id).ok())
            .is_some_and(|task| task.state == TaskState::Cancelled)
    };
    anyhow::ensure!(!cancelled(), "screenshot capture was cancelled");
    anyhow::ensure!(
        pixels.width > 0 && pixels.height > 0,
        "empty screenshot dimensions"
    );
    let row_bytes = pixels
        .width
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("screenshot row size overflow"))?;
    let expected = row_bytes
        .checked_mul(pixels.height)
        .ok_or_else(|| anyhow::anyhow!("screenshot buffer size overflow"))?;
    anyhow::ensure!(
        pixels.rgba.len() == expected,
        "unexpected screenshot buffer size: expected {expected}, got {}",
        pixels.rgba.len()
    );

    let rgba_top_down = if pixels.bottom_up {
        let mut normalized = vec![0; expected];
        for y in 0..pixels.height {
            let source = (pixels.height - 1 - y) * row_bytes;
            let destination = y * row_bytes;
            normalized[destination..destination + row_bytes]
                .copy_from_slice(&pixels.rgba[source..source + row_bytes]);
        }
        normalized
    } else {
        pixels.rgba
    };

    let parent = spec
        .path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if !spec.overwrite {
        anyhow::ensure!(
            !spec.path.exists(),
            "destination exists; pass overwrite=true to replace it"
        );
    }

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);
    let file_name = spec
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("screenshot.png");
    let mut opened = None;
    for _ in 0..32 {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{file_name}.odon-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                opened = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let (temporary_path, file) =
        opened.ok_or_else(|| anyhow::anyhow!("could not allocate screenshot temporary file"))?;

    let write_result = (|| -> anyhow::Result<()> {
        use image::ImageEncoder;

        let mut writer = BufWriter::new(file);
        image::codecs::png::PngEncoder::new(&mut writer).write_image(
            &rgba_top_down,
            u32::try_from(pixels.width)?,
            u32::try_from(pixels.height)?,
            image::ExtendedColorType::Rgba8,
        )?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        anyhow::ensure!(!cancelled(), "screenshot capture was cancelled");
        if spec.overwrite {
            fs::rename(&temporary_path, &spec.path)?;
        } else {
            // Creating a hard link is an atomic no-clobber commit on the same filesystem.
            fs::hard_link(&temporary_path, &spec.path)?;
            fs::remove_file(&temporary_path)?;
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result?;
    Ok(fs::metadata(&spec.path)?.len())
}

pub(in crate::control::actor) fn load_label_resource(
    document: &RenderDocument,
    name: &str,
) -> anyhow::Result<ControlLabelResource> {
    if document.dataset().is_root_label_mask() {
        let labels = LabelZarrDataset::from_root_dataset(document.dataset());
        anyhow::ensure!(
            labels.label_name == name,
            "top-level label mask is named '{}', not '{name}'",
            labels.label_name
        );
        return Ok(ControlLabelResource {
            dataset: labels,
            store: Arc::clone(document.store()),
        });
    }
    let dataset = LabelZarrDataset::try_open(Arc::clone(document.store()), name)?
        .ok_or_else(|| anyhow::anyhow!("no labels/{name} found in this ROI"))?;
    Ok(ControlLabelResource {
        dataset,
        store: Arc::clone(document.store()),
    })
}
