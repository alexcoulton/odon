mod app_support;
mod annotations;
mod app;
mod camera;
mod custom;
mod data;
mod debug_log;
mod features;
mod geometry;
mod imaging;
mod masks;
mod mosaic;
mod objects;
mod project;
mod render;
mod root_app;
mod spatialdata;
mod ui;
mod xenium;

use std::path::PathBuf;

use eframe::egui;

use crate::data::ome;

fn print_usage() {
    eprintln!(
        r#"odon

Usage:
  odon "/path/to/dataset.ome.zarr"
  odon --project "/path/to/project.json"
  odon --project "/path/to/project.json" --mosaic "TMA1v3,TMA2" [--mosaic-cols N]
  odon --mosaic-samplesheet "/path/to/samplesheet.csv" [--mosaic-cols N]

Other:
  --check      Run a small IO sanity check (single-dataset local only)
  --log-level  Set log level: error|warn|info|debug|trace
  --debug-io   Print extra IO/status logs (very verbose)
  -h, --help   Show this help

Notes:
  - `--mosaic-samplesheet` also accepts `--mosaic-samplesheet=/path/to.csv`.
  - `--mosaic-cols` also accepts `--mosaic-cols=N`.
  - `--mosaic "..."` filters ROIs by their `dataset` key in the Project JSON.
"#
    );
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            return args.get(i + 1).cloned();
        }
        let prefix = format!("{flag}=");
        if let Some(rest) = a.strip_prefix(&prefix) {
            return Some(rest.to_string());
        }
    }
    None
}

fn parse_usize_flag(args: &[String], flag: &str) -> Option<usize> {
    flag_value(args, flag).and_then(|v| v.parse::<usize>().ok())
}

fn positional_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        let a = &args[i];

        // Bool flags.
        if a == "--check" || a == "--debug-io" {
            i += 1;
            continue;
        }

        // Flags with optional value (we treat `--log` as bool, but accept `--log=...` and `--log ...`).
        if a == "--log" {
            if args.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        // Flags with value.
        if a == "--project"
            || a == "--mosaic"
            || a == "--mosaic-cols"
            || a == "--mosaic-samplesheet"
            || a == "--log-level"
        {
            i += 2;
            continue;
        }

        // `--flag=value` forms.
        if a.starts_with("--project=")
            || a.starts_with("--mosaic=")
            || a.starts_with("--mosaic-cols=")
            || a.starts_with("--mosaic-samplesheet=")
            || a.starts_with("--log-level=")
            || a.starts_with("--log=")
        {
            i += 1;
            continue;
        }

        // Any other flag: do not treat as positional.
        if a.starts_with("--") {
            i += 1;
            continue;
        }

        out.push(a.clone());
        i += 1;
    }
    out
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    // Logging (stderr). Example:
    //   cargo run -- --log-level debug --debug-io "/path/to/dataset"
    let mut log_level = flag_value(&args, "--log-level").or_else(|| flag_value(&args, "--log"));
    if log_level.is_none() && args.iter().any(|a| a == "--log") {
        log_level = Some("info".to_string());
    }
    let log_level = log_level.unwrap_or_else(|| "warn".to_string());
    let debug_io = args.iter().any(|a| a == "--debug-io");
    debug_log::init(
        match log_level.as_str() {
            "error" => debug_log::LogLevel::Error,
            "warn" => debug_log::LogLevel::Warn,
            "info" => debug_log::LogLevel::Info,
            "debug" => debug_log::LogLevel::Debug,
            "trace" => debug_log::LogLevel::Trace,
            _ => debug_log::LogLevel::Warn,
        },
        debug_io,
    );

    let check_only = args.first().is_some_and(|a| a == "--check");
    if check_only {
        args.remove(0);
    }

    #[derive(Clone)]
    enum Launch {
        Project { project_path: Option<PathBuf> },
        Single { dataset_root: PathBuf },
        Mosaic { args: mosaic::MosaicCliArgs },
    }

    let project_path = flag_value(&args, "--project").map(PathBuf::from);

    let has_mosaic = args
        .iter()
        .any(|a| a == "--mosaic" || a.starts_with("--mosaic="));
    let has_mosaic_samplesheet = args
        .iter()
        .any(|a| a == "--mosaic-samplesheet" || a.starts_with("--mosaic-samplesheet="));
    if has_mosaic && has_mosaic_samplesheet {
        anyhow::bail!("use only one of --mosaic or --mosaic-samplesheet");
    }

    let launch = if has_mosaic {
        if check_only {
            anyhow::bail!("--check is not supported with --mosaic");
        }
        let Some(project_path) = project_path.clone() else {
            anyhow::bail!("--mosaic requires --project /path/to/project.json");
        };

        let list = flag_value(&args, "--mosaic").unwrap_or_default();
        if list.is_empty() || list.starts_with("--") {
            anyhow::bail!("usage: --mosaic \"TMA1v3,TMA2\" [--mosaic-cols N]");
        }

        let columns: Option<usize> = parse_usize_flag(&args, "--mosaic-cols");

        let dataset_names = list
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        Launch::Mosaic {
            args: mosaic::MosaicCliArgs {
                dataset_names,
                columns,
                samplesheet_csv: None,
                project_path: Some(project_path),
            },
        }
    } else if has_mosaic_samplesheet {
        if check_only {
            anyhow::bail!("--check is not supported with --mosaic-samplesheet");
        }
        let sheet = flag_value(&args, "--mosaic-samplesheet").unwrap_or_default();
        if sheet.is_empty() || sheet.starts_with("--") {
            anyhow::bail!("usage: --mosaic-samplesheet /path/to/samplesheet.csv [--mosaic-cols N]");
        }

        let columns: Option<usize> = parse_usize_flag(&args, "--mosaic-cols");

        Launch::Mosaic {
            args: mosaic::MosaicCliArgs {
                dataset_names: Vec::new(),
                columns,
                samplesheet_csv: Some(PathBuf::from(sheet)),
                project_path: project_path.clone(),
            },
        }
    } else {
        let dataset_root: Option<PathBuf> = positional_args(&args).first().map(PathBuf::from);
        if let Some(dataset_root) = dataset_root {
            Launch::Single { dataset_root }
        } else {
            Launch::Project {
                project_path: project_path.clone(),
            }
        }
    };

    if check_only {
        match &launch {
            Launch::Single { dataset_root } => {
                let (dataset, _store) = ome::OmeZarrDataset::open_local(dataset_root)?;
                check_tile(&dataset)?;
                return Ok(());
            }
            Launch::Project { .. } => {
                anyhow::bail!("--check requires a dataset path");
            }
            Launch::Mosaic { .. } => {
                anyhow::bail!("--check is not supported with mosaic modes");
            }
        }
    }

    let viewport = load_app_icon()
        .map(|icon| egui::ViewportBuilder::default().with_icon(icon))
        .unwrap_or_default();
    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "odon",
        native_options,
        Box::new(move |cc| {
            Ok(match &launch {
                Launch::Project { project_path } => {
                    Box::new(root_app::RootApp::new_project(cc, project_path.clone())?)
                }
                Launch::Single { dataset_root } => {
                    match ome::OmeZarrDataset::open_local(dataset_root) {
                        Ok((dataset, store)) => Box::new(root_app::RootApp::new_single(
                            cc,
                            dataset,
                            store,
                            project_path.clone(),
                        )?),
                        Err(_err) => {
                            // Fall back to the project landing UI and queue an open. This allows
                            // opening SpatialData containers (and other non-image roots) via the
                            // chooser dialog, instead of failing at startup.
                            let mut app = root_app::RootApp::new_project(cc, project_path.clone())?;
                            app.add_paths_to_project(vec![dataset_root.clone()]);
                            app.queue_open_root(dataset_root.clone());
                            Box::new(app)
                        }
                    }
                }
                Launch::Mosaic { args } => {
                    let mosaic = mosaic::MosaicViewerApp::from_args(cc, args.clone())?;
                    Box::new(root_app::RootApp::new_mosaic(
                        cc,
                        mosaic,
                        project_path.clone(),
                    )?)
                }
            })
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}

fn load_app_icon() -> Option<egui::IconData> {
    let image = image::load_from_memory(include_bytes!("../assets/odon.png"))
        .ok()?
        .into_rgba8();
    let (width, height) = image.dimensions();
    Some(egui::IconData {
        rgba: image.into_raw(),
        width,
        height,
    })
}

fn check_tile(dataset: &ome::OmeZarrDataset) -> anyhow::Result<()> {
    use std::sync::Arc;

    use zarrs::array::{Array, ArraySubset};
    use zarrs::storage::ReadableStorageTraits;

    let Some(local_root) = dataset.source.local_path() else {
        anyhow::bail!("--check currently supports local datasets only");
    };
    let store: Arc<dyn ReadableStorageTraits> =
        Arc::new(zarrs::filesystem::FilesystemStore::new(local_root)?);

    let level = dataset.levels.last().expect("missing levels");
    let array = Array::open(store, &format!("/{}", level.path))?;

    let shape = &level.shape;
    let chunks = &level.chunks;
    let c_dim = dataset.dims.c;
    let y_dim = dataset.dims.y;
    let x_dim = dataset.dims.x;

    let y1 = chunks[y_dim].min(shape[y_dim]);
    let x1 = chunks[x_dim].min(shape[x_dim]);

    let mut ranges = Vec::with_capacity(shape.len());
    for dim in 0..shape.len() {
        if Some(dim) == c_dim {
            ranges.push(0..1);
        } else if dim == y_dim {
            ranges.push(0..y1);
        } else if dim == x_dim {
            ranges.push(0..x1);
        } else {
            ranges.push(0..1);
        }
    }

    let subset = ArraySubset::new_with_ranges(&ranges);
    let tile_shape = if dataset.is_root_label_mask() {
        let tile: ndarray::ArrayD<u32> = array.retrieve_array_subset(&subset)?;
        tile.shape().to_vec()
    } else {
        let tile = ome::retrieve_image_subset_u16(&array, &subset, &level.dtype)?;
        tile.shape().to_vec()
    };

    println!(
        "OK: loaded tile level {} path '{}' shape {:?} -> subset {:?}",
        level.index, level.path, shape, tile_shape
    );
    Ok(())
}
