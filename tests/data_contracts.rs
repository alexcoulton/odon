use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use odon::data::dataset_kind::{
    LocalDatasetKind, can_open_in_mosaic, classify_local_dataset_path, normalize_local_dataset_path,
};
use odon::data::samplesheet::{
    SampleRow, SampleSheet, load_samplesheet_csv, write_samplesheet_csv,
};
use odon::data::zarr_attrs::{normalize_ngff_attributes, read_node_attributes};
use odon::dataset_source::DatasetSource;
use odon::project_config::{
    ProjectChannelGroup, ProjectChannelGroupMember, ProjectConfig, ProjectMaskLayer, ProjectRoi,
};
use serde::Deserialize;

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let unique = format!(
            "odon-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock is before Unix epoch")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn samplesheet_loads_relative_paths_and_metadata() {
    let dir = TestDir::new("samplesheet-load");
    let path = dir.path().join("samples.csv");
    fs::write(
        &path,
        "id,path,cohort,segpath\n\
         ROI-1,data/roi-1.ome.zarr,A,objects/roi-1.parquet\n\
         ,,ignored,ignored\n\
         ROI-2,data/roi-2.ome.zarr,B,objects/roi-2.parquet\n",
    )
    .expect("write samplesheet fixture");

    let sheet = load_samplesheet_csv(&path).expect("load samplesheet");

    assert_eq!(sheet.meta_columns, vec!["cohort", "segpath"]);
    assert_eq!(sheet.rows.len(), 2);
    assert_eq!(sheet.rows[0].id, "ROI-1");
    assert_eq!(sheet.rows[0].path, dir.path().join("data/roi-1.ome.zarr"));
    assert_eq!(sheet.rows[0].meta["cohort"], "A");
    assert_eq!(sheet.rows[0].meta["segpath"], "objects/roi-1.parquet");
    assert_eq!(sheet.rows[1].id, "ROI-2");
}

#[test]
fn samplesheet_write_then_load_preserves_rows_and_sorted_metadata() {
    let dir = TestDir::new("samplesheet-roundtrip");
    let path = dir.path().join("roundtrip.csv");
    let sheet = SampleSheet {
        meta_columns: vec!["response".to_string(), "cohort".to_string()],
        rows: vec![
            SampleRow {
                id: "ROI-2".to_string(),
                path: PathBuf::from("images/roi-2.ome.zarr"),
                meta: HashMap::from([
                    ("cohort".to_string(), "B".to_string()),
                    ("response".to_string(), "responder".to_string()),
                ]),
            },
            SampleRow {
                id: "ROI-1".to_string(),
                path: PathBuf::from("images/roi-1.ome.zarr"),
                meta: HashMap::from([("cohort".to_string(), "A".to_string())]),
            },
        ],
    };

    write_samplesheet_csv(&path, &sheet).expect("write samplesheet");
    let loaded = load_samplesheet_csv(&path).expect("reload samplesheet");

    assert_eq!(loaded.meta_columns, vec!["cohort", "response"]);
    assert_eq!(loaded.rows.len(), 2);
    assert_eq!(loaded.rows[0].id, "ROI-2");
    assert_eq!(
        loaded.rows[0].path,
        dir.path().join("images/roi-2.ome.zarr")
    );
    assert_eq!(loaded.rows[0].meta["response"], "responder");
    assert_eq!(loaded.rows[1].meta["response"], "");
}

#[test]
fn samplesheet_rejects_missing_columns_and_empty_rows() {
    let dir = TestDir::new("samplesheet-errors");

    let missing_path = dir.path().join("missing-path-column.csv");
    fs::write(&missing_path, "id\nROI-1\n").expect("write invalid fixture");
    let error = load_samplesheet_csv(&missing_path).expect_err("one-column CSV must fail");
    assert!(
        error.to_string().contains("at least 2 columns"),
        "unexpected error: {error:#}"
    );

    let empty = dir.path().join("empty.csv");
    fs::write(&empty, "id,path\n,\n").expect("write empty fixture");
    let error = load_samplesheet_csv(&empty).expect_err("empty rows must fail");
    assert!(
        error.to_string().contains("no usable rows"),
        "unexpected error: {error:#}"
    );
}

#[derive(Deserialize)]
struct ProjectFixture {
    version: u32,
    config: ProjectConfig,
}

#[test]
fn checked_in_project_fixture_preserves_legacy_local_path_contract() {
    let project: ProjectFixture =
        serde_json::from_str(include_str!("../fixtures/synthetic_5ch.project.json"))
            .expect("parse checked-in project fixture");

    assert_eq!(project.version, 6);
    assert_eq!(
        project.config.default_dataset.as_deref(),
        Some("Synthetic examples")
    );
    assert_eq!(project.config.rois.len(), 1);

    let roi = &project.config.rois[0];
    assert_eq!(roi.id, "synthetic_5ch.ome.zarr");
    assert_eq!(roi.local_path(), Some(Path::new("synthetic_5ch.ome.zarr")));
    assert_eq!(
        roi.dataset_source(),
        Some(DatasetSource::Local(PathBuf::from(
            "synthetic_5ch.ome.zarr"
        )))
    );
    assert_eq!(roi.meta["synthetic"], "true");
}

#[test]
fn project_config_json_roundtrip_preserves_sources_masks_and_groups() {
    let mut roi = ProjectRoi {
        id: "roi-1".to_string(),
        dataset: Some("study".to_string()),
        display_name: Some("ROI 1".to_string()),
        segpath: Some(PathBuf::from("objects/roi-1.parquet")),
        channel_order: vec![2, 0, 1],
        meta: HashMap::from([("cohort".to_string(), "A".to_string())]),
        ..Default::default()
    };
    roi.set_dataset_source(DatasetSource::Http {
        base_url: "https://example.test/roi-1.ome.zarr".to_string(),
    });
    roi.mask_layers.push(ProjectMaskLayer {
        id: 7,
        name: "Exclusion".to_string(),
        visible: true,
        opacity: 0.6,
        width_screen_px: 2.0,
        display_mode: Some("outline".to_string()),
        color_rgb: [255, 32, 64],
        offset_world: [3.5, -2.0],
        editable: true,
        polygons_world: vec![vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]]],
        source_geojson: Some(PathBuf::from("masks/exclusion.geojson")),
    });

    let mut config = ProjectConfig {
        rois: vec![roi],
        default_dataset: Some("study".to_string()),
        ..Default::default()
    };
    config
        .layer_groups
        .channel_groups
        .push(ProjectChannelGroup {
            id: 11,
            name: "Immune".to_string(),
            expanded: false,
            color_rgb: [0, 255, 255],
        });
    config.layer_groups.channel_members.insert(
        "CD3".to_string(),
        ProjectChannelGroupMember {
            group_id: 11,
            inherit_color: false,
        },
    );

    let json = serde_json::to_string_pretty(&config).expect("serialize project config");
    let loaded: ProjectConfig = serde_json::from_str(&json).expect("deserialize project config");

    assert_eq!(loaded.default_dataset.as_deref(), Some("study"));
    assert_eq!(loaded.rois.len(), 1);
    let loaded_roi = &loaded.rois[0];
    assert_eq!(
        loaded_roi.dataset_source(),
        Some(DatasetSource::Http {
            base_url: "https://example.test/roi-1.ome.zarr".to_string(),
        })
    );
    assert_eq!(
        loaded_roi.segpath,
        Some(PathBuf::from("objects/roi-1.parquet"))
    );
    assert_eq!(loaded_roi.channel_order, vec![2, 0, 1]);
    assert_eq!(loaded_roi.mask_layers.len(), 1);
    assert_eq!(loaded_roi.mask_layers[0].id, 7);
    assert_eq!(loaded_roi.mask_layers[0].polygons_world[0].len(), 4);
    assert_eq!(config.layer_groups, loaded.layer_groups);
}

#[test]
fn local_dataset_routing_normalizes_supported_roots_and_metadata_files() {
    let dir = TestDir::new("dataset-routing");
    let ome_zarr = dir.path().join("image.ome.zarr");
    let xenium = dir.path().join("xenium-output");
    let tiff = dir.path().join("image.OME.TIFF");
    let unknown = dir.path().join("notes.txt");

    fs::create_dir_all(&ome_zarr).expect("create OME-Zarr directory");
    fs::write(ome_zarr.join(".zattrs"), "{}").expect("write .zattrs");
    fs::create_dir_all(&xenium).expect("create Xenium directory");
    fs::write(xenium.join("experiment.xenium"), "{}").expect("write Xenium manifest");
    fs::write(&tiff, []).expect("write TIFF placeholder");
    fs::write(&unknown, "not a dataset").expect("write unknown file");

    assert_eq!(
        classify_local_dataset_path(&ome_zarr),
        Some(LocalDatasetKind::OmeZarr)
    );
    assert_eq!(
        normalize_local_dataset_path(&ome_zarr.join(".zattrs")),
        Some(ome_zarr.clone())
    );
    assert!(can_open_in_mosaic(&ome_zarr));

    assert_eq!(
        classify_local_dataset_path(&xenium),
        Some(LocalDatasetKind::Xenium)
    );
    assert_eq!(
        normalize_local_dataset_path(&xenium.join("experiment.xenium")),
        Some(xenium.clone())
    );
    assert!(!can_open_in_mosaic(&xenium));

    assert_eq!(
        classify_local_dataset_path(&tiff),
        Some(LocalDatasetKind::Tiff)
    );
    assert_eq!(normalize_local_dataset_path(&tiff), Some(tiff));
    assert_eq!(normalize_local_dataset_path(&unknown), None);
}

#[test]
fn local_dataset_routing_supports_zarr_v3_metadata() {
    let dir = TestDir::new("dataset-routing-v3");
    let ome_zarr = dir.path().join("image.ome.zarr");
    fs::create_dir_all(&ome_zarr).expect("create OME-Zarr directory");
    fs::write(
        ome_zarr.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group","attributes":{}}"#,
    )
    .expect("write zarr.json");

    assert_eq!(
        classify_local_dataset_path(&ome_zarr),
        Some(LocalDatasetKind::OmeZarr)
    );
    assert_eq!(
        normalize_local_dataset_path(&ome_zarr.join("zarr.json")),
        Some(ome_zarr)
    );
}

#[test]
fn zarr_attribute_reader_handles_v2_v3_and_missing_metadata() {
    let dir = TestDir::new("zarr-attributes");
    let v2 = dir.path().join("v2");
    let v3 = dir.path().join("v3");
    let missing = dir.path().join("missing");
    fs::create_dir_all(&v2).expect("create v2 directory");
    fs::create_dir_all(&v3).expect("create v3 directory");
    fs::create_dir_all(&missing).expect("create missing directory");

    fs::write(
        v2.join(".zattrs"),
        r#"{"multiscales":[{"name":"image"}],"omero":{"name":"sample"}}"#,
    )
    .expect("write v2 attributes");
    fs::write(
        v3.join("zarr.json"),
        r#"{"zarr_format":3,"attributes":{"multiscales":[],"custom":42}}"#,
    )
    .expect("write v3 attributes");

    let attrs = read_node_attributes(&v2)
        .expect("read v2 attributes")
        .expect("v2 attributes exist");
    assert!(attrs["multiscales"].is_array());
    assert_eq!(attrs["omero"]["name"], "sample");

    let attrs = read_node_attributes(&v3)
        .expect("read v3 attributes")
        .expect("v3 attributes exist");
    assert_eq!(attrs["custom"], 42);
    assert_eq!(
        read_node_attributes(&missing).expect("read missing attributes"),
        None
    );
}

#[test]
fn ngff_attribute_normalization_unwraps_ome_and_preserves_outer_omero() {
    let attrs = serde_json::json!({
        "ome": {
            "version": "0.5",
            "multiscales": [{"name": "image"}]
        },
        "omero": {
            "name": "display metadata"
        }
    })
    .as_object()
    .expect("attributes object")
    .clone();

    let normalized = normalize_ngff_attributes(attrs);

    assert_eq!(normalized["version"], "0.5");
    assert!(normalized["multiscales"].is_array());
    assert_eq!(normalized["omero"]["name"], "display metadata");
    assert!(!normalized.contains_key("ome"));
}

#[test]
fn zarr_attribute_reader_rejects_non_object_v2_metadata() {
    let dir = TestDir::new("zarr-invalid-attributes");
    fs::write(dir.path().join(".zattrs"), "[]").expect("write invalid attributes");

    let error = read_node_attributes(dir.path()).expect_err("array .zattrs must fail");

    assert!(
        error.to_string().contains("must be a JSON object"),
        "unexpected error: {error:#}"
    );
}
