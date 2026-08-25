use super::*;

struct TestProjectDir(PathBuf);

impl TestProjectDir {
    fn new(label: &str) -> Self {
        let unique = format!(
            "odon-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is before Unix epoch")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("create test project directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestProjectDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn local_roi(path: &str, id: &str) -> ProjectRoi {
    let mut roi = ProjectRoi {
        id: id.to_string(),
        source: None,
        path: None,
        dataset: Some("default".to_string()),
        display_name: Some(id.to_string()),
        segpath: None,
        mask_layers: Vec::new(),
        channel_order: Vec::new(),
        meta: Default::default(),
    };
    roi.set_dataset_source(DatasetSource::Local(PathBuf::from(path)));
    roi
}

#[test]
fn roi_link_target_uses_sample_to_disambiguate_duplicate_roi_ids() {
    let mut ps = ProjectSpace::default();
    ps.config.rois = vec![
        local_roi("/data/18S1746/ROI1/ROI1.ome.zarr", "ROI1.ome.zarr"),
        local_roi("/data/18S1746/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
        local_roi("/data/19S4359/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
    ];

    let roi = ps
        .roi_for_link_target(Some("ROI2"), Some("18S1746"))
        .unwrap();
    assert!(roi.source_display().contains("18S1746/ROI2"));
}

#[test]
fn roi_link_target_accepts_specific_path_fragment() {
    let mut ps = ProjectSpace::default();
    ps.config.rois = vec![
        local_roi("/data/18S1746/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
        local_roi("/data/19S4359/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
    ];

    let roi = ps.roi_for_link_target(Some("19S4359/ROI2"), None).unwrap();
    assert!(roi.source_display().contains("19S4359/ROI2"));
}

#[test]
fn load_preserves_saved_rois_when_local_paths_are_unavailable() {
    let unique = format!(
        "odon-project-load-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let project_path = std::env::temp_dir().join(format!("{unique}.json"));
    let missing_roi_path = std::env::temp_dir()
        .join(unique)
        .join("18S1746/ROI2/ROI2.ome.zarr");

    let file = ProjectFileV6 {
        version: 6,
        config: ProjectConfig {
            rois: vec![local_roi(
                missing_roi_path.to_string_lossy().as_ref(),
                "ROI2.ome.zarr",
            )],
            ..Default::default()
        },
        state: ProjectState::default(),
    };
    fs::write(&project_path, serde_json::to_string(&file).unwrap()).unwrap();

    let mut ps = ProjectSpace::default();
    ps.load_from_file(&project_path).unwrap();
    let roi = ps.roi_for_link_target(Some("18S1746/ROI2"), None).unwrap();

    assert!(roi.source_display().contains("18S1746/ROI2"));
    let _ = fs::remove_file(project_path);
}

#[test]
fn load_resolves_relative_roi_paths_against_project_file() {
    let unique = format!(
        "odon-project-relative-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let project_dir = std::env::temp_dir().join(unique);
    let project_path = project_dir.join("synthetic_5ch.project.json");
    fs::create_dir_all(&project_dir).unwrap();

    let mut roi = local_roi("synthetic_5ch.ome.zarr", "Synthetic 5-channel");
    roi.segpath = Some(PathBuf::from("objects/cells.parquet"));
    let file = ProjectFileV6 {
        version: 6,
        config: ProjectConfig {
            rois: vec![roi],
            ..Default::default()
        },
        state: ProjectState::default(),
    };
    fs::write(&project_path, serde_json::to_string(&file).unwrap()).unwrap();

    let mut ps = ProjectSpace::default();
    ps.load_from_file(&project_path).unwrap();
    let roi = ps.roi_for_link_target(Some("synthetic_5ch"), None).unwrap();

    assert_eq!(
        roi.local_path(),
        Some(project_dir.join("synthetic_5ch.ome.zarr").as_path())
    );
    assert_eq!(
        roi.segpath.as_deref(),
        Some(project_dir.join("objects/cells.parquet").as_path())
    );

    let _ = fs::remove_dir_all(project_dir);
}

#[test]
fn roi_link_target_reports_ambiguous_matches() {
    let mut ps = ProjectSpace::default();
    ps.config.rois = vec![
        local_roi("/data/18S1746/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
        local_roi("/data/19S4359/ROI2/ROI2.ome.zarr", "ROI2.ome.zarr"),
    ];

    let err = ps.roi_for_link_target(Some("ROI2"), None).unwrap_err();
    assert!(err.contains("matches 2 project ROIs"));
}

#[test]
fn save_then_load_preserves_roi_view_masks_groups_and_browser_state() {
    let dir = TestProjectDir::new("project-roundtrip");
    let project_path = dir.path().join("roundtrip.project.json");
    let roi_path = dir.path().join("roi-1.ome.zarr");
    let source = DatasetSource::Local(roi_path.clone());

    let groups = ProjectLayerGroups {
        channel_groups: vec![crate::data::project_config::ProjectChannelGroup {
            id: 12,
            name: "Immune".to_string(),
            expanded: false,
            color_rgb: [0, 255, 255],
        }],
        ..Default::default()
    };
    let masks = vec![ProjectMaskLayer {
        id: 4,
        name: "Exclusion".to_string(),
        visible: true,
        opacity: 0.5,
        width_screen_px: 2.0,
        display_mode: Some("outline".to_string()),
        color_rgb: [255, 0, 0],
        offset_world: [2.0, -3.0],
        editable: true,
        polygons_world: vec![vec![[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 0.0]]],
        source_geojson: Some(PathBuf::from("masks/exclusion.geojson")),
    }];
    let view = ProjectRoiViewState {
        channel_order: vec![2, 0, 1],
        active_channel: Some(2),
        active_layer: Some("objects".to_string()),
        ..Default::default()
    };
    let preset_spec = ProjectViewSpec {
        channel: Some("CD3".to_string()),
        visible_channels: vec!["DAPI".to_string(), "CD3".to_string()],
        cell_color_by: Some("cell_type".to_string()),
        object_color_mapping: Some(odon::model::ObjectColorMapping::Continuous {
            property: "mean_channel_1".to_string(),
            palette: odon::model::ContinuousPalette::Named("magma".to_string()),
            domain: odon::model::ContinuousDomain::Fixed([100.0, 900.0]),
            scale: odon::model::ContinuousScale::Linear,
            reverse: true,
            out_of_range: odon::model::OutOfRangeMode::Hide,
            missing_color_rgb: Some([64, 64, 64]),
        }),
        camera: Some(ProjectCameraState {
            center_world_lvl0: [125.0, 250.0],
            zoom_screen_per_lvl0_px: 1.5,
        }),
        ..Default::default()
    };
    let mosaic = ProjectMosaicViewState {
        channel_order: vec![2, 0, 1],
        active_channel: Some(2),
        sort_by: Some("cohort".to_string()),
        group_by: Some("response".to_string()),
        layout_mode: Some("fit_cells".to_string()),
        label_columns: vec!["sample".to_string(), "cohort".to_string()],
        ..Default::default()
    };

    let mut project = ProjectSpace::default();
    project.add_roi_source(source.clone());
    project.update_layer_groups(|current| *current = groups.clone());
    project.set_roi_mask_layers(&roi_path, masks.clone());
    project.set_roi_view_state(&source, view.clone());
    project.save_view_preset("Review".to_string(), preset_spec.clone());
    project.set_mosaic_view_state(mosaic.clone());
    project
        .save_to_file(&project_path)
        .expect("save project round-trip fixture");

    let saved: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&project_path).expect("read saved project"))
            .expect("saved project is valid JSON");
    assert_eq!(saved["version"], 6);

    let mut loaded = ProjectSpace::default();
    loaded
        .load_from_file(&project_path)
        .expect("load saved project");

    assert_eq!(loaded.rois().len(), 1);
    assert_eq!(loaded.rois()[0].local_path(), Some(roi_path.as_path()));
    assert_eq!(loaded.layer_groups(), &groups);
    assert_eq!(loaded.roi_mask_layers(&roi_path), Some(masks.as_slice()));
    assert_eq!(loaded.roi_view_state(&source), Some(&view));
    assert_eq!(loaded.state.view_presets.len(), 1);
    assert_eq!(loaded.state.view_presets[0].name, "Review");
    assert_eq!(loaded.state.view_presets[0].spec, preset_spec);
    assert_eq!(loaded.mosaic_view_state(), Some(&mosaic));
    assert_eq!(
        loaded.focused_roi().and_then(ProjectRoi::source_key),
        Some(source.source_key())
    );
    assert_eq!(loaded.selected_rois().len(), 1);
    assert_eq!(loaded.saved_project_path(), Some(project_path));
}

#[test]
fn project_load_rejects_malformed_json_and_unsupported_versions_without_replacing_state() {
    let dir = TestProjectDir::new("project-errors");
    let malformed = dir.path().join("malformed.project.json");
    let unsupported = dir.path().join("unsupported.project.json");
    fs::write(&malformed, "{not valid JSON").expect("write malformed project");
    fs::write(
        &unsupported,
        r#"{"version":99,"config":{"rois":[]},"state":{}}"#,
    )
    .expect("write unsupported project");

    let mut project = ProjectSpace::default();
    project.add_roi_source(DatasetSource::Http {
        base_url: "https://example.test/original.ome.zarr".to_string(),
    });
    let original = project
        .rois()
        .iter()
        .map(|roi| (roi.id.clone(), roi.source_key()))
        .collect::<Vec<_>>();

    let malformed_error = project
        .load_from_file(&malformed)
        .expect_err("malformed project must fail");
    assert!(malformed_error.to_string().contains("Load failed"));
    assert_eq!(
        project
            .rois()
            .iter()
            .map(|roi| (roi.id.clone(), roi.source_key()))
            .collect::<Vec<_>>(),
        original
    );

    let version_error = project
        .load_from_file(&unsupported)
        .expect_err("unsupported project version must fail");
    assert!(
        version_error
            .to_string()
            .contains("Unsupported project version: 99")
    );
    assert_eq!(
        project
            .rois()
            .iter()
            .map(|roi| (roi.id.clone(), roi.source_key()))
            .collect::<Vec<_>>(),
        original
    );
}

#[test]
fn view_presets_validate_replace_and_keep_active_alias_consistent() {
    let mut project = ProjectSpace::default();
    project.save_view_preset("  ".to_string(), ProjectViewSpec::default());
    assert!(project.state.view_presets.is_empty());
    assert_eq!(project.status, "View preset name is empty.");

    let first = ProjectViewSpec {
        channel_ref: Some(ProjectViewChannelRef {
            label: "C002 - CD3 (FITC)".to_string(),
            alias: "stale".to_string(),
        }),
        visible_channel_refs: vec![ProjectViewChannelRef {
            label: "C002 - CD3 (FITC)".to_string(),
            alias: "cd3".to_string(),
        }],
        camera: Some(ProjectCameraState {
            center_world_lvl0: [10.0, 20.0],
            zoom_screen_per_lvl0_px: 0.5,
        }),
        ..Default::default()
    };
    project.save_view_preset(" Review ".to_string(), first);
    assert_eq!(project.state.view_presets.len(), 1);
    assert_eq!(project.state.view_presets[0].name, "Review");
    assert_eq!(
        project.state.view_presets[0]
            .spec
            .channel_ref
            .as_ref()
            .map(|channel| channel.alias.as_str()),
        Some("cd3")
    );

    let replacement = ProjectViewSpec {
        channel: Some("DAPI".to_string()),
        visible_channels: vec!["DAPI".to_string()],
        ..Default::default()
    };
    project.save_view_preset("Review".to_string(), replacement.clone());
    assert_eq!(project.state.view_presets.len(), 1);
    assert_eq!(project.state.view_presets[0].spec, replacement);
    assert_eq!(project.status, "Updated view preset 'Review'.");

    let request = project.state.view_presets[0]
        .spec
        .to_deep_link_request(Some("ROI-1".to_string()));
    assert_eq!(request.roi.as_deref(), Some("ROI-1"));
    assert_eq!(request.channel.as_deref(), Some("DAPI"));
    assert_eq!(request.visible_channels, vec!["DAPI"]);

    project
        .rename_view_preset(0, "Nuclear review".to_string())
        .expect("rename view preset");
    assert_eq!(project.view_presets()[0].name, "Nuclear review");
    assert!(
        project
            .rename_view_preset(1, "Missing".to_string())
            .is_err()
    );
    let removed = project.delete_view_preset(0).expect("delete view preset");
    assert_eq!(removed.name, "Nuclear review");
    assert!(project.view_presets().is_empty());
}

#[test]
fn roi_crud_order_selection_and_focus_use_stable_ids() {
    let mut project = ProjectSpace::default();
    for (id, path) in [("ROI-A", "/tmp/a.ome.zarr"), ("ROI-B", "/tmp/b.ome.zarr")] {
        let mut roi = ProjectRoi {
            id: id.to_string(),
            display_name: Some(id.to_string()),
            ..Default::default()
        };
        roi.set_dataset_source(DatasetSource::Local(PathBuf::from(path)));
        project.add_roi_record(roi).expect("add ROI");
    }
    assert_eq!(project.roi_index_by_id("ROI-B"), Ok(1));
    assert!(
        project
            .select_roi_ids(&["ROI-A".to_string()], "replace")
            .is_ok()
    );
    assert_eq!(
        project
            .selected_rois()
            .into_iter()
            .map(|roi| roi.id)
            .collect::<Vec<_>>(),
        vec!["ROI-A"]
    );
    project.step_focused_roi(1, true).expect("step focused ROI");
    assert_eq!(
        project.focused_roi().map(|roi| roi.id.as_str()),
        Some("ROI-B")
    );
    project
        .reorder_rois(&["ROI-B".to_string(), "ROI-A".to_string()])
        .expect("reorder ROIs");
    assert_eq!(project.rois()[0].id, "ROI-B");

    let mut replacement = project.rois()[0].clone();
    replacement.id = "ROI-C".to_string();
    project
        .update_roi_record("ROI-B", replacement)
        .expect("update ROI");
    assert_eq!(project.rois()[0].id, "ROI-C");
    let removed = project.remove_roi_by_id("ROI-C").expect("remove ROI");
    assert_eq!(removed.id, "ROI-C");
    assert_eq!(project.rois().len(), 1);
}

#[test]
fn actor_owned_project_edits_emit_commands_without_mutating_project_semantics() {
    let mut project = ProjectSpace::default();
    for (id, path) in [("ROI-A", "/tmp/a.ome.zarr"), ("ROI-B", "/tmp/b.ome.zarr")] {
        project
            .add_roi_record(local_roi(path, id))
            .expect("seed project ROI");
    }
    project.save_view_preset("Review".to_string(), ProjectViewSpec::default());
    let before = project.control_actor_project_delta_snapshot();
    project.set_control_actor_owned(true);

    project.add_segmentation_search_root(PathBuf::from("/tmp/segmentations"));
    project.save_view_preset("Overview".to_string(), ProjectViewSpec::default());
    project
        .add_roi_record(local_roi("/tmp/c.ome.zarr", "ROI-C"))
        .expect("queue ROI add");
    let mut replacement = project.rois()[0].clone();
    replacement.display_name = Some("Updated A".to_string());
    project
        .update_roi_record("ROI-A", replacement)
        .expect("queue ROI update");
    project
        .remove_roi_by_id("ROI-B")
        .expect("queue ROI removal");
    project
        .reorder_rois(&["ROI-B".to_string(), "ROI-A".to_string()])
        .expect("queue ROI reorder");
    project
        .select_roi_ids(&["ROI-B".to_string()], "replace")
        .expect("queue ROI selection");
    project.focus_roi_id("ROI-B").expect("queue ROI focus");

    let intents = project.take_control_intents();
    for intent in &intents {
        odon::control::ControlCommand::decode(intent.method, intent.params.clone())
            .expect("top-level project action emits a typed actor command");
    }
    assert_eq!(
        intents
            .iter()
            .map(|intent| intent.method)
            .collect::<Vec<_>>(),
        vec![
            "project.update_metadata",
            "project.views.create",
            "project.rois.add",
            "project.rois.update",
            "project.rois.remove",
            "project.rois.reorder",
            "project.rois.select",
            "project.rois.focus",
        ]
    );
    assert_eq!(
        serde_json::to_value(project.control_actor_project_delta_snapshot().rois).unwrap(),
        serde_json::to_value(before.rois).unwrap()
    );
    assert_eq!(
        project
            .control_actor_project_delta_snapshot()
            .mosaic_segmentation_search_roots,
        before.mosaic_segmentation_search_roots
    );
    assert_eq!(
        project.control_actor_project_delta_snapshot().view_presets,
        before.view_presets
    );
    assert_eq!(
        project
            .control_actor_project_delta_snapshot()
            .selected_source_keys,
        before.selected_source_keys
    );
    assert_eq!(
        project
            .control_actor_project_delta_snapshot()
            .focused_source_key,
        before.focused_source_key
    );
}

#[test]
fn actor_owned_top_level_actions_emit_direct_commands_without_host_relays() {
    let mut project = ProjectSpace::default();
    project.set_control_actor_owned(true);
    let roi = local_roi("/tmp/a.ome.zarr", "ROI-A");
    let actions = vec![
        ProjectSpaceAction::Open(roi.clone()),
        ProjectSpaceAction::OpenView(roi.clone(), ProjectViewSpec::default()),
        ProjectSpaceAction::OpenLocalPath(PathBuf::from("/tmp/image.ome.tif")),
        ProjectSpaceAction::OpenProject(PathBuf::from("/tmp/project.json")),
        ProjectSpaceAction::SaveProject(PathBuf::from("/tmp/saved.json")),
        ProjectSpaceAction::ForgetRecentProject(PathBuf::from("/tmp/old.json")),
        ProjectSpaceAction::ClearRecentProjects,
        ProjectSpaceAction::OpenMosaic,
        ProjectSpaceAction::PreloadObjectSegmentations(ObjectPreloadSettings {
            mode: ObjectPreloadMode::CentroidPoints,
            lazy_properties: false,
        }),
        ProjectSpaceAction::ClearObjectCache,
    ];
    for action in &actions {
        assert!(project.submit_action_control_intent(action));
    }

    let intents = project.take_control_intents();
    assert_eq!(
        intents
            .iter()
            .map(|intent| intent.method)
            .collect::<Vec<_>>(),
        vec![
            "project.rois.open",
            "deep_links.apply",
            "datasets.open_tiff",
            "project.open",
            "project.save_as",
            "app.recent_projects.forget",
            "app.recent_projects.clear",
            "project.rois.open_selected_mosaic",
            "project.objects.preload.start",
            "project.objects.preload.clear",
        ]
    );
    assert_eq!(intents[0].params["roi"], "ROI-A");
    assert_eq!(intents[1].params["request"]["roi"], "ROI-A");
    assert_eq!(intents[8].params["mode"], "centroid_points");
    assert_eq!(intents[8].params["lazy_properties"], false);
    assert!(!project.submit_action_control_intent(&ProjectSpaceAction::CaptureCurrentView));
    assert!(!project.submit_action_control_intent(&ProjectSpaceAction::OpenRemoteDialog));
    assert!(project.take_control_intents().is_empty());
}

#[test]
fn actor_owned_saved_view_rename_and_delete_emit_direct_commands() {
    let mut project = ProjectSpace::default();
    project.save_view_preset("Review".to_string(), ProjectViewSpec::default());
    let before = project.control_actor_project_delta_snapshot().view_presets;
    project.set_control_actor_owned(true);

    project
        .rename_view_preset(0, "Renamed".to_string())
        .expect("queue saved-view rename");
    let rename = project.take_control_intents();
    assert_eq!(rename.len(), 1);
    assert_eq!(rename[0].method, "project.views.rename");
    assert_eq!(
        rename[0].params,
        serde_json::json!({
            "name":"Review",
            "new_name":"Renamed",
        })
    );
    assert_eq!(
        project.control_actor_project_delta_snapshot().view_presets,
        before
    );

    project
        .delete_view_preset(0)
        .expect("queue saved-view deletion");
    let delete = project.take_control_intents();
    assert_eq!(delete.len(), 1);
    assert_eq!(delete[0].method, "project.views.delete");
    assert_eq!(delete[0].params, serde_json::json!({"name":"Review"}));
    assert_eq!(
        project.control_actor_project_delta_snapshot().view_presets,
        before
    );
}

#[test]
fn samplesheet_import_builds_project_and_export_round_trips_local_rois() {
    let dir = TestProjectDir::new("samplesheet-project");
    let image_a = dir.path().join("images/a.ome.zarr");
    let image_b = dir.path().join("images/b.ome.zarr");
    let objects_a = dir.path().join("objects/a.parquet");
    fs::create_dir_all(&image_a).expect("create image A");
    fs::create_dir_all(&image_b).expect("create image B");
    fs::create_dir_all(objects_a.parent().expect("objects parent"))
        .expect("create objects directory");
    fs::write(&objects_a, []).expect("create segmentation placeholder");
    let input = dir.path().join("input.samplesheet.csv");
    fs::write(
        &input,
        "id,path,dataset,segpath,cohort\n\
         ROI-A,images/a.ome.zarr,Study A,objects/a.parquet,treated\n\
         ROI-B,images/b.ome.zarr,Study B,,control\n",
    )
    .expect("write project samplesheet");

    let mut project = ProjectSpace::default();
    project
        .import_rois_from_csv(&input)
        .expect("import samplesheet into project");
    assert_eq!(project.rois().len(), 2);
    assert_eq!(project.rois()[0].id, "ROI-A");
    assert_eq!(project.rois()[0].dataset.as_deref(), Some("Study A"));
    let canonical_image_a = image_a.canonicalize().expect("canonical image A");
    let canonical_objects_a = objects_a.canonicalize().expect("canonical objects A");
    assert_eq!(
        project.rois()[0].local_path(),
        Some(canonical_image_a.as_path())
    );
    assert_eq!(
        project.rois()[0].segpath.as_deref(),
        Some(canonical_objects_a.as_path())
    );
    assert_eq!(project.rois()[0].meta["cohort"], "treated");
    assert_eq!(
        project.focused_roi().map(|roi| roi.id.as_str()),
        Some("ROI-A")
    );
    assert_eq!(project.selected_rois().len(), 1);

    project.config.rois.push({
        let mut remote = ProjectRoi {
            id: "remote".to_string(),
            ..Default::default()
        };
        remote.set_dataset_source(DatasetSource::Http {
            base_url: "https://example.test/remote.ome.zarr".to_string(),
        });
        remote
    });
    let output = dir.path().join("exported.samplesheet.csv");
    project
        .export_samplesheet_csv(&output)
        .expect("export project samplesheet");
    assert!(project.status.contains("skipped 1 non-local ROI"));

    let exported = load_samplesheet_csv(&output).expect("reload exported samplesheet");
    assert_eq!(exported.rows.len(), 2);
    assert_eq!(exported.rows[0].id, "ROI-A");
    assert_eq!(exported.rows[0].path, canonical_image_a);
    assert_eq!(exported.rows[0].meta["dataset"], "Study A");
    assert_eq!(
        exported.rows[0].meta["segpath"],
        canonical_objects_a.to_string_lossy()
    );
    assert_eq!(exported.rows[1].meta["cohort"], "control");
}

#[test]
fn actor_project_projection_updates_semantics_without_replacing_ui_state() {
    let mut project = ProjectSpace::default();
    project.view_preset_name_input = "draft remains local".to_string();
    let mut roi = ProjectRoi {
        id: "actor-roi".to_string(),
        display_name: Some("Actor ROI".to_string()),
        ..ProjectRoi::default()
    };
    roi.set_dataset_source(DatasetSource::Local(PathBuf::from("/tmp/actor.zarr")));
    let source_key = roi.source_key().unwrap();
    let snapshot = odon::model::ProjectModelSnapshot {
        rois: vec![roi],
        default_dataset: Some("actor-dataset".to_string()),
        selected_source_keys: vec![source_key.clone()],
        focused_source_key: Some(source_key),
        config_generation: 7,
        dirty: true,
        ..odon::model::ProjectModelSnapshot::default()
    };

    project.apply_control_actor_project_projection(&snapshot);
    assert_eq!(
        project.config.default_dataset.as_deref(),
        Some("actor-dataset")
    );
    assert_eq!(
        project.focused_roi().map(|roi| roi.id.as_str()),
        Some("actor-roi")
    );
    assert_eq!(project.selected_rois().len(), 1);
    assert_eq!(project.config_generation, 7);
    assert!(project.config_json_dirty);
    assert_eq!(project.view_preset_name_input, "draft remains local");
    assert_eq!(
        project
            .control_actor_project_snapshot()
            .selected_source_keys,
        vec![snapshot.selected_source_keys[0].clone()]
    );

    let mut full_config = ProjectConfig::default();
    full_config.default_dataset = Some("loaded-dataset".to_string());
    full_config.control_resources = vec![serde_json::json!({"id":"resource-from-file"})];
    let full = odon::model::ProjectModelSnapshot {
        config: full_config,
        state: serde_json::json!({
            "view_presets": [{"name":"Loaded view","spec":{}}],
            "browser": {}
        }),
        view_presets: vec![serde_json::json!({"name":"Loaded view","spec":{}})],
        view_count: 1,
        load_generation: 1,
        default_dataset: Some("loaded-dataset".to_string()),
        saved_path: Some(PathBuf::from("/tmp/loaded.odon.project.json")),
        ..odon::model::ProjectModelSnapshot::default()
    };
    project.apply_control_actor_project_projection(&full);
    assert_eq!(
        project.config.default_dataset.as_deref(),
        Some("loaded-dataset")
    );
    assert_eq!(project.config.control_resources.len(), 1);
    assert_eq!(project.view_presets()[0].name, "Loaded view");
    assert_eq!(project.view_preset_name_input, "draft remains local");
    assert_eq!(project.control_actor_load_generation, 1);
}
