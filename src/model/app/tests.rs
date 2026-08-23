use super::*;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/synthetic_5ch.ome.zarr")
}

#[test]
fn comparison_commands_advance_without_a_renderer() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let created = model.dispatch("viewer.viewports.clone", &json!({"source_viewport_id": left, "layout":"horizontal", "ratio":0.5, "title":"Right"})).unwrap().unwrap().response;
    let right = created["viewport_id"].as_str().unwrap().to_string();
    model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":left,"channels":[0],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":right,"channels":[1],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    let fitted = model
        .dispatch("viewer.viewports.camera.fit", &json!({"viewport_id":right}))
        .unwrap()
        .unwrap()
        .response;
    assert!(
        fitted["result"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    let workspace = model.workspace_snapshot().unwrap();
    assert_eq!(workspace["viewports"].as_array().unwrap().len(), 2);
    assert_eq!(workspace["layout"], "horizontal");
}

#[test]
fn invalid_workspace_ratios_are_rejected_before_any_mutation() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    for ratio in [json!("half"), json!(0.95)] {
        let error = model
            .dispatch(
                "viewer.viewports.clone",
                &json!({"layout":"horizontal", "ratio":ratio}),
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(error.kind, ControlErrorKind::InvalidParams);
        assert_eq!(
            model.workspace_snapshot().unwrap()["viewports"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"layout":"horizontal", "ratio":0.6}),
        )
        .unwrap()
        .unwrap();
    let before = model.layout_snapshot().unwrap();
    let error = model
        .dispatch(
            "viewer.workspace.layout.set",
            &json!({"layout":"vertical", "ratio":0.95}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.layout_snapshot().unwrap(), before);
}

#[test]
fn readiness_and_availability_queries_are_background_safe_in_every_mode() {
    let mut model = AppModel::project();
    for mode in [ModelMode::Project, ModelMode::Mosaic, ModelMode::Transition] {
        model.bootstrap_mode_from_renderer(mode);
        let loading = model
            .dispatch("app.get_loading_state", &json!({}))
            .expect("loading state is actor-owned")
            .unwrap()
            .response;
        assert_eq!(loading["mode"], mode.as_str());
        let availability = model
            .dispatch(
                "app.get_method_availability",
                &json!({"methods":["app.get_loading_state","viewer.camera.fit"]}),
            )
            .expect("availability is actor-owned")
            .unwrap()
            .response;
        assert_eq!(availability["mode"], mode.as_str());
        assert_eq!(availability["methods"].as_array().unwrap().len(), 2);
    }
}

#[test]
fn concurrent_resource_operations_have_independent_readiness() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let project_generation = model.begin_project_operation("Saving project");
    let (document_generation, label_generation, _) = model
        .begin_label_load(&json!({"name":"cells"}))
        .expect("label operation starts");
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"]["project_io"]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"]["labels"]["phase"],
        "pending"
    );

    assert!(model.fail_label_load_for_generation(
        document_generation,
        label_generation,
        "label fixture failed",
    ));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(loading["loading"]["resources_ready"], false);
    assert_eq!(loading["loading"]["status"], "Saving project");
    assert_eq!(
        loading["loading"]["operations"]["labels"]["phase"],
        "failed"
    );

    assert!(model.finish_project_operation_for_generation(project_generation));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], false);
    assert_eq!(loading["loading"]["resources_ready"], true);
}

#[test]
fn settings_writes_cannot_commit_out_of_order() {
    let mut model = AppModel::project();
    model.bootstrap_settings(
        AppSettings::default(),
        Some(PathBuf::from("/tmp/odon-settings-ordering.json")),
        Vec::new(),
    );
    let SettingsMutationOutcome::Persist(first) = model
        .prepare_settings_set(&json!({"fast_object_rendering":false}))
        .unwrap()
    else {
        panic!("first settings change should require persistence")
    };
    let error = model
        .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
        .unwrap_err();
    assert_eq!(error.kind, ControlErrorKind::NotReady);
    assert!(
        model
            .install_settings_for_generation(first.generation, first.settings, first.response,)
            .is_some()
    );
    let SettingsMutationOutcome::Persist(second) = model
        .prepare_settings_set(&json!({"auto_contrast":{"enabled_on_open":false}}))
        .unwrap()
    else {
        panic!("second settings change should start after the first commits")
    };
    assert!(second.generation > first.generation);
}

#[test]
fn viewport_filters_of_the_same_kind_have_independent_readiness() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let (document_generation, resource_generation) =
        model.begin_object_resource_load("objects.geojson");
    assert!(model.install_object_resource_for_generation(
        document_generation,
        resource_generation,
        Arc::new(ControlObjectResource {
            source: PathBuf::from("objects.geojson"),
            downsample_factor: 1.0,
            features: Arc::new(Vec::new()),
            property_names: Arc::new(vec!["id".to_string()]),
            renderer_payload: None,
        }),
    ));
    let left = model.workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let right = model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"horizontal"}),
        )
        .unwrap()
        .unwrap()
        .response["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();

    let left_work = model
        .begin_object_filter_evaluation(
            &json!({"viewport_id":left,"mode":"query","query":"id == 'a'"}),
        )
        .unwrap();
    let right_work = model
        .begin_object_filter_evaluation(
            &json!({"viewport_id":right,"mode":"query","query":"id == 'b'"}),
        )
        .unwrap();
    let loading = model.loading_state();
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{left}:segmentation_objects")]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{right}:segmentation_objects")]["phase"],
        "pending"
    );

    assert!(
        model
            .install_object_filter_for_generation(
                left_work.0,
                left_work.1,
                left_work.2,
                &left_work.3,
                left_work.4,
                left_work.5,
                ControlObjectFilterResult {
                    model: left_work.7,
                    matching_indices: Arc::new(Vec::new()),
                    active: true,
                },
            )
            .is_some()
    );
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{left}:segmentation_objects")]["phase"],
        "ready"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("object_filter:{right}:segmentation_objects")]["phase"],
        "pending"
    );
    assert!(model.fail_object_filter_for_generation(
        &right_work.3,
        right_work.4,
        right_work.2,
        "Right filter failed",
    ));
    assert_eq!(model.loading_state()["loading"]["busy"], false);
}

#[test]
fn spatial_shape_resources_support_the_complete_object_compute_surface() {
    use crate::model::ControlObjectFeature;

    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let resource = Arc::new(ControlObjectResource {
        source: PathBuf::from("shapes/cells.parquet"),
        downsample_factor: 1.0,
        features: Arc::new(vec![ControlObjectFeature {
            id: "shape-1".to_string(),
            bbox_world: [1.0, 1.0, 9.0, 9.0],
            centroid_world: [5.0, 5.0],
            polygons_world: Arc::new(vec![vec![
                [1.0, 1.0],
                [9.0, 1.0],
                [9.0, 9.0],
                [1.0, 9.0],
                [1.0, 1.0],
            ]]),
            point_position_world: None,
            area_px: 64.0,
            perimeter_px: 32.0,
            properties: serde_json::Map::from_iter([("score".to_string(), json!(2.5))]),
        }]),
        property_names: Arc::new(vec!["id".to_string(), "score".to_string()]),
        renderer_payload: None,
    });
    model
        .install_document_object_layers(&[DocumentObjectLayerResource {
            layer_id: "spatial_shape:7".to_string(),
            name: "Cells".to_string(),
            kind: "spatial_shape".to_string(),
            primary: false,
            resource,
        }])
        .expect("secondary object layer installs");

    let target = json!({"target":"spatial_shape","layer_id":7});
    let properties = model
        .dispatch("viewer.objects.properties.list", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(properties["target"], "spatial_shape");
    assert_eq!(properties["layer_id"], 7);
    assert_eq!(properties["total"], 2);

    let selected = model
        .dispatch(
            "viewer.objects.select_rect",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "world_rect":[0.0,0.0,10.0,10.0],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(selected["objects"]["target"], "spatial_shape");
    assert_eq!(
        selected["objects"]["result"]["selection"]["selection_count"],
        1
    );

    model
        .dispatch(
            "viewer.objects.style.set",
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "fill_cells":true,
                "color_property":"score",
            }),
        )
        .unwrap()
        .unwrap();
    let style = model
        .dispatch("viewer.objects.style.get", &target)
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(style["fill_cells"], true);
    assert_eq!(style["color_property"], "score");

    let filter = model
        .begin_object_filter_evaluation(&json!({
            "target":"spatial_shape",
            "layer_id":7,
            "mode":"query",
            "query":"score > 2",
        }))
        .expect("spatial filter starts");
    assert_eq!(filter.4, ObjectTarget::SpatialShape(7));
    let filtered = model.install_object_filter_for_generation(
        filter.0,
        filter.1,
        filter.2,
        &filter.3,
        filter.4,
        filter.5,
        ControlObjectFilterResult {
            model: json!({"mode":"query","query":"score > 2"}),
            matching_indices: Arc::new(vec![0]),
            active: true,
        },
    );
    assert_eq!(filtered.unwrap()["result"]["visible_count"], 1);
    let projected_layer = model
        .dispatch(
            "viewer.native_layers.get",
            &json!({"layer_id":"spatial_shape:7"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        projected_layer["layer"]["presentation"]["objects"]["fill_cells"],
        true
    );
    assert_eq!(
        projected_layer["layer"]["presentation"]["objects"]["filter"]["query"],
        "score > 2"
    );

    let analysis = model
        .prepare_analysis_resource_operation(&target, "histogram")
        .expect("spatial analysis starts");
    assert_eq!(analysis.target, ObjectTarget::SpatialShape(7));
    assert_eq!(analysis.resource.features.len(), 1);

    let measurement = model
        .prepare_measurement(&target)
        .expect("spatial measurement starts");
    assert_eq!(measurement.target, ObjectTarget::SpatialShape(7));
    assert_eq!(measurement.target_indices.as_ref(), &[0]);

    let export = model
        .prepare_object_export(
            &json!({
                "target":"spatial_shape",
                "layer_id":7,
                "scope":"selected",
            }),
            PathBuf::from("spatial-shape.csv"),
            Some(ObjectExportFormat::Csv),
        )
        .expect("spatial export starts");
    assert_eq!(export.target, ObjectTarget::SpatialShape(7));
    assert_eq!(export.row_indices.as_ref(), &[0]);

    model
        .dispatch(
            "viewer.native_layers.set_active",
            &json!({"layer_id":"spatial_shape:7"}),
        )
        .unwrap()
        .unwrap();
    let active = model
        .dispatch("viewer.objects.get_selection", &json!({"target":"active"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(active["objects"]["target"], "spatial_shape");
    assert_eq!(active["objects"]["layer_id"], 7);
}

#[test]
fn mask_io_readiness_is_scoped_and_cancelled_by_document_replacement() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let (document_generation, mask_generation, import_generation, import_scope) =
        model.begin_mask_import_operation().unwrap();
    let (export_generation, export_scope) = model.begin_mask_export_operation().unwrap();
    assert!(model.finish_mask_io_for_generation(
        &export_scope,
        export_generation,
        "Mask export ready",
    ));
    let loading = model.loading_state();
    assert_eq!(loading["loading"]["busy"], true);
    assert_eq!(
        loading["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
        "pending"
    );
    assert_eq!(
        loading["loading"]["operations"][format!("mask_io:{export_scope}")]["phase"],
        "ready"
    );

    let layer = model
        .dispatch("viewer.masks.layers.create", &json!({"name":"Drawn"}))
        .unwrap()
        .unwrap()
        .response;
    model
        .dispatch(
            "viewer.masks.polygons.add",
            &json!({"id":layer["id"],"vertices":[[0,0],[4,0],[4,4]]}),
        )
        .unwrap()
        .unwrap();
    let (_, _, append_generation, append_scope, _) = model.begin_mask_append_operation().unwrap();
    let duplicate_append = model.begin_mask_append_operation().unwrap_err();
    assert_eq!(duplicate_append.kind, ControlErrorKind::NotReady);
    assert!(model.finish_mask_io_for_generation(
        &append_scope,
        append_generation,
        "Mask append ready",
    ));

    model.begin_dataset_open("replacement");
    assert_eq!(
        model.loading_state()["loading"]["operations"][format!("mask_io:{import_scope}")]["phase"],
        "cancelled"
    );
    assert!(
        model
            .install_imported_masks_for_generation(
                document_generation,
                mask_generation,
                import_generation,
                &import_scope,
                "stale".to_string(),
                true,
                None,
                vec![vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
                PathBuf::from("stale.geojson"),
            )
            .is_none()
    );
}

#[test]
fn actor_model_enforces_scoped_viewport_revision_guards() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let workspace = model.workspace_snapshot().unwrap();
    let id = workspace["active_viewport_id"].as_str().unwrap();
    let navigation = workspace["viewports"][0]["navigation_revision"]
        .as_u64()
        .unwrap();
    let presentation = workspace["viewports"][0]["presentation_revision"]
        .as_u64()
        .unwrap();

    model
        .dispatch(
            "viewer.viewports.camera.set",
            &json!({"viewport_id":id,"zoom":2.0,"if_navigation_revision":navigation}),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        model
            .dispatch(
                "viewer.viewports.camera.set",
                &json!({"viewport_id":id,"zoom":3.0,"if_navigation_revision":navigation}),
            )
            .unwrap()
            .unwrap_err()
            .kind,
        ControlErrorKind::Conflict
    );

    model
        .dispatch(
            "viewer.viewports.channels.set_color",
            &json!({"viewport_id":id,"channel":0,"color_rgb":[1,2,3],"if_presentation_revision":presentation}),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        model
            .dispatch(
                "viewer.viewports.channels.set_color",
                &json!({"viewport_id":id,"channel":0,"color_rgb":[3,2,1],"if_presentation_revision":presentation}),
            )
            .unwrap()
            .unwrap_err()
            .kind,
        ControlErrorKind::Conflict
    );
}

#[test]
fn stale_dataset_worker_results_cannot_replace_a_newer_document_request() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    let stale = model.begin_dataset_open("first");
    let loading_error = model
        .dispatch("viewer.workspace.get", &json!({}))
        .expect("actor-owned methods never fall back while loading")
        .unwrap_err();
    assert_eq!(loading_error.kind, ControlErrorKind::NotReady);
    assert_eq!(
        loading_error.data.as_ref().unwrap()["loading"]["resources_ready"],
        false
    );
    let current = model.begin_dataset_open("second");
    assert!(!model.install_dataset_for_generation(stale, &dataset, Vec::new(), None));
    assert_eq!(model.mode(), ModelMode::Transition);
    assert!(model.install_dataset_for_generation(current, &dataset, Vec::new(), None));
    assert_eq!(model.mode(), ModelMode::Single);
}

#[test]
fn unequal_splits_derive_each_viewport_from_the_retained_workspace_geometry() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let left = model.render_workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    model.report_viewport_geometry(&left, 0.0, 0.0, 1200.0, 800.0);
    let right = model
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"horizontal","ratio":0.6}),
        )
        .unwrap()
        .unwrap()
        .response["viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    let workspace = model.render_workspace_snapshot().unwrap();
    let viewport = |id: &str| {
        workspace["viewports"]
            .as_array()
            .unwrap()
            .iter()
            .find(|viewport| viewport["viewport_id"] == id)
            .unwrap()
    };
    assert_eq!(
        viewport(&left)["camera"]["viewport"],
        json!([0.0, 0.0, 720.0, 800.0])
    );
    let right_width = viewport(&right)["camera"]["viewport"][2].as_f64().unwrap();
    assert!((right_width - 480.0).abs() < 1.0e-3);
    assert_eq!(viewport(&right)["camera"]["viewport"][3], json!(800.0));

    model
        .dispatch("viewer.viewports.remove", &json!({"viewport_id":left}))
        .unwrap()
        .unwrap();
    let remaining = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        remaining["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1200.0, 800.0])
    );
}

#[test]
fn dataset_replacement_preserves_observed_logical_geometry() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    model.report_viewport_geometry("viewport-1", 0.0, 0.0, 1234.0, 777.0);
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "observed"
    );

    model.install_dataset(&dataset);
    let workspace = model.render_workspace_snapshot().unwrap();
    assert_eq!(
        workspace["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1234.0, 777.0])
    );
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "observed"
    );
}

#[test]
fn panel_changes_derive_background_geometry_without_a_frame() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    model.report_viewport_geometry("viewport-1", 0.0, 0.0, 1000.0, 700.0);

    let hidden = model
        .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(hidden["result"]["changed"], true);
    assert_eq!(
        hidden["result"]["panels"],
        json!({"left":false,"right":false})
    );
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["camera"]["viewport"],
        json!([0.0, 0.0, 1740.0, 700.0])
    );
    assert_eq!(
        model.loading_state()["loading"]["geometry"]["source"],
        "derived"
    );

    model
        .dispatch("viewer.camera.fit", &json!({}))
        .unwrap()
        .unwrap();
    let fitted = model
        .dispatch("viewer.camera.get", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert!(
        fitted["camera"]["zoom_screen_per_lvl0_px"]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn renderer_bootstrap_atomically_replaces_workspace_and_supersedes_workers() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut source = AppModel::project();
    source.install_dataset(&dataset);
    let left = source.render_workspace_snapshot().unwrap()["active_viewport_id"]
        .as_str()
        .unwrap()
        .to_string();
    source
        .dispatch(
            "viewer.viewports.clone",
            &json!({"source_viewport_id":left,"layout":"vertical","ratio":0.7,"title":"Native second"}),
        )
        .unwrap()
        .unwrap();
    source
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":left,"channels":[3],"mode":"only"}),
        )
        .unwrap()
        .unwrap();
    let renderer_workspace = source.render_workspace_snapshot().unwrap();

    let mut target = AppModel::project();
    let stale_generation = target.begin_dataset_open("superseded");
    target
        .bootstrap_dataset_from_renderer(&dataset, &renderer_workspace)
        .expect("native renderer state bootstraps atomically");
    assert!(!target.install_dataset_for_generation(stale_generation, &dataset, Vec::new(), None));
    assert_eq!(
        target.render_workspace_snapshot().unwrap(),
        renderer_workspace
    );
}

#[test]
fn plane_commands_retain_per_axis_slices_and_clamp_to_dataset_extents() {
    let (mut dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    dataset.dims.z = Some(1);
    dataset.dims.y = 2;
    dataset.dims.x = 3;
    dataset.dims.ndim = 4;
    for level in &mut dataset.levels {
        level.shape.insert(1, 7);
        level.chunks.insert(1, 1);
        level.scale.insert(1, 1.0);
        level.translation.insert(1, 0.0);
    }
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let set = |model: &mut AppModel, mode: &str, slice: u64| {
        model
            .dispatch(
                "viewer.viewports.planes.set",
                &json!({"viewport_id":"viewport-1","mode":mode,"slice":slice}),
            )
            .unwrap()
            .unwrap()
            .response
    };
    let xy = set(&mut model, "xy", 99);
    assert_eq!(xy["result"]["plane"]["slice"], 6);
    assert_eq!(xy["result"]["plane"]["extent"], 7);
    assert_eq!(
        xy["result"]["plane"]["supported_modes"],
        json!(["xy", "xz", "yz"])
    );

    let xz = set(&mut model, "xz", 1234);
    assert_eq!(xz["result"]["plane"]["slice"], 511);
    assert_eq!(xz["result"]["plane"]["slice_axis"], "y");
    let yz = set(&mut model, "yz", 42);
    assert_eq!(yz["result"]["plane"]["slice"], 42);
    assert_eq!(yz["result"]["plane"]["slice_axis"], "x");

    let back_to_xy = model
        .dispatch(
            "viewer.viewports.planes.set",
            &json!({"viewport_id":"viewport-1","mode":"xy"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(back_to_xy["result"]["plane"]["slice"], 6);
}

#[test]
fn invalid_presentation_commands_are_atomic() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let before = model.render_workspace_snapshot().unwrap();

    let invalid_mode = model
        .dispatch(
            "viewer.viewports.channels.set_visible",
            &json!({"viewport_id":"viewport-1","channels":[0],"mode":"toggle"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid_mode.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.render_workspace_snapshot().unwrap(), before);

    let invalid_rendering = model
        .dispatch(
            "viewer.viewports.rendering.set",
            &json!({"viewport_id":"viewport-1","show_hud":"yes"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid_rendering.kind, ControlErrorKind::InvalidParams);
    assert_eq!(model.render_workspace_snapshot().unwrap(), before);
}

#[test]
fn complete_channel_presentation_executes_and_roundtrips_without_a_renderer() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);

    let note = model
        .dispatch(
            "viewer.channels.set_note",
            &json!({"channel": 1, "note": "T-cell marker"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(note["channel"]["note"], "T-cell marker");

    let transform = model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({
                "viewport_id": "viewport-1",
                "if_presentation_revision": 1,
                "channel": 1,
                "offset_world": [12.5, -3.0],
                "scale": [1.25, 0.75],
                "rotation_rad": 0.5,
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(transform["changed"], true);
    assert_eq!(transform["transform"]["offset_world"], json!([12.5, -3.0]));
    assert_eq!(
        model.render_workspace_snapshot().unwrap()["viewports"][0]["presentation_revision"],
        2
    );
    let stale_transform = model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({
                "viewport_id":"viewport-1",
                "if_presentation_revision":1,
                "channel":1,
                "offset_world":[0.0,0.0],
            }),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(stale_transform.kind, ControlErrorKind::Conflict);

    let order = model
        .dispatch(
            "viewer.viewports.channels.set_order",
            &json!({
                "viewport_id": "viewport-1",
                "channels": [4, 3, 2, 1, 0],
                "mode": "exact",
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(order["result"]["order"][0]["index"], 4);

    let group = model
        .dispatch(
            "viewer.viewports.channels.set_group",
            &json!({
                "viewport_id": "viewport-1",
                "channels": [1, 2],
                "name": "Immune",
                "color_rgb": [10, 20, 30],
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(group["result"]["groups"][0]["name"], "Immune");
    assert_eq!(
        group["result"]["groups"][0]["members"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    model
        .dispatch(
            "viewer.channels.presentation.set",
            &json!({"search": "CD", "sort": "visible_first"}),
        )
        .unwrap()
        .unwrap();
    let projection = model.render_workspace_snapshot().unwrap();
    assert_eq!(projection["channel_presentation"]["search"], "CD");
    assert_eq!(projection["channel_transforms"][1]["rotation_rad"], 0.5);
    assert_eq!(
        projection["viewports"][0]["channel_order"],
        json!([4, 3, 2, 1, 0])
    );

    let mut restored = AppModel::project();
    restored
        .bootstrap_dataset_from_renderer(&dataset, &projection)
        .expect("complete presentation projection roundtrips");
    assert_eq!(restored.render_workspace_snapshot().unwrap(), projection);

    let before = restored.render_workspace_snapshot().unwrap();
    let invalid = restored
        .dispatch(
            "viewer.channels.set_transform",
            &json!({"channel": 1, "scale": [0.0, 1.0]}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(invalid.kind, ControlErrorKind::InvalidParams);
    assert_eq!(restored.render_workspace_snapshot().unwrap(), before);
}

#[test]
fn stale_renderer_observation_cannot_revert_actor_owned_state() {
    let (dataset, _) = OmeZarrDataset::open_local(&fixture()).expect("fixture");
    let mut model = AppModel::project();
    model.install_dataset(&dataset);
    let stale = model.render_workspace_snapshot().unwrap();

    model
        .dispatch(
            "viewer.channels.set_note",
            &json!({"channel":1,"note":"new actor value"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.set_transform",
            &json!({"channel":1,"offset_world":[8.0,-4.0]}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.viewports.channels.set_order",
            &json!({
                "viewport_id":"viewport-1",
                "channels":[4,3,2,1,0],
                "mode":"exact",
            }),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch(
            "viewer.channels.presentation.set",
            &json!({"search":"actor search","sort":"visible_first"}),
        )
        .unwrap()
        .unwrap();
    model
        .dispatch("viewer.panels.set", &json!({"left":false,"right":false}))
        .unwrap()
        .unwrap();
    let current_revision = model.mark_projection_dirty();

    assert!(model.observe_renderer_workspace(&stale, current_revision - 1));
    let current = model.render_workspace_snapshot().unwrap();
    assert_eq!(current["channel_metadata"][1]["note"], "new actor value");
    assert_eq!(
        current["channel_transforms"][1]["offset_world"],
        json!([8.0, -4.0])
    );
    assert_eq!(
        current["viewports"][0]["channel_order"],
        json!([4, 3, 2, 1, 0])
    );
    assert_eq!(current["channel_presentation"]["search"], "actor search");
    assert_eq!(current["panels"], json!({"left":false,"right":false}));

    assert!(!model.observe_renderer_workspace(&stale, current_revision + 1));
    let mut wrong_document = stale;
    wrong_document["shared_resources"]["dataset_source"] = json!("another dataset");
    assert!(!model.observe_renderer_workspace(&wrong_document, current_revision));
    assert_eq!(model.render_workspace_snapshot().unwrap(), current);
}

#[test]
fn project_metadata_and_roi_transactions_execute_without_a_renderer() {
    let mut model = AppModel::project();
    let created = model
        .dispatch("project.create", &json!({"default_dataset":"cohort-a"}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(
        created["project"]["metadata"]["default_dataset"],
        "cohort-a"
    );

    for (id, path) in [("roi-a", "/tmp/a.ome.zarr"), ("roi-b", "/tmp/b.ome.zarr")] {
        let added = model
            .dispatch(
                "project.rois.add",
                &json!({"id":id,"path":path,"metadata":{"group":"test"}}),
            )
            .unwrap()
            .unwrap()
            .response;
        assert_eq!(added["roi"]["id"], id);
    }
    let selected = model
        .dispatch(
            "project.rois.select",
            &json!({"ids":["roi-b"],"mode":"replace"}),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(selected["selected"], json!(["roi-b"]));
    assert_eq!(selected["focused"], "roi-b");

    let updated = model
        .dispatch(
            "project.rois.update",
            &json!({
                "target_id":"roi-b",
                "changes":{"display_name":"B","segmentation_path":"/tmp/b.parquet"},
            }),
        )
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(updated["roi"]["display_name"], "B");
    model
        .dispatch("project.rois.reorder", &json!({"ids":["roi-b","roi-a"]}))
        .unwrap()
        .unwrap();
    let rois = model
        .dispatch("project.rois.list", &json!({}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(rois["rois"][0]["id"], "roi-b");
    assert_eq!(rois["rois"][0]["selected"], true);

    let before = rois.clone();
    let duplicate = model
        .dispatch(
            "project.rois.add",
            &json!({"id":"roi-a","path":"/tmp/c.ome.zarr"}),
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(duplicate.kind, ControlErrorKind::Conflict);
    assert_eq!(
        model
            .dispatch("project.rois.list", &json!({}))
            .unwrap()
            .unwrap()
            .response,
        before
    );

    let stepped = model
        .dispatch("project.rois.next", &json!({"wrap":true}))
        .unwrap()
        .unwrap()
        .response;
    assert_eq!(stepped["focused"], "roi-a");
}

#[test]
fn project_create_installs_a_complete_typed_config_atomically() {
    let mut roi = crate::data::project_config::ProjectRoi {
        id: "roi-from-config".to_string(),
        display_name: Some("Configured ROI".to_string()),
        ..Default::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        PathBuf::from("/tmp/configured.ome.zarr"),
    ));
    let mut config = crate::data::project_config::ProjectConfig {
        default_dataset: Some("configured-dataset".to_string()),
        secondary_dataset: Some("secondary".to_string()),
        default_threshold_marker: Some("DAPI".to_string()),
        rois: vec![roi.clone()],
        ..Default::default()
    };
    config
        .mosaic_segmentation_search_roots
        .push(PathBuf::from("/tmp/segmentations"));
    config
        .datasets
        .insert("configured-dataset".to_string(), Default::default());
    config.control_resources = vec![json!({"id":"resource-from-config"})];

    let mut model = AppModel::project();
    let response = model
        .dispatch("project.create", &json!({"config":config}))
        .unwrap()
        .unwrap()
        .response;
    let snapshot = model.project_snapshot();

    assert_eq!(response["created"], true);
    assert_eq!(snapshot.load_generation, 1);
    assert_eq!(
        serde_json::to_value(&snapshot.rois).unwrap(),
        serde_json::to_value([roi]).unwrap()
    );
    assert_eq!(
        snapshot.default_dataset.as_deref(),
        Some("configured-dataset")
    );
    assert_eq!(snapshot.secondary_dataset.as_deref(), Some("secondary"));
    assert_eq!(snapshot.default_threshold_marker.as_deref(), Some("DAPI"));
    assert_eq!(
        snapshot.mosaic_segmentation_search_roots,
        vec![PathBuf::from("/tmp/segmentations")]
    );
    assert_eq!(snapshot.dataset_keys, vec!["configured-dataset"]);
    assert_eq!(snapshot.config.control_resources.len(), 1);
    assert!(!snapshot.dirty);
}

#[test]
fn renderer_project_bootstrap_cannot_revert_actor_owned_commits() {
    let mut roi = crate::data::project_config::ProjectRoi {
        id: "roi".to_string(),
        display_name: Some("Initial".to_string()),
        ..Default::default()
    };
    roi.set_dataset_source(crate::data::dataset_source::DatasetSource::Local(
        PathBuf::from("/tmp/bootstrap.ome.zarr"),
    ));
    let bootstrap = ProjectModelSnapshot {
        rois: vec![roi],
        ..ProjectModelSnapshot::default()
    };
    let mut model = AppModel::project();
    assert!(model.bootstrap_project_from_renderer(bootstrap.clone()));
    model
        .dispatch(
            "project.rois.update",
            &json!({"target_id":"roi","changes":{"display_name":"Actor"}}),
        )
        .unwrap()
        .unwrap();

    assert!(!model.bootstrap_project_from_renderer(bootstrap));
    assert_eq!(
        model.project_snapshot().rois[0].display_name.as_deref(),
        Some("Actor")
    );
}

#[test]
fn renderer_project_bootstrap_normalizes_persisted_state_and_saved_views() {
    let mut model = AppModel::project();
    assert!(model.bootstrap_project_from_renderer(ProjectModelSnapshot {
        state: json!({
            "browser": "invalid legacy value",
            "view_presets": [{
                "name": "Actor view",
                "description": "",
                "spec": {"channel_ref": {"label": "DAPI"}},
            }],
        }),
        ..ProjectModelSnapshot::default()
    }));

    let snapshot = model.project_snapshot();
    assert!(snapshot.state["browser"].is_object());
    assert_eq!(snapshot.view_count, 1);
    assert_eq!(snapshot.view_presets[0]["name"], "Actor view");

    let replacement = ProjectModelSnapshot {
        state: Value::String("invalid legacy state".to_string()),
        view_count: 99,
        ..ProjectModelSnapshot::default()
    };
    let mut replacement_model = AppModel::project();
    assert!(replacement_model.bootstrap_project_from_renderer(replacement));
    let replacement = replacement_model.project_snapshot();
    assert!(replacement.state.is_object());
    assert_eq!(replacement.view_count, 0);
}

#[test]
fn native_object_layer_replacement_applies_full_display_and_legend_state() {
    let mut objects = default_object_snapshot();
    let changed = apply_native_object_layer_presentation(
        &mut objects,
        &json!({
            "visible":true,
            "opacity":0.6,
            "width_screen_px":2.0,
            "color_rgb":[4,5,6],
            "show_selection_overlay":false,
            "display":{
                "color_property_key":"phenotype",
                "color_level_overrides":{
                    "tumour":{"visible":false,"color_rgb":[220,40,60]}
                },
                "fill_cells":true,
                "fill_opacity":0.45,
                "selected_fill_opacity":0.8,
                "fast_rendering":false
            },
            "filter":{"mode":"query","query":"ignored by presentation replacement"}
        }),
    )
    .expect("native object presentation is valid");

    assert!(changed);
    assert_eq!(objects["visible"], true);
    assert_eq!(objects["opacity"], json!(0.6_f32));
    assert_eq!(objects["fill_cells"], true);
    assert_eq!(objects["fill_opacity"], json!(0.45_f32));
    assert_eq!(objects["selected_fill_opacity"], json!(0.8_f32));
    assert_eq!(objects["fast_rendering"], false);
    assert_eq!(objects["color_property"], "phenotype");
    assert_eq!(
        objects["color_level_overrides"]["tumour"],
        json!({"visible":false,"color_rgb":[220,40,60]})
    );
    assert_eq!(
        objects["filter"],
        default_object_filter_model(),
        "filter evaluation remains on its worker-backed command path"
    );
}
